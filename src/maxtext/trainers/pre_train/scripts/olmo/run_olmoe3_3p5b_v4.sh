#!/bin/bash
#
# Copyright 2023-2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# OLMo 3.5 Small (Option A: olmoe3-3p5b-e256k8-sh2x, 63.1B total / 3.74B active,
# E=256, K=8, F=3584, 2x Shared F_sh=3584; and Baseline: olmoe3-3p5b, 62.9B total /
# 3.48B active, E=512, K=16, F=1792, 1x Shared F_sh=1792) Cloud TPU v4 training and
# benchmark launcher for MaxText (supports Pathways single-controller and McJAX).
#
# Verified 100% quality-neutral (strict FP32 master weights, router logits, KDA state,
# and KDA 4-tap conv; SEQ_LEN=8192) performance on Cloud TPU v4 with Pathways:
#   - 3 x v4-2048 (3,072 chips, 25.17M tok/step, DCN_DATA=3, ICI_DATA=8, ICI_EP=128):
#       * Option A (olmoe3-3p5b-e256k8-sh2x): 2,343.7 ms steady mean / 2,330.2 ms min
#         (30.95% steady / 31.13% peak MFU, 3,495-3,516 tok/s/chip, 10.74M-10.80M tok/s cluster,
#         4.3 min cold compile / 0.04 s warm compile cache hit, 36.3-62.0 s worker quick-restart)
#       * Baseline (olmoe3-3p5b): 2,676.0 ms steady mean (25.10% MFU, 3,061 tok/s/chip, 9.40M tok/s)
#   - 1 x v4-2048 (1,024 chips, 8.39M tok/step, DCN_DATA=1, ICI_DATA=8, ICI_EP=128):
#       * Option A (olmoe3-3p5b-e256k8-sh2x): 2,010.8 ms steady mean (36.08% MFU, 4,074 tok/s/chip)
#       * Baseline (olmoe3-3p5b): 2,465.0 ms steady mean (27.25% MFU, 3,323 tok/s/chip)
#   - 1 x v4-256 (128 chips, 1.05M tok/step, DCN_DATA=1, ICI_DATA=1, ICI_EP=128):
#       * Option A (olmoe3-3p5b-e256k8-sh2x): 1,860.0 ms steady mean (39.00% MFU, 4,404 tok/s/chip)
#       * Baseline (olmoe3-3p5b): 2,371.0 ms steady mean (28.33% MFU, 3,455 tok/s/chip)
#   - 1 x v4-6144 re-sliced as 8x8x48 (3,072 chips, 25.17M tok/step, Option A, strict FP32):
#       * TOPOLOGY=8x8x48 EP=64 (ICI_DATA=48, ICI_EP=64 on closed 8x8 X-Y planes, Zero-1 optimizer sharding):
#         Pathways + real Dolma 3.5: 2,117 ms steady mean (34.26% MFU, 11.89M tok/s)
#         McJAX + real Dolma 3.5:    2,150 ms steady mean (33.74% MFU, 11.71M tok/s; synthetic: 2,159 ms / 33.61% MFU)
#       * TOPOLOGY=8x8x48 EP=128 (ICI_DATA=24, ICI_EP=128 on (8,8,2) slabs, no Zero-1):
#         Pathways + real Dolma 3.5: 2,195 ms (33.06% MFU, 11.47M tok/s)
#         McJAX + real Dolma 3.5:    2,253 ms (32.20% MFU, 11.17M tok/s)
#
# The 3 x v4-2048 Option A run above, under Pathways:
#   MODEL_NAME=olmoe3-3p5b-e256k8-sh2x ENABLE_SINGLE_CONTROLLER=true NUM_SLICES=3 DCN_DATA_PARALLELISM=3 \
#   ICI_DATA_PARALLELISM=8 ICI_EXPERT_PARALLELISM=128 NORM_METRICS=none \
#   bash src/maxtext/trainers/pre_train/scripts/olmo/run_olmoe3_3p5b_v4.sh
#
# 8x8x48 recipe (TOPOLOGY=8x8x48) pins every flag of the measured runs:
#   RUNTIME=pathways|mcjax   Pathways single controller (default) or McJAX (one process per TPU VM host)
#   DATA=real|synthetic      real = Dolma 3.5 through dataset_type=olmo_grain (default), synthetic = no input data
#   MODE=bench|hero          bench = 20 steps, no checkpoints (default); hero = long run with async checkpoints
#   EP=64|128                EP=64 + Zero-1 is the best config (default); EP=128 is the (8,8,2) variant
# Example (Pathways hero run, from the jax-driver container of the Pathways head pod):
#   TOPOLOGY=8x8x48 RUNTIME=pathways DATA=real MODE=hero OUTPUT_DIR=gs://<bucket>/<path> RUN_NAME=<name> \
#     bash src/maxtext/trainers/pre_train/scripts/olmo/run_olmoe3_3p5b_v4.sh
# See README_v4_8x8x48.md next to this script for the Kubernetes manifests and the Pathways head-pod placement
# (at 3,072 chips run the head pod on a TPU VM host of the slice nodepool or on a CPU node with >= 128 GB RAM).

set -euo pipefail

MAXTEXT_ROOT="$(cd "$(dirname "$0")/../../../../../.." && pwd)"
VENV_PATH="${VENV_PATH:-${MAXTEXT_ROOT}/maxtext_venv}"

if [[ -d "${VENV_PATH}" ]]; then
  # shellcheck disable=SC1090,SC1091
  source "${VENV_PATH}/bin/activate"
fi

export PYTHONPATH="${MAXTEXT_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OLMOE3_FAST_SC_DEVICE_PUT="${OLMOE3_FAST_SC_DEVICE_PUT:-1}"
export OLMOE3_FAST_HOST_LOOP="${OLMOE3_FAST_HOST_LOOP:-1}"

# -------------------------- Configuration Defaults --------------------------
# Model variant:
#   - olmoe3-3p5b-e256k8-sh2x: Recommended Option A (E=256, K=8, F=3584, F_shared=3584; 63.1B total / 3.74B active)
#   - olmoe3-3p5b            : Baseline (E=512, K=16, F=1792, F_shared=1792; 62.9B total / 3.48B active)
#   - olmoe3-3p5b-e256k8     : 2x wider routed experts (E=256, K=8, F=3584, F_shared=1792; 62.9B total / 3.46B active)
#   - olmoe3-3p5b-e256k8-sh4x: Option B (E=256, K=8, F=3584, F_shared=7168; 63.7B total / 4.30B active)
MODEL_NAME="${MODEL_NAME:-olmoe3-3p5b}"
SEQ_LEN="${SEQ_LEN:-8192}"
RUN_NAME="${RUN_NAME:-${MODEL_NAME//-/_}_v4_L${SEQ_LEN}}"
DEFAULT_OUTPUT_DIR="gs://cloud-tpu-multipod-dev-maxtext-output/olmoe3_3p5b"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"
DATASET_TYPE="${DATASET_TYPE:-synthetic}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1.0}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
STEPS="${STEPS:-20}"
LOG_PERIOD="${LOG_PERIOD:-5}"

# Pathways single-controller mode (set ENABLE_SINGLE_CONTROLLER=true when running under Pathways proxy):
ENABLE_SINGLE_CONTROLLER="${ENABLE_SINGLE_CONTROLLER:-false}"
# -1 detects the slice count.
NUM_SLICES="${NUM_SLICES:--1}"

# Parallelism (the defaults are the single-slice v4-128 setup; set these per topology):
#   - 3 x v4-2048 (3,072 chips, 3 slices of 8x8x16): NUM_SLICES=3, DCN_DATA=3, ICI_DATA=8, ICI_FSDP=1, ICI_EP=128
#   - 1 x v4-6144 (3,072 chips, 1 slice of 12x16x16): NUM_SLICES=1, DCN_DATA=1, ICI_DATA=24, ICI_FSDP=1, ICI_EP=128
#   - 1 x v4-2048 (1,024 chips, 1 slice of 8x8x16)  : NUM_SLICES=1, DCN_DATA=1, ICI_DATA=8, ICI_FSDP=1, ICI_EP=128
#   - 1 x v4-256  (128 chips, 1 slice of 4x4x8)     : NUM_SLICES=1, DCN_DATA=1, ICI_DATA=1, ICI_FSDP=1, ICI_EP=128
#   - 1 x v4-128  (64 chips, 1 slice of 4x4x4)      : NUM_SLICES=1, DCN_DATA=1, ICI_DATA=1, ICI_FSDP=1, ICI_EP=64
#   - 1 x v4-6144 as 8x8x48 (3,072 chips)          : TOPOLOGY=8x8x48 (EP=64: ICI_DATA=48 + Zero-1; EP=128: ICI_DATA=24)
TOPOLOGY="${TOPOLOGY:-}"
PRESET_ARGS=()
# Megablox tile dimensions along the routed-expert width F, WI_TILE_HALF = F / 4 and WI_TILE_FULL = F / 2 of the
# e256k8-sh2x models (3584 for olmoe3-3p5b-e256k8-sh2x, 3072 for olmo35-small-e256k8-sh2x). The 896 / 1792 pair also
# divides the baseline routed width (F=1792); all fit TPU v4's 16 MiB VMEM, and 896 / 1792 are the tiles verified on
# 1x and 3x v4-2048. Set once MODEL_NAME is final (the 8x8x48 preset can change it).
set_routed_tiles() {
  case "${MODEL_NAME}" in
    olmo35-small-e256k8-sh2x) WI_TILE_HALF=768; WI_TILE_FULL=1536 ;;
    *) WI_TILE_HALF=896; WI_TILE_FULL=1792 ;;
  esac
}
# XLA flags of the measured 8x8x48 runs. Under Pathways they must also be passed to the pathways resource-manager,
# proxy, and worker containers (see the JobSet templates next to this script); the TPUs ignore this driver's env.
V4_8X8X48_LIBTPU="--xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_bf16_emission_mode=NATIVE_EMISSION \
--xla_tpu_scoped_vmem_limit_kib=16384 --xla_tpu_enable_async_ragged_all_to_all=true \
--xla_tpu_enable_all_reduce_scatter_fusion=true --xla_tpu_use_minor_sharding_for_major_trivial_input=true \
--xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1 --xla_tpu_assign_all_reduce_scatter_layout=true"
case "${TOPOLOGY}" in
  "") ;;
  8x8x48)
    RUNTIME="${RUNTIME:-pathways}"
    DATA="${DATA:-real}"
    MODE="${MODE:-bench}"
    EP="${EP:-64}"
    case "${RUNTIME}" in
      pathways) ENABLE_SINGLE_CONTROLLER=true ;;
      mcjax) ENABLE_SINGLE_CONTROLLER=false ;;
      *) echo "RUNTIME must be pathways or mcjax, got ${RUNTIME}" >&2; exit 1 ;;
    esac
    # PRESET_MODEL_NAME swaps the model at the same recipe (e.g. olmo35-small-e256k8-sh2x, the 7:1 small, for an A/B).
    MODEL_NAME="${PRESET_MODEL_NAME:-olmoe3-3p5b-e256k8-sh2x}"
    set_routed_tiles
    SEQ_LEN=8192
    PER_DEVICE_BATCH=1.0
    GRAD_ACCUM_STEPS=1
    NUM_SLICES=1
    DCN_DATA_PARALLELISM=1
    ICI_FSDP_PARALLELISM=1
    case "${EP}" in
      64)
        # Expert groups are closed 8x8 X-Y planes (maxtext_utils.tpu_8x8xz_expert_mesh); DP=48 runs along Z.
        # Zero-1 under Pathways relies on the on-device reshard in sharding.reshard_train_state.
        ICI_DATA_PARALLELISM=48
        ICI_EXPERT_PARALLELISM=64
        PRESET_ARGS+=("shard_optimizer_over_data=true")
        ;;
      128)
        # Expert groups are (8, 8, 2) slabs; DP=24 runs along Z.
        ICI_DATA_PARALLELISM=24
        ICI_EXPERT_PARALLELISM=128
        ;;
      *) echo "EP must be 64 or 128 for TOPOLOGY=8x8x48, got ${EP}" >&2; exit 1 ;;
    esac
    MOE_A2A_TOKEN_CHUNKS=2
    # PRESET_RAGGED_BUFFER_FACTOR sizes the all-to-all receive buffer (rows beyond it are dropped; add
    # moe_log_a2a_drops=true to log how many).
    RAGGED_BUFFER_FACTOR="${PRESET_RAGGED_BUFFER_FACTOR:-1.0625}"
    BF16_KDA_STATE=false
    NORM_METRICS=none
    LIBTPU_OVERRIDE="${V4_8X8X48_LIBTPU}"
    # The measured runs used XLA's reduce-scatter fusion (the flags above), not the manual 2-stage reduce-scatter in
    # linears.dot_general_with_gathered_kernel, which is otherwise automatic at DP >= 8.
    export OLMOE3_ENABLE_2STAGE_RS="${OLMOE3_ENABLE_2STAGE_RS:-0}"
    TPU_MEGACORE="${TPU_MEGACORE:-megacore_dense}"
    PRESET_ARGS+=(
      "float32_gate_logits=true"
      "moe_router_logits=device" "mlpwi=device" "mlpwo=device" "out_proj=device" "attention_out=device"
      "query_proj=device" "key_proj=device" "value_proj=device" "gathered_weights=device"
      "kda_fused=remat" "kda_fused_state_stride=0" "kda_flat_head_layout=true" "kda_fused_proj_reduce_scatter=true"
      "olmoe3_replicate_router=true" "olmoe3_segments_for_emo_only=true"
      "moe_fused_swiglu_gmm=true" "moe_a2a_pipeline_chunks=false"
      "moe_fused_swiglu_bwd_tile_batch_seq=256" "moe_fused_swiglu_bwd_tile_embed_dim=768"
      "moe_fused_swiglu_bwd_tile_mlp_dim=${WI_TILE_HALF}" "moe_fused_swiglu_dwo_tile_embed_dim=768"
      "moe_fused_swiglu_dwo_tile_mlp_dim=${WI_TILE_HALF}"
      "fused_lm_head_ce=true" "fused_lm_head_ce_block_t=512" "fused_lm_head_ce_block_v=1024"
      "fused_lm_head_ce_block_t_bwd=512" "fused_lm_head_ce_block_v_bwd=128" "logits_dot_in_fp32=false"
      "colocated_python_data_input=false"
    )
    if [[ "${RUNTIME}" == "mcjax" ]]; then
      PRESET_ARGS+=("jax_distributed_initialization_timeout=600")
    fi
    case "${DATA}" in
      real)
        # Dolma 3.5 (OLMo-core order) through gcsfuse mounts; override the paths for other buckets.
        DATASET_TYPE=olmo_grain
        INDEX_PATH="${INDEX_PATH:-/mnt/olmo-3p5-maxtext/indices/dolma3p5-14t_seq8192.json}"
        GCS_BASE="${GCS_BASE:-gs://pretraining-data/dolma3p5-14t-e6c5a51bd6b9/}"
        LOCAL_MOUNT="${LOCAL_MOUNT:-/mnt/dolma3p5-14t/}"
        GLOBAL_INDICES_PATH="${GLOBAL_INDICES_PATH:-/mnt/olmo-3p5-maxtext/indices/global_indices_dataset_size1708983195_epoch1_seed928543231_v1.npy}"
        DATA_SHUFFLE_SEED="${DATA_SHUFFLE_SEED:-928543231}"
        PRESET_ARGS+=(
          "olmo_global_indices_path=${GLOBAL_INDICES_PATH}"
          "data_shuffle_seed=${DATA_SHUFFLE_SEED}"
          "reuse_example_batch=0"
          "synthetic_data_reuse_batch=false"
        )
        if [[ "${RUNTIME}" == "pathways" ]]; then
          # One driver process loads the whole 25.17M-token batch.
          PRESET_ARGS+=("grain_worker_count=${GRAIN_WORKER_COUNT:-8}" "grain_per_worker_buffer_size=1")
        else
          PRESET_ARGS+=("grain_worker_count=${GRAIN_WORKER_COUNT:-1}" "grain_per_worker_buffer_size=4")
        fi
        ;;
      synthetic)
        DATASET_TYPE=synthetic
        PRESET_ARGS+=("reuse_example_batch=1")
        ;;
      *) echo "DATA must be real or synthetic, got ${DATA}" >&2; exit 1 ;;
    esac
    case "${MODE}" in
      bench) ;;
      hero)
        if [[ "${OUTPUT_DIR}" != gs://* || "${OUTPUT_DIR}" == "${DEFAULT_OUTPUT_DIR}" ]]; then
          echo "MODE=hero needs your own OUTPUT_DIR=gs://<bucket>/<path> for checkpoints" >&2
          exit 1
        fi
        STEPS="${HERO_STEPS:-100000}"
        LOG_PERIOD="${LOG_PERIOD_HERO:-100}"
        PRESET_ARGS+=(
          "enable_checkpointing=true" "async_checkpointing=true" "enable_orbax_v1=false"
          "save_checkpoint_on_completion=true" "checkpoint_period=${CHECKPOINT_PERIOD:-1000}"
        )
        # Profiling 768 Pathways workers currently crashes the slice; keep the profiler off for Pathways hero runs.
        if [[ "${RUNTIME}" == "pathways" ]]; then
          PROFILER=""
        fi
        ;;
      *) echo "MODE must be bench or hero, got ${MODE}" >&2; exit 1 ;;
    esac
    ;;
  *)
    echo "Unknown TOPOLOGY=${TOPOLOGY} (supported: 8x8x48; leave unset for the manual parallelism variables)" >&2
    exit 1
    ;;
esac
DCN_DATA_PARALLELISM="${DCN_DATA_PARALLELISM:-1}"
ICI_DATA_PARALLELISM="${ICI_DATA_PARALLELISM:-1}"
ICI_FSDP_PARALLELISM="${ICI_FSDP_PARALLELISM:-1}"
ICI_EXPERT_PARALLELISM="${ICI_EXPERT_PARALLELISM:-64}"

# Precision mode:
#   Default (BF16_KDA_STATE=false): 100% quality-neutral strict FP32 KDA state & conv (30.95%-31.13% MFU on 3x v4-2048)
#   BF16_KDA_STATE=true: bf16 KDA state & conv (requires 10B-token ablation validation)
BF16_KDA_STATE="${BF16_KDA_STATE:-false}"
if [[ "${BF16_KDA_STATE}" == "true" ]]; then
  GDN_STATE_DTYPE="${GDN_STATE_DTYPE:-bfloat16}"
  KDA_CONV_IN_COMPUTE_DTYPE="${KDA_CONV_IN_COMPUTE_DTYPE:-true}"
else
  GDN_STATE_DTYPE="${GDN_STATE_DTYPE:-float32}"
  KDA_CONV_IN_COMPUTE_DTYPE="${KDA_CONV_IN_COMPUTE_DTYPE:-false}"
fi

KDA_PALLAS_FUSED="${KDA_PALLAS_FUSED:-true}"
KDA_FUSED_TOKEN_MAJOR="${KDA_FUSED_TOKEN_MAJOR:-true}"
KDA_OUTER_BLOCK="${KDA_OUTER_BLOCK:-32}"
KDA_INVERSE_PRECISION="${KDA_INVERSE_PRECISION:-high}"

if (( SEQ_LEN <= 4096 )); then
  MOE_A2A_TOKEN_CHUNKS="${MOE_A2A_TOKEN_CHUNKS:-2}"
  NUM_VOCAB_TILING="${NUM_VOCAB_TILING:-1}"
  VOCAB_TILING_AG_ONCE="${VOCAB_TILING_AG_ONCE:-false}"
elif [[ "${MODEL_NAME}" == *"e256k8"* ]]; then
  # At K=8 (65,536 routed tokens/layer at 8k vs. 131,072 at K=16), 2 chunks give the
  # same per-chunk buffer as 4 chunks at K=16 and compile 2.2x-2.4x faster.
  MOE_A2A_TOKEN_CHUNKS="${MOE_A2A_TOKEN_CHUNKS:-2}"
  NUM_VOCAB_TILING="${NUM_VOCAB_TILING:-4}"
  VOCAB_TILING_AG_ONCE="${VOCAB_TILING_AG_ONCE:-true}"
else
  MOE_A2A_TOKEN_CHUNKS="${MOE_A2A_TOKEN_CHUNKS:-4}"
  NUM_VOCAB_TILING="${NUM_VOCAB_TILING:-4}"
  VOCAB_TILING_AG_ONCE="${VOCAB_TILING_AG_ONCE:-true}"
fi
RAGGED_BUFFER_FACTOR="${RAGGED_BUFFER_FACTOR:-1.125}"

EMO_ENABLED="${EMO_ENABLED:-true}"
# norm_metrics=none skips the grad/param norm reductions (faster under Pathways).
NORM_METRICS="${NORM_METRICS:-grad}"

# Profiler (set PROFILER=xplane to capture .xplane.pb hardware trace):
PROFILER="${PROFILER:-}"
SKIP_FIRST_N_STEPS_FOR_PROFILER="${SKIP_FIRST_N_STEPS_FOR_PROFILER:-5}"
PROFILER_STEPS="${PROFILER_STEPS:-3}"

# TPU v4 MegaCore & XLA / libtpu flags:
export TPU_MEGACORE="${TPU_MEGACORE:-MEGACORE_DENSE}"
LIBTPU="--xla_tpu_spmd_rng_bit_generator_unsafe=true   --xla_tpu_bf16_emission_mode=NATIVE_EMISSION   --xla_tpu_scoped_vmem_limit_kib=16384   --xla_tpu_enable_async_ragged_all_to_all=true   --xla_tpu_overlap_compute_collective_tc=true   --xla_enable_async_all_gather=true   --xla_enable_async_collective_permute=true   --xla_jf_rematerialization_percent_shared_memory_limit=98   --xla_tpu_user_reserved_hbm_bytes=0"
if (( SEQ_LEN <= 4096 )); then
  LIBTPU="${LIBTPU} --xla_enable_async_all_gather=false"
fi
if [[ -n "${LIBTPU_OVERRIDE:-}" ]]; then
  LIBTPU="${LIBTPU_OVERRIDE}"
fi
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-} ${LIBTPU}"

set_routed_tiles

echo "=== OLMo 3.5 Small (${MODEL_NAME}) TPU v4 Run ==="
echo "  model_name               : ${MODEL_NAME}"
echo "  run_name                 : ${RUN_NAME}"
echo "  output_dir               : ${OUTPUT_DIR}"
echo "  dataset_type             : ${DATASET_TYPE}"
echo "  seq_len                  : ${SEQ_LEN}"
echo "  per_device_batch_size    : ${PER_DEVICE_BATCH}"
echo "  enable_single_controller : ${ENABLE_SINGLE_CONTROLLER}"
echo "  num_slices               : ${NUM_SLICES}"
echo "  dcn_data_parallelism     : ${DCN_DATA_PARALLELISM}"
echo "  ici_data_parallelism     : ${ICI_DATA_PARALLELISM}"
echo "  ici_fsdp_parallelism     : ${ICI_FSDP_PARALLELISM}"
echo "  ici_expert_parallelism   : ${ICI_EXPERT_PARALLELISM}"
echo "  topology preset          : ${TOPOLOGY:-<none>} (runtime=${RUNTIME:-}, data=${DATA:-}, mode=${MODE:-}, ep=${EP:-})"
echo "  gdn_state_dtype          : ${GDN_STATE_DTYPE} (conv_bf16=${KDA_CONV_IN_COMPUTE_DTYPE}, pallas_fused=${KDA_PALLAS_FUSED}, tm=${KDA_FUSED_TOKEN_MAJOR}, ob=${KDA_OUTER_BLOCK})"
echo "  moe_a2a_token_chunks     : ${MOE_A2A_TOKEN_CHUNKS} (rbf=${RAGGED_BUFFER_FACTOR})"
echo "  emo_enabled              : ${EMO_ENABLED}"
echo "  norm_metrics             : ${NORM_METRICS}"
echo "  profiler                 : ${PROFILER:-<disabled>}"
echo

EXTRA_ARGS=()
if (( SEQ_LEN <= 4096 )); then
  EXTRA_ARGS+=("moe_mlpwi_0=device")
else
  # At EP=128 (25.49 GiB / 31.75 GiB HBM), saving kda_wy, kda_proj, mlpwi_0, and mlpwi_1
  # on device eliminates backward recomputation of KDA input projections and shared-expert
  # SwiGLU branches (-55 to -86 ms/step on 1x and 3x v4-2048).
  EXTRA_ARGS+=(
    "kda_wy=device"
    "kda_proj=device"
    "mlpwi_0=device"
    "mlpwi_1=device"
  )
fi
if [[ -n "${PROFILER}" ]]; then
  EXTRA_ARGS+=("profiler=${PROFILER}")
fi
if [[ "${DATASET_TYPE}" == "olmo_grain" ]]; then
  : "${INDEX_PATH:?INDEX_PATH is required when DATASET_TYPE=olmo_grain}"
  : "${GCS_BASE:?GCS_BASE is required when DATASET_TYPE=olmo_grain}"
  : "${LOCAL_MOUNT:?LOCAL_MOUNT is required when DATASET_TYPE=olmo_grain}"
  EXTRA_ARGS+=(
    "olmo_index_path=${INDEX_PATH}"
    "olmo_path_remap_from=${GCS_BASE%/}/"
    "olmo_path_remap_to=${LOCAL_MOUNT%/}/"
    "olmo_apply_ngram_filter=True"
  )
fi

python3 -m maxtext.trainers.pre_train.train   "${MAXTEXT_ROOT}/src/maxtext/configs/base.yml"   model_name="${MODEL_NAME}"   override_model_config=true   run_name="${RUN_NAME}"   base_output_directory="${OUTPUT_DIR}"   dataset_type="${DATASET_TYPE}"   enable_single_controller="${ENABLE_SINGLE_CONTROLLER}"   num_slices="${NUM_SLICES}"   steps="${STEPS}"   log_period="${LOG_PERIOD}"   per_device_batch_size="${PER_DEVICE_BATCH}"   gradient_accumulation_steps="${GRAD_ACCUM_STEPS}"   max_target_length="${SEQ_LEN}"   dcn_data_parallelism="${DCN_DATA_PARALLELISM}"   ici_data_parallelism="${ICI_DATA_PARALLELISM}"   ici_fsdp_parallelism="${ICI_FSDP_PARALLELISM}"   ici_expert_parallelism="${ICI_EXPERT_PARALLELISM}"   shard_exp_on_fsdp=false   allow_split_physical_axes=true   scan_layers=false   olmoe3_per_layer_remat=true   remat_policy=custom   decoder_layer_input=device   moe_routing=device   moe_x_sorted=device   moe_combine=device   num_vocab_tiling="${NUM_VOCAB_TILING}"   vocab_tiling_ag_once="${VOCAB_TILING_AG_ONCE}"   vocab_tiling_seq_major=true   moe_global_load_balancing=true   dtype=bfloat16   weight_dtype=float32   use_tokamax_kda=false   kda_chunked_impl=subblock   kda_sub_block=16   gdn_chunk_size=128   gdn_state_dtype="${GDN_STATE_DTYPE}"   kda_conv_in_compute_dtype="${KDA_CONV_IN_COMPUTE_DTYPE}"   kda_pallas_fused="${KDA_PALLAS_FUSED}"   kda_fused_token_major="${KDA_FUSED_TOKEN_MAJOR}"   kda_outer_block="${KDA_OUTER_BLOCK}"   kda_inverse_precision="${KDA_INVERSE_PRECISION}"   kda_fused_input_proj=true   capacity_factor=-1   moe_lean_routing=true   moe_topk_pallas=true   emo_threshold_by_bisection=true   moe_a2a_expert_major=true   moe_a2a_token_chunks="${MOE_A2A_TOKEN_CHUNKS}"   ragged_buffer_factor="${RAGGED_BUFFER_FACTOR}"   megablox=true   sparse_matmul=true   use_tokamax_gmm=false   use_gmm_v2=false   wi_tile_fwd_batch_seq=512   wi_tile_fwd_embed_dim=768   wi_tile_fwd_mlp_dim="${WI_TILE_HALF}"   wi_tile_dlhs_batch_seq=512   wi_tile_dlhs_mlp_dim="${WI_TILE_FULL}"   wi_tile_dlhs_embed_dim=384   wi_tile_drhs_batch_seq=512   wi_tile_drhs_embed_dim=768   wi_tile_drhs_mlp_dim="${WI_TILE_HALF}"   wo_tile_fwd_batch_seq=512   wo_tile_fwd_mlp_dim="${WI_TILE_FULL}"   wo_tile_fwd_embed_dim=384   wo_tile_dlhs_batch_seq=512   wo_tile_dlhs_embed_dim=768   wo_tile_dlhs_mlp_dim="${WI_TILE_HALF}"   wo_tile_drhs_batch_seq=512   wo_tile_drhs_mlp_dim="${WI_TILE_HALF}"   wo_tile_drhs_embed_dim=384   attention=flash   emo_enabled="${EMO_ENABLED}"   norm_metrics="${NORM_METRICS}"   synthetic_data_reuse_batch=true   skip_first_n_steps_for_profiler="${SKIP_FIRST_N_STEPS_FOR_PROFILER}"   profiler_steps="${PROFILER_STEPS}"   upload_all_profiler_results=false   profile_cleanly=false   learning_rate=1e-5   abort_on_nan_loss=false   abort_on_inf_loss=false   enable_checkpointing=false   async_checkpointing=false   enable_tensorboard=false   enable_goodput_recording=false   monitor_goodput=false   enable_checkpoint_cloud_logger=false   "${EXTRA_ARGS[@]}"   "${PRESET_ARGS[@]}"   "$@"
