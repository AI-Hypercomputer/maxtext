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
#
# The 3 x v4-2048 Option A run above, under Pathways:
#   MODEL_NAME=olmoe3-3p5b-e256k8-sh2x ENABLE_SINGLE_CONTROLLER=true NUM_SLICES=3 DCN_DATA_PARALLELISM=3 \
#   ICI_DATA_PARALLELISM=8 ICI_EXPERT_PARALLELISM=128 NORM_METRICS=none \
#   bash src/maxtext/trainers/pre_train/scripts/olmo/run_olmoe3_3p5b_v4.sh

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
OUTPUT_DIR="${OUTPUT_DIR:-gs://cloud-tpu-multipod-dev-maxtext-output/olmoe3_3p5b}"
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
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-} ${LIBTPU}"

# Megablox tile dimensions: the 256-aligned 896 / 1792 N-tiles divide both the baseline
# routed width (F=1792) and the 2x wider routed width (F=3584: 3584 = 4 x 896 = 2 x 1792),
# fit TPU v4's 16 MiB VMEM, and are the tiles verified on 1x and 3x v4-2048.
WI_TILE_HALF=896
WI_TILE_FULL=1792

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

python3 -m maxtext.trainers.pre_train.train   "${MAXTEXT_ROOT}/src/maxtext/configs/base.yml"   model_name="${MODEL_NAME}"   override_model_config=true   run_name="${RUN_NAME}"   base_output_directory="${OUTPUT_DIR}"   dataset_type="${DATASET_TYPE}"   enable_single_controller="${ENABLE_SINGLE_CONTROLLER}"   num_slices="${NUM_SLICES}"   steps="${STEPS}"   log_period="${LOG_PERIOD}"   per_device_batch_size="${PER_DEVICE_BATCH}"   gradient_accumulation_steps="${GRAD_ACCUM_STEPS}"   max_target_length="${SEQ_LEN}"   dcn_data_parallelism="${DCN_DATA_PARALLELISM}"   ici_data_parallelism="${ICI_DATA_PARALLELISM}"   ici_fsdp_parallelism="${ICI_FSDP_PARALLELISM}"   ici_expert_parallelism="${ICI_EXPERT_PARALLELISM}"   shard_exp_on_fsdp=false   allow_split_physical_axes=true   scan_layers=false   olmoe3_per_layer_remat=true   remat_policy=custom   decoder_layer_input=device   moe_routing=device   moe_x_sorted=device   moe_combine=device   num_vocab_tiling="${NUM_VOCAB_TILING}"   vocab_tiling_ag_once="${VOCAB_TILING_AG_ONCE}"   vocab_tiling_seq_major=true   moe_global_load_balancing=true   dtype=bfloat16   weight_dtype=float32   use_tokamax_kda=false   kda_chunked_impl=subblock   kda_sub_block=16   gdn_chunk_size=128   gdn_state_dtype="${GDN_STATE_DTYPE}"   kda_conv_in_compute_dtype="${KDA_CONV_IN_COMPUTE_DTYPE}"   kda_pallas_fused="${KDA_PALLAS_FUSED}"   kda_fused_token_major="${KDA_FUSED_TOKEN_MAJOR}"   kda_outer_block="${KDA_OUTER_BLOCK}"   kda_inverse_precision="${KDA_INVERSE_PRECISION}"   kda_fused_input_proj=true   capacity_factor=-1   moe_lean_routing=true   moe_topk_pallas=true   emo_threshold_by_bisection=true   moe_a2a_expert_major=true   moe_a2a_token_chunks="${MOE_A2A_TOKEN_CHUNKS}"   ragged_buffer_factor="${RAGGED_BUFFER_FACTOR}"   megablox=true   sparse_matmul=true   use_tokamax_gmm=false   use_gmm_v2=false   wi_tile_fwd_batch_seq=512   wi_tile_fwd_embed_dim=768   wi_tile_fwd_mlp_dim="${WI_TILE_HALF}"   wi_tile_dlhs_batch_seq=512   wi_tile_dlhs_mlp_dim="${WI_TILE_FULL}"   wi_tile_dlhs_embed_dim=384   wi_tile_drhs_batch_seq=512   wi_tile_drhs_embed_dim=768   wi_tile_drhs_mlp_dim="${WI_TILE_HALF}"   wo_tile_fwd_batch_seq=512   wo_tile_fwd_mlp_dim="${WI_TILE_FULL}"   wo_tile_fwd_embed_dim=384   wo_tile_dlhs_batch_seq=512   wo_tile_dlhs_embed_dim=768   wo_tile_dlhs_mlp_dim="${WI_TILE_HALF}"   wo_tile_drhs_batch_seq=512   wo_tile_drhs_mlp_dim="${WI_TILE_HALF}"   wo_tile_drhs_embed_dim=384   attention=flash   emo_enabled="${EMO_ENABLED}"   norm_metrics="${NORM_METRICS}"   synthetic_data_reuse_batch=true   skip_first_n_steps_for_profiler="${SKIP_FIRST_N_STEPS_FOR_PROFILER}"   profiler_steps="${PROFILER_STEPS}"   upload_all_profiler_results=false   profile_cleanly=false   learning_rate=1e-5   abort_on_nan_loss=false   abort_on_inf_loss=false   enable_checkpointing=false   async_checkpointing=false   enable_tensorboard=false   enable_goodput_recording=false   monitor_goodput=false   enable_checkpoint_cloud_logger=false   "${EXTRA_ARGS[@]}"   "$@"
