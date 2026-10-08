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
# OLMo 3.5 Small (olmoe3-3p5b: 62.9B total / 3.48B active, hybrid 4:1 KDA + Full Attention,
# 512-expert top-16 LatentMoE + EMo) Cloud TPU v4 (v4-128 / 4x4x4, v4-256 / 4x4x8,
# v4-1024 / 8x8x8, v4-2048 / 8x8x16) training and benchmark launcher for MaxText.
#
# Verified performance on Cloud TPU v4-128 (64 chips, 4x4x4 torus):
#   - With fused Pallas token-major KDA kernel (kda_pallas_fused=true,
#     kda_fused_token_major=true, kda_outer_block=32, kda_inverse_precision=high):
#       * SEQ_LEN=4096 (bf16 KDA state): 1.194 s / step (70.4 TF/s/dev, 25.6% logged MFU / 27.8% guide MFU, 3,430 tok/s/chip)
#       * SEQ_LEN=8192 (bf16 KDA state): 2.403 s / step (71.0 TF/s/dev, 25.8% logged MFU / 28.0% guide MFU, 3,409 tok/s/chip)
#       * SEQ_LEN=8192 (strict FP32 KDA state & conv, BF16_KDA_STATE=false): 2.559 s / step (66.7 TF/s/dev, 24.3% logged MFU / 26.2% guide MFU, 3,201 tok/s/chip)
#
# Key optimizations:
#   - Fused Pallas KDA kernel (kernels/kda_fused.py + kernels/kda_intra.py):
#     kda_pallas_fused=true, kda_fused_token_major=true, kda_outer_block=32,
#     kda_inverse_precision=high, gdn_chunk_size=128, kda_fused_input_proj=true.
#   - Unrolled layers with per-layer remat boundary (scan_layers=false,
#     olmoe3_per_layer_remat=true, remat_policy=custom) and both MoE saves
#     (moe_x_sorted=device, moe_combine=device, moe_routing=device,
#     decoder_layer_input=device), skipping both the dispatch and combine
#     ragged_all_to_all collectives in backward recomputation.
#   - Megablox Pallas GMM/TGMM (megablox=true, sparse_matmul=true) with two N-tiles
#     per call (896 for wi, 384 for wo, m=512) so grid[0]=2 ("parallel") utilizes both
#     MegaCore TensorCores concurrently (~83% of MXU peak).
#   - Expert-major ragged all-to-all (moe_a2a_expert_major=true) with sequence token
#     chunking (moe_a2a_token_chunks=2 at 4k, 4 at 8k) and async ragged all-to-all
#     (--xla_tpu_enable_async_ragged_all_to_all=true).
#   - Lean routing with Pallas Top-K and bisection EMo threshold (moe_lean_routing=true,
#     moe_topk_pallas=true, emo_threshold_by_bisection=true) and single-pass grad norm
#     (norm_metrics=grad).

set -euo pipefail

MAXTEXT_ROOT="$(cd "$(dirname "$0")/../../../../../.." && pwd)"
VENV_PATH="${VENV_PATH:-${MAXTEXT_ROOT}/maxtext_venv}"

if [[ -d "${VENV_PATH}" ]]; then
  # shellcheck disable=SC1090,SC1091
  source "${VENV_PATH}/bin/activate"
fi

export PYTHONPATH="${MAXTEXT_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# -------------------------- Configuration Defaults --------------------------
SEQ_LEN="${SEQ_LEN:-8192}"
RUN_NAME="${RUN_NAME:-olmoe3_3p5b_v4_L${SEQ_LEN}}"
OUTPUT_DIR="${OUTPUT_DIR:-gs://cloud-tpu-multipod-dev-maxtext-output/olmoe3_3p5b}"
DATASET_TYPE="${DATASET_TYPE:-synthetic}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1.0}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
STEPS="${STEPS:-20}"
LOG_PERIOD="${LOG_PERIOD:-5}"

# Parallelism:
#   1x v4-128 (64 chips, 4x4x4): DCN_DATA=1, ICI_FSDP=1, ICI_EP=64
#   1x v4-256 (128 chips, 4x4x8): DCN_DATA=1, ICI_FSDP=1, ICI_EP=128 (or FSDP=2, EP=64)
#   1x v4-1024 (512 chips, 8x8x8): DCN_DATA=1, ICI_DATA=2..4, ICI_FSDP=2, ICI_EP=64..128
#   1x v4-2048 (1024 chips, 8x8x16): DCN_DATA=1, ICI_DATA=4..8, ICI_FSDP=2, ICI_EP=64..128
DCN_DATA_PARALLELISM="${DCN_DATA_PARALLELISM:-1}"
ICI_FSDP_PARALLELISM="${ICI_FSDP_PARALLELISM:-1}"
ICI_EXPERT_PARALLELISM="${ICI_EXPERT_PARALLELISM:-64}"

# Precision mode:
#   Default (BF16_KDA_STATE=false): strict FP32 KDA state & conv (2.559s @ 8k on v4-128)
#   BF16_KDA_STATE=true: bf16 KDA state & conv (1.194s @ 4k, 2.403s @ 8k on v4-128)
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
else
  MOE_A2A_TOKEN_CHUNKS="${MOE_A2A_TOKEN_CHUNKS:-4}"
  NUM_VOCAB_TILING="${NUM_VOCAB_TILING:-4}"
  VOCAB_TILING_AG_ONCE="${VOCAB_TILING_AG_ONCE:-true}"
fi
RAGGED_BUFFER_FACTOR="${RAGGED_BUFFER_FACTOR:-1.125}"

EMO_ENABLED="${EMO_ENABLED:-true}"

# Profiler (set PROFILER=xplane to capture .xplane.pb hardware trace):
PROFILER="${PROFILER:-}"
SKIP_FIRST_N_STEPS_FOR_PROFILER="${SKIP_FIRST_N_STEPS_FOR_PROFILER:-5}"
PROFILER_STEPS="${PROFILER_STEPS:-3}"

# TPU v4 MegaCore & XLA / libtpu flags:
export TPU_MEGACORE="${TPU_MEGACORE:-MEGACORE_DENSE}"
LIBTPU="--xla_tpu_spmd_rng_bit_generator_unsafe=true \
  --xla_tpu_bf16_emission_mode=NATIVE_EMISSION \
  --xla_tpu_scoped_vmem_limit_kib=16384 \
  --xla_tpu_enable_async_ragged_all_to_all=true"
if (( SEQ_LEN <= 4096 )); then
  LIBTPU="${LIBTPU} --xla_enable_async_all_gather=false"
fi
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-} ${LIBTPU}"

echo "=== OLMo 3.5 Small (olmoe3-3p5b) TPU v4 Run ==="
echo "  run_name               : ${RUN_NAME}"
echo "  output_dir             : ${OUTPUT_DIR}"
echo "  dataset_type           : ${DATASET_TYPE}"
echo "  seq_len                : ${SEQ_LEN}"
echo "  per_device_batch_size  : ${PER_DEVICE_BATCH}"
echo "  dcn_data_parallelism   : ${DCN_DATA_PARALLELISM}"
echo "  ici_fsdp_parallelism   : ${ICI_FSDP_PARALLELISM}"
echo "  ici_expert_parallelism : ${ICI_EXPERT_PARALLELISM}"
echo "  gdn_state_dtype        : ${GDN_STATE_DTYPE} (conv_bf16=${KDA_CONV_IN_COMPUTE_DTYPE}, pallas_fused=${KDA_PALLAS_FUSED}, tm=${KDA_FUSED_TOKEN_MAJOR}, ob=${KDA_OUTER_BLOCK})"
echo "  moe_a2a_token_chunks   : ${MOE_A2A_TOKEN_CHUNKS} (rbf=${RAGGED_BUFFER_FACTOR})"
echo "  emo_enabled            : ${EMO_ENABLED}"
echo "  profiler               : ${PROFILER:-<disabled>}"
echo

EXTRA_ARGS=()
if (( SEQ_LEN <= 4096 )); then
  EXTRA_ARGS+=("moe_mlpwi_0=device")
else
  EXTRA_ARGS+=("kda_wy=device")
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

python3 -m maxtext.trainers.pre_train.train \
  "${MAXTEXT_ROOT}/src/maxtext/configs/base.yml" \
  model_name=olmoe3-3p5b \
  override_model_config=true \
  run_name="${RUN_NAME}" \
  base_output_directory="${OUTPUT_DIR}" \
  dataset_type="${DATASET_TYPE}" \
  steps="${STEPS}" \
  log_period="${LOG_PERIOD}" \
  per_device_batch_size="${PER_DEVICE_BATCH}" \
  gradient_accumulation_steps="${GRAD_ACCUM_STEPS}" \
  max_target_length="${SEQ_LEN}" \
  dcn_data_parallelism="${DCN_DATA_PARALLELISM}" \
  ici_fsdp_parallelism="${ICI_FSDP_PARALLELISM}" \
  ici_expert_parallelism="${ICI_EXPERT_PARALLELISM}" \
  shard_exp_on_fsdp=false \
  allow_split_physical_axes=true \
  scan_layers=false \
  olmoe3_per_layer_remat=true \
  remat_policy=custom \
  decoder_layer_input=device \
  moe_routing=device \
  moe_x_sorted=device \
  moe_combine=device \
  num_vocab_tiling="${NUM_VOCAB_TILING}" \
  vocab_tiling_ag_once="${VOCAB_TILING_AG_ONCE}" \
  dtype=bfloat16 \
  weight_dtype=float32 \
  use_tokamax_kda=false \
  kda_chunked_impl=subblock \
  kda_sub_block=16 \
  gdn_chunk_size=128 \
  gdn_state_dtype="${GDN_STATE_DTYPE}" \
  kda_conv_in_compute_dtype="${KDA_CONV_IN_COMPUTE_DTYPE}" \
  kda_pallas_fused="${KDA_PALLAS_FUSED}" \
  kda_fused_token_major="${KDA_FUSED_TOKEN_MAJOR}" \
  kda_outer_block="${KDA_OUTER_BLOCK}" \
  kda_inverse_precision="${KDA_INVERSE_PRECISION}" \
  kda_fused_input_proj=true \
  capacity_factor=-1 \
  moe_lean_routing=true \
  moe_topk_pallas=true \
  emo_threshold_by_bisection=true \
  moe_a2a_expert_major=true \
  moe_a2a_token_chunks="${MOE_A2A_TOKEN_CHUNKS}" \
  ragged_buffer_factor="${RAGGED_BUFFER_FACTOR}" \
  megablox=true \
  sparse_matmul=true \
  use_tokamax_gmm=false \
  use_gmm_v2=false \
  wi_tile_fwd_batch_seq=512 \
  wi_tile_fwd_embed_dim=768 \
  wi_tile_fwd_mlp_dim=896 \
  wi_tile_dlhs_batch_seq=512 \
  wi_tile_dlhs_mlp_dim=1792 \
  wi_tile_dlhs_embed_dim=384 \
  wi_tile_drhs_batch_seq=512 \
  wi_tile_drhs_embed_dim=768 \
  wi_tile_drhs_mlp_dim=896 \
  wo_tile_fwd_batch_seq=512 \
  wo_tile_fwd_mlp_dim=1792 \
  wo_tile_fwd_embed_dim=384 \
  wo_tile_dlhs_batch_seq=512 \
  wo_tile_dlhs_embed_dim=768 \
  wo_tile_dlhs_mlp_dim=896 \
  wo_tile_drhs_batch_seq=512 \
  wo_tile_drhs_mlp_dim=896 \
  wo_tile_drhs_embed_dim=384 \
  attention=flash \
  emo_enabled="${EMO_ENABLED}" \
  norm_metrics=grad \
  synthetic_data_reuse_batch=true \
  skip_first_n_steps_for_profiler="${SKIP_FIRST_N_STEPS_FOR_PROFILER}" \
  profiler_steps="${PROFILER_STEPS}" \
  upload_all_profiler_results=false \
  profile_cleanly=false \
  learning_rate=1e-5 \
  abort_on_nan_loss=false \
  abort_on_inf_loss=false \
  enable_checkpointing=false \
  async_checkpointing=false \
  enable_tensorboard=false \
  enable_goodput_recording=false \
  monitor_goodput=false \
  enable_checkpoint_cloud_logger=false \
  "${EXTRA_ARGS[@]}" \
  "$@"
