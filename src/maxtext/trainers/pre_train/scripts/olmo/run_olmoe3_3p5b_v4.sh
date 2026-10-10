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
# OLMo 3.5 Small (olmoe3-3p5b: 62.9B total / 3.48B active, hybrid 3:1 KDA + SWA,
# 512-expert top-32 LatentMoE) high-MFU TPU v4 (v4-128 / 4x4x4, v4-256 / 4x4x8)
# training and benchmark launcher for MaxText.
#
# Incorporates all verified TPU v4 hardware & compiler optimizations:
#   - 2D physical mesh (ICI FSDP=4, EP=16 on v4-128; FSDP=8, EP=16 on v4-256)
#     with allow_split_physical_axes=false
#   - 0.00 GiB host offload at both SEQ_LEN=4096 and SEQ_LEN=8192 (100% pure device HBM)
#   - Analytical custom VJP for KDA WY chunked scan (1-GEMM Gram-Adjoint Collapse +
#     0-matmul state reconstruction)
#   - Fused 3-GMM Ragged LatentMoE with Pallas Top-K (moe_topk_pallas=true,
#     moe_lean_routing=true, ragged_buffer_factor=1.125)
#   - Tiled Vocab Cross-Entropy with single FSDP all-gather (num_vocab_tiling=8,
#     vocab_tiling_ag_once=true)
#   - TPU v4 TensorCore HBM & async collective overlap flags

set -euo pipefail

MAXTEXT_ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
VENV_PATH="${VENV_PATH:-${MAXTEXT_ROOT}/maxtext_venv}"

if [[ -d "${VENV_PATH}" ]]; then
  # shellcheck disable=SC1090,SC1091
  source "${VENV_PATH}/bin/activate"
fi

export PYTHONPATH="${MAXTEXT_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# -------------------------- Configuration Defaults --------------------------
RUN_NAME="${RUN_NAME:-olmoe3_3p5b_v4_128_L8192_hc5}"
OUTPUT_DIR="${OUTPUT_DIR:-gs://cloud-tpu-multipod-dev-maxtext-output/olmoe3_3p5b}"
DATASET_TYPE="${DATASET_TYPE:-synthetic}"
SEQ_LEN="${SEQ_LEN:-8192}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1.0}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
STEPS="${STEPS:-30}"
LOG_PERIOD="${LOG_PERIOD:-10}"

# Parallelism:
#   1x v4-128 (64 chips, 4x4x4): DCN_DATA=1, ICI_FSDP=4, ICI_EP=16
#   1x v4-256 (128 chips, 4x4x8): DCN_DATA=1, ICI_FSDP=8, ICI_EP=16
#   4x v4-128 (256 chips across 4 node pools): DCN_DATA=4, ICI_FSDP=4, ICI_EP=16
DCN_DATA_PARALLELISM="${DCN_DATA_PARALLELISM:-1}"
ICI_FSDP_PARALLELISM="${ICI_FSDP_PARALLELISM:-4}"
ICI_EXPERT_PARALLELISM="${ICI_EXPERT_PARALLELISM:-16}"

# Profiler (set PROFILER=xplane to capture .xplane.pb hardware trace):
PROFILER="${PROFILER:-}"
SKIP_FIRST_N_STEPS_FOR_PROFILER="${SKIP_FIRST_N_STEPS_FOR_PROFILER:-6}"
PROFILER_STEPS="${PROFILER_STEPS:-2}"

# Sequence-length-specific zero-offload remat policy:
#   At SEQ_LEN=4096 (opt_lean_v3): context=device,moe_dispatch=device fits in 29.09 GiB
#   At SEQ_LEN=8192 (hc4/hc5):     context=remat,moe_dispatch=remat,moe_x_sorted=device
#                                  fits in ~27.89-29.61 GiB with 0.00 GiB host offload!
if (( SEQ_LEN <= 4096 )); then
  CONTEXT_POLICY="${CONTEXT_POLICY:-device}"
  MOE_DISPATCH_POLICY="${MOE_DISPATCH_POLICY:-device}"
  MOE_X_SORTED_POLICY="${MOE_X_SORTED_POLICY:-device}"
else
  CONTEXT_POLICY="${CONTEXT_POLICY:-remat}"
  MOE_DISPATCH_POLICY="${MOE_DISPATCH_POLICY:-remat}"
  MOE_X_SORTED_POLICY="${MOE_X_SORTED_POLICY:-device}"
fi

# TPU v4 XLA / libtpu flags:
#   - 64 MiB scoped VMEM for Pallas GMM & FlashAttention kernels
#   - Disable async collective fusion (prevents holding multiple concurrent all-to-all
#     staging buffers in TensorCore HBM)
#   - Enable TensorCore compute/collective overlap & async all-gather / collective-permute
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-} \
  --xla_tpu_scoped_vmem_limit_kib=65536 \
  --xla_tpu_enable_async_collective_fusion=false \
  --xla_tpu_overlap_compute_collective_tc=true \
  --xla_enable_async_all_gather=true \
  --xla_enable_async_collective_permute=true \
  --xla_tpu_spmd_rng_bit_generator_unsafe=true"

echo "=== OLMo 3.5 Small (olmoe3-3p5b) TPU v4 Run ==="
echo "  run_name               : ${RUN_NAME}"
echo "  output_dir             : ${OUTPUT_DIR}"
echo "  dataset_type           : ${DATASET_TYPE}"
echo "  seq_len                : ${SEQ_LEN}"
echo "  per_device_batch_size  : ${PER_DEVICE_BATCH}"
echo "  dcn_data_parallelism   : ${DCN_DATA_PARALLELISM}"
echo "  ici_fsdp_parallelism   : ${ICI_FSDP_PARALLELISM}"
echo "  ici_expert_parallelism : ${ICI_EXPERT_PARALLELISM}"
echo "  remat (context/disp)   : context=${CONTEXT_POLICY}, moe_dispatch=${MOE_DISPATCH_POLICY}, moe_x_sorted=${MOE_X_SORTED_POLICY}"
echo "  profiler               : ${PROFILER:-<disabled>}"
echo

EXTRA_ARGS=()
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
  allow_split_physical_axes=false \
  scan_layers=true \
  remat_policy=custom \
  olmoe3_per_layer_remat=true \
  decoder_layer_input=device \
  context="${CONTEXT_POLICY}" \
  query_proj=device \
  key_proj=device \
  value_proj=device \
  out_proj=device \
  qkv_proj=remat \
  mlpwi_0=remat \
  mlpwi_1=remat \
  mlpwo=remat \
  moe_mlpwi_0=remat \
  moe_mlpwi_1=remat \
  moe_mlpwo=remat \
  moe_x_sorted="${MOE_X_SORTED_POLICY}" \
  moe_dispatch="${MOE_DISPATCH_POLICY}" \
  moe_routing=device \
  moe_router_logits=remat \
  num_vocab_tiling=8 \
  vocab_tiling_ag_once=true \
  dtype=bfloat16 \
  weight_dtype=float32 \
  gdn_chunk_size=64 \
  gdn_state_dtype=float32 \
  kda_conv_in_compute_dtype=false \
  moe_lean_routing=true \
  moe_topk_pallas=true \
  ragged_buffer_factor=1.125 \
  wi_tile_fwd_batch_seq=256 \
  wi_tile_fwd_embed_dim=768 \
  wi_tile_fwd_mlp_dim=1792 \
  wi_tile_dlhs_batch_seq=256 \
  wi_tile_dlhs_mlp_dim=1792 \
  wi_tile_dlhs_embed_dim=768 \
  wi_tile_drhs_batch_seq=256 \
  wi_tile_drhs_embed_dim=768 \
  wi_tile_drhs_mlp_dim=896 \
  wo_tile_fwd_batch_seq=256 \
  wo_tile_fwd_mlp_dim=1792 \
  wo_tile_fwd_embed_dim=768 \
  wo_tile_dlhs_batch_seq=256 \
  wo_tile_dlhs_embed_dim=768 \
  wo_tile_dlhs_mlp_dim=1792 \
  wo_tile_drhs_batch_seq=256 \
  wo_tile_drhs_mlp_dim=896 \
  wo_tile_drhs_embed_dim=768 \
  megablox=false \
  sparse_matmul=true \
  use_tokamax_gmm=false \
  prefuse_moe_weights=false \
  attention=flash \
  emo_enabled=false \
  profiler="${PROFILER}" \
  skip_first_n_steps_for_profiler="${SKIP_FIRST_N_STEPS_FOR_PROFILER}" \
  profiler_steps="${PROFILER_STEPS}" \
  upload_all_profiler_results=false \
  profile_cleanly=false \
  learning_rate=1e-5 \
  abort_on_nan_loss=false \
  abort_on_inf_loss=false \
  enable_checkpointing=false \
  enable_tensorboard=false \
  enable_goodput_recording=false \
  monitor_goodput=false \
  enable_checkpoint_cloud_logger=false \
  "${EXTRA_ARGS[@]}" \
  "$@"
