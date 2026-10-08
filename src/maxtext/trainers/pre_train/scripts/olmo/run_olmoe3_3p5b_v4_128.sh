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
# olmoe3-3p5b (OLMo 3.5 small) on one TPU v4-128 slice (4x4x4, 64 chips, one megacore device per chip):
# the fastest measured configuration, run on every worker of the slice.
#
#   SEQ_LEN=8192 bash src/maxtext/trainers/pre_train/scripts/olmo/run_olmoe3_3p5b_v4_128.sh
#   SEQ_LEN=4096 bash src/maxtext/trainers/pre_train/scripts/olmo/run_olmoe3_3p5b_v4_128.sh
#
# Measured (median of steps 10-19, synthetic data, per_device_batch_size=1):
#   seq 4096: 1.520 s/step, 20.4% MFU (85.17 TF per chip per step, scripts/olmoe3_flops_check.py)
#   seq 8192: 3.095 s/step, 20.3% MFU (172.81 TF per chip per step)
#   F32_KDA_STATE=1 (strict precision): 1.558 s / 19.9% and 3.167 s / 19.8%
#   SAFE_BUFFER=1 (ragged buffer 1.25): 1.581 s and 3.223 s
# Flags and kernels are explained in olmoe3-3p5b-v4-128-config.md.
#
# Knobs (environment):
#   SEQ_LEN        4096 or 8192 (default 8192)
#   F32_KDA_STATE  1: KDA recurrent state and depthwise conv in float32 (strictly loss-neutral; ~2.5% slower)
#   SAFE_BUFFER    1: ragged_buffer_factor=1.25 instead of 1.125 (no token drops on skewed routing)
#   PROFILE        1: capture an xplane trace of steps 6-8
#   STEPS, RUN_NAME, OUTPUT_DIR, DATASET_TYPE   as usual; extra MaxText flags can follow on the command line.

set -euo pipefail

MAXTEXT_ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
export PYTHONPATH="${MAXTEXT_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

SEQ_LEN="${SEQ_LEN:-8192}"
STEPS="${STEPS:-20}"
RUN_NAME="${RUN_NAME:-olmoe3_3p5b_v4_128_s${SEQ_LEN}}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/olmoe3_3p5b_v4}"
DATASET_TYPE="${DATASET_TYPE:-synthetic}"

# One megacore device per chip: +1 GiB of HBM for the TensorCores.
export TPU_MEGACORE=MEGACORE_DENSE
# Async ragged all-to-all lets one token chunk's exchange overlap another chunk's expert GEMMs.
LIBTPU="--xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_bf16_emission_mode=NATIVE_EMISSION \
--xla_tpu_scoped_vmem_limit_kib=16384 --xla_tpu_enable_async_ragged_all_to_all=true"

# Shared by both sequence lengths.
FLAGS=(
  # model, precision, batch
  model_name=olmoe3-3p5b override_model_config=True
  dtype=bfloat16 weight_dtype=float32
  per_device_batch_size=1 max_target_length="${SEQ_LEN}"
  # layout: 8 of 512 experts per chip, no expert-weight all-gathers
  ici_expert_parallelism=64 ici_fsdp_parallelism=1 shard_exp_on_fsdp=False
  # unrolled layers with a per-layer remat boundary (scanning keeps every layer's buffers live at once)
  scan_layers=False olmoe3_per_layer_remat=True remat_policy=custom
  moe_routing=device moe_x_sorted=device moe_combine=device decoder_layer_input=device
  # routed experts: megablox (Pallas GMM); tokamax does not run on v4
  sparse_matmul=True megablox=True use_tokamax_gmm=False use_gmm_v2=False
  capacity_factor=-1 moe_lean_routing=True moe_topk_pallas=True emo_threshold_by_bisection=True
  moe_a2a_expert_major=True
  # megablox tiles (m, k, n): two n-tiles per GMM so both megacore TensorCores run
  wi_tile_fwd_batch_seq=512 wi_tile_fwd_embed_dim=768 wi_tile_fwd_mlp_dim=896
  wi_tile_dlhs_batch_seq=512 wi_tile_dlhs_mlp_dim=1792 wi_tile_dlhs_embed_dim=384
  wi_tile_drhs_batch_seq=512 wi_tile_drhs_embed_dim=768 wi_tile_drhs_mlp_dim=896
  wo_tile_fwd_batch_seq=512 wo_tile_fwd_mlp_dim=1792 wo_tile_fwd_embed_dim=384
  wo_tile_dlhs_batch_seq=512 wo_tile_dlhs_embed_dim=768 wo_tile_dlhs_mlp_dim=896
  wo_tile_drhs_batch_seq=512 wo_tile_drhs_mlp_dim=896 wo_tile_drhs_embed_dim=384
  # KDA: pure-JAX sub-block chunked delta rule (tokamax KDA does not run on v4)
  use_tokamax_kda=False kda_chunked_impl=subblock gdn_chunk_size=128 kda_fused_input_proj=True
  # metrics: one shared grad norm instead of three full passes (training unchanged)
  norm_metrics=grad
  # the run
  run_name="${RUN_NAME}" base_output_directory="${OUTPUT_DIR}" dataset_type="${DATASET_TYPE}" steps="${STEPS}"
  enable_checkpointing=False async_checkpointing=False
)

if (( SEQ_LEN <= 4096 )); then
  FLAGS+=(moe_a2a_token_chunks=2 num_vocab_tiling=1 moe_mlpwi_0=device)
  # With MEGACORE_DENSE, XLA otherwise turns the weight all-gathers async, which loses ~50 ms at 4k.
  LIBTPU="${LIBTPU} --xla_enable_async_all_gather=false"
else
  # 4 chunks keep each permute gather under v4's 128 MiB CMEM; at 8k the weight all-gathers must stay async.
  FLAGS+=(moe_a2a_token_chunks=4 num_vocab_tiling=4 vocab_tiling_ag_once=True kda_wy=device)
fi

if [[ "${F32_KDA_STATE:-0}" == "1" ]]; then
  FLAGS+=(gdn_state_dtype=float32 kda_conv_in_compute_dtype=False)
else
  FLAGS+=(gdn_state_dtype=bfloat16 kda_conv_in_compute_dtype=True)
fi

if [[ "${SAFE_BUFFER:-0}" == "1" ]]; then
  FLAGS+=(ragged_buffer_factor=1.25)
else
  FLAGS+=(ragged_buffer_factor=1.125)
fi

if [[ "${DATASET_TYPE}" == "synthetic" ]]; then
  # The synthetic batch never changes; reuse it instead of resharding it every step.
  FLAGS+=(synthetic_data_reuse_batch=True)
fi

if [[ "${PROFILE:-0}" == "1" ]]; then
  FLAGS+=(profiler=xplane skip_first_n_steps_for_profiler=5 profiler_steps=3)
fi

export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-} ${LIBTPU}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "TPU_MEGACORE=${TPU_MEGACORE}"
  echo "LIBTPU_INIT_ARGS=${LIBTPU_INIT_ARGS}"
  printf '%s\n' "${FLAGS[@]}" "$@"
  exit 0
fi

exec python3 -m maxtext.trainers.pre_train.train "${MAXTEXT_ROOT}/src/maxtext/configs/base.yml" "${FLAGS[@]}" "$@"
