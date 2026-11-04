#!/bin/bash
# Copyright 2023–2026 Google LLC
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

# Baseline Gemma 4 26B-A4B (MoE) pre-training recipe on TPU v6e-128 (16k context).
#
# Direct port of the public Gemma 4 26B recipe to TPU v6e-128 at 16k sequence length
# (32,768 tokens/chip/step, global batch 4.19M tokens), using only stock upstream MaxText
# flags so it runs identically on unmodified `AI-Hypercomputer/maxtext:main` (7.579 s/step,
# 121.8 TFLOP/s/chip, 13.3% MFU) for side-by-side throughput and training-loss comparison
# against `src/maxtext/configs/tpu/v6e/gemma4_26b.sh` (3.638 s/step, 253.8 TFLOP/s/chip,
# 27.6% MFU — 2.08x speedup).
#
# Notes on comparing training losses vs. `gemma4_26b.sh`:
#   1. Random initialization (`LOAD_PARAMETERS_PATH=""`):
#      - Set `MATCH_PREFUSED_INIT=true` when comparing step-by-step losses from Step 0.
#        `prefuse_moe_weights=true` and `fused_mlp=true` fuse `(wi_0, wi_1)` into a single
#        `wi` parameter leaf at init, which otherwise changes the Flax NNX PRNG draw order
#        when weights are initialized randomly. With `MATCH_PREFUSED_INIT=true`, both runs
#        start from the exact same initial weights and match step-by-step (~13.004 +/- 0.001).
#      - With default `MATCH_PREFUSED_INIT=false` (exact 7.579 s/step baseline), both runs
#        draw independent random initializations from the same distribution and follow the
#        same convergence curve.
#   2. Pre-converted checkpoint (`LOAD_PARAMETERS_PATH="gs://..."`):
#      - Pass the same `LOAD_PARAMETERS_PATH` to both scripts (for HF checkpoints converted
#        with split `wi_0`/`wi_1`, pass `PREFUSE_MOE_WEIGHTS=false FUSED_MLP=false` to
#        `gemma4_26b.sh`).
#
# Example invocations:
#   # Exact 7.579 s/step baseline on upstream main:
#   bash src/maxtext/configs/tpu/v6e/gemma4_26b_base.sh \
#     RUN_NAME="gemma4-26b-base-16k" OUTPUT_PATH="gs://<your_bucket>/gemma4"
#
#   # Step-by-step loss parity baseline (identical random init parameter tree as gemma4_26b.sh):
#   bash src/maxtext/configs/tpu/v6e/gemma4_26b_base.sh \
#     MATCH_PREFUSED_INIT=true STEPS=50 RUN_NAME="gemma4-26b-base-loss" OUTPUT_PATH="gs://<your_bucket>/gemma4"

set -euo pipefail

export EXECUTABLE="${EXECUTABLE:-train}" # train or train_compile
export RUN_PREFLIGHT="${RUN_PREFLIGHT:-true}"
export SEQ_LEN="${SEQ_LEN:-16384}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
export ICI_FSDP_PARALLELISM="${ICI_FSDP_PARALLELISM:-128}"
export STEPS="${STEPS:-30}"
export DATASET_TYPE="${DATASET_TYPE:-synthetic}"
export DATASET_PATH="${DATASET_PATH:-}"
export LOAD_PARAMETERS_PATH="${LOAD_PARAMETERS_PATH:-}"
export ENABLE_CHECKPOINTING="${ENABLE_CHECKPOINTING:-false}"
export ENABLE_DROPOUT="${ENABLE_DROPOUT:-false}"
export INIT_WEIGHTS_SEED="${INIT_WEIGHTS_SEED:-0}"
export DATA_SHUFFLE_SEED="${DATA_SHUFFLE_SEED:-0}"
export MATCH_PREFUSED_INIT="${MATCH_PREFUSED_INIT:-false}"
export EXTRA_MAXTEXT_ARGS="${EXTRA_MAXTEXT_ARGS:-}"

for ARGUMENT in "$@"; do
  IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
  export "$KEY"="$VALUE"
done

OUTPUT_PATH="${OUTPUT_PATH:-${BASE_OUTPUT_DIRECTORY:-}}"
if [ -z "$OUTPUT_PATH" ]; then
  echo "Error: OUTPUT_PATH (or BASE_OUTPUT_DIRECTORY) must be set, e.g. OUTPUT_PATH=gs://<your_bucket>/gemma4" >&2
  exit 1
fi

RUN_NAME="${RUN_NAME:-${JOBSET_NAME:-gemma4-26b-base-$(date +%Y%m%d-%H%M%S)}}"
export M_RUN_NAME="$RUN_NAME"

if [ "$RUN_PREFLIGHT" = "true" ] && [ -f "src/dependencies/scripts/preflight.sh" ]; then
  bash src/dependencies/scripts/preflight.sh
fi

export LIBTPU_INIT_ARGS="\
--xla_tpu_scoped_vmem_limit_kib=131072 \
--xla_tpu_bf16_emission_mode=NATIVE_EMISSION \
--xla_tpu_enable_sparse_core_reduce_scatter_v2=true \
--xla_tpu_enable_sparse_core_collective_offload_all_gather=true \
--xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true \
--xla_tpu_use_single_sparse_core_for_all_gather_offload=true \
--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true \
--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true \
--xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true \
--xla_tpu_enable_all_gather_offload_tracing=true \
--xla_tpu_use_tc_device_shape_on_sc=True \
--xla_sc_disable_megacore_partitioning=True \
--xla_enable_async_all_gather=true \
--xla_tpu_prefer_async_allgather_to_allreduce=true \
--xla_tpu_enable_latency_hiding_layer_scheduler=true \
--xla_tpu_scheduler_percent_shared_memory_limit=150 \
--xla_tpu_enable_layer_scheduler_for_dependent_collectives=true \
--xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true \
--xla_tpu_enable_sparse_core_collective_aggregator=true"

OPTIONAL_ARGS=()
WI_MLP_TILE=704
if [ "$MATCH_PREFUSED_INIT" = "true" ]; then
  OPTIONAL_ARGS+=("prefuse_moe_weights=true" "fused_mlp=true")
  # When wi_0 and wi_1 are prefused into wi (2 * 704 = 1408), Pallas requires
  # the MLP block dim to either equal 1408 or be divisible by 128.
  WI_MLP_TILE=1408
fi
if [ -n "$DATASET_PATH" ]; then
  OPTIONAL_ARGS+=("dataset_path=${DATASET_PATH}")
fi
if [ -n "$LOAD_PARAMETERS_PATH" ]; then
  OPTIONAL_ARGS+=("load_parameters_path=${LOAD_PARAMETERS_PATH}")
fi
if [ -n "$EXTRA_MAXTEXT_ARGS" ]; then
  # shellcheck disable=SC2206
  OPTIONAL_ARGS+=(${EXTRA_MAXTEXT_ARGS})
fi

CONFIGS_DIR="${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"

python3 -m "maxtext.trainers.pre_train.${EXECUTABLE}" "${CONFIGS_DIR}/base.yml" \
  model_name=gemma4-26b \
  base_output_directory="${OUTPUT_PATH}" \
  dataset_type="${DATASET_TYPE}" \
  steps="${STEPS}" \
  enable_checkpointing="${ENABLE_CHECKPOINTING}" \
  enable_dropout="${ENABLE_DROPOUT}" \
  init_weights_seed="${INIT_WEIGHTS_SEED}" \
  data_shuffle_seed="${DATA_SHUFFLE_SEED}" \
  dtype=bfloat16 \
  opt_type=adamw \
  per_device_batch_size="${PER_DEVICE_BATCH_SIZE}" \
  max_target_length="${SEQ_LEN}" \
  num_vocab_tiling=16 \
  remat_policy=full \
  decoder_layer_input=device \
  allow_split_physical_axes=true \
  ici_fsdp_transpose_parallelism=1 \
  ici_fsdp_parallelism="${ICI_FSDP_PARALLELISM}" \
  use_iota_embed=true \
  megablox=true \
  sparse_matmul=true \
  use_custom_sort_vjp=true \
  wi_tile_fwd_embed_dim=2816 wi_tile_dlhs_embed_dim=2816 wi_tile_drhs_embed_dim=2816 \
  wo_tile_fwd_embed_dim=2816 wo_tile_dlhs_embed_dim=2816 wo_tile_drhs_embed_dim=2816 \
  wi_tile_fwd_mlp_dim="${WI_MLP_TILE}" wi_tile_dlhs_mlp_dim="${WI_MLP_TILE}" wi_tile_drhs_mlp_dim="${WI_MLP_TILE}" \
  wo_tile_fwd_mlp_dim=704 wo_tile_dlhs_mlp_dim=704 wo_tile_drhs_mlp_dim=704 \
  attention=flash \
  use_tokamax_splash=true \
  sa_use_fused_bwd_kernel=true \
  sa_block_q=1024 \
  sa_block_kv=1024 \
  sa_block_kv_compute=512 \
  sa_block_q_dkv=1024 \
  sa_block_kv_dkv=1024 \
  sa_block_kv_dkv_compute=256 \
  "${OPTIONAL_ARGS[@]}"
