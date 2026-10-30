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

# Optimized Gemma 4 26B-A4B (MoE) pre-training recipe on TPU v6e-128 (16k -> 256k context).
#
# Combines:
#   - Part 1 (#5572): Layer/model optimizations (split-half RoPE, seq-axis vocab tiling,
#     gate/up custom VJP, FP32 norm/router scales, top_k_dense_vjp, moe_unsliced_prefused_wi)
#   - Part 2 (#5573): Fused TensorCore Pallas MoE combine & permute kernels
#     (moe_combine_kernel=tc, moe_permute_kernel=tc)
#   - Part 3 (#5608): Per-layer Context Parallelism & Halo Sliding-Window Attention
#     (local_context_parallel_strategy=halo) for 32k -> 256k context lengths
#
# Measured throughput on TPU v6e-128 (32,768 tokens/chip/step, global batch 4.19M tokens):
#   - SEQ_LEN=16384  (FSDP=128, CP=1, pdbs=2):     3.779 s/step, 244.3 TFLOP/s/chip (unfused Splash bwd; 3.638 s/step with USE_TOKAMAX_SPLASH=true)
#   - SEQ_LEN=32768  (FSDP=64,  CP=2, pdbs=1):     4.086 s/step, 258.4 TFLOP/s/chip (28.1% MFU)
#   - SEQ_LEN=65536  (FSDP=64,  CP=2, pdbs=0.5):   4.547 s/step, 290.3 TFLOP/s/chip (31.6% MFU)
#   - SEQ_LEN=131072 (FSDP=32,  CP=4, pdbs=0.25):  5.686 s/step, 325.0 TFLOP/s/chip (35.4% MFU)
#   - SEQ_LEN=262144 (FSDP=16,  CP=8, pdbs=0.125): 7.837 s/step, 370.5 TFLOP/s/chip (40.4% MFU)
#
# Example invocations:
#   # 16k default (synthetic data):
#   bash src/maxtext/configs/tpu/v6e/gemma4_26b.sh \
#     RUN_NAME="gemma4-26b-opt-16k" OUTPUT_PATH="gs://<your_bucket>/gemma4"
#
#   # 128k long-context with Halo CP:
#   bash src/maxtext/configs/tpu/v6e/gemma4_26b.sh \
#     SEQ_LEN=131072 RUN_NAME="gemma4-26b-opt-128k" OUTPUT_PATH="gs://<your_bucket>/gemma4"
#
#   # Compare step-by-step loss against gemma4_26b_base.sh (same initial random weights):
#   bash src/maxtext/configs/tpu/v6e/gemma4_26b_base.sh \
#     MATCH_PREFUSED_INIT=true STEPS=50 RUN_NAME="gemma4-26b-base-loss" OUTPUT_PATH="gs://<your_bucket>/gemma4"
#   bash src/maxtext/configs/tpu/v6e/gemma4_26b.sh \
#     STEPS=50 RUN_NAME="gemma4-26b-opt-loss" OUTPUT_PATH="gs://<your_bucket>/gemma4"

set -euo pipefail

export EXECUTABLE="${EXECUTABLE:-train}" # train or train_compile
export RUN_PREFLIGHT="${RUN_PREFLIGHT:-true}"
export SEQ_LEN="${SEQ_LEN:-16384}"
export STEPS="${STEPS:-30}"
export DATASET_TYPE="${DATASET_TYPE:-synthetic}"
export DATASET_PATH="${DATASET_PATH:-}"
export LOAD_PARAMETERS_PATH="${LOAD_PARAMETERS_PATH:-}"
export ENABLE_CHECKPOINTING="${ENABLE_CHECKPOINTING:-false}"
export ENABLE_DROPOUT="${ENABLE_DROPOUT:-false}"
export INIT_WEIGHTS_SEED="${INIT_WEIGHTS_SEED:-0}"
export DATA_SHUFFLE_SEED="${DATA_SHUFFLE_SEED:-0}"
export PREFUSE_MOE_WEIGHTS="${PREFUSE_MOE_WEIGHTS:-true}"
export FUSED_MLP="${FUSED_MLP:-true}"
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

RUN_NAME="${RUN_NAME:-${JOBSET_NAME:-gemma4-26b-opt-$(date +%Y%m%d-%H%M%S)}}"
export M_RUN_NAME="$RUN_NAME"

if [ "$RUN_PREFLIGHT" = "true" ] && [ -f "src/dependencies/scripts/preflight.sh" ]; then
  bash src/dependencies/scripts/preflight.sh
fi

# Context-length topology & memory presets on TPU v6e-128 (32,768 tokens/chip/step)
SCHED_SHARED_MEM_LIMIT=150
CP_ARGS=()
case "$SEQ_LEN" in
  16384)
    PDBS="${PER_DEVICE_BATCH_SIZE:-2}"
    FSDP="${ICI_FSDP_PARALLELISM:-128}"
    CP=1
    VOCAB_TILING="${NUM_VOCAB_TILING:-4}"
    UNSLICED_WI="${MOE_UNSLICED_PREFUSED_WI:-true}"
    # Unfused Splash backward reduces dQ in FP32 inside VMEM at 16k (CP=1) for tighter loss parity.
    TOKAMAX_SPLASH="${USE_TOKAMAX_SPLASH:-false}"
    ;;
  32768)
    PDBS="${PER_DEVICE_BATCH_SIZE:-1}"
    FSDP="${ICI_FSDP_PARALLELISM:-64}"
    CP=2
    VOCAB_TILING="${NUM_VOCAB_TILING:-4}"
    UNSLICED_WI="${MOE_UNSLICED_PREFUSED_WI:-true}"
    TOKAMAX_SPLASH="${USE_TOKAMAX_SPLASH:-true}"
    CP_ARGS=(
      "allow_split_physical_axes=true"
      "context_parallel_strategy=all_gather"
      "local_context_parallel_strategy=halo"
      "context_parallel_load_balance=true"
    )
    ;;
  65536)
    PDBS="${PER_DEVICE_BATCH_SIZE:-0.5}"
    FSDP="${ICI_FSDP_PARALLELISM:-64}"
    CP=2
    VOCAB_TILING="${NUM_VOCAB_TILING:-4}"
    UNSLICED_WI="${MOE_UNSLICED_PREFUSED_WI:-true}"
    TOKAMAX_SPLASH="${USE_TOKAMAX_SPLASH:-true}"
    CP_ARGS=(
      "allow_split_physical_axes=true"
      "context_parallel_strategy=all_gather"
      "local_context_parallel_strategy=halo"
      "context_parallel_load_balance=true"
    )
    ;;
  131072)
    PDBS="${PER_DEVICE_BATCH_SIZE:-0.25}"
    FSDP="${ICI_FSDP_PARALLELISM:-32}"
    CP=4
    VOCAB_TILING="${NUM_VOCAB_TILING:-16}"
    UNSLICED_WI="${MOE_UNSLICED_PREFUSED_WI:-true}"
    TOKAMAX_SPLASH="${USE_TOKAMAX_SPLASH:-true}"
    CP_ARGS=(
      "allow_split_physical_axes=true"
      "context_parallel_strategy=all_gather"
      "local_context_parallel_strategy=halo"
      "context_parallel_load_balance=true"
    )
    ;;
  262144)
    PDBS="${PER_DEVICE_BATCH_SIZE:-0.125}"
    FSDP="${ICI_FSDP_PARALLELISM:-16}"
    CP=8
    VOCAB_TILING="${NUM_VOCAB_TILING:-16}"
    # At FSDP=16, keeping moe_unsliced_prefused_wi=false saves 610 MiB HBM so context=device fits in 28.0 GiB.
    UNSLICED_WI="${MOE_UNSLICED_PREFUSED_WI:-false}"
    TOKAMAX_SPLASH="${USE_TOKAMAX_SPLASH:-true}"
    CP_ARGS=(
      "allow_split_physical_axes=true"
      "context_parallel_strategy=all_gather"
      "local_context_parallel_strategy=halo"
      "context_parallel_load_balance=true"
    )
    ;;
  *)
    echo "Unsupported SEQ_LEN=${SEQ_LEN}. Expected one of: 16384, 32768, 65536, 131072, 262144." >&2
    exit 1
    ;;
esac

if [ "$PREFUSE_MOE_WEIGHTS" != "true" ]; then
  UNSLICED_WI="false"
fi

if [ "$TOKAMAX_SPLASH" = "true" ]; then
  ATTN_ARGS=(
    "use_tokamax_splash=true"
    "sa_use_fused_bwd_kernel=true"
    "dq_reduction_steps=3"
  )
else
  ATTN_ARGS=(
    "use_tokamax_splash=false"
    "sa_use_fused_bwd_kernel=false"
  )
fi

export LIBTPU_INIT_ARGS="\
--xla_tpu_scoped_vmem_limit_kib=81920 \
--xla_tpu_bf16_emission_mode=NATIVE_EMISSION \
--xla_tpu_enable_sparse_core_reduce_scatter_v2=true \
--xla_tpu_enable_sparse_core_collective_offload_all_gather=true \
--xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true \
--xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true \
--xla_tpu_use_single_sparse_core_for_all_gather_offload=true \
--xla_tpu_enable_all_gather_offload_tracing=true \
--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true \
--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true \
--xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true \
--xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false \
--xla_tpu_enable_sparse_core_collective_aggregator=true \
--xla_tpu_use_tc_device_shape_on_sc=True \
--xla_sc_disable_megacore_partitioning=True \
--xla_enable_async_all_gather=true \
--xla_enable_async_all_reduce=true \
--xla_enable_async_collective_permute=true \
--xla_tpu_prefer_async_allgather_to_allreduce=true \
--xla_jf_spmd_threshold_for_windowed_einsum_mib=1000000 \
--xla_latency_hiding_scheduler_rerun=2 \
--xla_tpu_enable_latency_hiding_layer_scheduler=true \
--xla_tpu_enable_layer_scheduler_for_dependent_collectives=true \
--xla_tpu_scheduler_percent_shared_memory_limit=${SCHED_SHARED_MEM_LIMIT} \
--xla_tpu_enable_all_experimental_scheduler_features=true \
--xla_tpu_enable_scheduler_memory_pressure_tracking=true \
--xla_tpu_aggressive_opt_barrier_removal=ENABLED \
--xla_lhs_prioritize_async_depth_over_stall=ENABLED \
--xla_should_allow_loop_variant_parameter_in_chain=ENABLED \
--xla_should_add_loop_invariant_op_in_chain=ENABLED \
--xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1 \
--xla_tpu_assign_all_reduce_scatter_layout=true \
--xla_tpu_data_parallel_opt_different_sized_ops=true \
--xla_tpu_enable_data_parallel_all_reduce_opt=true \
--xla_tpu_overlap_compute_collective_tc=true \
--xla_tpu_use_minor_sharding_for_major_trivial_input=true \
--xla_tpu_enable_sparse_core_offload_queuing_in_lhs=true \
--xla_tpu_sparse_core_offload_queuing_overlap_limit=8 \
--xla_tpu_sparse_core_all_gather_latency_multiplier=3 \
--xla_tpu_offload_gather_to_sparsecore=true \
--xla_tpu_offload_all_supported_gathers_to_sparsecore=true"

OPTIONAL_ARGS=()
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
  per_device_batch_size="${PDBS}" \
  max_target_length="${SEQ_LEN}" \
  ici_fsdp_parallelism="${FSDP}" \
  ici_context_parallelism="${CP}" \
  ici_expert_parallelism=1 \
  ici_tensor_parallelism=1 \
  shard_exp_on_fsdp=true \
  remat_policy=custom \
  decoder_layer_input=device \
  context=device \
  use_iota_embed=false \
  vocab_tiling_ag_once=true \
  num_vocab_tiling="${VOCAB_TILING}" \
  attention=flash \
  "${ATTN_ARGS[@]}" \
  sa_block_q=1024 sa_block_kv=1024 sa_block_kv_compute=1024 \
  sa_block_q_dkv=1024 sa_block_kv_dkv=1024 sa_block_kv_dkv_compute=1024 \
  sa_block_q_dq=1024 sa_block_kv_dq=1024 \
  megablox=true \
  sparse_matmul=true \
  use_tokamax_gmm=true \
  use_gmm_v2=true \
  use_gmm_v2_heuristic_tiling=false \
  moe_gmm_v2_dlhs_transpose_rhs=true \
  moe_use_direct_token_gather=true \
  prefuse_moe_weights="${PREFUSE_MOE_WEIGHTS}" \
  moe_unsliced_prefused_wi="${UNSLICED_WI}" \
  fused_mlp="${FUSED_MLP}" \
  wi_tile_fwd_batch_seq=512 wi_tile_fwd_embed_dim=2816 wi_tile_fwd_mlp_dim=1408 \
  wo_tile_fwd_batch_seq=1024 wo_tile_fwd_embed_dim=2816 wo_tile_fwd_mlp_dim=768 \
  wi_tile_dlhs_batch_seq=512 wi_tile_dlhs_embed_dim=2816 wi_tile_dlhs_mlp_dim=1408 \
  wo_tile_dlhs_batch_seq=1024 wo_tile_dlhs_embed_dim=2816 wo_tile_dlhs_mlp_dim=768 \
  wi_tile_drhs_batch_seq=512 wi_tile_drhs_embed_dim=2816 wi_tile_drhs_mlp_dim=1408 \
  wo_tile_drhs_batch_seq=1024 wo_tile_drhs_embed_dim=2816 wo_tile_drhs_mlp_dim=768 \
  moe_combine_kernel=tc \
  moe_permute_kernel=tc \
  "${CP_ARGS[@]}" \
  "${OPTIONAL_ARGS[@]}"
