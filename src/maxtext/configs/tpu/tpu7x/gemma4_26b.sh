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

# Gemma4-26b (MoE, 128 experts top-8 + shared expert) pre-training on TPU v7x (Ironwood).
# Measured on tpu7x-128 (4x4x4, 64 chips): seq 8192, per-device batch 2 (2.1M tokens/step),
# 1.83 s/step, 20.3% MFU.
#
# Command Flags:
# OUTPUT_PATH (Required, unless base_output_directory is already set in base.yml)
# RUN_NAME (Required, unless run_name is already set in base.yml or the JOBSET_NAME environment variable is set)
#
# Example to invoke this script:
# bash src/maxtext/configs/tpu/tpu7x/gemma4_26b.sh RUN_NAME="<your_run_name>" OUTPUT_PATH="gs://<your_output_path>"
#

# Stop execution if any command exits with error
set -e

export EXECUTABLE="train" # or train_compile
export RUN_PREFLIGHT="true"
export STEPS="${STEPS:-40}"

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

# The setup accommodates two cases:
# 1) Passing the 'RUN_NAME' variable at runtime
# 2) Propagating the 'M_RUN_NAME' variable within an Airflow sweeping workflow
if [ -n "$RUN_NAME" ];
then
    export M_RUN_NAME=$RUN_NAME
fi

# Set up network optimizations
if [ "$RUN_PREFLIGHT" = "true" ]; then
    bash src/dependencies/scripts/preflight.sh
fi

# XLA / libtpu flags: collective offload to SparseCore, async all-gather and the latency-hiding
# layer scheduler (the TPU v7x MoE flag set). One flag per line.
XLA_FLAGS_LIST=(
  --xla_tpu_dvfs_p_state=7
  --xla_tpu_scoped_vmem_limit_kib=65472
  --xla_tpu_bf16_emission_mode=NATIVE_EMISSION
  --xla_tpu_enable_sparse_core_reduce_scatter_v2=true
  --xla_tpu_enable_sparse_core_collective_offload_all_gather=true
  --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true
  --xla_tpu_enable_all_gather_offload_tracing=true
  --xla_tpu_use_tc_device_shape_on_sc=true
  --xla_sc_disable_megacore_partitioning=true
  --xla_tpu_enable_async_collective_fusion_fuse_all_gather=false
  --xla_enable_async_all_gather=true
  --xla_tpu_prefer_async_allgather_to_allreduce=true
  --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true
  --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true
  --xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true
  --xla_tpu_use_single_sparse_core_for_all_gather_offload=true
  --xla_tpu_enable_concurrent_sparse_core_offloading=true
  --xla_tpu_aggressive_opt_barrier_removal=true
  --xla_tpu_enable_offloading_gather_to_sparsecore=true
  --xla_tpu_sparse_core_all_gather_latency_multiplier=1
  --xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3
  --xla_tpu_enable_sparse_core_collective_aggregator=true
  --xla_tpu_enable_latency_hiding_layer_scheduler=true
  --xla_tpu_scheduler_percent_shared_memory_limit=150
  --xla_tpu_enable_layer_scheduler_for_dependent_collectives=true
  --xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true
  --xla_tpu_pcie_bandwidth_multiplier=0.03
  --xla_tpu_enable_sparse_core_offload_queuing_in_lhs=true
  --xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=false
  --xla_tpu_enable_3d_reduce_scatter_decomposer=false
)
export LIBTPU_INIT_ARGS="${XLA_FLAGS_LIST[*]}"

# MaxText overrides of base.yml. One flag per line.
MAXTEXT_ARGS=(
  model_name=gemma4-26b
  per_device_batch_size=2
  max_target_length=8192
  steps="${STEPS}"
  dataset_type=synthetic
  reuse_example_batch=1
  enable_checkpointing=false
  gcs_metrics=true
  base_output_directory="${OUTPUT_PATH}"
  dtype=bfloat16
  # Rematerialization.
  remat_policy=full
  # Sharding: FSDP over every device.
  ici_fsdp_parallelism=-1
  ici_expert_parallelism=1
  ici_tensor_parallelism=1
  # Routed experts: dropless sparse matmul with the Tokamax grouped matmul.
  megablox=true
  sparse_matmul=true
  use_tokamax_gmm=true
  # Tokamax GMM v2 with tiles for Gemma4's expert shapes (embed 2816, expert mlp 704 tiled at 768 =
  # 6 x 128 lanes); the backward DLHS reads the gathered expert weight without a transposed copy.
  use_gmm_v2=true
  use_gmm_v2_heuristic_tiling=false
  moe_gmm_v2_dlhs_transpose_rhs=true
  wi_tile_fwd_batch_seq=1024
  wi_tile_fwd_embed_dim=2816
  wi_tile_fwd_mlp_dim=768
  wi_tile_dlhs_batch_seq=1024
  wi_tile_dlhs_embed_dim=2816
  wi_tile_dlhs_mlp_dim=768
  wi_tile_drhs_batch_seq=512
  wi_tile_drhs_embed_dim=2816
  wi_tile_drhs_mlp_dim=768
  wo_tile_fwd_batch_seq=1024
  wo_tile_fwd_embed_dim=2816
  wo_tile_fwd_mlp_dim=768
  wo_tile_dlhs_batch_seq=1024
  wo_tile_dlhs_embed_dim=2816
  wo_tile_dlhs_mlp_dim=768
  wo_tile_drhs_batch_seq=512
  wo_tile_drhs_embed_dim=2816
  wo_tile_drhs_mlp_dim=768
  # Attention: splash (flash) attention for both sliding-window and global layers.
  attention=flash
  scan_layers=true
)

python3 -m maxtext.trainers.pre_train.$EXECUTABLE "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"/base.yml \
  "${MAXTEXT_ARGS[@]}"
