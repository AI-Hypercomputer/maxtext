#!/bin/bash
# Parameterized copy of repro.sh — unique run name so it doesn't collide with
# the hardcoded mlperf-ub3uaw. XLA_FLAGS and MAXTEXT_ARGS are verbatim from
# repro.sh (the 16.24s baseline). Override RUN_TAG to change the run name.

VENV=/usr/local/google/home/surajkolla/VENVS/ubench/bin/activate
[ -f "$VENV" ] && source "$VENV"

set -e
set -o pipefail

RUN_TAG="${RUN_TAG:-ds-raggedoff-0617}"

# --- Environment Variables ---
export PROJECT_ID="cloud-tpu-multipod-dev"
export CLUSTER_NAME="bodaborg-super-xpk-x8p"
export ZONE="us-central1-a"
export BASE_OUTPUT_DIR="gs://ubench-logs"
export ARTIFACT_DIR="gs://ubench-logs/${RUN_TAG}"
export WORKLOAD_IMAGE="gcr.io/cloud-tpu-multipod-dev/integrate_v2:gmmv2_bf16_log27"
export WORKLOAD_NAME="${RUN_TAG}"


# XLA Flags
XLA_FLAGS=" \
  --xla_tpu_dvfs_p_state=7 \
  --xla_tpu_scoped_vmem_limit_kib=65536 \
  --xla_tpu_bf16_emission_mode=NATIVE_EMISSION \
  --xla_tpu_enable_sparse_core_reduce_scatter_v2=true \
  --xla_tpu_enable_sparse_core_collective_offload_all_gather=true \
  --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true \
  --xla_tpu_enable_all_gather_offload_tracing=true \
  --xla_tpu_use_tc_device_shape_on_sc=True \
  --xla_sc_disable_megacore_partitioning=True \
  --xla_tpu_enable_async_collective_fusion_fuse_all_gather=false \
  --xla_enable_async_all_gather=true \
  --xla_tpu_prefer_async_allgather_to_allreduce=true \
  --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true \
  --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true \
  --xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true \
  --xla_tpu_use_single_sparse_core_for_all_gather_offload=true \
  --xla_tpu_enable_concurrent_sparse_core_offloading=true \
  --xla_tpu_aggressive_opt_barrier_removal=true \
  --xla_tpu_enable_offloading_gather_to_sparsecore=true \
  --xla_tpu_sparse_core_all_gather_latency_multiplier=1 \
  --xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3 \
  --xla_tpu_enable_sparse_core_collective_aggregator=true \
  --xla_tpu_enable_latency_hiding_layer_scheduler=true \
  --xla_tpu_scheduler_percent_shared_memory_limit=150 \
  --xla_tpu_enable_layer_scheduler_for_dependent_collectives=true \
  --xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true \
  --xla_tpu_pcie_bandwidth_multiplier=0.03 \
  --xla_tpu_enable_sparse_core_offload_queuing_in_lhs=true \
  --xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=false \
  --xla_tpu_enable_3d_reduce_scatter_decomposer=false "

# MaxText Workload Overrides
MAXTEXT_ARGS="\
model_name=deepseek3-671b \
per_device_batch_size=4.0 \
max_target_length=4096 \
ici_fsdp_parallelism=128 \
ici_expert_parallelism=4 \
ici_data_parallelism=1 \
ici_fsdp_transpose_parallelism=1 \
dcn_data_parallelism=-1 \
dcn_pipeline_parallelism=1 \
ici_pipeline_parallelism=1 \
shard_exp_on_fsdp=False \
use_iota_embed=True \
tokenizer_path=assets/tokenizer.mistral-v3 \
dataset_type=synthetic \
dataset_path=gs://max-datasets-rogue \
opt_type=adamw \
mu_dtype=bfloat16 \
grad_dtype=bfloat16 \
dtype=bfloat16 \
sa_use_fused_bwd_kernel=True \
megablox=True \
sparse_matmul=True \
use_tokamax_gmm=True \
use_tokamax_splash=True \
use_max_logit_estimate=-1 \
cost_estimate_flops_fwd=5000000000000 \
cost_estimate_flops_bwd=5000000000000 \
float32_weight_sum=False \
remat_policy=custom \
allow_split_physical_axes=False \
decoder_layer_input=device \
enable_tpu_profiling_options=True \
async_checkpointing=False \
enable_checkpointing=False \
attention=flash \
sa_block_q=2048 \
sa_block_kv=2048 \
sa_block_kv_compute=2048 \
sa_block_q_dkv=2048 \
sa_block_kv_dkv=2048 \
sa_block_kv_dkv_compute=2048 \
sa_block_kv_dq=2048 \
sa_block_q_dq=2048 \
use_random_routing=True \
use_ring_of_experts=False \
use_custom_sort_vjp=True \
use_ragged_sort=False \
merge_gating_gmm=False \
wi_tile_fwd_batch_seq=256 \
wi_tile_fwd_embed_dim=7168 \
wi_tile_fwd_mlp_dim=1024 \
wi_tile_dlhs_batch_seq=256 \
wi_tile_dlhs_embed_dim=3584 \
wi_tile_dlhs_mlp_dim=2048 \
wi_tile_drhs_batch_seq=512 \
wi_tile_drhs_embed_dim=1792 \
wi_tile_drhs_mlp_dim=2048 \
wo_tile_fwd_batch_seq=512 \
wo_tile_fwd_embed_dim=3584 \
wo_tile_fwd_mlp_dim=2048 \
wo_tile_dlhs_batch_seq=512 \
wo_tile_dlhs_embed_dim=1792 \
wo_tile_dlhs_mlp_dim=2048 \
wo_tile_drhs_batch_seq=512 \
wo_tile_drhs_embed_dim=1792 \
wo_tile_drhs_mlp_dim=2048 \
skip_jax_distributed_system=True \
steps=20 \
base_output_directory=${BASE_OUTPUT_DIR} \
run_name=${WORKLOAD_NAME} \
profiler=xplane \
skip_first_n_steps_for_profiler=5 \
profiler_steps=3"



echo "=== Creating XPK Workload: $WORKLOAD_NAME ==="
xpk workload create \
  --cluster=$CLUSTER_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --priority=very-high \
  --max-restarts=0 \
  --device-type=tpu7x-4x8x8 \
  --num-slices=1 \
  --docker-image="${WORKLOAD_IMAGE}" \
  --enable-debug-logs \
  --no-use-parallel-containers \
   \
  --workload="${WORKLOAD_NAME}" \
  --output-manifest-file=/tmp/ubench_recipe/${WORKLOAD_NAME}/recipe_for_github/k8s/k8s_manifest.yaml \
   \
  --command="set -e && set -o pipefail && export ENABLE_PATHWAYS_PERSISTENCE='1' && \
export LIBTPU_INIT_ARGS='${XLA_FLAGS}' && \
export ARTIFACT_DIR='${ARTIFACT_DIR}' && \
export JAX_PLATFORMS='tpu,cpu' && export ENABLE_PJRT_COMPATIBILITY='true' && \
 \
 \
python3 -m maxtext.trainers.pre_train.train maxtext/configs/base.yml ${MAXTEXT_ARGS} | tee train.log && \
gcloud storage cp --no-user-output-enabled train.log ${ARTIFACT_DIR}/logs/train-\${TPU_WORKER_ID}.log"
