#!/bin/bash

# This script launches a Reinforcement Learning (RL) training workload for the
# Qwen3-30B-A3B model on a GKE cluster using XPK.

set -ex

# --- Environment Setup ---
if ! pip show xpk &> /dev/null; then
    echo "xpk not found in the environment. Please install it by running:"
    echo "uv pip install -e .[runner] --resolution=lowest"
    exit 1
fi

export RUN_ID=${1:-$(date +%Y-%m-%d-%H-%M-%S)}
export PROJECT_ID="${PROJECT_ID:-cloud-tpu-multipod-dev}" # GCP project ID where the cluster is deployed
export CLUSTER_NAME="${CLUSTER_NAME:-v5p-64-bodaborg-europe-west4-b}" # Name of your cluster
export ZONE="${ZONE:-europe-west4}" # Zone where your cluster is deployed
export BASE_OUTPUT_DIRECTORY="${BASE_OUTPUT_DIRECTORY:-gs://runner-maxtext-logs/qwen3-30b-a3b-base}"
export BASE_DOCKER_IMAGE="${BASE_DOCKER_IMAGE:-gcr.io/tpu-prod-env-multipod/maxtext_post_training_nightly:latest}" 
export MAXTEXT_CKPT_PATH="${MAXTEXT_CKPT_PATH:-${BASE_OUTPUT_DIRECTORY}/to_maxtext/unscanned/${RUN_ID}/0/items}"
export TPU_TYPE="v5p-64"
export WORKLOAD_NAME="rl-${RUN_ID}"
export MODEL_NAME="${MODEL_NAME:-qwen3-30b-a3b-base}"
export use_pathways="${use_pathways:-True}"
export run_id="${RUN_ID}"

# XLA Flags
XLA_FLAGS="--xla_tpu_dvfs_p_state=7 \
--xla_tpu_scoped_vmem_limit_kib=65536 \
--xla_tpu_num_sparse_cores_for_gather_offloading=1 \
--xla_tpu_bf16_emission_mode=NATIVE_EMISSION \
--xla_tpu_enable_sparse_core_reduce_scatter_v2=true \
--xla_tpu_enable_sparse_core_collective_offload_all_gather=true \
--xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true \
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
--xla_tpu_enable_3d_reduce_scatter_decomposer=false"

# Use base64 encoding to prevent any YAML formatting errors with xpk
MAXTEXT_COMMAND_B64=$(cat "$(dirname "$0")/test_qwen3_rl.sh" | base64 -w 0)

# Workload Creation
xpk workload create-pathways \
  --cluster=$CLUSTER_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --priority=medium \
  --max-restarts=0 \
  --tpu-type=$TPU_TYPE \
  --num-slices=1 \
  --base-docker-image="${BASE_DOCKER_IMAGE}" \
  --workload="${WORKLOAD_NAME}" \
  --custom-pathways-proxy-server-args="${XLA_FLAGS}" \
  --command="echo ${MAXTEXT_COMMAND_B64} | base64 -d | bash -s -- ${RUN_ID} ${use_pathways}"
