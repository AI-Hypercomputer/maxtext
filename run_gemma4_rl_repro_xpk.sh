#!/bin/bash

# This script launches a Reinforcement Learning (RL) training workload for the
# Gemma4-e2b model on a GKE cluster using XPK (Base image to reproduce multi-host decode bug).

set -e

# --- Environment Setup ---
XPK_BIN="/usr/local/google/home/mazumdera/.venv-py312/bin/xpk"
if [ ! -f "$XPK_BIN" ]; then
    echo "xpk not found at $XPK_BIN. Please verify the virtual environment."
    exit 1
fi

# --- Environment Variables ---
export PROJECT_ID="${PROJECT_ID:-cloud-tpu-multipod-dev}"
export CLUSTER_NAME="${CLUSTER_NAME:-bodaborg-tpu7x-auto-nap2}"
export ZONE="${ZONE:-us-central1-c}"
export BASE_OUTPUT_DIRECTORY="${BASE_OUTPUT_DIRECTORY:-gs://mazumdera-test-bucket-europe-west4/gemma4-e2b}"
export BASE_DOCKER_IMAGE="${BASE_DOCKER_IMAGE:-gcr.io/cloud-tpu-multipod-dev/anisha-tmvp/anisha-0630:gemma4_2b_repro}"
export MAXTEXT_CKPT_PATH="${MAXTEXT_CKPT_PATH:-gs://maxtext-gemma/gemma4/e2b/converted/2026-07-01-21-56/0/items}"
export TPU_TYPE="${TPU_TYPE:-tpu7x-128}"
export WORKLOAD_NAME="${WORKLOAD_NAME:-maz-gm4-repro-$RANDOM}"

# --- Variable Validation ---
if [ -z "$PROJECT_ID" ]; then
    echo "Error: PROJECT_ID is not set."
    exit 1
fi
if [ -z "$CLUSTER_NAME" ]; then
    echo "Error: CLUSTER_NAME is not set."
    exit 1
fi
if [ -z "$ZONE" ]; then
    echo "Error: ZONE is not set."
    exit 1
fi
if [ -z "$BASE_OUTPUT_DIRECTORY" ]; then
    echo "Error: BASE_OUTPUT_DIRECTORY is not set."
    exit 1
fi
if [ -z "$BASE_DOCKER_IMAGE" ]; then
    echo "Error: BASE_DOCKER_IMAGE is not set."
    exit 1
fi
if [ -z "$MAXTEXT_CKPT_PATH" ]; then
    echo "Error: MAXTEXT_CKPT_PATH is not set."
    exit 1
fi

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

# MaxText command for Gemma4-e2b RL
MAXTEXT_COMMAND="HF_TOKEN='' \
SKIP_JAX_PRECOMPILE=1 \
JAX_RANDOM_WEIGHTS=1 \
NEW_MODEL_DESIGN=1 \
TPU_MIN_LOG_LEVEL=0 \
TF_CPP_MIN_LOG_LEVEL=0 \
TPU_STDERR_LOG_LEVEL=0 \
JAX_PLATFORMS=proxy,cpu \
NUM_PRECOMPILE_WORKERS=1 \
JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 \
ENABLE_PATHWAYS_PERSISTENCE=1 \
PYTHONPATH=/deps/src:/deps \
python3 -m maxtext.trainers.post_train.rl.train_rl \
model_name=gemma4-e2b \
load_parameters_path=$MAXTEXT_CKPT_PATH \
run_name=$WORKLOAD_NAME \
base_output_directory=$BASE_OUTPUT_DIRECTORY \
chips_per_vm=8 \
num_batches=10 \
trainer_devices_fraction=0.5 \
sampler_devices_fraction=0.5 \
rollout_data_parallelism=8 \
rollout_tensor_parallelism=8 \
scan_layers=False \
rl.use_agentic_rollout=False \
vllm_hf_overrides='{\"architectures\": [\"MaxTextForCausalLM\"]}' \
vllm_additional_config='{\"maxtext_config\": {\"model_name\": \"gemma4-e2b\", \"model_call_mode\": \"inference\", \"enable_dp_attention\": false, \"allow_split_physical_axes\": true, \"log_config\": false, \"weight_dtype\": \"bfloat16\", \"scan_layers\": false}}'"

echo "Submitting workload via xpk..."
$XPK_BIN workload create-pathways \
  --cluster="$CLUSTER_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --priority=very-high \
  --max-restarts=10 \
  --tpu-type="$TPU_TYPE" \
  --num-slices=1 \
  --docker-image="$BASE_DOCKER_IMAGE" \
  --server-image="us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:20260623-jax_0.10.1" \
  --proxy-server-image="us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:20260623-jax_0.10.1" \
  --workload="$WORKLOAD_NAME" \
  --custom-pathways-proxy-server-args="$XLA_FLAGS" \
  --custom-pathways-server-args="" \
  --env RPA_D_BLOCK_SIZES="1,4096,1,4096" \
  --command="cd /deps && pip install git+https://github.com/AI-Hypercomputer/pathways-utils.git --no-deps && $MAXTEXT_COMMAND"
