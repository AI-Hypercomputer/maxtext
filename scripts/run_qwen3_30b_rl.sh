#!/bin/bash

# This script launches a Reinforcement Learning (RL) training workload for the
# Qwen3-30B-A3B model on a GKE cluster using XPK.

set -e

# --- Environment Setup ---
if ! pip show xpk &> /dev/null; then
    echo "xpk not found in the environment. Please install it by running:"
    echo "uv pip install -e .[runner] --resolution=lowest"
    exit 1
fi

# --- Environment Variables ---
export PROJECT_ID="${PROJECT_ID:-}" # GCP project ID where the Ironwood cluster is deployed
export CLUSTER_NAME="${CLUSTER_NAME:-}" # Name of your Ironwood cluster
export ZONE="${ZONE:-}" # Zone where your Ironwood cluster is deployed
export BASE_OUTPUT_DIRECTORY="${BASE_OUTPUT_DIRECTORY:-}" # GCS bucket path for outputs (e.g., gs://my-bucket/outputs)
export DOCKER_IMAGE="${DOCKER_IMAGE:-}" # Full path to the Docker image you pushed (e.g., gcr.io/my-project/my-image:tag)
export MAXTEXT_CKPT_PATH="${MAXTEXT_CKPT_PATH:-}" # GCS path of the MaxText checkpoint you want to fine-tune from (e.g., gs://my-bucket/checkpoints/maxtext-ckpt)
export TPU_TYPE="tpu7x-128"
export WORKLOAD_NAME="rl-$(date +%Y%m%d-%H%M)"

# --- Variable Validation ---
if [ -z "$PROJECT_ID" ]; then
    echo "Error: PROJECT_ID is not set. Please set it in the script or as an environment variable."
    exit 1
fi
if [ -z "$CLUSTER_NAME" ]; then
    echo "Error: CLUSTER_NAME is not set. Please set it in the script or as an environment variable."
    exit 1
fi
if [ -z "$ZONE" ]; then
    echo "Error: ZONE is not set. Please set it in the script or as an environment variable."
    exit 1
fi
if [ -z "$BASE_OUTPUT_DIRECTORY" ]; then
    echo "Error: BASE_OUTPUT_DIRECTORY is not set. Please set it in the script or as an environment variable."
    exit 1
fi
if [ -z "$DOCKER_IMAGE" ]; then
    echo "Error: DOCKER_IMAGE is not set. Please set it in the script or as an environment variable."
    exit 1
fi

if [ -z "$MAXTEXT_CKPT_PATH" ]; then
    echo "MAXTEXT_CKPT_PATH is not set. Please set it in the script or as an environment variable."
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

# MaxText command
MAXTEXT_COMMAND="JAX_RANDOM_WEIGHTS=1 \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
NEW_MODEL_DESIGN=1 \
TPU_MIN_LOG_LEVEL=0 \
TF_CPP_MIN_LOG_LEVEL=0 \
TPU_STDERR_LOG_LEVEL=0 \
JAX_PLATFORMS=proxy,cpu \
JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 \
ENABLE_PATHWAYS_PERSISTENCE=1 \
python3 -m maxtext.trainers.post_train.rl.train_rl \
model_name=qwen3-30b-a3b \
tokenizer_path=Qwen/Qwen3-30B-A3B-Base \
run_name=$WORKLOAD_NAME \
async_scheduling=True \
base_output_directory=$BASE_OUTPUT_DIRECTORY \
chips_per_vm=8 \
num_batches=500 \
num_test_batches=10 \
rl.num_generations=8 \
rl.grpo_beta=0.05 \
rl.grpo_epsilon=0.2 \
gradient_clipping_threshold=1.0 \
decode_sampling_temperature=0.8 \
decode_sampling_top_k=50 \
decode_sampling_nucleus_p=0.95 \
dataset_name=nvidia/OpenMathInstruct-2 \
hf_train_files=hf://datasets/nvidia/OpenMathInstruct-2/data/train_1M-*.parquet \
train_split=train_1M \
eval_dataset_name=nvidia/OpenMathInstruct-2 \
eval_mode=pass_at_1 \
num_eval_passes=4 \
max_target_length=8192 \
max_prefill_predict_length=512 \
learning_rate=1e-6 \
batch_size=128 \
train_micro_batch_size=16 \
rollout_micro_batch_size=128 \
rollout_data_parallelism=16 \
rollout_tensor_parallelism=4 \
enable_dp_attention=True \
hbm_utilization_vllm=0.75 \
max_num_seqs=256 \
max_num_batched_tokens=8192 \
scan_layers=True \
allow_split_physical_axes=True \
enable_tunix_perf_metrics=True \
checkpoint_period=2 \
max_num_checkpoints_to_keep=1000 \
enable_checkpointing=true \
load_parameters_path=$MAXTEXT_CKPT_PATH"

# Workload Creation
xpk workload create-pathways \
  --cluster=$CLUSTER_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --priority=medium \
  --max-restarts=0 \
  --tpu-type=$TPU_TYPE \
  --num-slices=1 \
  --docker-image="${DOCKER_IMAGE}" \
  --workload="${WORKLOAD_NAME}" \
  --custom-pathways-proxy-server-args='${XLA_FLAGS}' \
  --command="${MAXTEXT_COMMAND}"
