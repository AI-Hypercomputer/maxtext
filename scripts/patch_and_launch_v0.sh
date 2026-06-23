#!/bin/bash

# This script launches a Reinforcement Learning (RL) training workload for the
# Qwen3-0.6B model on a GKE cluster using XPK.

set -e

# --- Environment Setup ---
if ! /usr/local/google/home/mohitkhatwani/max_venv/bin/pip show xpk &> /dev/null; then
    echo "xpk not found in the environment. Please install it by running:"
    echo "uv pip install -e .[runner] --resolution=lowest"
    exit 1
fi

# --- Environment Variables ---
export PROJECT_ID="${PROJECT_ID:-cloud-tpu-multipod-dev}" # GCP project ID where the Ironwood cluster is deployed
export CLUSTER_NAME="${CLUSTER_NAME:-bodaborg-super-xpk-x8p}" # Name of your Ironwood cluster
export ZONE="${ZONE:-us-central1-ai1a}" # Zone where your Ironwood cluster is deployed
export BASE_OUTPUT_DIRECTORY="${BASE_OUTPUT_DIRECTORY:-gs://runner-maxtext-logs}" # GCS bucket path for outputs
export DOCKER_IMAGE="${DOCKER_IMAGE:-gcr.io/tpu-prod-env-multipod/maxtext_post_training_nightly:2026-06-12}" # Full path to the Docker image you pushed (e.g., gcr.io/my-project/my-image:tag)
export MAXTEXT_CKPT_PATH="${MAXTEXT_CKPT_PATH:-gs://mohitkhatwani_multipods/qwen3-0.6b/pathways-compat/0/items}" # GCS path of the MaxText checkpoint to fine-tune from
export TPU_TYPE="tpu7x-128"
export WORKLOAD_NAME="rl-qwen3-$RANDOM"

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
SKIP_JAX_PRECOMPILE=0 \
TPU_MULTIPROCESS_DP=0 \
NEW_MODEL_DESIGN=1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
TPU_BACKEND_TYPE=jax \
TPU_MIN_LOG_LEVEL=0 \
TF_CPP_MIN_LOG_LEVEL=0 \
TPU_STDERR_LOG_LEVEL=0 \
JAX_PLATFORMS=proxy,cpu \
NUM_PRECOMPILE_WORKERS=4 \
JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 \
ENABLE_PATHWAYS_PERSISTENCE=1 \
PYTHONPATH=/app/src \
python3 -m maxtext.trainers.post_train.rl.train_rl \
model_name=qwen3-0.6b \
tokenizer_path=Qwen/Qwen3-0.6B \
run_name=$WORKLOAD_NAME-run \
checkpoint_storage_use_ocdbt=False \
async_scheduling=True \
base_output_directory=$BASE_OUTPUT_DIRECTORY \
chips_per_vm=8 \
num_batches=4 \
num_test_batches=0 \
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
max_target_length=24576 \
max_prefill_predict_length=16384 \
learning_rate=1e-6 \
batch_size=480 \
train_micro_batch_size=4 \
rollout_micro_batch_size=480 \
rollout_data_parallelism=16 \
rollout_tensor_parallelism=4 \
enable_dp_attention=false \
hbm_utilization_vllm=0.4 \
max_num_seqs=256 \
max_num_batched_tokens=24832 \
scan_layers=True \
allow_split_physical_axes=True \
enable_tunix_perf_metrics=True \
checkpoint_period=200 \
max_num_checkpoints_to_keep=1000 \
enable_checkpointing=true \
load_checkpoint_only_once=True \
load_parameters_path=$MAXTEXT_CKPT_PATH \
vllm_hf_overrides='{architectures: [\"MaxTextForCausalLM\"]}' \
vllm_additional_config='{\"maxtext_config\": {\"model_name\": \"qwen3-0.6b\", \"log_config\": \"false\"}}' \
convert_checkpoint_if_possible=False"

# 1. Run xpk build/upload command (without --dry-run) to build and upload the patched container image
echo "Building and uploading Docker container image..."
UPLOAD_LOG=$(/usr/local/google/home/mohitkhatwani/max_venv/bin/xpk workload create-pathways \
  --cluster=$CLUSTER_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --priority=very-high \
  --max-restarts=0 \
  --tpu-type=$TPU_TYPE \
  --num-slices=1 \
  --base-docker-image="gcr.io/tpu-prod-env-multipod/mohitkhatwani-rl:0717" \
  --server-image="us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:20260608-jax_0.10.1" \
  --proxy-server-image="us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:20260608-jax_0.10.1" \
  --workload="${WORKLOAD_NAME}" \
  --custom-pathways-proxy-server-args="${XLA_FLAGS} --temporary_flags_for_debugging=temporary_flag_for_debugging_pipe_unreachable_timeout=30m" \
  --custom-pathways-server-args="${XLA_FLAGS} --temporary_flags_for_debugging=temporary_flag_for_debugging_pipe_unreachable_timeout=30m" \
  --command="${MAXTEXT_COMMAND}" 2>&1 || true)

# 2. Extract the uploaded image tag from the log
UPLOADED_TAG=$(echo "$UPLOAD_LOG" | grep -oE "gcr.io/cloud-tpu-multipod-dev/mohitkhatwani-runner:[a-zA-Z0-9_-]+" | head -n 1)

if [ -z "$UPLOADED_TAG" ]; then
    echo "Failed to extract uploaded image tag from xpk output! Logs:"
    echo "$UPLOAD_LOG"
    exit 1
fi
echo "Successfully built and uploaded image: $UPLOADED_TAG"

# Clean up conflicting JobSet resource created during upload phase
kubectl delete jobset $WORKLOAD_NAME --ignore-not-found=true --wait=true

# 3. Run xpk create-pathways with --dry-run to generate the manifest
echo "Generating dry-run manifest..."
/usr/local/google/home/mohitkhatwani/max_venv/bin/xpk workload create-pathways \
  --cluster=$CLUSTER_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --priority=very-high \
  --max-restarts=0 \
  --tpu-type=$TPU_TYPE \
  --num-slices=1 \
  --base-docker-image="gcr.io/tpu-prod-env-multipod/mohitkhatwani-rl:0717" \
  --server-image="us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:20260608-jax_0.10.1" \
  --proxy-server-image="us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:20260608-jax_0.10.1" \
  --workload="${WORKLOAD_NAME}" \
  --custom-pathways-proxy-server-args="${XLA_FLAGS} --temporary_flags_for_debugging=temporary_flag_for_debugging_pipe_unreachable_timeout=30m" \
  --custom-pathways-server-args="--temporary_flags_for_debugging=temporary_flag_for_debugging_pipe_unreachable_timeout=30m" \
  --command="${MAXTEXT_COMMAND}" \
  --dry-run \
  --output-manifest-file=generated_manifest.yaml

# 4. Replace dry-run image placeholder tag with the uploaded tag in the manifest
echo "Replacing dry-run image placeholder in generated_manifest.yaml with $UPLOADED_TAG..."
sed -i "s|image: gcr.io/cloud-tpu-multipod-dev/dry-run-runner:prefix-current|image: $UPLOADED_TAG|g" generated_manifest.yaml

# 5. Apply GKE Warden bypass patches to generated_manifest.yaml
echo "Applying Warden webhook bypass patch to generated_manifest.yaml..."
python3 scripts/patch_manifest.py generated_manifest.yaml

# 6. Deploy the patched workload manifest
echo "Deploying the patched workload manifest..."
kubectl apply -f generated_manifest.yaml


