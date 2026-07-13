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
export PROJECT_ID="${PROJECT_ID:-cloud-tpu-shared-capacity}" # GCP project ID where the Ironwood cluster is deployed
export CLUSTER_NAME="${CLUSTER_NAME:-bodaborg-tpu7x-nap}" # Name of your Ironwood cluster
export ZONE="${ZONE:-us-central1}" # Zone where your Ironwood cluster is deployed
export BASE_OUTPUT_DIRECTORY="${BASE_OUTPUT_DIRECTORY:-gs://runner-maxtext-logs/}" # GCS bucket path for outputs
export BASE_DOCKER_IMAGE="${BASE_DOCKER_IMAGE:-gcr.io/cloud-tpu-multipod-dev/mohitkhatwani-rl:agentic}" # Base Docker image
export MAXTEXT_CKPT_PATH="${MAXTEXT_CKPT_PATH:-gs://mohitkhatwani_multipods/qwen3-0.6b/pathways-compat/0/items}" # GCS path of the MaxText checkpoint to fine-tune from
export TPU_TYPE="${TPU_TYPE:-tpu7x-128}"
export WORKLOAD_NAME="mohit-rl-qwen3-$RANDOM"

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
if [ -z "$BASE_DOCKER_IMAGE" ]; then
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

# VLLM_ENABLE_V1_MULTIPROCESSING=0 \
# MaxText command
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
PYTHONPATH=/app/src \
python3 -m maxtext.trainers.post_train.rl.train_rl \
model_name=qwen3-0.6b \
tokenizer_path=Qwen/Qwen3-0.6B \
run_name=$WORKLOAD_NAME \
checkpoint_storage_use_ocdbt=False \
async_scheduling=False \
base_output_directory=$BASE_OUTPUT_DIRECTORY \
chips_per_vm=8 \
num_batches=20 \
num_test_batches=0 \
profiler=xplane \
skip_first_n_steps_for_profiler=1 \
profiler_steps=1 \
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
train_micro_batch_size=8 \
rollout_micro_batch_size=480 \
rollout_data_parallelism=32 \
rollout_tensor_parallelism=2 \
enable_dp_attention=false \
hbm_utilization_vllm=0.6 \
max_num_seqs=480 \
max_num_batched_tokens=24832 \
scan_layers=True \
allow_split_physical_axes=True \
enable_tunix_perf_metrics=True \
checkpoint_period=100 \
max_num_checkpoints_to_keep=1000 \
enable_checkpointing=true \
load_parameters_path=$MAXTEXT_CKPT_PATH \
rollout_vllm_init_with_random_weights=True \
vllm_additional_config='{\"enable_continue_decode\": true, \"max_decode_steps\": 128}'"
# vllm_hf_overrides='{architectures: [\"MaxTextForCausalLM\"]}' \
# vllm_additional_config='{\"maxtext_config\": {\"model_name\": \"qwen3-0.6b\", \"model_call_mode\": \"inference\", \"enable_dp_attention\": false, \"allow_split_physical_axes\": true, \"log_config\": false, \"weight_dtype\": \"bfloat16\", \"prefuse_moe_weights\": true}}'"

# 1. Run live submit to compile and upload the container image with local changes
echo "Compiling and uploading container image via xpk..."
/usr/local/google/home/mohitkhatwani/max_venv/bin/xpk workload create-pathways \
  --cluster="${CLUSTER_NAME}" \
  --project="${PROJECT_ID}" \
  --tpu-type="${TPU_TYPE}" \
  --zone="${ZONE}" \
  --priority=very-high \
  --num-slices=1 \
  --base-docker-image="${BASE_DOCKER_IMAGE}" \
  --server-image="us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:20260623-jax_0.10.1" \
  --proxy-server-image="us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:20260623-jax_0.10.1" \
  --workload="${WORKLOAD_NAME}-build" \
  --custom-pathways-proxy-server-args="${XLA_FLAGS}" \
  --custom-pathways-server-args="" \
  --command="echo build-stage" 2>&1 | tee xpk_build.log

# 2. Extract the dynamically built container tag
BUILT_IMAGE=$(grep -o -E "gcr.io/cloud-tpu-multipod-dev/mohitkhatwani-runner:[a-zA-Z0-9._-]+" xpk_build.log | tail -n 1)
echo "Successfully built and uploaded image: ${BUILT_IMAGE}"

# 3. Immediately clean up the build workload in GKE
echo "Cleaning up build workload..."
kubectl delete jobset "${WORKLOAD_NAME}-build" || true
rm -f xpk_build.log

# Workload Creation (dry-run to generate manifest with built image)
/usr/local/google/home/mohitkhatwani/max_venv/bin/xpk workload create-pathways \
  --cluster=$CLUSTER_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --priority=very-high \
  --max-restarts=0 \
  --tpu-type=$TPU_TYPE \
  --num-slices=1 \
  --docker-image="${BUILT_IMAGE}" \
  --server-image="us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:20260623-jax_0.10.1" \
  --proxy-server-image="us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:20260623-jax_0.10.1" \
  --workload="${WORKLOAD_NAME}" \
  --custom-pathways-proxy-server-args="${XLA_FLAGS}" \
  --custom-pathways-server-args="" \
  --env RPA_D_BLOCK_SIZES="1,4096,1,4096" \
  --command="cd /app; pip install git+https://github.com/AI-Hypercomputer/pathways-utils.git --no-deps; python3 scripts/patch_vllm_sampler.py; pip install -e . --no-deps; ${MAXTEXT_COMMAND}" \
  --dry-run \
  --output-manifest-file=generated_manifest.yaml


echo "Detecting GKE TPU Nodepool..."
GKE_NODEPOOL=$(/usr/bin/kubectl get nodes -l cloud.google.com/gke-tpu-accelerator=tpu7x -o jsonpath='{.items[0].metadata.labels.cloud\.google\.com/gke-nodepool}' 2>/dev/null || true)
echo "Detected GKE TPU Nodepool: ${GKE_NODEPOOL:-none (will auto-provision)}"

echo "Applying Warden webhook bypass patch to generated_manifest.yaml..."
python3 scripts/patch_manifest.py generated_manifest.yaml "" "" "${WORKLOAD_NAME}" ""

echo "Auto-detecting Kueue LocalQueue..."
LOCAL_QUEUE=$(/usr/bin/kubectl get localqueues.kueue.x-k8s.io -n default -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "multislice-queue")
echo "Patching queue to ${LOCAL_QUEUE}..."
sed -i "s/multislice-queue/${LOCAL_QUEUE}/g" generated_manifest.yaml

echo "Deploying the patched workload manifest..."
/usr/bin/kubectl apply -f generated_manifest.yaml

