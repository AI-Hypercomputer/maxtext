#!/bin/bash

# This script launches a Reinforcement Learning (RL) training workload for the
# gemma4-e2b model on a GKE cluster using Cluster Toolkit (gcluster).

set -e

# --- Environment Setup ---
if ! command -v gcluster &> /dev/null; then
    echo "gcluster not found in the environment. Please install Cluster Toolkit before running this script."
    exit 1
fi

# --- Environment Variables ---
export PROJECT_ID="${PROJECT_ID:-}" # GCP project ID where the TPU v7x (Ironwood) cluster is deployed
export CLUSTER_NAME="${CLUSTER_NAME:-}" # Name of your TPU v7x (Ironwood) cluster
export ZONE="${ZONE:-}" # Zone where your TPU v7x cluster is deployed
export BASE_OUTPUT_DIRECTORY="${BASE_OUTPUT_DIRECTORY:-}" # GCS bucket path for outputs (e.g., gs://my-bucket/outputs)
export DOCKER_IMAGE="${DOCKER_IMAGE:-}" # Full path to the Docker image you pushed (e.g., us-docker.pkg.dev/<project_id>/<repository_name>/<image_name>:<tag>)
export MAXTEXT_CKPT_PATH="${MAXTEXT_CKPT_PATH:-}" # GCS path of the MaxText checkpoint you want to fine-tune from (e.g., gs://my-bucket/checkpoints/maxtext-ckpt)
export COMPUTE_TYPE="${COMPUTE_TYPE:-tpu7x-standard-4t}" # TPU v7x (Ironwood), 4 chips (8 devices) per VM
export TOPOLOGY="${TOPOLOGY:-4x4x4}" # TPU v7x 4x4x4 = 64-chip slice
export RUN_NAME="${RUN_NAME:-rl-$(date +%Y%m%d-%H%M)}"
export NUM_BATCHES="${NUM_BATCHES:-500}" # GRPO updates (use 20 for a quick performance check)
export HF_TOKEN="${HF_TOKEN:-}" # Hugging Face token: google/gemma-4-E2B-it (tokenizer) is a gated repository

# Pathways component images, pinned to specific versions (mirrors the reference config).
# Note: PATHWAYS_SERVER_IMAGE is used for BOTH the Pathways resource-manager server and the
# workers; in the reference config the worker and pathwaysServer images are identical.
export PATHWAYS_SERVER_IMAGE="${PATHWAYS_SERVER_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server@sha256:bf2da6814806695b70724f61c5c5b53f5e10b018d3d51bd05e36530d95bd3e32}"
export PATHWAYS_PROXY_SERVER_IMAGE="${PATHWAYS_PROXY_SERVER_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server@sha256:ddf6315fc10cf8ce050b0b11b7109ec78ae1a5931c29f3b3f8585fc2d0f9aeda}"

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

# XLA flags (TPU v7x / Ironwood: SparseCore collective offload, DVFS p-state 6)
XLA_FLAGS="--xla_tpu_dvfs_p_state=6 \
--xla_tpu_scoped_vmem_limit_kib=65536 \
--xla_tpu_bf16_emission_mode=NATIVE_EMISSION \
--xla_tpu_num_sparse_cores_for_gather_offloading=1 \
--xla_tpu_enable_sparse_core_reduce_scatter_v2=true \
--xla_tpu_enable_sparse_core_collective_offload_all_gather=true \
--xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true \
--xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true \
--xla_tpu_use_single_sparse_core_for_all_gather_offload=true \
--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true \
--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true \
--xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true \
--xla_tpu_enable_sparse_core_collective_aggregator=true \
--xla_tpu_use_tc_device_shape_on_sc=True \
--xla_sc_disable_megacore_partitioning=True \
--xla_enable_async_all_gather=true \
--xla_tpu_prefer_async_allgather_to_allreduce=true \
--xla_tpu_enable_latency_hiding_layer_scheduler=true \
--xla_tpu_scheduler_percent_shared_memory_limit=150 \
--xla_tpu_enable_layer_scheduler_for_dependent_collectives=true \
--xla_tpu_enable_concurrent_sparse_core_offloading=true \
--xla_tpu_enable_offloading_gather_to_sparsecore=true \
--xla_tpu_enable_sparse_core_offload_queuing_in_lhs=true \
--xla_tpu_enable_3d_reduce_scatter_decomposer=false \
--xla_tpu_enable_async_collective_fusion_fuse_all_gather=false \
--xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=false \
--xla_tpu_pcie_bandwidth_multiplier=0.03 \
--xla_tpu_sparse_core_all_gather_latency_multiplier=1 \
--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3 \
--xla_tpu_use_enhanced_launch_barrier=true"

# Rollout settings: continue-decode runs each 1024-step decode window to its full depth
# (CONTINUE_DECODE_EOS_CHECK_INTERVAL=0) instead of returning to the host on the first EOS; round-robin DP routing
# (DP_SCHED_ROUTING=round_robin) avoids a per-request query of all 128 data-parallel ranks. The routing knob needs
# tpu-inference 1224e9fb (2026-09-18) or newer, which the pin in src/dependencies/extra_deps/post_train_github_deps.txt
# provides; an older tpu-inference silently falls back to least_loaded routing.
MAXTEXT_COMMAND="VLLM_ENABLE_V1_MULTIPROCESSING=0 \
NEW_MODEL_DESIGN=1 \
TF_CPP_MIN_LOG_LEVEL=0 \
ENABLE_PJRT_COMPATIBILITY=true \
ENABLE_PATHWAYS_PERSISTENCE=1 \
TPU_BACKEND_TYPE=jax \
VLLM_LOGGING_LEVEL=WARNING \
JAX_PLATFORMS=proxy,cpu \
JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 \
NUM_SLICES=1 \
VLLM_ENGINE_READY_TIMEOUT_S=7200 \
TOKENIZERS_PARALLELISM=False \
CONTINUE_DECODE_EOS_CHECK_INTERVAL=0 \
DP_SCHED_ROUTING=round_robin \
python3 -m maxtext.trainers.post_train.rl.train_rl \
model_name=gemma4-e2b \
tokenizer_type=huggingface \
tokenizer_path=google/gemma-4-E2B-it \
hf_access_token=${HF_TOKEN} \
run_name=$RUN_NAME \
base_output_directory=$BASE_OUTPUT_DIRECTORY \
load_parameters_path=$MAXTEXT_CKPT_PATH \
chips_per_vm=8 \
async_scheduling=False \
data_template_path=maxtext/examples/chat_templates/openmathinstruct2_rl.json \
hf_name=default \
dataset_name=nvidia/OpenMathInstruct-2 \
hf_train_files=hf://datasets/nvidia/OpenMathInstruct-2/data/train_1M-*.parquet \
train_split=train_1M \
num_batches=${NUM_BATCHES} \
num_test_batches=0 \
batch_size=256 \
train_micro_batch_size=8 \
rollout_micro_batch_size=256 \
rl.num_generations=8 \
rl.grpo_beta=0.0 \
rl.grpo_epsilon=0.2 \
rl.epsilon_high=0.28 \
learning_rate=1e-6 \
gradient_clipping_threshold=1.0 \
decode_sampling_temperature=0.8 \
decode_sampling_top_k=50 \
decode_sampling_nucleus_p=0.95 \
max_prefill_predict_length=4096 \
max_target_length=12288 \
kv_cache_buffer=12288 \
max_seq_token_per_tpu=12288 \
attention=flash \
local_sa_block_q=512 \
local_sa_block_kv=512 \
local_sa_block_kv_compute=512 \
local_sa_block_q_dkv=512 \
local_sa_block_kv_dkv=512 \
local_sa_block_kv_dkv_compute=512 \
local_sa_block_q_dq=512 \
local_sa_block_kv_dq=512 \
eval_local_sa_block_q=512 \
eval_local_sa_block_kv=512 \
eval_local_sa_block_kv_compute=512 \
local_sa_q_layout=HEAD_DIM_MINOR \
local_sa_k_layout=HEAD_DIM_MINOR \
local_sa_v_layout=HEAD_DIM_MINOR \
eval_local_sa_q_layout=HEAD_DIM_MINOR \
eval_local_sa_k_layout=HEAD_DIM_MINOR \
eval_local_sa_v_layout=HEAD_DIM_MINOR \
local_sa_use_fused_bwd_kernel=false \
local_sa_fuse_reciprocal=true \
local_sa_use_base2_exp=true \
scan_layers=False \
allow_split_physical_axes=True \
trainer_devices_fraction=1.0 \
sampler_devices_fraction=1.0 \
rollout_data_parallelism=128 \
rollout_tensor_parallelism=1 \
enable_dp_attention=False \
hbm_utilization_vllm=0.3 \
max_num_seqs=256 \
max_num_batched_tokens=33280 \
vllm_block_size=128 \
free_kv_cache_during_weight_sync=False \
math_verify_num_procs=64 \
enable_tunix_perf_metrics=True \
enable_checkpointing=True \
checkpoint_period=100 \
vllm_hf_overrides='{architectures: [\"MaxTextForCausalLM\"]}' \
vllm_additional_config='{\"enable_continue_decode\": true, \"max_decode_steps\": 1024, \"maxtext_config\": {\"model_name\": \"gemma4-e2b\", \"model_call_mode\": \"inference\", \"log_config\": false, \"weight_dtype\": \"bfloat16\", \"allow_split_physical_axes\": true}}'"

# Workload Creation
# Configure target cluster in gcluster job config (gcluster job submit reads project/cluster/location from job config)
gcluster job config set project "$PROJECT_ID"
gcluster job config set cluster "$CLUSTER_NAME"
gcluster job config set location "$ZONE"

# Submit Pathways workload (see `gcluster job submit --help` for --pathways-* flag options)
gcluster job submit \
  --image="${DOCKER_IMAGE}" \
  --name="${RUN_NAME}" \
  --pathways \
  --compute-type="${COMPUTE_TYPE}" \
  --topology="${TOPOLOGY}" \
  --num-slices=1 \
  --priority=medium \
  --restarts=0 \
  --pathways-gcs-location="${BASE_OUTPUT_DIRECTORY}" \
  --pathways-server-image="${PATHWAYS_SERVER_IMAGE}" \
  --pathways-worker-image="${PATHWAYS_SERVER_IMAGE}" \
  --pathways-proxy-server-image="${PATHWAYS_PROXY_SERVER_IMAGE}" \
  --pathways-proxy-args="${XLA_FLAGS}" \
  --command="${MAXTEXT_COMMAND}"
