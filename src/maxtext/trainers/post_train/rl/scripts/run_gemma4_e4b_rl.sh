#!/bin/bash

# This script launches a Reinforcement Learning (RL) training workload for the
# gemma4-e4b model on a GKE cluster using XPK.

set -e

# --- Environment Setup ---
if ! pip show xpk &> /dev/null; then
    echo "xpk not found in the environment. Please install it by running:"
    echo "uv pip install -e .[runner] --resolution=lowest"
    exit 1
fi

# --- Environment Variables ---
export PROJECT_ID="${PROJECT_ID:-}" # GCP project ID where the v6e cluster is deployed
export CLUSTER_NAME="${CLUSTER_NAME:-}" # Name of your v6e (Trillium) cluster
export ZONE="${ZONE:-}" # Zone where your v6e cluster is deployed
export BASE_OUTPUT_DIRECTORY="${BASE_OUTPUT_DIRECTORY:-}" # GCS bucket path for outputs (e.g., gs://my-bucket/outputs)
export DOCKER_IMAGE="${DOCKER_IMAGE:-}" # Full path to the Docker image you pushed (e.g., gcr.io/my-project/my-image:tag)
export MAXTEXT_CKPT_PATH="${MAXTEXT_CKPT_PATH:-}" # GCS path of the MaxText checkpoint you want to fine-tune from (e.g., gs://my-bucket/checkpoints/maxtext-ckpt)
export TPU_TYPE="v6e-32"
export WORKLOAD_NAME="rl-$(date +%Y%m%d-%H%M)"

# Pathways component images, pinned to specific versions (mirrors the reference config).
# Note: --server-image is used for BOTH the Pathways resource-manager server and the
# workers; in the reference config the worker and pathwaysServer images are identical.
export PATHWAYS_SERVER_IMAGE="us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server@sha256:d03068b39a8a2fab0621086ccb7c9445ce17ad34f1520159ca4ecd395346a162"
export PATHWAYS_PROXY_SERVER_IMAGE="us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server@sha256:6342a3cae2818d2f887d396e9ae4156b3316f79fef186d71ff16d535b44e1724"

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

# XLA Flags (tuned for v6e / Trillium)
XLA_FLAGS="--xla_tpu_scoped_vmem_limit_kib=65536 \
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

# MaxText command
MAXTEXT_COMMAND="JAX_RANDOM_WEIGHTS=1 \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
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
RPA_D_BLOCK_SIZES=1,1024,1,512 \
TOKENIZERS_PARALLELISM=False \
python3 -m maxtext.trainers.post_train.rl.train_rl \
model_name=gemma4-e4b \
tokenizer_type=huggingface \
tokenizer_path=google/gemma-4-E4B \
run_name=$WORKLOAD_NAME \
async_scheduling=True \
base_output_directory=$BASE_OUTPUT_DIRECTORY \
chips_per_vm=4 \
data_template_path=maxtext/examples/chat_templates/openmathinstruct2_rl.json \
chat_template_path=maxtext/examples/chat_templates/gemma-3-27b-chat_template.json \
remat_policy=full \
decoder_layer_input=offload \
megablox=True \
sparse_matmul=True \
use_custom_sort_vjp=True \
attention=flash \
use_tokamax_splash=True \
sa_use_fused_bwd_kernel=True \
sa_block_q=1024 \
sa_block_kv=1024 \
sa_block_kv_compute=512 \
sa_block_q_dkv=1024 \
sa_block_kv_dkv=1024 \
sa_block_kv_dkv_compute=256 \
mu_dtype=bfloat16 \
grad_dtype=bfloat16 \
use_iota_embed=True \
batch_size=128 \
num_batches=500 \
learning_rate_schedule_steps=500 \
num_test_batches=5 \
eval_interval=100 \
rl.num_generations=8 \
rl.num_iterations=1 \
rl.grpo_beta=0.05 \
rl.grpo_epsilon=0.2 \
debug.rl=True \
gradient_clipping_threshold=1.0 \
decode_sampling_temperature=0.8 \
decode_sampling_top_k=50 \
decode_sampling_nucleus_p=0.95 \
hf_name=default \
dataset_name=nvidia/OpenMathInstruct-2 \
hf_train_files=hf://datasets/nvidia/OpenMathInstruct-2/data/train_1M-*.parquet \
train_split=train_1M \
eval_dataset_name=nvidia/OpenMathInstruct-2 \
eval_split=test \
learning_rate=1e-6 \
max_prefill_predict_length=512 \
max_target_length=4096 \
kv_cache_buffer=512 \
max_num_seqs=16 \
max_num_batched_tokens=2048 \
rollout_data_parallelism=8 \
rollout_tensor_parallelism=2 \
rollout_expert_parallelism=1 \
rl.reshard_chunk_size=420 \
enable_dp_attention=True \
hbm_utilization_vllm=0.5 \
scan_layers=False \
allow_split_physical_axes=True \
enable_tunix_perf_metrics=True \
checkpoint_period=25 \
max_num_checkpoints_to_keep=30 \
enable_checkpointing=True \
train_micro_batch_size=1 \
rollout_micro_batch_size=16 \
ici_tensor_parallelism=2 \
load_parameters_path=$MAXTEXT_CKPT_PATH \
vllm_hf_overrides='{architectures: [\"MaxTextForCausalLM\"]}' \
vllm_additional_config='{\"maxtext_config\": {\"model_name\": \"gemma4-e4b\", \"model_call_mode\": \"inference\", \"enable_dp_attention\": true, \"allow_split_physical_axes\": true, \"log_config\": false, \"weight_dtype\": \"bfloat16\", \"prefuse_moe_weights\": true}}'"

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
  --server-image="${PATHWAYS_SERVER_IMAGE}" \
  --proxy-server-image="${PATHWAYS_PROXY_SERVER_IMAGE}" \
  --custom-pathways-proxy-server-args='${XLA_FLAGS}' \
  --command="${MAXTEXT_COMMAND}"
