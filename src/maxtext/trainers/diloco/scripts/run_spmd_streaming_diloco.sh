#!/bin/bash
# ==============================================================================
# SPMD Streaming DiLoCo Multi-Slice Training Runner
# ==============================================================================
# This script builds a container image with local MaxText workspace code and submits
# a multi-slice SPMD Streaming DiLoCo training workload on GKE TPU clusters via XPK.
#
# Reference Paper:
#   "Streaming DiLoCo with overlapping communication: Towards a Distributed Free Lunch"
#   https://arxiv.org/abs/2501.18512
#
# Key Concepts:
#   - Island / Replica: Independent TPU slice running local training steps (dcn_diloco_parallelism).
#   - Fragment: Model weights partitioned into K disjoint subsets (num_diloco_fragments).
#   - Streaming Sync: Communicates 1 fragment per inner step to overlap DCN all-reduce with compute.
#
# ------------------------------------------------------------------------------
# Example Invocations:
# ------------------------------------------------------------------------------
# 1. Standard Multi-Slice Pretraining (2x v5p-128, Qwen3-8B):
#      CLUSTER="mlperf-v5p" ZONE="europe-west4-b" PROJECT="cloud-tpu-multipod-dev" \
#      DEVICE_TYPE="v5p-128" NUM_SLICES="2" RUNNAME="dlco-qwen8b-01" \
#      BASE_OUTPUT_DIRECTORY="gs://chriszuo-maxtext-logs" DATASET_PATH="gs://chriszuo-maxtext-datasets" \
#      RESERVATION="cloudtpu-20240716121201-595617744" \
#      STEPS="1000" CHECKPOINT_PERIOD="100" DILOCO_SYNC_PERIOD="37" DILOCO_NUM_FRAGMENTS="37" \
#      bash src/maxtext/trainers/diloco/scripts/run_spmd_streaming_diloco.sh
#
# 2. Fast Smoke / Integration Test Run (2x v5p-16, short workload name < 20 chars):
#      CLUSTER="mlperf-v5p" ZONE="europe-west4-b" PROJECT="cloud-tpu-multipod-dev" \
#      DEVICE_TYPE="v5p-16" NUM_SLICES="2" RUNNAME="dlco-smk-01" XPK_WORKLOAD="dlco-smk-01" \
#      BASE_OUTPUT_DIRECTORY="gs://chriszuo-maxtext-logs" DATASET_PATH="gs://chriszuo-maxtext-datasets" \
#      RESERVATION="cloudtpu-20240716121201-595617744" \
#      STEPS="20" CHECKPOINT_PERIOD="10" DILOCO_SYNC_PERIOD="3" DILOCO_NUM_FRAGMENTS="3" \
#      bash src/maxtext/trainers/diloco/scripts/run_spmd_streaming_diloco.sh
#
# 3. Automatic Checkpoint Resumption (Reuse same RUNNAME, zero explicit path flags):
#      CLUSTER="mlperf-v5p" ZONE="europe-west4-b" PROJECT="cloud-tpu-multipod-dev" \
#      DEVICE_TYPE="v5p-16" NUM_SLICES="2" RUNNAME="dlco-smk-01" XPK_WORKLOAD="dlco-resm-01" \
#      BASE_OUTPUT_DIRECTORY="gs://chriszuo-maxtext-logs" DATASET_PATH="gs://chriszuo-maxtext-datasets" \
#      RESERVATION="cloudtpu-20240716121201-595617744" \
#      STEPS="40" CHECKPOINT_PERIOD="10" LOAD_FULL_STATE_PATH="" \
#      bash src/maxtext/trainers/diloco/scripts/run_spmd_streaming_diloco.sh
# ==============================================================================
set -e

# cluster
CLUSTER="${CLUSTER:-mlperf-v5p}"
PROJECT="${PROJECT:-cloud-tpu-multipod-dev}"
ZONE="${ZONE:-europe-west4-b}"

# specify resource
NUM_SLICES="${NUM_SLICES:-2}"
DEVICE_TYPE="${DEVICE_TYPE:-v5p-128}"

# command
RUNNAME="${RUNNAME:-spmd-dlco-$(date +%H%M)}"
XPK_WORKLOAD="${XPK_WORKLOAD:-$RUNNAME}"
DOCKER_IMAGE_BASE="${DOCKER_IMAGE_BASE:-gcr.io/tpu-prod-env-multipod/maxtext_jax_stable:latest}"
MY_IMAGE="gcr.io/${PROJECT}/$(whoami)-runner:${XPK_WORKLOAD}"

if [ -z "${BASE_OUTPUT_DIRECTORY:-}" ]; then
  echo "Error: BASE_OUTPUT_DIRECTORY is not set. Please set it as an environment variable (e.g. export BASE_OUTPUT_DIRECTORY=gs://your-bucket/maxtext-logs)."
  exit 1
fi
DATASET_PATH="${DATASET_PATH:-gs://maxtext-dataset}"
DILOCO_SYNC_PERIOD="${DILOCO_SYNC_PERIOD:-37}"
DILOCO_OUTER_LR="${DILOCO_OUTER_LR:-0.1}"
DILOCO_OUTER_MOMENTUM="${DILOCO_OUTER_MOMENTUM:-0.9}"
DILOCO_NUM_FRAGMENTS="${DILOCO_NUM_FRAGMENTS:-37}" # 36 decoder layers + 1 embedding fragment = 37
DILOCO_USE_SEQUENTIAL_LAYERS="${DILOCO_USE_SEQUENTIAL_LAYERS:-false}"
DILOCO_NUM_COMM_OVERLAP_STEPS="${DILOCO_NUM_COMM_OVERLAP_STEPS:-2}"
DILOCO_COMM_OVERLAP_ALPHA="${DILOCO_COMM_OVERLAP_ALPHA:-0.0}"
MODEL_NAME="${MODEL_NAME:-qwen3-8b}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-2048}"
STEPS="${STEPS:-100}"
CHECKPOINT_PERIOD="${CHECKPOINT_PERIOD:-100}"
ENABLE_CHECKPOINTING="${ENABLE_CHECKPOINTING:-true}"
SAVE_CHECKPOINT_ON_COMPLETION="${SAVE_CHECKPOINT_ON_COMPLETION:-true}"
ASYNC_CHECKPOINTING="${ASYNC_CHECKPOINTING:-false}"

LOAD_PARAM_ARG=""
if [ -n "${LOAD_FULL_STATE_PATH}" ]; then
  LOAD_PARAM_ARG="load_full_state_path=${LOAD_FULL_STATE_PATH}"
fi

LIBTPU_INIT_ARGS=" \
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

CMD="export PYTHONPATH=/app/src:\$PYTHONPATH && unset XLA_FLAGS && export LIBTPU_INIT_ARGS=\"${LIBTPU_INIT_ARGS}\" && cd /app/src/ && python3 maxtext/trainers/pre_train/train.py \
             maxtext/configs/base.yml \
             run_name=${RUNNAME} \
             save_config_to_gcs=true \
             base_output_directory=${BASE_OUTPUT_DIRECTORY} \
             dataset_path=${DATASET_PATH} \
             dataset_name='c4/en:3.0.1' \
             eval_dataset_name='c4/en:3.0.1' \
             model_name=${MODEL_NAME} \
             tokenizer_type=huggingface \
             tokenizer_path=maxtext/assets/tokenizers/qwen3-tokenizer \
             per_device_batch_size=${PER_DEVICE_BATCH_SIZE} \
             max_target_length=${MAX_TARGET_LENGTH} \
             enable_diloco=true \
             enable_streaming_diloco=true \
             pure_nnx=true \
             num_diloco_fragments=${DILOCO_NUM_FRAGMENTS} \
             use_sequential_layers=${DILOCO_USE_SEQUENTIAL_LAYERS} \
             num_communication_overlapping_steps=${DILOCO_NUM_COMM_OVERLAP_STEPS} \
             communication_overlapping_alpha=${DILOCO_COMM_OVERLAP_ALPHA} \
             dcn_diloco_parallelism=${NUM_SLICES} \
             diloco_sync_period=${DILOCO_SYNC_PERIOD} \
             diloco_outer_lr=${DILOCO_OUTER_LR} \
             diloco_outer_momentum=${DILOCO_OUTER_MOMENTUM} \
             enable_checkpointing=${ENABLE_CHECKPOINTING} \
             checkpoint_period=${CHECKPOINT_PERIOD} \
             save_checkpoint_on_completion=${SAVE_CHECKPOINT_ON_COMPLETION} \
             async_checkpointing=${ASYNC_CHECKPOINTING} \
             profiler=xplane \
             skip_first_n_steps_for_profiler=5 \
             profiler_steps=5 \
             upload_all_profiler_results=true \
             jax_distributed_initialization_timeout=3600 \
             steps=${STEPS}"

if [ -n "${LOAD_FULL_STATE_PATH}" ]; then
  CMD="${CMD} load_full_state_path=${LOAD_FULL_STATE_PATH}"
fi

# 1. Build and push the docker image manually containing your local changes
echo "Building docker image containing local changes..."
docker build -t "${MY_IMAGE}" -f - . <<EOF
FROM ${DOCKER_IMAGE_BASE}
WORKDIR /app
COPY . .
RUN find /app -name "*.pyc" -delete && find /app -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
EOF

echo "Pushing image ${MY_IMAGE}..."
docker push "${MY_IMAGE}"

# 2. Create the workload directly using xpk
echo "Creating workload: ${XPK_WORKLOAD}"
XPK_ARGS=(
  --workload "${XPK_WORKLOAD}"
  --docker-image "${MY_IMAGE}"
  --command "${CMD}"
  --num-slices "${NUM_SLICES}"
  --priority "${PRIORITY:-medium}"
  --enable-debug-logs
  --cluster "${CLUSTER}"
  --tpu-type "${DEVICE_TYPE}"
  --project "${PROJECT}"
  --zone "${ZONE}"
)

if [ -n "${RESERVATION:-}" ] && [ "${RESERVATION}" != "NONE" ]; then
  XPK_ARGS+=(--reservation "${RESERVATION}")
fi

xpk workload create "${XPK_ARGS[@]}"
