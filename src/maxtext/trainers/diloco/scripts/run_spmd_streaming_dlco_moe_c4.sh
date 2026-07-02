#!/bin/bash
set -e

# Configuration for MoE (Streaming) DiLoCo training job logging all 5 diagnostic metrics:
# 1. Inter-Replica Router Distance (d_router / diloco/inter_replica_router_distance)
# 2. Top-K Token Routing Overlap (J_route / diloco/topk_token_routing_overlap)
# 3. Jensen-Shannon Routing Divergence Index (RDI / diloco/js_routing_divergence_index)
# 4. Post-Sync Loss Spike Severity (Delta L_sync / diloco/post_sync_loss_spike_severity)
# 5. Post-Sync Expert Utilization Entropy (EUE / diloco/post_sync_expert_utilization_entropy)
#
# ------------------------------------------------------------------------------
# Example Invocations:
# ------------------------------------------------------------------------------
# 1. Standard Multi-Slice MoE DiLoCo Training (2x v5p-128, Qwen3-30B-A3B on C4):
#      CLUSTER="mlperf-v5p" ZONE="europe-west4-b" PROJECT="cloud-tpu-multipod-dev" \
#      DEVICE_TYPE="v5p-128" NUM_SLICES="2" RUNNAME="dlco-moe-01" \
#      BASE_OUTPUT_DIRECTORY="gs://chriszuo-maxtext-logs" DATASET_PATH="gs://chriszuo-maxtext-datasets" \
#      RESERVATION="cloudtpu-20240716121201-595617744" \
#      STEPS="100" DILOCO_SYNC_PERIOD="49" DILOCO_NUM_FRAGMENTS="49" \
#      bash src/maxtext/trainers/diloco/scripts/run_spmd_streaming_dlco_moe_c4.sh
# ------------------------------------------------------------------------------

CLUSTER="${CLUSTER:-mlperf-v5p}"
PROJECT="${PROJECT:-cloud-tpu-multipod-dev}"
ZONE="${ZONE:-europe-west4-b}"

NUM_SLICES="${NUM_SLICES:-2}"
DEVICE_TYPE="${DEVICE_TYPE:-v5p-128}"

RUNNAME="${RUNNAME:-dlco-moe-$(date +%H%M)}"
DOCKER_IMAGE_BASE="${DOCKER_IMAGE_BASE:-gcr.io/tpu-prod-env-multipod/maxtext_jax_stable:latest}"
MY_IMAGE="gcr.io/${PROJECT}/$(whoami)-runner:${RUNNAME}"

if [ -z "${BASE_OUTPUT_DIRECTORY:-}" ]; then
  echo "Error: BASE_OUTPUT_DIRECTORY is not set. Please set it as an environment variable (e.g. export BASE_OUTPUT_DIRECTORY=gs://your-bucket/maxtext-logs)."
  exit 1
fi
DATASET_PATH="${DATASET_PATH:-gs://maxtext-dataset}"

# DiLoCo Hyperparameters
DILOCO_SYNC_PERIOD="${DILOCO_SYNC_PERIOD:-49}"
DILOCO_OUTER_LR="${DILOCO_OUTER_LR:-0.1}"
DILOCO_OUTER_MOMENTUM="${DILOCO_OUTER_MOMENTUM:-0.9}"
ENABLE_STREAMING_DILOCO="${ENABLE_STREAMING_DILOCO:-true}"
DILOCO_NUM_FRAGMENTS="${DILOCO_NUM_FRAGMENTS:-49}"
DILOCO_USE_SEQUENTIAL_LAYERS="${DILOCO_USE_SEQUENTIAL_LAYERS:-false}"

# MoE Model Parameters
MODEL_NAME="${MODEL_NAME:-qwen3-30b-a3b}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-2048}"
STEPS="${STEPS:-100}"

# Learning Rate & Optimizer Parameters
LEARNING_RATE="${LEARNING_RATE:-1.0e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
LR_SCHEDULE_STEPS="${LR_SCHEDULE_STEPS:-18596}"
COSINE_FINAL_FRAC="${COSINE_FINAL_FRAC:-0.1}"
WARMUP_FRAC=$(python3 -c "print(${WARMUP_STEPS}/${LR_SCHEDULE_STEPS})")
ADAM_B1="${ADAM_B1:-0.9}"
ADAM_B2="${ADAM_B2:-0.95}"
ADAM_EPS="${ADAM_EPS:-1e-8}"
ADAM_WD="${ADAM_WD:-0.1}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"

# Z-Loss & MoE Load Balancing Parameters
Z_LOSS="${Z_LOSS:-1.0e-5}"
LOAD_BALANCE_LOSS_WEIGHT="${LOAD_BALANCE_LOSS_WEIGHT:-0.01}"

LIBTPU_INIT_ARGS=" \
  --xla_tpu_scoped_vmem_limit_kib=65472 \
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
             learning_rate=${LEARNING_RATE} \
             learning_rate_schedule_steps=${LR_SCHEDULE_STEPS} \
             learning_rate_final_fraction=${COSINE_FINAL_FRAC} \
             warmup_steps_fraction=${WARMUP_FRAC} \
             adam_b1=${ADAM_B1} \
             adam_b2=${ADAM_B2} \
             adam_eps=${ADAM_EPS} \
             adam_weight_decay=${ADAM_WD} \
             gradient_clipping_threshold=${GRAD_CLIP} \
             z_loss_multiplier=${Z_LOSS} \
             load_balance_loss_weight=${LOAD_BALANCE_LOSS_WEIGHT} \
             enable_diloco=true \
             enable_streaming_diloco=${ENABLE_STREAMING_DILOCO} \
             dcn_diloco_parallelism=${NUM_SLICES} \
             diloco_sync_period=${DILOCO_SYNC_PERIOD} \
             diloco_outer_lr=${DILOCO_OUTER_LR} \
             diloco_outer_momentum=${DILOCO_OUTER_MOMENTUM} \
             num_diloco_fragments=${DILOCO_NUM_FRAGMENTS} \
             use_sequential_layers=${DILOCO_USE_SEQUENTIAL_LAYERS} \
             jax_distributed_initialization_timeout=1200 \
             steps=${STEPS}"

if [ -n "${LOAD_FULL_STATE_PATH:-}" ]; then
  CMD="${CMD} load_full_state_path=${LOAD_FULL_STATE_PATH}"
elif [ -n "${LOAD_PARAMETERS_PATH:-}" ]; then
  CMD="${CMD} load_parameters_path=${LOAD_PARAMETERS_PATH}"
fi

docker build -t "${MY_IMAGE}" -f - . << EOF
FROM ${DOCKER_IMAGE_BASE}
WORKDIR /app
COPY . .
RUN find /app -name "*.pyc" -delete && find /app -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
EOF

echo "Pushing docker image: ${MY_IMAGE}"
docker push "${MY_IMAGE}"

echo "Submitting MoE DiLoCo training workload using XPK..."
XPK_ARGS=(
  --workload "${RUNNAME}"
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

echo "MoE DiLoCo workload submission complete!"
