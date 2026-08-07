#!/bin/bash
set -e

# Configuration for Gemma4-26B MoE (Streaming) DiLoCo training job logging diagnostic metrics
# Setup: 2-Slice v5p-256 GKE cluster (64 nodes / 256 TPU chips)
# Model: gemma4-26b (MoE: 128 routed experts + 1 shared expert)

CLUSTER="${CLUSTER:-mlperf-v5p}"
PROJECT="${PROJECT:-cloud-tpu-multipod-dev}"
ZONE="${ZONE:-europe-west4}"

NUM_SLICES="${NUM_SLICES:-2}"
DEVICE_TYPE="${DEVICE_TYPE:-v5p-256}"

RUNNAME="${RUNNAME:-dlco-gemma4-moe}"
DOCKER_IMAGE_BASE="${DOCKER_IMAGE_BASE:-gcr.io/tpu-prod-env-multipod/maxtext_jax_stable:2026-07-17}"
MY_IMAGE="gcr.io/${PROJECT}/$(whoami)-runner:${RUNNAME}"

BASE_OUTPUT_DIRECTORY="${BASE_OUTPUT_DIRECTORY:-gs://chriszuo-maxtext-logs}"
DATASET_PATH="${DATASET_PATH:-gs://chriszuo-maxtext-datasets}"

# DiLoCo Hyperparameters (Optimal parameters from MoE sweep)
DILOCO_SYNC_PERIOD="${DILOCO_SYNC_PERIOD:-31}"
DILOCO_OUTER_LR="${DILOCO_OUTER_LR:-0.3}"
DILOCO_OUTER_MOMENTUM="${DILOCO_OUTER_MOMENTUM:-0.9}"
ENABLE_STREAMING_DILOCO="${ENABLE_STREAMING_DILOCO:-true}"
DILOCO_NUM_FRAGMENTS="${DILOCO_NUM_FRAGMENTS:-31}"
DILOCO_USE_SEQUENTIAL_LAYERS="${DILOCO_USE_SEQUENTIAL_LAYERS:-false}"

# Gemma4-26B Model Parameters
MODEL_NAME="${MODEL_NAME:-gemma4-26b}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-2048}"
STEPS="${STEPS:-1024}"

# Learning Rate & Optimizer Parameters (Identical to pre-training runs)
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
LOAD_BALANCE_LOSS_WEIGHT="${LOAD_BALANCE_LOSS_WEIGHT:-0.001}"

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
             tokenizer_path=maxtext/assets/tokenizers/tokenizer.gemma3 \
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
             pure_nnx=true \
             dcn_diloco_parallelism=${NUM_SLICES} \
             diloco_sync_period=${DILOCO_SYNC_PERIOD} \
             diloco_outer_lr=${DILOCO_OUTER_LR} \
             diloco_outer_momentum=${DILOCO_OUTER_MOMENTUM} \
             num_diloco_fragments=${DILOCO_NUM_FRAGMENTS} \
             use_sequential_layers=${DILOCO_USE_SEQUENTIAL_LAYERS} \
             steps=${STEPS}"

docker build -t "${MY_IMAGE}" -f - . << 'EOF'
FROM gcr.io/tpu-prod-env-multipod/maxtext_jax_stable:2026-07-17
WORKDIR /app
COPY . .
RUN find /app -name "*.pyc" -delete && find /app -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
EOF

echo "Pushing docker image: ${MY_IMAGE}"
docker push "${MY_IMAGE}"

echo "Submitting Gemma4-26B MoE Streaming DiLoCo training workload using XPK..."
xpk workload create --workload "${RUNNAME}" \
  --docker-image "${MY_IMAGE}" \
  --command "${CMD}" \
  --num-slices="${NUM_SLICES}" \
  --enable-debug-logs \
  --cluster "${CLUSTER}" --tpu-type "${DEVICE_TYPE}" --project "${PROJECT}" --zone "${ZONE}"

echo "Gemma4-26B MoE Streaming DiLoCo workload submission complete!"
