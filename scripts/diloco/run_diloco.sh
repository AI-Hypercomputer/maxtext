#!/bin/bash

# This script launches a DiLoCo pre-training workload on a GKE cluster using XPK.

set -e

# --- Cluster Parameters ---
export PROJECT_ID="${PROJECT_ID:-}"
export CLUSTER_NAME="${CLUSTER_NAME:-}"
export ZONE="${ZONE:-}"
export RESERVATION="${RESERVATION:-}"  # optional
export BASE_OUTPUT_DIRECTORY="${BASE_OUTPUT_DIRECTORY:-}" # change to your own GCS bucket for logging and checkpointing
export DATASET_PATH="${DATASET_PATH:-}" # change to your own GSC bucket for datasets. Make sure datasets exists
export DOCKER_IMAGE="${DOCKER_IMAGE:-}" # Full path to the Docker image you pushed (e.g., gcr.io/tpu-prod-env-multipod/maxtext_jax_stable:2026-06-22)
export TPU_TYPE="${TPU_TYPE:-}"  # At least v5p-32 is needed to run Qwen3-30b-a3b. v5p-8 for qwen3-8b
export NUM_SLICES="${NUM_SLICES:-}"  # you need at least two slices to let diloco take effect
export WORKLOAD_NAME="${WORKLOAD_NAME:-$(whoami)-diloco-${TPU_TYPE}-$(date +%Y%m%d-%H%M%S)}" # this will be the name of run, for logging purposes

# --- Model Parameters ---
export MODEL_NAME="${MODEL_NAME:-}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
export MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-2048}"
export TRAINING_STEPS="${TRAINING_STEPS:-20}"

# --- DiLoCo Parameters ---
export DILOCO_SYNC_PERIOD="${DILOCO_SYNC_PERIOD:-10}"
export DILOCO_OUTER_LR="${DILOCO_OUTER_LR:-0.1}"
export DILOCO_OUTER_MOMENTUM="${DILOCO_OUTER_MOMENTUM:-0.9}"

# --- XLA Flags ---
export XLA_FLAGS="${XLA_FLAGS:- \
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
  --xla_tpu_enable_3d_reduce_scatter_decomposer=false }"

if [ "$NUM_SLICES" -lt 2 ]; then
    echo "Warning: NUM_SLICES is less than 2. DiLoCo will not take effect."
fi

# MaxText command
MAXTEXT_COMMAND="cd /deps/src/ && \
LIBTPU_INIT_ARGS='${XLA_FLAGS}' \
python3 -m maxtext.trainers.pre_train.train \
maxtext/configs/base.yml \
run_name=$WORKLOAD_NAME \
save_config_to_gcs=true \
base_output_directory=$BASE_OUTPUT_DIRECTORY \
dataset_path=$DATASET_PATH \
dataset_name='c4/en:3.0.1' \
eval_dataset_name='c4/en:3.0.1' \
model_name=$MODEL_NAME \
tokenizer_type=huggingface \
tokenizer_path=maxtext/assets/tokenizers/qwen3-tokenizer \
per_device_batch_size=$PER_DEVICE_BATCH_SIZE \
max_target_length=$MAX_TARGET_LENGTH \
enable_diloco=true \
dcn_diloco_parallelism=$NUM_SLICES \
diloco_sync_period=$DILOCO_SYNC_PERIOD \
diloco_outer_lr=$DILOCO_OUTER_LR \
diloco_outer_momentum=$DILOCO_OUTER_MOMENTUM \
steps=$TRAINING_STEPS"

# Workload Creation
xpk workload create \
  --cluster="$CLUSTER_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --priority=medium \
  --max-restarts=0 \
  --tpu-type="$TPU_TYPE" \
  --num-slices="$NUM_SLICES" \
  --docker-image="${DOCKER_IMAGE}" \
  --workload="${WORKLOAD_NAME}" \
  ${RESERVATION:+--reservation="$RESERVATION"} \
  --command="${MAXTEXT_COMMAND}"
