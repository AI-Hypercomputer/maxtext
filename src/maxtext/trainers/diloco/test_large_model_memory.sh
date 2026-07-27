#!/bin/bash
set -e

# ==============================================================================
# DiLoCo Large Model CPU Memory Headroom Acceptance Test Script (Pathways Single-Controller)
#
# Tests non-SPMD / Streaming DiLoCo memory management on full-scale models
# (e.g. Qwen3-8B or Qwen3-30B-A3B) with Pathways single controller to verify that
# host CPU OOM fixes (parameter donation, coordinator-only submesh placement,
# array resharding, bounded transport) function correctly under low CPU memory headroom.
# ==============================================================================

CLUSTER="${CLUSTER:-mlperf-v5p}"
PROJECT="${PROJECT:-cloud-tpu-multipod-dev}"
ZONE="${ZONE:-europe-west4-b}"
RESERVATION="${RESERVATION:-cloudtpu-20240716121201-595617744}"

NUM_SLICES="${NUM_SLICES:-2}"
DEVICE_TYPE="${DEVICE_TYPE:-v5p-8}" # 4 chips per slice (8 chips total)

# IMPORTANT (GEMINI.md): Keep XPK workload names short (< 20 chars) to avoid JobSet label truncation errors.
# Format: j-mem-MMDDHHMM (14 chars)
RUNNAME="${XPK_WORKLOAD:-"j-mem-$(date +%d%H%M)"}"
DOCKER_IMAGE_BASE="${DOCKER_IMAGE_BASE:-gcr.io/tpu-prod-env-multipod/maxtext_jax_stable:2026-06-22}"
MY_IMAGE="gcr.io/${PROJECT}/jzuo-runner:${RUNNAME}"

BASE_OUTPUT_DIRECTORY="${BASE_OUTPUT_DIRECTORY:-gs://chriszuo-maxtext-logs}"
DATASET_PATH="${DATASET_PATH:-gs://chriszuo-maxtext-datasets}"

# Large Model Parameters (Qwen3-8B)
MODEL_NAME="${MODEL_NAME:-qwen3-8b}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-1024}"
STEPS="${STEPS:-100}"

# DiLoCo Hyperparameters
DILOCO_SYNC_PERIOD="${DILOCO_SYNC_PERIOD:-36}"
DILOCO_NUM_FRAGMENTS="${DILOCO_NUM_FRAGMENTS:-36}"
DILOCO_OUTER_LR="${DILOCO_OUTER_LR:-0.3}"
DILOCO_OUTER_MOMENTUM="${DILOCO_OUTER_MOMENTUM:-0.9}"
ENABLE_STREAMING_DILOCO="${ENABLE_STREAMING_DILOCO:-true}"
ENABLE_NON_SPMD_DILOCO="${ENABLE_NON_SPMD_DILOCO:-true}"
ENABLE_SINGLE_CONTROLLER="${ENABLE_SINGLE_CONTROLLER:-true}"
DILOCO_USE_SEQUENTIAL_LAYERS="${DILOCO_USE_SEQUENTIAL_LAYERS:-true}"
DILOCO_NUM_COMM_OVERLAP_STEPS="${DILOCO_NUM_COMM_OVERLAP_STEPS:-2}"
DILOCO_COMM_OVERLAP_ALPHA="${DILOCO_COMM_OVERLAP_ALPHA:-0.0}"

# XLA Flags
XLA_FLAGS=" \
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

CMD="export PYTHONPATH=/app/src:\$PYTHONPATH && export JAX_NUM_CPU_DEVICES=8 && export LIBTPU_INIT_ARGS='${XLA_FLAGS}' && cd /app/src/ && python3 maxtext/trainers/pre_train/train.py \
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
             dtype=bfloat16 \
             weight_dtype=bfloat16 \
             enable_diloco=true \
             enable_streaming_diloco=${ENABLE_STREAMING_DILOCO} \
             enable_non_spmd_diloco=${ENABLE_NON_SPMD_DILOCO} \
             enable_single_controller=${ENABLE_SINGLE_CONTROLLER} \
             pure_nnx=true \
             dcn_diloco_parallelism=${NUM_SLICES} \
             diloco_sync_period=${DILOCO_SYNC_PERIOD} \
             diloco_outer_lr=${DILOCO_OUTER_LR} \
             diloco_outer_momentum=${DILOCO_OUTER_MOMENTUM} \
             num_diloco_fragments=${DILOCO_NUM_FRAGMENTS} \
             use_sequential_layers=${DILOCO_USE_SEQUENTIAL_LAYERS} \
             num_communication_overlapping_steps=${DILOCO_NUM_COMM_OVERLAP_STEPS} \
             communication_overlapping_alpha=${DILOCO_COMM_OVERLAP_ALPHA} \
             steps=${STEPS}"

echo "Building docker image containing local changes..."
docker build --platform linux/amd64 --no-cache -t "${MY_IMAGE}" -f - . <<EOF
FROM ${DOCKER_IMAGE_BASE}
WORKDIR /app
COPY . .
EOF

echo "Pushing image ${MY_IMAGE}..."
docker push "${MY_IMAGE}"

echo "Creating Pathways workload via XPK: ${RUNNAME}"
xpk workload create-pathways --workload "${RUNNAME}" \
  --docker-image "${MY_IMAGE}" \
  --command "${CMD}" \
  --num-slices="${NUM_SLICES}" \
  --cluster "${CLUSTER}" --tpu-type "${DEVICE_TYPE}" --project "${PROJECT}" --zone "${ZONE}" --reservation "${RESERVATION}"

echo "Workload ${RUNNAME} submitted successfully!"
