#!/bin/bash
set -e

# Cluster parameters
CLUSTER=mlperf-v5p
PROJECT=cloud-tpu-multipod-dev
ZONE=europe-west4

# Topology
NUM_SLICES=2
DEVICE_TYPE=v5p-8

# Run configuration
RUNNAME="${RUNNAME:-dlco-ns-$(date +%d%H%M%S)}"
DOCKER_IMAGE_BASE="gcr.io/tpu-prod-env-multipod/maxtext_jax_stable:2026-07-17"
MY_IMAGE="gcr.io/${PROJECT}/jzuo-runner:${RUNNAME}"

BASE_OUTPUT_DIRECTORY="gs://chriszuo-maxtext-logs"
DATASET_PATH="gs://chriszuo-maxtext-datasets"
DILOCO_SYNC_PERIOD=37
DILOCO_OUTER_LR=0.1
DILOCO_OUTER_MOMENTUM=0.9
DILOCO_NUM_FRAGMENTS=37
DILOCO_USE_SEQUENTIAL_LAYERS=false
DILOCO_NUM_COMM_OVERLAP_STEPS=5
DILOCO_COMM_OVERLAP_ALPHA=0.0
MODEL_NAME="qwen3-8b"
PER_DEVICE_BATCH_SIZE=8
MAX_TARGET_LENGTH=2048
STEPS=200

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

TC_CMD="(for iface in \$(ip -o link show | awk -F': ' '{print \$2}' | awk -F'@' '{print \$1}' | grep -E '^eth|^ens'); do tc qdisc replace dev \$iface root tbf rate 10gbit burst 32mbit latency 50ms 2>/dev/null || tc qdisc add dev \$iface root tbf rate 10gbit burst 32mbit latency 50ms 2>/dev/null || true; done; tc qdisc show || true)"

CMD="${TC_CMD} && export PYTHONPATH=/app/src:\$PYTHONPATH && export JAX_NUM_CPU_DEVICES=8 && export LIBTPU_INIT_ARGS='${XLA_FLAGS}' && cd /app/src/ && python3 maxtext/trainers/pre_train/train.py \
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
             weight_dtype=bfloat16 \
             per_device_batch_size=${PER_DEVICE_BATCH_SIZE} \
             max_target_length=${MAX_TARGET_LENGTH} \
             enable_diloco=true \
             enable_streaming_diloco=true \
             enable_non_spmd_diloco=true \
             enable_single_controller=true \
             pure_nnx=true \
             enable_checkpointing=false \
             log_period=20 \
             num_diloco_fragments=${DILOCO_NUM_FRAGMENTS} \
             use_sequential_layers=${DILOCO_USE_SEQUENTIAL_LAYERS} \
             num_communication_overlapping_steps=${DILOCO_NUM_COMM_OVERLAP_STEPS} \
             communication_overlapping_alpha=${DILOCO_COMM_OVERLAP_ALPHA} \
             dcn_diloco_parallelism=${NUM_SLICES} \
             diloco_sync_period=${DILOCO_SYNC_PERIOD} \
             diloco_outer_lr=${DILOCO_OUTER_LR} \
             diloco_outer_momentum=${DILOCO_OUTER_MOMENTUM} \
             steps=${STEPS}"

# 1. Build and push image from current workspace (/usr/local/google/home/jzuo/maxtext_work)
echo "Building docker image for Non-SPMD DiLoCo..."
docker build -t "${MY_IMAGE}" -f - . <<INNER_EOF
FROM ${DOCKER_IMAGE_BASE}
WORKDIR /app
COPY . .
RUN find /app -name "*.pyc" -delete && find /app -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
INNER_EOF

echo "Pushing image ${MY_IMAGE}..."
docker push "${MY_IMAGE}"

# 2. Create workload using Pathways
echo "Creating Non-SPMD workload: ${RUNNAME}"
/usr/local/google/home/jzuo/xpk_venv/bin/xpk workload create-pathways --workload "${RUNNAME}" \
  --docker-image "${MY_IMAGE}" \
  --command "${CMD}" \
  --num-slices=$NUM_SLICES \
  --priority very-high \
  --enable-debug-logs \
  --cluster "${CLUSTER}" --tpu-type "${DEVICE_TYPE}" --project "${PROJECT}" --zone "${ZONE}"

echo "Non-SPMD Workload ${RUNNAME} created successfully."
