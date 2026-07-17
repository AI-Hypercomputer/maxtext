#!/bin/bash
set -e

# cluster
CLUSTER=v5p-8-bodaborg-europe-west4-b
PROJECT=cloud-tpu-multipod-dev
ZONE=europe-west4

# specify resource
NUM_SLICES=2
DEVICE_TYPE=v5p-8 # 4 chips per slice (8 chips total)

# command
RUNNAME="j-ns-${DEVICE_TYPE}x${NUM_SLICES}-$(date +%d%H%M)"
DOCKER_IMAGE_BASE="gcr.io/tpu-prod-env-multipod/maxtext_jax_stable:latest"
MY_IMAGE="gcr.io/${PROJECT}/jzuo-runner:${RUNNAME}"

BASE_OUTPUT_DIRECTORY="gs://chriszuo-maxtext-logs"
DATASET_PATH="gs://chriszuo-maxtext-datasets"
DILOCO_SYNC_PERIOD=9
DILOCO_OUTER_LR=0.3
DILOCO_OUTER_MOMENTUM=0.9
DILOCO_NUM_FRAGMENTS=8 # matches the overridden base_num_decoder_layers
DILOCO_USE_SEQUENTIAL_LAYERS=false
DILOCO_NUM_COMM_OVERLAP_STEPS=2
DILOCO_COMM_OVERLAP_ALPHA=0.0
MODEL_NAME="qwen3-8b"
PER_DEVICE_BATCH_SIZE=2
MAX_TARGET_LENGTH=1024
STEPS=100

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

# a must to use enable_single_controller=true to use single client pathways

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
             enable_diloco=true \
             enable_streaming_diloco=false \
             enable_non_spmd_diloco=true \
             enable_single_controller=true \
             pure_nnx=true \
             num_diloco_fragments=${DILOCO_NUM_FRAGMENTS} \
             use_sequential_layers=${DILOCO_USE_SEQUENTIAL_LAYERS} \
             num_communication_overlapping_steps=${DILOCO_NUM_COMM_OVERLAP_STEPS} \
             communication_overlapping_alpha=${DILOCO_COMM_OVERLAP_ALPHA} \
             dcn_diloco_parallelism=${NUM_SLICES} \
             diloco_sync_period=${DILOCO_SYNC_PERIOD} \
             diloco_outer_lr=${DILOCO_OUTER_LR} \
             diloco_outer_momentum=${DILOCO_OUTER_MOMENTUM} \
             override_model_config=True \
             base_num_decoder_layers=8 \
             base_emb_dim=256 \
             base_mlp_dim=512 \
             base_num_query_heads=4 \
             base_num_kv_heads=4 \
             head_dim=32 \
             steps=${STEPS}"

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
echo "Creating workload: ${RUNNAME}"
xpk workload create-pathways --workload "${RUNNAME}" \
--docker-image "${MY_IMAGE}" \
--command "${CMD}" \
--num-slices=$NUM_SLICES \
--cluster "${CLUSTER}" --tpu-type "${DEVICE_TYPE}" --project "${PROJECT}" --zone "${ZONE}"

