source ~/VENVS/ubench/bin/activate

export PROJECT_ID="diesel-patrol-382622"
export BASE_OUTPUT_DIRECTORY="gs://mdonati-uscentral1/maxtext-logs/"
export CLUSTER_NAME=mdonati-xpk-v7-spot
export ZONE=us-central1-c
export BASE_DOCKER_IMAGE="gcr.io/diesel-patrol-382622/maxtextrl-mdonati:latest" # rebuilt on 8/6/26
export WORKLOAD_NAME=gemmae2b-sm-min



# # General open-XLA compiler flags
# export XLA_FLAGS="--xla_enable_async_all_gather=true"

# TPU-specific and SparseCore backend flags
export LIBTPU_INIT_ARGS="\
--xla_tpu_dvfs_p_state=7 \
--xla_tpu_scoped_vmem_limit_kib=65536 \
--xla_tpu_num_sparse_cores_for_gather_offloading=1 \
--xla_tpu_bf16_emission_mode=NATIVE_EMISSION \
--xla_tpu_enable_sparse_core_reduce_scatter_v2=true \
--xla_tpu_enable_sparse_core_collective_offload_all_gather=true \
--xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true \
--xla_tpu_use_tc_device_shape_on_sc=True \
--xla_sc_disable_megacore_partitioning=True \
--xla_tpu_enable_async_collective_fusion_fuse_all_gather=false \
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
--xla_tpu_enable_3d_reduce_scatter_decomposer=false \
--xla_enable_async_all_gather=true"

# Safely inject the variables by exporting them inside the container's shell session
export MAXTEXT_COMMAND="export LIBTPU_INIT_ARGS=\"$LIBTPU_INIT_ARGS\" && \
HF_TOKEN='' \
JAX_RANDOM_WEIGHTS=1 \
NEW_MODEL_DESIGN=1 \
TPU_MIN_LOG_LEVEL=0 \
TF_CPP_MIN_LOG_LEVEL=0 \
TPU_STDERR_LOG_LEVEL=0 \
PYTHONPATH=/app/src \
python3 -m maxtext.trainers.pre_train.train \
src/maxtext/configs/base.yml \
model_name=gemma4-e2b \
run_name=$WORKLOAD_NAME \
base_output_directory=$BASE_OUTPUT_DIRECTORY \
steps=10 \
skip_first_n_steps_for_profiler=5 \
profiler_steps=5 \
profiler=xplane \
gradient_clipping_threshold=1.0 \
remat_policy=minimal \
attention=flash \
max_target_length=8192 \
learning_rate=1e-6 \
per_device_batch_size=1 \
allow_split_physical_axes=True \
checkpoint_period=100 \
max_num_checkpoints_to_keep=1000 \
enable_checkpointing=False \
dataset_type=synthetic \
scan_layers=False \
num_vocab_tiling=8"

xpk workload create \
  --cluster="${CLUSTER_NAME}" \
  --project="${PROJECT_ID}" \
  --tpu-type=tpu7x-8 \
  --zone="${ZONE}" \
  --num-slices=1 \
  --base-docker-image="${BASE_DOCKER_IMAGE}" \
  --script-dir="/usr/local/google/home/mattdonati/common_files/maxtext" \
  --workload="${WORKLOAD_NAME}" \
  --command="${MAXTEXT_COMMAND}"