# gemma4e2b_checkpoint_fix'

source ~/VENVS/ubench/bin/activate

export PROJECT_ID="diesel-patrol-382622"
export BASE_OUTPUT_DIRECTORY="gs://mdonati-uscentral1/maxtext-logs/"
export CLUSTER_NAME=mdonati-xpk-v7-spot
export ZONE=us-central1-c
export BASE_DOCKER_IMAGE="gcr.io/diesel-patrol-382622/maxtextrl-mdonati:latest" # rebuilt on 8/6/26
#export WORKLOAD_NAME=gemmae2b-recipe-scan-full
#export WORKLOAD_NAME=gemmae2b-recipe-nscan-full
export WORKLOAD_NAME=gemmae2b-recipe-scan-full-bs8
# export WORKLOAD_NAME=gemmae2b-recipe-nscan-none
export LIBTPU_INIT_ARGS=" \
  --xla_tpu_scoped_vmem_limit_kib=65536 \
  --xla_tpu_bf16_emission_mode=NATIVE_EMISSION \
  --xla_tpu_enable_sparse_core_reduce_scatter_v2=true \
  --xla_tpu_enable_sparse_core_collective_offload_all_gather=true \
  --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true \
  --xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true \
  --xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true \
  --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true \
  --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true \
  --xla_tpu_use_single_sparse_core_for_all_gather_offload=true \
  --xla_tpu_use_tc_device_shape_on_sc=true \
  --xla_sc_disable_megacore_partitioning=true \
  --xla_enable_async_all_gather=true \
  --xla_tpu_prefer_async_allgather_to_allreduce=true \
  --xla_tpu_enable_latency_hiding_layer_scheduler=true \
  --xla_tpu_enable_layer_scheduler_for_dependent_collectives=true \
  --xla_tpu_scheduler_percent_shared_memory_limit=150 \
  --xla_tpu_enable_sparse_core_collective_aggregator=true \
  --xla_tpu_enable_all_gather_offload_tracing=true"

export XLA_FLAGS="--xla_dump_to=${BASE_OUTPUT_DIRECTORY}/HLO_dumps/${WORKLOAD_NAME}/ --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=.*"

# MaxText Workload Overrides
export MAXTEXT_COMMAND="export LIBTPU_INIT_ARGS=\"$LIBTPU_INIT_ARGS\" && \
export XLA_FLAGS=\"$XLA_FLAGS\" && \
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
skip_jax_distributed_system=True \
scan_layers=True \
dtype=bfloat16 \
per_device_batch_size=8 \
max_target_length=8192 \
async_checkpointing=False \
enable_checkpointing=False \
use_iota_embed=True \
num_vocab_tiling=8 \
remat_policy=full \
allow_split_physical_axes=True \
attention=flash \
use_tokamax_splash=True \
sa_use_fused_bwd_kernel=True \
sa_block_q=1024 \
sa_block_kv=1024 \
sa_block_kv_compute=512 \
sa_block_q_dkv=1024 \
sa_block_kv_dkv=1024 \
sa_block_kv_dkv_compute=256 \
dataset_type=synthetic \
opt_type=adamw \
steps=20 \
base_output_directory=${BASE_OUTPUT_DIRECTORY} \
run_name=${WORKLOAD_NAME} \
profiler=xplane \
skip_first_n_steps_for_profiler=5 \
profiler_steps=3 \
gcs_metrics=True"


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