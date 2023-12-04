export PROJECT_ID=tpu-prod-env-multipod
export ACCELERATOR_TYPE=v5p-8
export ZONE=us-east5-a
export RUNTIME_VERSION=v2-alpha-tpuv5
export NODE_COUNT=2
export TPU_NAME=tonyjohnchen-tpu-v5p-64-2-slices-dcn-9k-tony
export NETWORK=${USER}-mtu9k
export GCS_PATH="${USER}-1vm/dcn_benchmarks"

gcloud auth list
gcloud config set project ${PROJECT_ID}
gcloud config set compute/zone ${ZONE}

RUN_NAME=${USER}-maxtext-$(date +%Y-%m-%d-%H-%M-%S)

# install libtpu-nightly
python3 multihost_runner.py --TPU_PREFIX=$TPU_NAME \
--COMMAND="bash setup.sh MODE=nightly; \
LIBTPU_INIT_ARGS=\"--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true\"
XLA_FLAGS=\"--xla_dump_to=/tmp/xla_dump/\" \
python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME \
base_output_directory=gs://maxtext-experiments-tpem/ dataset_path=gs://max-datasets-rogue \
per_device_batch_size=6 reuse_example_batch=1 \
steps=50 enable_checkpointing=false enable_profiler=true profile_start_step=5 \
attention=mha int8_training=false;"

# # already installed libtpu-nightly
# python3 multihost_runner.py --TPU_PREFIX=$TPU_NAME \
# --COMMAND=" \
# LIBTPU_INIT_ARGS=\"--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true\"
# XLA_FLAGS=\"--xla_dump_to=/tmp/xla_dump/\" \
# python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME \
# base_output_directory=gs://maxtext-experiments-tpem/ dataset_path=gs://max-datasets-rogue \
# per_device_batch_size=6 reuse_example_batch=1 \
# steps=50 enable_checkpointing=false enable_profiler=true profile_start_step=5 \
# attention=mha int8_training=false;"