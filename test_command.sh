export PROJECT_ID=tpu-prod-env-multipod
export ACCELERATOR_TYPE=v5p-32
export ZONE=us-east5-a
export RUNTIME_VERSION=v2-alpha-tpuv5
export NODE_COUNT=8
export TPU_NAME=tonyjohnchen-tpu-${ACCELERATOR_TYPE}-${NODE_COUNT}slices-mtu9k
export NETWORK=${USER}-mtu9k

gcloud auth list
gcloud config set project ${PROJECT_ID}
gcloud config set compute/zone ${ZONE}

# yes | gcloud alpha compute tpus queued-resources delete $TPU_NAME --force

# Create with QR in mtu9k network
gcloud alpha compute tpus queued-resources create ${TPU_NAME} \
--node-prefix ${TPU_NAME} \
--node-count ${NODE_COUNT} \
--project ${PROJECT_ID} \
--zone ${ZONE} \
--network ${NETWORK} \
--accelerator-type ${ACCELERATOR_TYPE} \
--runtime-version ${RUNTIME_VERSION} --best-effort
gcloud alpha compute tpus queued-resources list --filter=tonyjohnchen


DATETIME=$(date +%Y-%m-%d-%H-%M-%S)
RUN_NAME=${USER}-mxla-steptime-debug-${ACCELERATOR_TYPE}-${NODE_COUNT}slices-${DATETIME}
python3 multihost_runner.py --TPU_PREFIX=$TPU_NAME \
--COMMAND="bash setup.sh MODE=nightly; \
megascale_crash_on_stale_unmatched_keys=true megascale_transport_type=grpc tpu_link_up_check_timeout=15s xla_tpu_use_megascale_host_reduction=true \
XLA_FLAGS=\"--xla_dump_to=/tmp/xla_dump/\" \
python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME \
base_output_directory=gs://tonyjohnchen-mxla-debug/ dataset_path=gs://max-datasets-rogue \
dataset_type=synthetic \
per_device_batch_size=6 reuse_example_batch=1 \
global_parameter_scale=1 \
steps=50 enable_checkpointing=false enable_profiler=true profile_start_step=0 gcs_metrics=false;"

# DATETIME=$(date +%Y-%m-%d-%H-%M-%S)
# RUN_NAME=${USER}-offload-collective-${ACCELERATOR_TYPE}-${NODE_COUNT}slices-${DATETIME}
# python3 multihost_runner.py --TPU_PREFIX=$TPU_NAME \
# --COMMAND="bash setup.sh MODE=nightly; \
# LIBTPU_INIT_ARGS=\"--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true\"
# XLA_FLAGS=\"--xla_dump_to=/tmp/xla_dump/\" \
# python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME \
# base_output_directory=gs://tonyjohnchen-offload-collective/ dataset_path=gs://max-datasets-rogue \
# per_device_batch_size=6 reuse_example_batch=1 \
# steps=50 enable_checkpointing=false enable_profiler=true profile_start_step=0 gcs_metrics=true \
# attention=mha int8_training=false;"

# _source_path=/tmp/${DATETIME}/output_slice_0000_worker_0000.txt 
# _dest_path=/cns/pl-d/home/tonyjohnchen/mxla-benchmark-logs/offload/${DATETIME}
# /google/data/ro/projects/cloud/bigstore/fileutil_bs mkdir -p "${_dest_path}"
# /google/data/ro/projects/cloud/bigstore/fileutil_bs cp "${_source_path}" "${_dest_path}"
# echo "check $_dest_path/output_slice_0000_worker_0000.txt"