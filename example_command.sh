# TPU configs

export PROJECT_ID=tpu-prod-env-multipod

export ACCELERATOR_TYPE=v4-8
export ZONE=us-central2-b
export RUNTIME_VERSION=tpu-ubuntu2204-base

export NODE_COUNT=2
export TPU_NAME=${USER}-tpu-${ACCELERATOR_TYPE}-${NODE_COUNT}slices-mtu9k
export NETWORK=mtu9k

gcloud auth list
gcloud config set project ${PROJECT_ID}
gcloud config set compute/zone ${ZONE}

## Delete QR and associated if needed
# yes | gcloud alpha compute tpus queued-resources delete $TPU_NAME --force

## Create Multislice in mtu9k network
gcloud alpha compute tpus queued-resources create ${TPU_NAME} \
--node-prefix ${TPU_NAME} \
--node-count ${NODE_COUNT} \
--project ${PROJECT_ID} \
--zone ${ZONE} \
--network ${NETWORK} \
--accelerator-type ${ACCELERATOR_TYPE} \
--runtime-version ${RUNTIME_VERSION} --best-effort
gcloud alpha compute tpus queued-resources list --filter=${USER}

# ## Create single slice.
# TPU_NAME=${USER}-tpu-${ACCELERATOR_TYPE}-1slices-mtu9k
# gcloud alpha compute tpus queued-resources create ${TPU_NAME} \
# --node-id ${TPU_NAME} \
# --project ${PROJECT_ID} \
# --zone ${ZONE} \
# --network ${NETWORK} \
# --accelerator-type ${ACCELERATOR_TYPE} \
# --runtime-version ${RUNTIME_VERSION}
# gcloud alpha compute tpus queued-resources list --filter=tonyjohnchen

#Find the cl that libtpu-nightly built:
DATETIME=$(date +%Y-%m-%d-%H-%M-%S)
RUN_NAME=${USER}-mxla-steptime-debug-${TPU_NAME}-${DATETIME}
python3 multihost_runner.py --TPU_PREFIX=$TPU_NAME \
--COMMAND="bash setup.sh MODE=nightly; python3 -c 'import jax; print(jax.lib.xla_bridge.get_backend().platform_version)'"

# Build custom libtpu:
# http://go/custom-libtpu

# Update custom libtpu and Run maxtext
DATETIME=$(date +%Y-%m-%d-%H-%M-%S)
RUN_NAME=${USER}-mxla-steptime-debug-${TPU_NAME}-${DATETIME}
python3 multihost_runner.py --TPU_PREFIX=$TPU_NAME \
--COMMAND="bash setup.sh MODE=nightly LIBTPU_GCS_PATH=gs://libtpu_internal/tonyjohnchen/viperfish/2024-02-04-21:21:35-libtpu.so && sudo apt install numactl && dpkg -l | grep numactl; \
bash preflight.sh PLATFORM=gce; \
XLA_FLAGS=\"--xla_dump_to=/tmp/xla_dump/\" \
LIBTPU_INIT_ARGS=\"--xla_tpu_enable_megascale_barrier=true\" \
TPU_LIBRARY_PATH=\$HOME/custom_libtpu/libtpu.so TPU_NAME=local \
TPU_STDERR_LOG_LEVEL=0 TPU_MIN_LOG_LEVEL=0 TPU_VMODULE=tpu_configuration_ops_impl=4 TF_CPP_MIN_LOG_LEVEL=0 \
numactl --membind 0 --cpunodebind=0 --strict python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME \
base_output_directory=gs://tonyjohnchen-mxla-debug/ dataset_path=gs://max-datasets-rogue \
dataset_type=synthetic \
per_device_batch_size=6 reuse_example_batch=1 \
global_parameter_scale=1 \
metrics_file='metrics.txt' \
steps=50 enable_checkpointing=false gcs_metrics=true enable_profiler=true profiler_steps=5 upload_all_profiler_results=false && \
python3 end_to_end/eval_assert.py metrics_average metrics.txt 0.0 perf/step_time_seconds;"