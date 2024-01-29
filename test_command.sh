export PROJECT_ID=tpu-prod-env-multipod

# export ACCELERATOR_TYPE=v5p-8
# export ZONE=us-east5-a
# export RUNTIME_VERSION=v2-alpha-tpuv5

export ACCELERATOR_TYPE=v4-8
export ZONE=us-central2-b
export RUNTIME_VERSION=tpu-ubuntu2204-base

# export ACCELERATOR_TYPE=v5litepod-16
# export ZONE=us-east5-b
# export RUNTIME_VERSION=v2-alpha-tpuv5-lite

export NODE_COUNT=8
export TPU_NAME=tonyjohnchen-tpu-${ACCELERATOR_TYPE}-${NODE_COUNT}slices-mtu9k
export NETWORK=${USER}-mtu9k

gcloud auth list
gcloud config set project ${PROJECT_ID}
gcloud config set compute/zone ${ZONE}

## Delete QR and TPU.
# TPU_NAME=tonyjohnchen-tpu-v4-16-2slices-mtu9k
# yes | gcloud alpha compute tpus queued-resources delete $TPU_NAME --force


## Create single slice.
TPU_NAME=tonyjohnchen-tpu-${ACCELERATOR_TYPE}-1slices-mtu9k
gcloud alpha compute tpus queued-resources create ${TPU_NAME} \
--node-id ${TPU_NAME} \
--project ${PROJECT_ID} \
--zone ${ZONE} \
--network ${NETWORK} \
--accelerator-type ${ACCELERATOR_TYPE} \
--runtime-version ${RUNTIME_VERSION}
gcloud alpha compute tpus queued-resources list --filter=tonyjohnchen

## Create Multislice in mtu9k network
gcloud alpha compute tpus queued-resources create ${TPU_NAME} \
--node-prefix ${TPU_NAME} \
--node-count ${NODE_COUNT} \
--project ${PROJECT_ID} \
--zone ${ZONE} \
--network ${NETWORK} \
--accelerator-type ${ACCELERATOR_TYPE} \
--runtime-version ${RUNTIME_VERSION} --best-effort

gcloud alpha compute tpus queued-resources list --filter=tonyjohnchen

# DATETIME=$(date +%Y-%m-%d-%H-%M-%S)
# RUN_NAME=${USER}-mxla-steptime-debug-${ACCELERATOR_TYPE}-${NODE_COUNT}slices-${DATETIME}
# python3 multihost_runner.py --TPU_PREFIX=$TPU_NAME \
# --COMMAND="bash setup.sh MODE=nightly LIBTPU_GCS_PATH=gs://libtpu_internal/tonyjohnchen/viperfish/2024-01-23-23:02:29-libtpu.so && sudo apt install numactl && dpkg -l | grep numactl \
# && sudo bash MaxText/network_setting.sh; \
# XLA_FLAGS=\"--xla_dump_to=/tmp/xla_dump/\" \
# LIBTPU_INIT_ARGS=\"--xla_tpu_enable_megascale_barrier=true grpc_experiments=event_engine_client,event_engine_listener,trace_record_callops megascale_grpc_enable_xor_tracer=true census_enabled=false megascale_crash_on_stale_unmatched_keys=true megascale_transport_type=grpc tpu_link_up_check_timeout=15s xla_tpu_use_megascale_host_reduction=true megascale_use_insecure_grpc=true grpc_filter_insecure_rpc=false enforce_kernel_ipv6_support=false\" \
# TPU_LIBRARY_PATH=\$HOME/custom_libtpu/libtpu.so TPU_NAME=local JAX_USE_PJRT_C_API_ON_TPU=1 \
# TPU_STDERR_LOG_LEVEL=0 TPU_MIN_LOG_LEVEL=0 TPU_VMODULE=tpu_configuration_ops_impl=3 TF_CPP_MIN_LOG_LEVEL=0 \
# numactl --membind 0 --cpunodebind=0 --strict python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME \
# base_output_directory=gs://tonyjohnchen-mxla-debug/ dataset_path=gs://max-datasets-rogue \
# dataset_type=synthetic \
# per_device_batch_size=6 reuse_example_batch=1 \
# global_parameter_scale=1 \
# metrics_file='metrics.txt' \
# steps=50 enable_checkpointing=false gcs_metrics=true enable_profiler=true profiler_steps=5 && \
# python3 end_to_end/eval_assert.py metrics_average metrics.txt 0.0 perf/step_time_seconds;"

DATETIME=$(date +%Y-%m-%d-%H-%M-%S)
RUN_NAME=${USER}-preflight-${TPU_NAME}-${DATETIME}
python3 multihost_runner.py --TPU_PREFIX=$TPU_NAME \
--COMMAND="bash preflight.sh PLATFORM=gce"

DATETIME=$(date +%Y-%m-%d-%H-%M-%S)
RUN_NAME=${USER}-mxla-steptime-debug-${TPU_NAME}-${DATETIME}
python3 multihost_runner.py --TPU_PREFIX=$TPU_NAME \
--COMMAND="bash setup.sh MODE=nightly && sudo apt install numactl && dpkg -l | grep numactl; \
sudo bash MaxText/network_setting.sh; \
XLA_FLAGS=\"--xla_dump_to=/tmp/xla_dump/\" \
LIBTPU_INIT_ARGS=\"--xla_tpu_enable_megascale_barrier=true\" \
TPU_NAME=local \
TPU_STDERR_LOG_LEVEL=0 TPU_MIN_LOG_LEVEL=0 TPU_VMODULE=tpu_configuration_ops_impl=3 TF_CPP_MIN_LOG_LEVEL=0 \
numactl --membind 0 --cpunodebind=0 --strict python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME \
base_output_directory=gs://tonyjohnchen-mxla-debug/ dataset_path=gs://max-datasets-rogue \
dataset_type=synthetic \
per_device_batch_size=24 reuse_example_batch=1 \
global_parameter_scale=1 \
metrics_file='metrics.txt' \
steps=50 enable_checkpointing=false gcs_metrics=true enable_profiler=true profiler_steps=5 upload_all_profiler_results=true && \
python3 end_to_end/eval_assert.py metrics_average metrics.txt 0.0 perf/step_time_seconds;"


TPU_NAME_ssh=$TPU_NAME-0
gcloud compute tpus tpu-vm ssh $TPU_NAME_ssh --zone=$ZONE --project=$PROJECT_ID

gs://tonyjohnchen-mxla-debug/10_steps_xplane

gcloud compute tpus tpu-vm ssh $TPU_NAME --worker=0 --zone=$ZONE --project=$PROJECT_ID
gcloud compute tpus tpu-vm ssh $TPU_NAME --worker=1 --zone=$ZONE --project=$PROJECT_ID




