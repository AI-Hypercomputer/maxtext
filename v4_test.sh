export PROJECT_ID=tpu-prod-env-multipod
export ACCELERATOR_TYPE=v4-8
export ZONE=us-central2-b
export RUNTIME_VERSION=tpu-ubuntu2204-base
export NODE_COUNT=2
export TPU_NAME=tonyjohnchen-tpu-${ACCELERATOR_TYPE}-${NODE_COUNT}slices
export NETWORK=${USER}-mtu9k

gcloud auth list
gcloud config set project ${PROJECT_ID}
gcloud config set compute/zone ${ZONE}

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

#asan
blaze build -c opt \
    --config=asan \
    --config=gce \
    --copt=-mavx \
    --copt=-mavx2 \
    --copt=-mno-fma \
    --copt=-DPLATFORM_CLOUD_TPU \
    --copt=-DTPU_GCS_FS \
    --copt='-DSTRIP_FLAG_HELP=1' \
    --copt=-DLIBTPU_EXCLUDE_C_API_IMPL \
    --copt=-DTF_CAPI_WEAK \
    --//learning/brain/tfrc/executor:max_gce_tpu_version=${TPU_FISHNAME} \
    --//learning/brain/tfrc/executor:enable_sparsecore \
    --//third_party/tf_runtime:eigen_mkldnn_contraction_kernel=false \
    --define=tensorflow_mkldnn_contraction_kernel=0 \
    --define=with_tpu_support=true \
    --define=tpu=1 \
    ${EXTRA_LIBTPU_BUILD_DEFINES} \
    //learning/brain/tfrc/executor:libtpu.so

ls blaze-bin/learning/brain/tfrc/executor/libtpu.so
gsutil cp blaze-bin/learning/brain/tfrc/executor/libtpu.so gs://libtpu_internal/tonyjohnchen/mxla/libtpu_stable_with_asan.so


DATETIME=$(date +%Y-%m-%d-%H-%M-%S)
RUN_NAME=${USER}-mxla-debug-${ACCELERATOR_TYPE}-${NODE_COUNT}slices-${DATETIME}
python3 multihost_runner.py --TPU_PREFIX=$TPU_NAME \
--COMMAND="bash setup.sh MODE=stable LIBTPU_GCS_PATH=gs://libtpu_internal/tonyjohnchen/mxla/libtpu_stable_with_asan.so; \
TPU_LIBRARY_PATH=\$HOME/custom_libtpu/libtpu.so \
TPU_STDERR_LOG_LEVEL=0 TPU_MIN_LOG_LEVEL=0 TPU_VMODULE=tpu_configuration_ops_impl=3 TF_CPP_MIN_LOG_LEVEL=0 \
python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME \
base_output_directory=gs://tonyjohnchen-mxla-debug/ dataset_path=gs://max-datasets-rogue \
per_device_batch_size=1 \
steps=50 enable_checkpointing=false \
attention=mha int8_training=false;"


# no-asan
blaze build -c opt \
    --config=gce \
    --copt=-mavx \
    --copt=-mavx2 \
    --copt=-mno-fma \
    --copt=-DPLATFORM_CLOUD_TPU \
    --copt=-DTPU_GCS_FS \
    --copt='-DSTRIP_FLAG_HELP=1' \
    --copt=-DLIBTPU_EXCLUDE_C_API_IMPL \
    --copt=-DTF_CAPI_WEAK \
    --//learning/brain/tfrc/executor:max_gce_tpu_version=${TPU_FISHNAME} \
    --//learning/brain/tfrc/executor:enable_sparsecore \
    --//third_party/tf_runtime:eigen_mkldnn_contraction_kernel=false \
    --define=tensorflow_mkldnn_contraction_kernel=0 \
    --define=with_tpu_support=true \
    --define=tpu=1 \
    ${EXTRA_LIBTPU_BUILD_DEFINES} \
    //learning/brain/tfrc/executor:libtpu.so

ls blaze-bin/learning/brain/tfrc/executor/libtpu.so
gsutil cp blaze-bin/learning/brain/tfrc/executor/libtpu.so gs://libtpu_internal/tonyjohnchen/mxla/libtpu_stable_without_asan.so


DATETIME=$(date +%Y-%m-%d-%H-%M-%S)
RUN_NAME=${USER}-mxla-debug-${ACCELERATOR_TYPE}-${NODE_COUNT}slices-${DATETIME}
python3 multihost_runner.py --TPU_PREFIX=$TPU_NAME \
--COMMAND="bash setup.sh MODE=stable LIBTPU_GCS_PATH=gs://libtpu_internal/tonyjohnchen/mxla/libtpu_stable_without_asan.so; \
TPU_LIBRARY_PATH=\$HOME/custom_libtpu/libtpu.so \
TPU_STDERR_LOG_LEVEL=0 TPU_MIN_LOG_LEVEL=0 TPU_VMODULE=tpu_configuration_ops_impl=3 TF_CPP_MIN_LOG_LEVEL=0 \
python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME \
base_output_directory=gs://tonyjohnchen-mxla-debug/ dataset_path=gs://max-datasets-rogue \
per_device_batch_size=1 \
steps=50 enable_checkpointing=false \
attention=mha int8_training=false;"


TPU_NAME=tonyjohnchen-tpu-v4-8-2slices-1
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE

TPU_LIBRARY_PATH=/home/tonyjohnchen/custom_libtpu/libtpu.so

DATETIME=$(date +%Y-%m-%d-%H-%M-%S)
RUN_NAME=${USER}-mxla-debug-${ACCELERATOR_TYPE}-${NODE_COUNT}slices-${DATETIME}
python3 multihost_runner.py --TPU_PREFIX=$TPU_NAME \
--COMMAND="bash setup.sh MODE=stable; \
python3 -c 'import jax; print(jax.lib.xla_bridge.get_backend().platform_version)'"


