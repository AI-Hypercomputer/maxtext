export PROJECT_ID=tpu-prod-env-multipod
export ACCELERATOR_TYPE=v4-8
export ZONE=us-central2-b
export RUNTIME_VERSION=tpu-ubuntu2204-base
export NODE_COUNT=2
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
--runtime-version ${RUNTIME_VERSION}  --best-effort
gcloud alpha compute tpus queued-resources list --filter=tonyjohnchen


DATETIME=$(date +%Y-%m-%d-%H-%M-%S)
RUN_NAME=${USER}-mxla-debug-${ACCELERATOR_TYPE}-${NODE_COUNT}slices-${DATETIME}
python3 multihost_runner.py --TPU_PREFIX=$TPU_NAME \
--COMMAND="bash setup.sh MODE=nightly LIBTPU_GCS_PATH=gs://libtpu_internal/tonyjohnchen/mxla/libtpu.so; \
TPU_LIBRARY_PATH=\$HOME/custom_libtpu/libtpu.so \
python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME \
base_output_directory=gs://tonyjohnchen-mxla-debug/ dataset_path=gs://max-datasets-rogue \
per_device_batch_size=1 \
steps=50 enable_checkpointing=false \
attention=mha int8_training=false;"