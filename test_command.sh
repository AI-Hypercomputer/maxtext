export PROJECT_ID=tpu-prod-env-multipod
export ACCELERATOR_TYPE=v4-8
export ZONE=us-central2-b
export RUNTIME_VERSION=tpu-ubuntu2204-base
export NODE_COUNT=2
export TPU_NAME=${USER}-tpu-${ACCELERATOR_TYPE}-${NODE_COUNT}-slices
export NETWORK=${USER}-mtu9k

gcloud auth list
gcloud config set project ${PROJECT_ID}
gcloud config set compute/zone ${ZONE}

# # # Remove QR if it exists.
# yes | gcloud alpha compute tpus queued-resources delete $TPU_NAME --force

# # # Create with QR in default network.
# gcloud alpha compute tpus queued-resources create ${TPU_NAME} \
# --node-prefix ${TPU_NAME} \
# --node-count ${NODE_COUNT} \
# --project ${PROJECT_ID} \
# --zone ${ZONE} \
# --accelerator-type ${ACCELERATOR_TYPE} \
# --runtime-version ${RUNTIME_VERSION}

RUN_NAME=${USER}-maxtext-$(date +%Y-%m-%d-%H-%M-%S)

python3 multihost_runner.py --TPU_PREFIX=$TPU_NAME \
--COMMAND="bash setup.sh MODE=nightly; \
XLA_FLAGS=\"--xla_dump_to=/tmp/xla_dump/\" \
python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME \
base_output_directory=gs://maxtext-experiments-tpem/ dataset_path=gs://max-datasets-rogue \
steps=10 enable_checkpointing=false enable_profiler=false \
attention=mha int8_training=false;"
