# export PROJECT_ID=tpu-prod-env-multipod
# export ACCELERATOR_TYPE=v5p-8
# export ZONE=us-east5-a
# export RUNTIME_VERSION=v2-alpha-tpuv5
# export NODE_COUNT=2
# export TPU_NAME=tonyjohnchen-v5-tpu-v5p-8-2-slices-dcn-9k-jh
# export NETWORK=${USER}-mtu9k
# export GCS_PATH="${USER}-1vm/dcn_benchmarks"

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

RUN_NAME=${USER}-maxtext-$(date +%Y-%m-%d-%H-%M-%S)

python3 multihost_runner.py --TPU_PREFIX=$TPU_NAME \
--COMMAND="bash setup.sh MODE=stable; \
XLA_FLAGS=\"--xla_dump_to=/tmp/xla_dump/\" \
python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME \
base_output_directory=gs://maxtext-experiments-tpem/ dataset_path=gs://max-datasets-rogue \
steps=10 enable_checkpointing=false enable_profiler=false \
enable_flash_attention=false int8_training=false;"

# #install libtpu-nightly
# python3 multihost_runner.py --TPU_PREFIX=$TPU_NAME \
# --COMMAND="bash setup.sh MODE=nightly; \
# EMIT_MEGASCALE_METRICS=true \
# python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME \
# base_output_directory=gs://maxtext-experiments-tpem/ dataset_path=gs://max-datasets-rogue \
# steps=50 enable_checkpointing=false enable_profiler=false \
# enable_flash_attention=false int8_training=false;"

# #already installed libtpu-nightly
# python3 multihost_runner.py --TPU_PREFIX=$TPU_NAME \
# --COMMAND="EMIT_MEGASCALE_METRICS=true \
# python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME \
# base_output_directory=gs://maxtext-experiments-tpem/ dataset_path=gs://max-datasets-rogue \
# steps=50 enable_checkpointing=false enable_profiler=false \
# enable_flash_attention=false int8_training=false;"
