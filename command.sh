# AOT Command:
python3 MaxText/train_compile.py MaxText/configs/base.yml model_name=subsup_small per_device_batch_size=3 compile_topology=v5p-256 compile_topology_num_slices=1

python3 MaxText/train_compile.py MaxText/configs/base.yml model_name=subsup_large per_device_batch_size=3 compile_topology=v5p-2048 compile_topology_num_slices=1

PROJECT_ID=cloud-tpu-best-effort-colo 
ZONE=europe-west1-c
TPU_NAME=v5p-256-moe-test
QR_NAME=$TPU_NAME
NODE_PREFIX=${QR_NAME}
TPU_TYPE=v5p-256
VERSION=v2-alpha-tpuv5

gcloud config set project ${PROJECT_ID}
gcloud config set compute/zone ${ZONE}


gcloud alpha compute tpus queued-resources create ${QR_NAME} \
--accelerator-type=${TPU_TYPE} \
--project=${PROJECT_ID} \
--zone=${ZONE} \
--runtime-version=${VERSION} \
--node-id=${QR_NAME} \
--description noteardown \
--reserved 

gcloud alpha compute tpus queued-resources list --filter=$QR_NAME

# Training Command:
# PROJECT_ID=tpu-prod-env-multipod
# ZONE=us-east5-a
PROJECT_ID=cloud-tpu-best-effort-colo 
ZONE=europe-west1-c
TPU_NAME=v5p-256-moe-test

gcloud config set project ${PROJECT_ID}
gcloud config set compute/zone ${ZONE}

# Install deps
python3 multihost_runner.py --TPU_PREFIX=$TPU_NAME \
--COMMAND="bash setup.sh MODE=stable;"

# Start training small model
BATCH_SIZE=4
RUN_NAME=v5p_256_subsup_small_test_per_device_batch_size$BATCH_SIZE
MAXTEXT_OUTPUT_PATH=gs://tony-moe

python3 multihost_runner.py --TPU_PREFIX=$TPU_NAME \
--COMMAND="\
    python3 MaxText/train.py MaxText/configs/base.yml model_name=subsup_small \
    base_output_directory=${MAXTEXT_OUTPUT_PATH} run_name=${RUN_NAME} \
    enable_checkpointing=false async_checkpointing=false \
    per_device_batch_size=${BATCH_SIZE} \
    skip_first_n_steps_for_profiler=5 steps=10 \
    tokenizer_path=assets/tokenizer.mistral dataset_type=synthetic \
    profiler=xplane"


