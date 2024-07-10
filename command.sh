# AOT Command:
python3 MaxText/train_compile.py MaxText/configs/base.yml model_name=subsup_small per_device_batch_size=3 compile_topology=v5p-256 compile_topology_num_slices=1

python3 MaxText/train_compile.py MaxText/configs/base.yml model_name=subsup_large per_device_batch_size=3 compile_topology=v5p-2048 compile_topology_num_slices=1

# Training Command:
PROJECT_ID=tpu-prod-env-multipod
ZONE=us-east5-a
TPU_NAME=v5p-256-test

gcloud config set project ${PROJECT_ID}
gcloud config set compute/zone ${ZONE}

# Install deps
python3 multihost_runner.py --TPU_PREFIX=$TPU_NAME \
--COMMAND="\
    git clone -b v5p_moe https://github.com/google/maxtext.git;\
    cd maxtext && bash setup.sh;"

# Start training small model
BATCH_SIZE=3
RUN_NAME=subsup_small_test_per_device_batch_size$BATCH_SIZE
MAXTEXT_OUTPUT_PATH=gs://tony-moe

python3 multihost_runner.py --TPU_PREFIX=$TPU_NAME \
--COMMAND="\
    python3 MaxText/train.py MaxText/configs/MaxText/configs/base.yml model_name=subsup_small \
    base_output_directory=${MAXTEXT_OUTPUT_PATH} run_name=${RUN_NAME} \
    enable_checkpointing=false async_checkpointing=false \
    per_device_batch_size=${BATCH_SIZE} \
    skip_first_n_steps_for_profiler=5 steps=20 \
    tokenizer_path=assets/tokenizer.mistral dataset_type=synthetic \
    profiler=xplane"


