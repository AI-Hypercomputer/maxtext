#!/bin/bash

# Validates the Llama3.1-8b 1-node GPU pre-training and decoding pipeline using a pre-converted MaxText checkpoint.

# The flow of this script is as follows:
# 1. Run inference decoding on the pre-converted checkpoint.
# 2. Run 5 steps of 1-node GPU pre-training starting from the pre-converted checkpoint.
# 3. Run inference decoding on the checkpoint produced by the pre-training run.

# Usage:
# export HF_TOKEN=<your Hugging Face access token>
# export RUN_ID=$(date +%Y-%m-%d-%H-%M-%S)
# bash test_llama3.1_8b_to_mt.sh $RUN_ID
# bash test_llama3.1_8b_1node.sh $RUN_ID


set -ex

# Avoid conflict between M_RUN_NAME env var and CLI run_name argument in pyconfig
unset M_RUN_NAME

run_id=${1:?"Error: run_id (argument 1) is required"}
train_run_name=${2:-${run_id}}
MODEL_NAME='llama3.1-8b'

# Non-Googlers please remember to point `BASE_OUTPUT_DIRECTORY` to the GCS paths where you have the scanned and unscanned checkpoints stored
BASE_OUTPUT_DIRECTORY="${BASE_OUTPUT_DIRECTORY:-gs://runner-maxtext-logs/${MODEL_NAME}}"
UNSCANNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/to_maxtext/unscanned/${run_id}/0/items

# Non-Googlers please remember to point `DATASET_PATH` to the GCS bucket where you have your training data
DATASET_PATH="${DATASET_PATH:-gs://maxtext-dataset}"

# Step 1: Run inference on the original checkpoint converted from Hugging Face
python3 -m maxtext.inference.decode \
    model_name=${MODEL_NAME} tokenizer_type="huggingface" \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    per_device_batch_size=1 run_name=${run_id} \
    max_prefill_predict_length=8 max_target_length=16 steps=1 async_checkpointing=false \
    checkpoint_storage_use_zarr3=False checkpoint_storage_use_ocdbt=False \
    scan_layers=false hardware=gpu prompt='I love to' attention='dot_product'

# Step 2: Run 5 steps of 1-node GPU pre-training on the converted checkpoint
# We can also run training by using the scanned converted checkpoint
# Note that scanned checkpoint helps with efficient training
python3 -m maxtext.trainers.pre_train.train \
    base_output_directory=${BASE_OUTPUT_DIRECTORY}/train \
    dataset_type=grain \
    grain_file_type=tfrecord \
    dataset_path=${DATASET_PATH} tokenizer_type="huggingface" \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    per_device_batch_size=1 run_name=${train_run_name} \
    max_target_length=2048 steps=5 async_checkpointing=false \
    checkpoint_storage_use_zarr3=False checkpoint_storage_use_ocdbt=False \
    model_name=${MODEL_NAME} hardware=gpu attention=cudnn_flash_te \
    max_segments_per_seq=32 \
    scan_layers=false

# Step 3: Run inference on the checkpoint generated from the previous run (saved at step 4 for steps=5)
python3 -m maxtext.inference.decode \
    model_name=${MODEL_NAME} tokenizer_type="huggingface" \
    load_parameters_path=${BASE_OUTPUT_DIRECTORY}/train/${train_run_name}/checkpoints/4/items \
    per_device_batch_size=1 run_name=${train_run_name} \
    max_prefill_predict_length=8 max_target_length=16 steps=1 async_checkpointing=false \
    checkpoint_storage_use_zarr3=False checkpoint_storage_use_ocdbt=False \
    scan_layers=false hardware=gpu prompt='I love to' attention='dot_product'
