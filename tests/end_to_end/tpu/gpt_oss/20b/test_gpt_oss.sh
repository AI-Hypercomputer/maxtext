#!/bin/bash

# Validates the GPT-OSS-20b pre-training pipeline using a pre-converted MaxText checkpoint.

# The flow of this script is as follows:
# 1. Run inference on the pre-converted checkpoint.
# 2. Run pre-training starting from the pre-converted checkpoint.
# 3. Run inference on the checkpoint produced by the pre-training run.

# Usage:
# export HF_TOKEN=<your Hugging Face access token>
# export RUN_ID=$(date +%Y-%m-%d-%H-%M-%S)
# bash test_gpt_oss_to_mt.sh $RUN_ID
# bash test_gpt_oss.sh $RUN_ID

set -ex

if [ -z "$1" ]; then
  echo "Error: run_id argument is required."
  exit 1
fi
run_id=$1
MODEL_NAME='gpt-oss-20b'

# Non-Googlers please remember to point BASE_OUTPUT_DIRECTORY to the GCS paths where you have the scanned and unscanned checkpoints stored
BASE_OUTPUT_DIRECTORY=gs://runner-maxtext-logs/${MODEL_NAME}
UNSCANNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/to_maxtext/unscanned/${run_id}/0/items

# Non-Googlers please remember to point DATASET_PATH to the GCS bucket where you have your training data
DATASET_PATH=gs://maxtext-dataset

# Step 1: Run inference on the original checkpoint converted from Hugging Face
    python3 -m maxtext.inference.decode \
    model_name=${MODEL_NAME} \
    tokenizer_type="huggingface" \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    per_device_batch_size=1 \
    run_name=${run_id} \
    max_prefill_predict_length=8 \
    max_target_length=16 \
    steps=1 \
    async_checkpointing=false \
    checkpoint_storage_use_zarr3=False \
    checkpoint_storage_use_ocdbt=False \
    scan_layers=false \
    prompt='I love to' \
    attention="dot_product"

# Step 2: Run Pre-training on the converted checkpoint
# We can also run training by using the scanned converted checkpoint
# Note that scanned checkpoint helps with efficient training
python3 -m maxtext.trainers.pre_train.train \
    base_output_directory=${BASE_OUTPUT_DIRECTORY}/train \
    dataset_type=grain \
    grain_file_type=tfrecord \
    dataset_path=${DATASET_PATH} \
    tokenizer_type="huggingface" \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    per_device_batch_size=1 \
    run_name=${run_id} \
    max_target_length=1024 \
    steps=2 \
    weight_dtype=bfloat16 \
    async_checkpointing=false \
    checkpoint_storage_use_zarr3=False \
    checkpoint_storage_use_ocdbt=False \
    model_name=${MODEL_NAME} \
    scan_layers=false \
    use_multimodal=false

# Step 3: Run inference on the checkpoint generated from the previous run
    python3 -m maxtext.inference.decode \
    model_name=${MODEL_NAME} \
    tokenizer_type="huggingface" \
    load_parameters_path=${BASE_OUTPUT_DIRECTORY}/train/${run_id}/checkpoints/1/items \
    per_device_batch_size=1 \
    run_name=${run_id} \
    max_prefill_predict_length=8 \
    max_target_length=16 \
    steps=1 \
    async_checkpointing=false \
    checkpoint_storage_use_zarr3=False \
    checkpoint_storage_use_ocdbt=False \
    scan_layers=false \
    prompt='I love to' \
    attention="dot_product"