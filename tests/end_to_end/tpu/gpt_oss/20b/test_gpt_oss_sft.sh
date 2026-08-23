#!/bin/bash

# Validates the GPT-OSS-20b SFT pipeline using a pre-converted MaxText checkpoint.

# The flow of this script is as follows:
# 1. Run SFT starting from the pre-converted checkpoint.

# Usage:
# export HF_TOKEN=<your Hugging Face access token>
# export RUN_ID=$(date +%Y-%m-%d-%H-%M-%S)
# bash test_gpt_oss_to_mt.sh $RUN_ID
# bash test_gpt_oss_sft.sh $RUN_ID


set -ex

if [ -z "$1" ]; then
  echo "Error: run_id argument is required."
  exit 1
fi
run_id=$1
use_pathways=${2:-false}
export MODEL_NAME='gpt-oss-20b'

# Non-Googlers please remember to point BASE_OUTPUT_DIRECTORY to the GCS paths where you have the scanned and unscanned checkpoints stored
BASE_OUTPUT_DIRECTORY=gs://runner-maxtext-logs/${MODEL_NAME}
SCANNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/to_maxtext/scanned/${run_id}/0/items

# Step 1: Run SFT on the converted checkpoint
python3 -m maxtext.trainers.post_train.sft.train_sft \
    base_output_directory=${BASE_OUTPUT_DIRECTORY}/sft \
    load_parameters_path=${SCANNED_CKPT_PATH} \
    per_device_batch_size=1 \
    run_name=${run_id} \
    steps=2 \
    scan_layers=true \
    model_name=${MODEL_NAME} \
    tokenizer_path='unsloth/gpt-oss-20b-BF16' \
    enable_single_controller=${use_pathways} \
    checkpoint_storage_use_zarr3=False \
    checkpoint_storage_use_ocdbt=False