#!/bin/bash

# Validates the Gemma3-4B pre-training pipeline using a pre-converted MaxText checkpoint.

# The flow of this script is as follows:
# 1. Run inference on the pre-converted checkpoint.
# 2. Run pre-training starting from the pre-converted checkpoint.
# 3. Run inference on the checkpoint produced by the pre-training run.
# 4. Convert the checkpoint produced by the pre-training run back to HuggingFace format.

# Usage:
# export HF_TOKEN=<your Hugging Face access token>
# export RUN_ID=$(date +%Y-%m-%d-%H-%M)
# bash test_gemma3_to_mt.sh $RUN_ID
# bash test_gemma3.sh $RUN_ID


set -ex

run_id=${1:-$(date +%Y-%m-%d-%H-%M)}
MODEL_NAME='gemma3-4b'

# To convert the multimodal model, make sure the use_multimodal is set to be true
USE_MULTIMODAL=false

# Non-Googlers please remember to point `BASE_OUTPUT_DIRECTORY` to the GCS paths where you have the scanned and unscanned checkpoints stored
BASE_OUTPUT_DIRECTORY=gs://runner-maxtext-logs/${MODEL_NAME}
UNSCANNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/to_maxtext/unscanned/${run_id}/0/items

# Non-Googlers please remember to point `DATASET_PATH` to the GCS bucket where you have your training data
DATASET_PATH=gs://maxtext-dataset

# Step 1: Install torch
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# Step 2: Run inference on the original checkpoint converted from Hugging Face
if [ ${USE_MULTIMODAL} == true ]; then
    python3 -m maxtext.inference.decode \
    model_name=${MODEL_NAME} tokenizer_type="huggingface" \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    per_device_batch_size=1 run_name=${run_id} \
    max_prefill_predict_length=272 max_target_length=300 steps=1 async_checkpointing=false \
    scan_layers=false use_multimodal=true \
    checkpoint_storage_use_zarr3=False checkpoint_storage_use_ocdbt=False \
    prompt=\'Describe\ image\ \<start_of_image\>\' image_path=\'tests/assets/test_image.jpg\' attention=\'dot_product\'
else
    python3 -m maxtext.inference.decode \
    model_name=${MODEL_NAME} tokenizer_type="huggingface" \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    per_device_batch_size=1 run_name=${run_id} \
    max_prefill_predict_length=8 max_target_length=16 steps=1 async_checkpointing=false \
    checkpoint_storage_use_zarr3=False checkpoint_storage_use_ocdbt=False \
    scan_layers=false prompt='I love to' attention=\'dot_product\'
fi

# Step 3: Run Pre-training on the converted checkpoint
# We can also run training by using the scanned converted checkpoint
# Note that scanned checkpoint helps with efficient training
python3 -m maxtext.trainers.pre_train.train \
    base_output_directory=${BASE_OUTPUT_DIRECTORY}/train \
    dataset_path=${DATASET_PATH} tokenizer_type="huggingface" \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    per_device_batch_size=1 run_name=${run_id} \
    max_target_length=8192 steps=5 async_checkpointing=false \
    checkpoint_storage_use_zarr3=False checkpoint_storage_use_ocdbt=False \
    model_name=${MODEL_NAME} scan_layers=false use_multimodal=${USE_MULTIMODAL}

# Step 4: Run inference on the checkpoint generated from the previous run
if [ ${USE_MULTIMODAL} == true ]; then
    python3 -m maxtext.inference.decode \
    model_name=${MODEL_NAME} tokenizer_type="huggingface" \
    load_parameters_path=${BASE_OUTPUT_DIRECTORY}/train/${run_id}/checkpoints/4/items \
    per_device_batch_size=1 run_name=${run_id} \
    max_prefill_predict_length=272 max_target_length=300 steps=1 async_checkpointing=false \
    scan_layers=false use_multimodal=true \
    checkpoint_storage_use_zarr3=False checkpoint_storage_use_ocdbt=False \
    prompt=\'Describe\ image\ \<start_of_image\>\' image_path=\'tests/assets/test_image.jpg\' attention=\'dot_product\'
else
    python3 -m maxtext.inference.decode \
    model_name=${MODEL_NAME} tokenizer_type="huggingface" \
    load_parameters_path=${BASE_OUTPUT_DIRECTORY}/train/${run_id}/checkpoints/4/items \
    per_device_batch_size=1 run_name=${run_id} \
    max_prefill_predict_length=8 max_target_length=16 steps=1 async_checkpointing=false \
    checkpoint_storage_use_zarr3=False checkpoint_storage_use_ocdbt=False \
    scan_layers=false prompt='I love to' attention=\'dot_product\'
fi

# Step 5: Convert the checkpoint from MaxText format to Hugging Face format
python3 -m maxtext.checkpoint_conversion.to_huggingface \
    model_name=${MODEL_NAME} tokenizer_type="huggingface" \
    load_parameters_path=${BASE_OUTPUT_DIRECTORY}/train/${run_id}/checkpoints/4/items \
    base_output_directory=${BASE_OUTPUT_DIRECTORY}/to_huggingface/unscanned/${run_id} \
    use_multimodal=${USE_MULTIMODAL} \
    scan_layers=false
