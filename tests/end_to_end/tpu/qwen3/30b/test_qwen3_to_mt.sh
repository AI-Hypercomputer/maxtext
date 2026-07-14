#!/bin/bash

# Converts Qwen3-30B HuggingFace checkpoint to MaxText format and validates logit correctness.

# The flow of this script is as follows:
# 1. Install PyTorch (CPU) required for checkpoint conversion.
# 2. Convert the HuggingFace checkpoint to MaxText format in both unscanned and scanned formats.
# 3. Run a forward pass logits check to verify the converted checkpoint matches the original HF model.

# Usage:
# export HF_TOKEN=<your Hugging Face access token>
# export RUN_ID=$(date +%Y-%m-%d-%H-%M-%S)
# bash test_qwen3_to_mt.sh $RUN_ID

set -ex

run_id=${1:-$(date +%Y-%m-%d-%H-%M-%S)}
MODEL_NAME='qwen3-30b-a3b-base'
HF_GOLDEN_MODEL='Qwen/Qwen3-30B-A3B-Base'


BASE_OUTPUT_DIRECTORY=gs://runner-maxtext-logs/${MODEL_NAME}/to_maxtext

# Step 1: Install torch
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# Step 2: Convert the checkpoint from Hugging Face
python3 -m maxtext.checkpoint_conversion.to_maxtext \
    model_name=${MODEL_NAME} \
    base_output_directory=${BASE_OUTPUT_DIRECTORY}/unscanned/${run_id} \
    scan_layers=false \
    hardware=cpu skip_jax_distributed_system=True \
    checkpoint_storage_use_zarr3=False checkpoint_storage_use_ocdbt=False \
    --lazy_load_tensors=False \
    --eager_load_method='safetensors'

UNSCANNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/unscanned/${run_id}/0/items
echo "Unscanned checkpoint path: ${UNSCANNED_CKPT_PATH}"

# Convert to scanned format
python3 -m maxtext.checkpoint_conversion.to_maxtext \
    model_name=${MODEL_NAME} \
    base_output_directory=${BASE_OUTPUT_DIRECTORY}/scanned/${run_id} \
    scan_layers=true \
    hardware=cpu skip_jax_distributed_system=True \
    checkpoint_storage_use_zarr3=False checkpoint_storage_use_ocdbt=False \
    --lazy_load_tensors=False \
    --eager_load_method='safetensors'

SCANNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/scanned/${run_id}/0/items
echo "Scanned checkpoint path: ${SCANNED_CKPT_PATH}"

# Step 3: Run forward pass logits check
python3 -m tests.utils.forward_pass_logit_checker \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    model_name=${MODEL_NAME} \
    scan_layers=false \
    --hf_model_path=${HF_GOLDEN_MODEL} \
    --max_kl_div=0.03 \
    --run_hf_model=true \
    hardware=cpu skip_jax_distributed_system=True
