#!/bin/bash

# Converts Qwen3.5-35B HuggingFace checkpoint to MaxText format and validates logit correctness.

# The flow of this script is as follows:
# 1. Install PyTorch (CPU) required for checkpoint conversion.
# 2. Convert the HuggingFace checkpoint to MaxText format in both unscanned and scanned formats.
# 3. Run a forward pass logits check to verify the converted checkpoint matches the original HF model.

# Usage:
# export HF_TOKEN=<your Hugging Face access token>
# export RUN_ID=$(date +%Y-%m-%d-%H-%M-%S)
# bash test_qwen3.5_to_mt.sh $RUN_ID

set -ex

run_id=${1:-$(date +%Y-%m-%d-%H-%M-%S)}
MODEL_NAME='qwen3.5-35b-a3b'


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
if [ ! -f /tmp/golden_data_qwen3.5-35b-a3b.jsonl ]; then
  gcloud storage cp gs://maxtext-test-assets/golden_data_qwen3.5-35b-a3b.jsonl /tmp/golden_data_qwen3.5-35b-a3b.jsonl
fi

python3 -m tests.utils.forward_pass_logit_checker \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    model_name=${MODEL_NAME} \
    scan_layers=false \
    per_device_batch_size=1 \
    max_target_length=4 \
    dtype=float32 \
    attention=dot_product \
    --golden_logits_path=/tmp/golden_data_qwen3.5-35b-a3b.jsonl \
    --atol=1.5 \
    --rtol=1.5 \
    --max_kl_div=0.2 \
    hardware=cpu \
    skip_jax_distributed_system=True