#!/bin/bash

# Converts GPT-OSS-20b HuggingFace checkpoint to MaxText format and validates logit correctness.

# The flow of this script is as follows:
# 1. Install PyTorch (CPU) required for checkpoint conversion.
# 2. Convert the HuggingFace checkpoint to MaxText format in both unscanned and scanned formats.
# 3. Run a forward pass logits check to verify the converted checkpoint matches the original HF model.

# Usage:
# export HF_TOKEN=<your Hugging Face access token>
# export RUN_ID=$(date +%Y-%m-%d-%H-%M-%S)
# bash test_gpt_oss_to_mt.sh $RUN_ID - to convert the checkpoint and run logit check

set -ex

export PYTHONPATH=src

run_id=${1:-$(date +%Y-%m-%d-%H-%M-%S)}
MODEL_NAME='gpt-oss-20b'

# Non-Googlers please remember to point BASE_OUTPUT_DIRECTORY to the GCS paths where you want to store scanned and unscanned checkpoints
BASE_OUTPUT_DIRECTORY=gs://runner-maxtext-logs/${MODEL_NAME}/to_maxtext

# Step 1: Install torch
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# Step 2: Convert the checkpoint from Hugging Face to make it compatible with MaxText

# Step 2.a: Convert to unscanned checkpoint (for inference)
python3 -m maxtext.checkpoint_conversion.to_maxtext \
    model_name=${MODEL_NAME} \
    --hf_model_path="unsloth/gpt-oss-20b-BF16" \
    base_output_directory=${BASE_OUTPUT_DIRECTORY}/unscanned/${run_id} \
    use_multimodal=false \
    scan_layers=false \
    hardware=cpu \
    skip_jax_distributed_system=True \
    checkpoint_storage_use_zarr3=False \
    checkpoint_storage_use_ocdbt=False \
    attention="dot_product"

UNSCANNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/unscanned/${run_id}/0/items
echo "Unscanned checkpoint path: ${UNSCANNED_CKPT_PATH}"

# Step 2.b: Convert to scanned checkpoint (for training)
python3 -m maxtext.checkpoint_conversion.to_maxtext \
    model_name=${MODEL_NAME} \
    --hf_model_path="unsloth/gpt-oss-20b-BF16" \
    base_output_directory=${BASE_OUTPUT_DIRECTORY}/scanned/${run_id} \
    use_multimodal=false \
    scan_layers=true \
    hardware=cpu \
    skip_jax_distributed_system=True \
    checkpoint_storage_use_zarr3=False \
    checkpoint_storage_use_ocdbt=False \
    attention="dot_product"

SCANNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/scanned/${run_id}/0/items
# Step 3: Test whether the forward pass logits match the original HF model
# to get higher precision (eg. float32) run on CPU with JAX_PLATFORMS=cpu
if [ ! -f /tmp/golden_data_gpt-oss-20b.jsonl ]; then
  gcloud storage cp gs://maxtext-test-assets/golden_data_gpt-oss-20b.jsonl /tmp/golden_data_gpt-oss-20b.jsonl
fi

python3 -m tests.utils.forward_pass_logit_checker \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    model_name=${MODEL_NAME} \
    use_multimodal=false \
    scan_layers=false \
    global_batch_size_to_train_on=1 \
    per_device_batch_size=1 \
    max_target_length=512 \
    --golden_logits_path=/tmp/golden_data_gpt-oss-20b.jsonl \
    --max_kl_div=0.01 \
    hardware=cpu \
    skip_jax_distributed_system=True