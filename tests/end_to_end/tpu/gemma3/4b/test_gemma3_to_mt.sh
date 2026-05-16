#!/bin/bash

# Converts Gemma3-4B HuggingFace checkpoint to MaxText format and validates logit correctness.

# The flow of this script is as follows:
# 1. Install PyTorch (CPU) required for checkpoint conversion.
# 2. Convert the HuggingFace checkpoint to MaxText format in both unscanned and scanned formats.
# 3. Run a forward pass logits check to verify the converted checkpoint matches the original HF model.

# Pre-requisites:
# 1. Set HF_TOKEN environment variable to your Hugging Face access token with read permissions
# export HF_TOKEN=<Hugging Face access token>


set -ex

run_id=${1:-$(date +%Y-%m-%d-%H-%M)}
MODEL_NAME='gemma3-4b'
HF_GOLDEN_MODEL='google/gemma-3-4b-it'

# To convert the multimodal model, make sure the use_multimodal is set to be true
USE_MULTIMODAL=false

# Non-Googlers please remember to point `BASE_OUTPUT_DIRECTORY` to the GCS paths where you want to store scanned and unscanned checkpoints
BASE_OUTPUT_DIRECTORY=gs://runner-maxtext-logs/${MODEL_NAME}/to_maxtext

# Step 1: Install torch
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# Step 2: Convert the checkpoint from Hugging Face to make it compatible with MaxText

# Step 2.a: Convert to unscanned checkpoint (for inference)
python3 -m maxtext.checkpoint_conversion.to_maxtext \
    model_name=${MODEL_NAME} \
    base_output_directory=${BASE_OUTPUT_DIRECTORY}/unscanned/${run_id} \
    use_multimodal=${USE_MULTIMODAL} \
    scan_layers=false \
    hardware=cpu skip_jax_distributed_system=True \
    checkpoint_storage_use_zarr3=False checkpoint_storage_use_ocdbt=False \
    --eager_load_method='transformers'

UNSCANNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/unscanned/${run_id}/0/items
echo "Unscanned checkpoint path: ${UNSCANNED_CKPT_PATH}"

# Step 2.b: Convert to scanned checkpoint (for training)
python3 -m maxtext.checkpoint_conversion.to_maxtext \
    model_name=${MODEL_NAME} \
    base_output_directory=${BASE_OUTPUT_DIRECTORY}/scanned/${run_id} \
    use_multimodal=${USE_MULTIMODAL} \
    scan_layers=true \
    hardware=cpu skip_jax_distributed_system=True \
    checkpoint_storage_use_zarr3=False checkpoint_storage_use_ocdbt=False \
    --eager_load_method='transformers'

SCANNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/scanned/${run_id}/0/items
echo "Scanned checkpoint path: ${SCANNED_CKPT_PATH}"

# Step 3: Test whether the forward pass logits match the original HF model
# to get higher precision (eg. float32) run on CPU with `JAX_PLATFORMS=cpu`
# ToDo: improve forward_pass_logit_checker to test multi-modal prompt
python3 -m tests.utils.forward_pass_logit_checker \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    model_name=${MODEL_NAME} \
    use_multimodal=${USE_MULTIMODAL} \
    scan_layers=false \
    --hf_model_path=${HF_GOLDEN_MODEL} \
    --max_kl_div=0.03 \
    --run_hf_model=true \
    hardware=cpu skip_jax_distributed_system=True
