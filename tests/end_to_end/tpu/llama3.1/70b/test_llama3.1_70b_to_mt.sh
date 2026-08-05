#!/bin/bash

# Converts Llama3.1-70b HuggingFace checkpoint to MaxText format and validates logit correctness.

# The flow of this script is as follows:
# 1. Install PyTorch (CPU) required for checkpoint conversion.
# 2. Convert the HuggingFace checkpoint to MaxText format in both unscanned and scanned formats.
# 3. Run a forward pass logits check to verify the converted checkpoint matches the original HF model.

# Usage:
# export HF_TOKEN=<your Hugging Face access token>
# export RUN_ID=$(date +%Y-%m-%d-%H-%M-%S)
# bash test_llama3.1_70b_to_mt.sh $RUN_ID - to convert the checkpoint and run logit check for non-multimodal version
# bash test_llama3.1_70b_to_mt.sh $RUN_ID true - to convert the checkpoint and run logit check for multimodal version


set -ex

run_id=${1:-$(date +%Y-%m-%d-%H-%M-%S)}
MODEL_NAME='llama3.1-70b'
HF_GOLDEN_MODEL='meta-llama/Llama-3.1-70B'

# To convert the multimodal model, make sure the use_multimodal is set to be true
USE_MULTIMODAL=${2:-false}

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
    scan_layers=false --lazy_load_tensors=True \
    hardware=cpu skip_jax_distributed_system=True \
    checkpoint_storage_use_zarr3=False checkpoint_storage_use_ocdbt=False

UNSCANNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/unscanned/${run_id}/0/items
echo "Unscanned checkpoint path: ${UNSCANNED_CKPT_PATH}"

# Step 2.b: Convert to scanned checkpoint (for training)
python3 -m maxtext.checkpoint_conversion.to_maxtext \
    model_name=${MODEL_NAME} \
    base_output_directory=${BASE_OUTPUT_DIRECTORY}/scanned/${run_id} \
    use_multimodal=${USE_MULTIMODAL} \
    scan_layers=true --lazy_load_tensors=True \
    hardware=cpu skip_jax_distributed_system=True \
    checkpoint_storage_use_zarr3=False checkpoint_storage_use_ocdbt=False

SCANNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/scanned/${run_id}/0/items
echo "Scanned checkpoint path: ${SCANNED_CKPT_PATH}"

# Step 3: Test whether the forward pass logits match the original HF model
# to get higher precision (eg. float32) run on CPU with `JAX_PLATFORMS=cpu`
# ToDo: improve forward_pass_logit_checker to test multi-modal prompt
if [ "${USE_MULTIMODAL}" = "false" ]; then
    python3 -m tests.utils.forward_pass_logit_checker \
        load_parameters_path=${UNSCANNED_CKPT_PATH} \
        model_name=${MODEL_NAME} \
        use_multimodal=${USE_MULTIMODAL} \
        scan_layers=false \
        weight_dtype=bfloat16 \
        --hf_model_path=${HF_GOLDEN_MODEL} \
        --max_kl_div=0.03 \
        --run_hf_model=true \
        hardware=cpu skip_jax_distributed_system=True
fi