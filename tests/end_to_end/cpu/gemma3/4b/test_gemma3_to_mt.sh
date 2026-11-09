#!/bin/bash

# Converts Gemma3-4B HuggingFace checkpoint to MaxText format and validates logit correctness.

# The flow of this script is as follows:
# 1. Install PyTorch (CPU) required for checkpoint conversion.
# 2. Convert the HuggingFace checkpoint to MaxText format in both unscanned and scanned formats.
# 3. Run a forward pass logits check to verify the converted checkpoint matches the original HF model.

# Usage:
# export HF_TOKEN=<your Hugging Face access token>
# export RUN_ID=$(date +%Y-%m-%d-%H-%M-%S)
# bash tests/end_to_end/cpu/gemma3/4b/test_gemma3_to_mt.sh $RUN_ID
# bash tests/end_to_end/cpu/gemma3/4b/test_gemma3_to_mt.sh $RUN_ID true


set -ex

# Avoid conflict between M_RUN_NAME env var and CLI run_name argument in pyconfig
unset M_RUN_NAME

run_id=${1:-$(date +%Y-%m-%d-%H-%M-%S)}
MODEL_NAME='gemma3-4b'
HF_GOLDEN_MODEL='google/gemma-3-4b-it'

# To convert the multimodal model, make sure use_multimodal is set to true
USE_MULTIMODAL=${2:-false}

# Non-Googlers please remember to point `BASE_OUTPUT_DIRECTORY` to the GCS paths where you want to store scanned and unscanned checkpoints
BASE_OUTPUT_DIRECTORY="${BASE_OUTPUT_DIRECTORY:-gs://runner-maxtext-logs/${MODEL_NAME}}"

# Step 1: Install torch if not already installed
if ! python3 -c "import torch" &>/dev/null; then
    python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

# Step 2: Convert the checkpoint from Hugging Face to make it compatible with MaxText

# Step 2.a: Convert to unscanned checkpoint (for inference)
python3 -m maxtext.checkpoint_conversion.to_maxtext \
    model_name=${MODEL_NAME} \
    base_output_directory=${BASE_OUTPUT_DIRECTORY}/to_maxtext/unscanned/${run_id} \
    use_multimodal=${USE_MULTIMODAL} \
    scan_layers=false \
    hardware=cpu \
    skip_jax_distributed_system=True \
    checkpoint_storage_use_zarr3=False \
    checkpoint_storage_use_ocdbt=False \
    --lazy_load_tensors=False \
    --eager_load_method='transformers'

UNSCANNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/to_maxtext/unscanned/${run_id}/0/items
echo "Unscanned checkpoint path: ${UNSCANNED_CKPT_PATH}"

# Step 2.b: Convert to scanned multimodal checkpoint (for multimodal training)
python3 -m maxtext.checkpoint_conversion.to_maxtext \
    model_name=${MODEL_NAME} \
    base_output_directory=${BASE_OUTPUT_DIRECTORY}/to_maxtext/scanned_multimodal/${run_id} \
    use_multimodal=true \
    scan_layers=true \
    hardware=cpu \
    skip_jax_distributed_system=True \
    checkpoint_storage_use_zarr3=False \
    checkpoint_storage_use_ocdbt=False \
    --lazy_load_tensors=False \
    --eager_load_method='transformers'

MULTIMODAL_SCANNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/to_maxtext/scanned_multimodal/${run_id}/0/items
echo "Multimodal Scanned checkpoint path: ${MULTIMODAL_SCANNED_CKPT_PATH}"

# Step 2.c: Convert to scanned checkpoint (for training)
python3 -m maxtext.checkpoint_conversion.to_maxtext \
    model_name=${MODEL_NAME} \
    base_output_directory=${BASE_OUTPUT_DIRECTORY}/to_maxtext/scanned/${run_id} \
    use_multimodal=${USE_MULTIMODAL} \
    scan_layers=true \
    hardware=cpu \
    skip_jax_distributed_system=True \
    checkpoint_storage_use_zarr3=False \
    checkpoint_storage_use_ocdbt=False \
    --lazy_load_tensors=False \
    --eager_load_method='transformers'

SCANNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/to_maxtext/scanned/${run_id}/0/items
echo "Scanned checkpoint path: ${SCANNED_CKPT_PATH}"

# Step 3: Test whether the forward pass logits match the original HF model
# to get higher precision (eg. float32) run on CPU with `hardware=cpu`
if [ "${USE_MULTIMODAL}" = "false" ]; then
    python3 -m tests.utils.forward_pass_logit_checker \
        load_parameters_path=${UNSCANNED_CKPT_PATH} \
        model_name=${MODEL_NAME} \
        use_multimodal=${USE_MULTIMODAL} \
        scan_layers=false \
        dtype=float32 \
        checkpoint_storage_use_zarr3=False \
        checkpoint_storage_use_ocdbt=False \
        --hf_model_path=${HF_GOLDEN_MODEL} \
        --max_kl_div=0.05 \
        --run_hf_model=true \
        attention=dot_product \
        hardware=cpu \
        skip_jax_distributed_system=True
fi
