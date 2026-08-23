#!/bin/bash

# Converts Qwen3-VL-2B HuggingFace checkpoint to MaxText format and validates logit correctness.

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
MODEL_NAME='qwen3-vl-2b'
HF_GOLDEN_MODEL=Qwen/Qwen3-VL-2B-Instruct

# To convert the multimodal model, make sure the use_multimodal is set to be true

BASE_OUTPUT_DIRECTORY=gs://runner-maxtext-logs/${MODEL_NAME}/to_maxtext

# Step 1: Install torch
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install decord

# Step 2: Convert to scanned multimodal checkpoint (for multimodal training)
python3 -m maxtext.checkpoint_conversion.to_maxtext \
    model_name=${MODEL_NAME} \
    base_output_directory=${BASE_OUTPUT_DIRECTORY}/unscanned_multimodal/${run_id} \
    use_multimodal=true \
    scan_layers=false \
    hardware=cpu \
    skip_jax_distributed_system=True \
    checkpoint_storage_use_zarr3=False \
    checkpoint_storage_use_ocdbt=False \
    --lazy_load_tensors=False \
    --eager_load_method='safetensors'

MULTIMODAL_UNSCANNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/unscanned_multimodal/${run_id}/0/items
echo "Multimodal Unscanned checkpoint path: ${MULTIMODAL_UNSCANNED_CKPT_PATH}"

# Step 3: Test whether the forward pass logits match the original HF model
# to get higher precision (eg. float32) run on CPU with `JAX_PLATFORMS=cpu`
TEST_PROMPT='Describe this image'
TEST_IMAGE='tests/assets/test_image.jpg'
export GOLDEN_LOGITS_PATH=/tmp/golden_qwen3_vl_2b_vision.jsonl

python3 -m tests.assets.logits_generation.generate_hf_golden_logits \
    --model-id=${HF_GOLDEN_MODEL} \
    --output-path=${GOLDEN_LOGITS_PATH} \
    --prompts="${TEST_PROMPT}" \
    --image-paths=${TEST_IMAGE} \
    --hf-model-path=${HF_GOLDEN_MODEL} \
    --apply-chat-template \
    --output-format=json

echo "=== Running MaxText Multimodal Forward Pass Logit Checker ==="
python3 -m tests.utils.forward_pass_logit_checker \
    tokenizer_path=${HF_GOLDEN_MODEL} \
    load_parameters_path=${MULTIMODAL_UNSCANNED_CKPT_PATH} \
    model_name=${MODEL_NAME} \
    use_multimodal=true \
    scan_layers=false \
    dtype=float32 \
    matmul_precision=highest \
    per_device_batch_size=1 \
    attention=dot_product \
    prompt="${TEST_PROMPT}" \
    image_path=${TEST_IMAGE} \
    --max_kl_div=0.03 \
    --golden_logits_path=${GOLDEN_LOGITS_PATH} \
    override_model_config=true \
    hardware=cpu \
    skip_jax_distributed_system=True