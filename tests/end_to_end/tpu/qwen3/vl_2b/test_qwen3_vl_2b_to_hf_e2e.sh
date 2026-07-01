#!/bin/bash

# This script is both an end-to-end test and documentation for converting a
# Qwen3-VL-2B MaxText checkpoint to Hugging Face format. Can be run on a v4-8.

# The flow of this script is as follows:
# 1. Convert an original Hugging Face model checkpoint to MaxText format.
# 2. Convert the resulting MaxText checkpoint back to Hugging Face format.
# 3. Run a forward pass check to compare the logits and KL divergence between
#    the MaxText checkpoint and the Hugging Face checkpoint.

# Pre-requisites:
# 1. Set HF_TOKEN environment variable to your Hugging Face access token.
#    export HF_TOKEN=<Hugging Face access token>
# 2. Configure USE_MULTIMODAL (true for multimodal, false for text-only).
# 3. Configure USE_SCAN_LAYERS (true if checkpoint was trained with scanned layers, false otherwise).

set -ex


MODEL_NAME='qwen3-vl-2b'
export MODEL_VARIATION='vl_2b'
export HF_MODEL=Qwen/Qwen3-VL-2B-Instruct

idx=$(date +%Y-%m-%d-%H-%M)

# Set USE_SCAN_LAYERS=true if the checkpoint was trained with scanned layers
USE_SCAN_LAYERS=false
if ${USE_SCAN_LAYERS}; then export CHECKPOINT_TYPE=scanned; else export CHECKPOINT_TYPE=unscanned; fi

USE_MULTIMODAL=true


export HF_TOKEN=<hf_token>

export MODEL_BUCKET=<your_gcs_bucket_path>

# Path to Maxtext converted to HF checkpoint
export LOCAL_PATH=<your_local_path>/hf/${MODEL_NAME}/${idx}


# Installing torch for deps in forward_pass_logit_checker.py
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install decord

# Check point conversion
python3 -m maxtext.checkpoint_conversion.to_maxtext \
    "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"/base.yml \
    model_name=${MODEL_NAME} \
    base_output_directory=${MODEL_BUCKET}/${MODEL_NAME}/${CHECKPOINT_TYPE}/${idx} \
    scan_layers=false \
    hf_access_token=${HF_TOKEN} \
    weight_dtype=bfloat16 \
    hardware=cpu \
    skip_jax_distributed_system=True \
    checkpoint_storage_use_ocdbt=False \
    checkpoint_storage_use_zarr3=False \
    --eager_load_method=safetensors \
    --lazy_load_tensors=False

# Path to MaxText checkpoint
export CKPT_PATH=${MODEL_BUCKET}/${MODEL_NAME}/${CHECKPOINT_TYPE}/${idx}/0/items

python3 -m maxtext.checkpoint_conversion.to_huggingface \
    "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"/base.yml \
    model_name=${MODEL_NAME} \
    hf_access_token=${HF_TOKEN} \
    load_parameters_path=${CKPT_PATH} \
    base_output_directory=${LOCAL_PATH} \
    use_multimodal=${USE_MULTIMODAL} \
    scan_layers=${USE_SCAN_LAYERS} \
    override_model_config=true

# Run forward pass logit checker to validate the converted checkpoint.
if [ "${USE_MULTIMODAL}" == true ]; then
    TEST_PROMPT='Describe this image'
    TEST_IMAGE='tests/assets/test_image.jpg'
    export GOLDEN_LOGITS_PATH=/tmp/golden_qwen3_vl_2b_vision.jsonl

    python3 -m tests.assets.logits_generation.generate_hf_golden_logits \
        --model-id=${HF_MODEL} \
        --output-path=${GOLDEN_LOGITS_PATH} \
        --prompts="${TEST_PROMPT}" \
        --image-paths=${TEST_IMAGE} \
        --hf-model-path=${LOCAL_PATH} \
        --apply-chat-template \
        --output-format=json

    echo "=== Running MaxText Forward Pass Logit Checker ==="
    python3 -m tests.utils.forward_pass_logit_checker \
        "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"/base.yml \
        tokenizer_path=${HF_MODEL} \
        load_parameters_path=${CKPT_PATH} \
        model_name=${MODEL_NAME} \
        use_multimodal=${USE_MULTIMODAL} \
        scan_layers=${USE_SCAN_LAYERS} \
        dtype=float32 \
        wi_tile_fwd_embed_dim=512 \
        wi_tile_fwd_mlp_dim=512 \
        wo_tile_fwd_embed_dim=512 \
        wo_tile_fwd_mlp_dim=512 \
        matmul_precision=highest \
        per_device_batch_size=1 \
        attention=dot_product \
        prompt="${TEST_PROMPT}" \
        image_path=${TEST_IMAGE} \
        --max_kl_div=0.1 \
        --golden_logits_path=${GOLDEN_LOGITS_PATH} \
        override_model_config=true
else
    echo "=== Running MaxText Forward Pass Logit Checker ==="
    python3 -m tests.utils.forward_pass_logit_checker \
        "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"/base.yml \
        tokenizer_path=${HF_MODEL} \
        load_parameters_path=${CKPT_PATH} \
        model_name=${MODEL_NAME} \
        use_multimodal=${USE_MULTIMODAL} \
        scan_layers=${USE_SCAN_LAYERS} \
        per_device_batch_size=1 \
        dtype=float32 \
        wi_tile_fwd_embed_dim=512 \
        wi_tile_fwd_mlp_dim=512 \
        wo_tile_fwd_embed_dim=512 \
        wo_tile_fwd_mlp_dim=512 \
        --max_kl_div=0.1 \
        --run_hf_model=true \
        --hf_model_path=${LOCAL_PATH} \
        override_model_config=true
fi
