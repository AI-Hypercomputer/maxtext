#!/bin/bash

# This script is both an end-to-end test and documentation for converting
# Qwen3-VL models between MaxText and Hugging Face formats. Can be run on a v4-8.
#
# TODO: rename this file to test_qwen3_vl_to_hf_e2e.sh and move it to ../

# The flow of this script is as follows:
# 1. Convert an original Hugging Face model checkpoint to MaxText format.
# 2. Run a forward pass check to compare the logits and KL divergence between
#    the MaxText checkpoint and the Hugging Face checkpoint.
# 3. (Optional) Convert the resulting MaxText checkpoint back to Hugging Face format.

# Required flags:
# 1. --hf_token: Your Hugging Face access token.
# 2. --model_bucket: Your GCS bucket path for storing models.
#
# Sample usage:
# export HF_TOKEN=<hf_token>
# export MODEL_BUCKET=<your_gcs_bucket_path>
# ./test_qwen3_vl_30b_to_hf_e2e.sh --hf_token=${HF_TOKEN} --model_bucket=${MODEL_BUCKET}


maxtext_folder_path="${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}}"

# Source shflags library (silencing logging during source to keep output clean)
. "${maxtext_folder_path}/tests/end_to_end/tpu/qwen3/vl_30b/shflags"

# Declare flags with default values
DEFINE_boolean 'run_convert_to_maxtext' true 'Run checkpoint conversion to MaxText' 'c'
DEFINE_boolean 'run_forward_pass' true 'Run forward pass check' 'f'
DEFINE_boolean 'run_convert_back_to_hf' false 'Run checkpoint conversion back to HF' 'b'

DEFINE_string 'use_scan_layers' 'false' 'Whether the checkpoint was trained with scanned layers' 's'
DEFINE_string 'use_multimodal' 'true' 'Use multimodal processing' 'u'
DEFINE_string 'max_kl_div' '0.1' 'Maximum KL divergence allowed' 'k'

DEFINE_string 'model_name' 'qwen3-vl-30b-a3b' 'Model name for conversion' 'm'
DEFINE_string 'hf_token' '' 'Hugging Face access token' 't'
DEFINE_string 'model_bucket' '' 'GCS model bucket to export to' 'g'

DEFINE_string 'idx' '' 'Unique index or timestamp for checkpoint directories' 'i'
DEFINE_string 'local_path' '' 'Local path to HF checkpoint' 'l'
DEFINE_string 'hf_home' '' 'Hugging Face home cache directory' 'o'


# Parse flags
FLAGS "$@" || exit $?
eval set -- "${FLAGS_ARGV}"

# Check required flags
for flag in FLAGS_hf_token FLAGS_model_bucket; do
    [ -z "${!flag}" ] && { echo "Error: --${flag#FLAGS_} is required."; exit 1; }
done

# Set default flag values
# Set default flag values
FLAGS_idx=${FLAGS_idx:-$(date +%Y-%m-%d-%H-%M)}
FLAGS_local_path=${FLAGS_local_path:-"/dev/shm/hf/${FLAGS_model_name}/${FLAGS_idx}"}


if [ "${FLAGS_use_scan_layers}" = "true" ]; then export CHECKPOINT_TYPE=scanned; else export CHECKPOINT_TYPE=unscanned; fi

set -x # Enable tracing only for the main logic

# Set MODEL_VARIATION and HF_MODEL based on model_name
# TODO: Add support for other qwen3-vl models (e.g., 8b) if needed
if [ "${FLAGS_model_name}" = "qwen3-vl-30b-a3b" ]; then
    MODEL_VARIATION="vl_30b"
    HF_MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct"
elif [ "${FLAGS_model_name}" = "qwen3-vl-2b" ]; then
    MODEL_VARIATION="vl_2b"
    HF_MODEL="Qwen/Qwen3-VL-2B-Instruct"
else
    echo "Unsupported model name: ${FLAGS_model_name}"
    exit 1
fi




# Optional: Redirect HF cache to RAM disk
if [ -n "${FLAGS_hf_home}" ]; then
    export HF_HOME="${FLAGS_hf_home}"
fi



# ==================== Sub-functions ====================

convert_checkpoint_to_maxtext() {
    # Checkpoint conversion
    python3 -m maxtext.checkpoint_conversion.to_maxtext \
        "${maxtext_folder_path}/src/maxtext/configs/base.yml" \
        model_name="${FLAGS_model_name}" \
        base_output_directory="${FLAGS_model_bucket}/${FLAGS_model_name}/${CHECKPOINT_TYPE}/${FLAGS_idx}" \
        scan_layers=false \
        hf_access_token=${FLAGS_hf_token} \
        weight_dtype=bfloat16 \
        hardware=cpu \
        skip_jax_distributed_system=True \
        checkpoint_storage_use_ocdbt=False \
        checkpoint_storage_use_zarr3=False \
        --eager_load_method=safetensors \
        --lazy_load_tensors=False
}

# For multimodal forward pass
generate_hf_golden_logits() {
    python3 -m tests.assets.logits_generation.generate_hf_golden_logits \
        --model-id="${HF_MODEL}" \
        --output-path=${GOLDEN_LOGITS_PATH} \
        --prompts="${TEST_PROMPT}" \
        --image-paths=${TEST_IMAGE} \
        --hf-model-path="${HF_MODEL}" \
        --apply-chat-template \
        --output-format=json
}

run_forward_pass_logit_checker_multimodal() {
    echo "=== Running MaxText Forward Pass Logit Checker ==="
    # Note: matmul_precision, attention, prompt, image_path, and golden_logits_path are Multimodal only
    python3 -m tests.utils.forward_pass_logit_checker \
        "${maxtext_folder_path}/src/maxtext/configs/base.yml" \
        tokenizer_path="${HF_MODEL}" \
        load_parameters_path=${CKPT_PATH} \
        model_name="${FLAGS_model_name}" \
        use_multimodal=${FLAGS_use_multimodal} \
        scan_layers=${FLAGS_use_scan_layers} \
        dtype=float32 \
        matmul_precision=highest \
        per_device_batch_size=1 \
        attention=dot_product \
        prompt="${TEST_PROMPT}" \
        image_path=${TEST_IMAGE} \
        --max_kl_div=${FLAGS_max_kl_div} \
        --golden_logits_path=${GOLDEN_LOGITS_PATH} \
        override_model_config=true
}

run_forward_pass_logit_checker() {
    echo "=== Running MaxText Forward Pass Logit Checker ==="
    # Note: run_hf_model and hf_model_path are Non-multimodal only
    python3 -m tests.utils.forward_pass_logit_checker \
        "${maxtext_folder_path}/src/maxtext/configs/base.yml" \
        tokenizer_path="${HF_MODEL}" \
        load_parameters_path=${CKPT_PATH} \
        model_name="${FLAGS_model_name}" \
        use_multimodal=${FLAGS_use_multimodal} \
        scan_layers=${FLAGS_use_scan_layers} \
        per_device_batch_size=1 \
        dtype=float32 \
        --max_kl_div=${FLAGS_max_kl_div} \
        --run_hf_model=true \
        --hf_model_path="${HF_MODEL}" \
        override_model_config=true
}

# Optional: Convert checkpoint back to HF format
convert_checkpoint_back_to_hf() {
    python3 -m maxtext.checkpoint_conversion.to_huggingface \
        "${maxtext_folder_path}/src/maxtext/configs/base.yml" \
        model_name="${FLAGS_model_name}" \
        hf_access_token=${FLAGS_hf_token} \
        load_parameters_path=${CKPT_PATH} \
        base_output_directory=${FLAGS_local_path} \
        use_multimodal=${FLAGS_use_multimodal} \
        scan_layers=${FLAGS_use_scan_layers} \
        override_model_config=true
}




# ==================== Main Caller ====================

# Installing torch for deps in forward_pass_logit_checker.py
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install decord

if [ "${FLAGS_run_convert_to_maxtext}" -eq "${FLAGS_TRUE}" ]; then
    convert_checkpoint_to_maxtext
fi

# Path to MaxText checkpoint
export CKPT_PATH="${FLAGS_model_bucket}/${FLAGS_model_name}/${CHECKPOINT_TYPE}/${FLAGS_idx}/0/items"


if [ "${FLAGS_run_forward_pass}" -eq "${FLAGS_TRUE}" ]; then
    if [ "${FLAGS_use_multimodal}" = "true" ]; then
        TEST_PROMPT='Describe this image'
        TEST_IMAGE='tests/assets/test_image.jpg'
        export GOLDEN_LOGITS_PATH=/tmp/golden_qwen3_vl_30b_vision.jsonl
        generate_hf_golden_logits
        run_forward_pass_logit_checker_multimodal
    else
        run_forward_pass_logit_checker
    fi
fi

if [ "${FLAGS_run_convert_back_to_hf}" -eq "${FLAGS_TRUE}" ]; then
    convert_checkpoint_back_to_hf
fi
