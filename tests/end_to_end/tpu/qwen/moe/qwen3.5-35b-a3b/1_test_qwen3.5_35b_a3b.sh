#!/bin/bash

# This file is documentation for how to get started with Qwen3.5-35B.

# This file runs Step 1 on CPU.
# 1. Convert the HuggingFace checkpoint (bf16) to MaxText-compatible checkpoint (bf16): 
#    Scanned format is better for training; unscanned format is better for decoding.
# 2. Run logit check, pre-training, fine-tuning, and decoding.
# ---
# Example Usage:
#
# bash tests/end_to_end/tpu/qwen/moe/qwen3.5-35b-a3b/1_test_qwen3.5_35b_a3b.sh
# ---

set -ex

export MODEL_NAME="${MODEL_NAME:-qwen3.5-35b-a3b}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-Qwen/Qwen3.5-35B-A3B}"

# (Optional) Path to your local Hugging Face checkpoint
export HF_MODEL_PATH="${HF_MODEL_PATH:-}" 

# Base output path for MaxText generations. 
export BASE_OUTPUT_PATH="${BASE_OUTPUT_PATH:-gs://your-gcs-bucket/qwen3.5-35b-a3b_maxtext_ckpt}"

if [ -z "${HF_TOKEN}" ]; then
  echo "Error: HF_TOKEN environment variable is not set. Please export your Hugging Face token."
  echo "Example: export HF_TOKEN=hf_..."
  exit 1
fi

# Strip trailing slash from base path to avoid malformed URIs
BASE_OUTPUT_PATH=${BASE_OUTPUT_PATH%/}
echo "Using BASE_OUTPUT_PATH = ${BASE_OUTPUT_PATH}"

# Install torch for checkpoint conversion and forward_pass_logit_checker.py
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# Setup local HF path argument if one was provided
HF_LOCAL_ARG=""
if [ -n "${HF_MODEL_PATH}" ]; then
  HF_LOCAL_ARG="hf_model_path=${HF_MODEL_PATH}"
fi

# 1.1 Convert checkpoint to `scanned` format, more suitable for training 
JAX_PLATFORMS=cpu python3 -m maxtext.checkpoint_conversion.to_maxtext src/maxtext/configs/base.yml \
    model_name=${MODEL_NAME} \
    base_output_directory=${BASE_OUTPUT_PATH}/scanned \
    hf_access_token=${HF_TOKEN} \
    scan_layers=true \
    use_multimodal=false \
    ${HF_LOCAL_ARG}

# 1.2 Convert checkpoint to `unscanned` format, more suitable for decoding
JAX_PLATFORMS=cpu python3 -m maxtext.checkpoint_conversion.to_maxtext src/maxtext/configs/base.yml \
    model_name=${MODEL_NAME} \
    base_output_directory=${BASE_OUTPUT_PATH}/unscanned \
    hf_access_token=${HF_TOKEN} \
    scan_layers=false \
    use_multimodal=false \
    ${HF_LOCAL_ARG}