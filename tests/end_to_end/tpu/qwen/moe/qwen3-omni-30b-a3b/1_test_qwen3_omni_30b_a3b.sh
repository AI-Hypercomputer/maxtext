#!/bin/bash

# This file is documentation for how to get started with Qwen3-Omni-30B-A3B.

# This file runs Step 1 on CPU.
# 1. Convert the HuggingFace checkpoint (bf16) to MaxText-compatible checkpoint (bf16):
#    Unscanned format is used here as it is better suited for decoding.
# ---
# Example Usage:
#
# export HF_TOKEN=<your_hf_token>
# export BASE_OUTPUT_PATH=gs://your-gcs-bucket/qwen3-omni-30b-a3b_maxtext_ckpt
# bash tests/end_to_end/tpu/qwen/moe/qwen3-omni-30b-a3b/1_test_qwen3_omni_30b_a3b.sh
# ---

set -ex

export MODEL_NAME="${MODEL_NAME:-qwen3-omni-30b-a3b}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"

# (Optional) Path to your local Hugging Face checkpoint
export HF_MODEL_PATH="${HF_MODEL_PATH:-}"

# Base output path for MaxText checkpoint.
export BASE_OUTPUT_PATH="${BASE_OUTPUT_PATH:-gs://your-gcs-bucket/qwen3-omni-30b-a3b_maxtext_ckpt}"

if [ -z "${HF_TOKEN}" ]; then
  echo "Error: HF_TOKEN environment variable is not set. Please export your Hugging Face token."
  echo "Example: export HF_TOKEN=hf_..."
  exit 1
fi

# Strip trailing slash from base path to avoid malformed URIs
BASE_OUTPUT_PATH=${BASE_OUTPUT_PATH%/}
echo "Using BASE_OUTPUT_PATH = ${BASE_OUTPUT_PATH}"

# Install torch for checkpoint conversion
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# Setup local HF path argument if one was provided
HF_LOCAL_ARG=""
if [ -n "${HF_MODEL_PATH}" ]; then
  HF_LOCAL_ARG="hf_model_path=${HF_MODEL_PATH}"
fi

# ---
# Step 1: Checkpoint Conversion
# Convert HuggingFace checkpoint to MaxText unscanned format (better for decoding).
# use_multimodal=true is required to include vision/audio encoder weights.
# ---
JAX_PLATFORMS=cpu python3 -m maxtext.checkpoint_conversion.to_maxtext src/maxtext/configs/base.yml \
    model_name=${MODEL_NAME} \
    base_output_directory=${BASE_OUTPUT_PATH}/unscanned \
    hf_access_token=${HF_TOKEN} \
    scan_layers=false \
    use_multimodal=true \
    --lazy_load_tensors=False \
    ${HF_LOCAL_ARG}


