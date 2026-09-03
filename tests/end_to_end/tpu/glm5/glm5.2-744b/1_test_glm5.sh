#!/bin/bash

# This file is documentation for how to get started with GLM-5.2 (Cross-Layer IndexShare).

# This file runs Step 1 on CPU.
# 1. Convert the HuggingFace checkpoint (bf16) to MaxText-compatible checkpoint (bf16):
#    Scanned format is better for training; unscanned format is better for decoding.
# 2. Run logit check, pre-training, fine-tuning, and decoding.

set -ex

export MODEL_NAME='glm5.2-744b'
export TOKENIZER_PATH='zai-org/GLM-5.2'

# Installing torch for checkpoint conversion and forward_pass_logit_checker.py
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

if [ -z "${BASE_OUTPUT_PATH}" ]; then
  export BASE_OUTPUT_PATH=gs://runner-maxtext-logs/$(date +%Y-%m-%d-%H-%M)
  echo "BASE_OUTPUT_PATH is not set"
fi
BASE_OUTPUT_PATH=${BASE_OUTPUT_PATH%/}
echo using BASE_OUTPUT_PATH = ${BASE_OUTPUT_PATH}

# Step 1: Checkpoint conversion
# HF checkpoint: https://huggingface.co/zai-org/GLM-5.2
BF16_HF_PATH=${BF16_HF_PATH:-gs://maxtext-glm5-europe-west4/glm5.2_raw}
if [ -z "${BF16_LOCAL_PATH}" ] && [ ! -d "/home/rishabhbaghel_google_com/glm5.2_raw" ]; then
  export BF16_LOCAL_PATH=/tmp/glm5.2_raw
  gcloud storage cp -r ${BF16_HF_PATH} /tmp || true
fi
BF16_LOCAL_PATH=${BF16_LOCAL_PATH:-/home/rishabhbaghel_google_com/glm5.2_raw}

# scanned
python3 -m maxtext.checkpoint_conversion.to_maxtext src/maxtext/configs/base.yml \
  model_name=${MODEL_NAME} scan_layers=true \
  base_output_directory=${BASE_OUTPUT_PATH}/scanned hf_access_token=$HF_TOKEN \
  hardware=cpu skip_jax_distributed_system=True \
  checkpoint_storage_concurrent_gb=1024 \
  --hf_model_path=$BF16_LOCAL_PATH \
  --lazy_load_tensors=False \
  --eager_load_method=safetensors \
  --save_dtype=bfloat16

# unscanned
python3 -m maxtext.checkpoint_conversion.to_maxtext src/maxtext/configs/base.yml \
  model_name=${MODEL_NAME} scan_layers=false \
  base_output_directory=${BASE_OUTPUT_PATH}/unscanned hf_access_token=$HF_TOKEN \
  hardware=cpu skip_jax_distributed_system=True \
  checkpoint_storage_concurrent_gb=1024 \
  --hf_model_path=$BF16_LOCAL_PATH \
  --lazy_load_tensors=False \
  --eager_load_method=safetensors \
  --save_dtype=bfloat16
