#!/bin/bash

# Qwen3-Omni 30B-A3B Decode Script
# Based on configuration from qwen3_omni_decode.log

set -ex

# --- Configuration ---

# Path to the MaxText checkpoint (REQUIRED)
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/disks/tpu-disk/test_conversion_noscan/0/items}"

# Image and audio paths (optional)
IMAGE_PATH="${IMAGE_PATH:-/home/eporat/maxtext/image.jpg}"
# --- Run Decode ---

echo "Running Qwen3-Omni 30B-A3B decode"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Image: ${IMAGE_PATH}"

PYTHONPATH=/home/eporat/maxtext/src python3 src/MaxText/decode.py \
  src/MaxText/configs/base.yml \
  model_name=qwen3-omni-30b-a3b \
  load_parameters_path="${CHECKPOINT_PATH}" \
  tokenizer_path=src/MaxText/assets/qwen3-tokenizer \
  tokenizer_type=huggingface \
  image_path="${IMAGE_PATH}" \
  max_target_length=4096 \
  max_prefill_predict_length=2048 \
  per_device_batch_size=1.0 \
  scan_layers=false \
  use_multimodal=true \
  use_audio=false \
  megablox=true \
  sparse_matmul=true \
  ici_tensor_parallelism=4 \
  ici_fsdp_parallelism=1 \
  ici_data_parallelism=1 \
  ici_expert_parallelism=1 \
  ici_autoregressive_parallelism=1 \
  prompt="<|image|>What is in this image?"

echo "Decode complete."
