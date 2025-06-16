#!/bin/bash

export HF_AUTH_TOKEN=""

DATE=$(date +%Y-%m-%d)
# config.base_output_directory
OUTPUT_BASE_DIR=""
# Model name must be consistent in utils/utils.py
MODEL_NAME="gemma2-2b"
# Tokenizer path
TOKENIZER_PATH="assets/tokenizer.gemma"

PER_DEVICE_BATCH_SIZE=1
ASYNC_CHECKPOINTING=false
SCAN_LAYERS=false
PROMPT="I love to"

# --- Step 1: Convert Checkpoint to MaxText Format ---
echo "--- Starting Checkpoint Conversion ---"
python3 -m "MaxText.utils.ckpt_conversion.to_maxtext" \
  "MaxText/configs/base.yml" \
  model_name="${MODEL_NAME}" \
  base_output_directory="${OUTPUT_BASE_DIR}" \
  per_device_batch_size="${PER_DEVICE_BATCH_SIZE}" \
  run_name="mt_gemma2" \
  async_checkpointing="${ASYNC_CHECKPOINTING}" \
  scan_layers="${SCAN_LAYERS}" \
  prompt="${PROMPT}" \
  attention='dot_product'

echo "--- Checkpoint Conversion Complete ---"

# --- Step 2: Decode using the Converted Checkpoint ---

echo "--- Starting Decoding ---"
python3 -m "MaxText.decode" \
  "MaxText/configs/base.yml" \
  model_name="${MODEL_NAME}" \
  tokenizer_path="${TOKENIZER_PATH}" \
  load_parameters_path="${OUTPUT_BASE_DIR}/0/items" \
  per_device_batch_size="${PER_DEVICE_BATCH_SIZE}" \
  run_name="mt_gemma2_check" \
  max_prefill_predict_length=8 \
  max_target_length=16 \
  dataset_type=synthetic \
  steps=1 \
  async_checkpointing="${ASYNC_CHECKPOINTING}" \
  scan_layers="${SCAN_LAYERS}" \
  prompt="${PROMPT}"

echo "--- Decoding Complete ---"