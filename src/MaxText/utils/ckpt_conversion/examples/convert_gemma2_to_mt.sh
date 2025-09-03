#!/bin/bash
# set -e

export HF_AUTH_TOKEN=""

DATE=$(date +%Y-%m-%d)
# config.base_output_directory
OUTPUT_BASE_DIR=""
# Model name must be consistent in utils/utils.py
MODEL_NAME="gemma2-2b"
# HF model id as golden model for verification
HF_MODEL_ID="google/gemma-2-2b-it"
# Tokenizer path for decoding
TOKENIZER_PATH="${MAXTEXT_ASSETS_ROOT:-${MAXTEXT_REPO_ROOT:-$PWD}/assets}/tokenizer.gemma"

PER_DEVICE_BATCH_SIZE=1
ASYNC_CHECKPOINTING=false
SCAN_LAYERS=false
PROMPT="I love to"

# --- Step 1: Convert Checkpoint to MaxText Format ---
echo "--- Starting Checkpoint Conversion ---"
python3 -m "MaxText.utils.ckpt_conversion.to_src/MaxText" \
  "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}configs/base.yml" \
  model_name="${MODEL_NAME}" \
  base_output_directory="${OUTPUT_BASE_DIR}" \
  per_device_batch_size="${PER_DEVICE_BATCH_SIZE}" \
  run_name="run_to_mt" \
  async_checkpointing="${ASYNC_CHECKPOINTING}" \
  scan_layers="${SCAN_LAYERS}" 

echo "--- Checkpoint Conversion Complete ---"

# --- Step 2 (Optional): Decode using the Converted Checkpoint ---

echo "--- Starting Decoding ---"
python3 -m "MaxText.decode" \
  "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}configs/base.yml" \
  model_name="${MODEL_NAME}" \
  tokenizer_path="${TOKENIZER_PATH}" \
  load_parameters_path="${OUTPUT_BASE_DIR}/0/items" \
  per_device_batch_size="${PER_DEVICE_BATCH_SIZE}" \
  run_name="run_decode" \
  max_prefill_predict_length=8 \
  max_target_length=16 \
  dataset_type=synthetic \
  steps=1 \
  async_checkpointing="${ASYNC_CHECKPOINTING}" \
  scan_layers="${SCAN_LAYERS}" \
  prompt="${PROMPT}"

echo "--- Decoding Complete ---"

# --- Step 3: Compare the HF and MT Checkpoint ---

echo "--- Starting Comparing Logits and Predicted Tokens ---"
python3 -m "tests.forward_pass_logit_checker" \
    "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}/configs/base.yml" \
    tokenizer_path=${TOKENIZER_PATH} \
    load_parameters_path="${OUTPUT_BASE_DIR}/0/items"\
    run_name=forward_pass_test_${MODEL_NAME}\
    per_device_batch_size=${PER_DEVICE_BATCH_SIZE} \
    model_name=${MODEL_NAME} \
    max_prefill_predict_length=4 \
    max_target_length=4 \
    dataset_type=synthetic \
    scan_layers=${scan_layers} \
    attention=dot_product \
    --max_kl_div=0.015 \
    --run_hf_model=True \
    --hf_model_path=${HF_MODEL_ID} \

echo "--- Decoding Complete ---"