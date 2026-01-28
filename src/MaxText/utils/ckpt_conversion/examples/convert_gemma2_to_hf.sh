#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

export HF_AUTH_TOKEN=""

DATE=$(date +%Y-%m-%d)
# Define variables for paths and arguments
HF_CHECKPOINT_GCS_PATH="gs://maxtext-model-checkpoints/HuggingFace/gemma2-2b/${DATE}" # (optional)GCS path for HF model
MAXTEXT_CHECKPOINT_DIR="gs://maxtext-model-checkpoints/gemma2-2b-it/2025-02-20-18-01/unscanned/checkpoints/0/items"
LOCAL_HF_CHECKPOINT_DIR="/tmp/hf_gemma2-2b_output" # HF requires a local dir
TOKENIZER_PATH="${MAXTEXT_ASSETS_ROOT:-${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/assets/tokenizers}}/tokenizer.gemma"
MODEL_NAME="gemma2-2b"
PER_DEVICE_BATCH_SIZE=1
SCAN_LAYERS=false


# --- Step 1: Run the Hugging Face Conversion ---
echo "Starting Hugging Face model conversion for gemma2-2b..."

python3 -m MaxText.utils.ckpt_conversion.to_huggingface \
    "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}configs/base.yml" \
    model_name="${MODEL_NAME}" \
    tokenizer_path="${MAXTEXT_ASSETS_ROOT:-${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/assets/tokenizers}}/tokenizer.gemma" \
    load_parameters_path="${MAXTEXT_CHECKPOINT_DIR}" \
    per_device_batch_size=${PER_DEVICE_BATCH_SIZE} \
    max_prefill_predict_length=8 \
    max_target_length=16 \
    steps=1 \
    async_checkpointing=false \
    scan_layers=${SCAN_LAYERS} \
    prompt="'I love to'" \
    attention="'dot_product'" \
    base_output_directory="${HF_CHECKPOINT_GCS_PATH}"

echo "Hugging Face model conversion finished."

# --- Step 2: Run the Verification Script ---
echo "Starting verification for the converted gemma2-2b model..."

# Create local directory for checkpoints and download from GCS
echo "Creating local directory for HF checkpoints: ${LOCAL_HF_CHECKPOINT_DIR}"
mkdir -p "${LOCAL_HF_CHECKPOINT_DIR}"
echo "Downloading HF checkpoints from ${HF_CHECKPOINT_GCS_PATH} to ${LOCAL_HF_CHECKPOINT_DIR}..."
gsutil -m cp -r "${HF_CHECKPOINT_GCS_PATH}/*" "${LOCAL_HF_CHECKPOINT_DIR}/"
echo "Download complete."

python3 -m tests.utils.forward_pass_logit_checker \
    "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}configs/base.yml" \
    tokenizer_path=${TOKENIZER_PATH} \
    load_parameters_path="${MAXTEXT_CHECKPOINT_DIR}"\
    run_name=forward_pass_test_${MODEL_NAME}\
    per_device_batch_size=${PER_DEVICE_BATCH_SIZE} \
    model_name=${MODEL_NAME} \
    max_prefill_predict_length=4 \
    max_target_length=4 \
    dataset_type=synthetic \
    scan_layers=${SCAN_LAYERS} \
    attention=dot_product \
    --max_kl_div=0.015 \
    --run_hf_model=True \
    --hf_model_path=${LOCAL_HF_CHECKPOINT_DIR} \

# Optional: Clean up the local checkpoint directory
echo "Cleaning up local HF checkpoint directory: ${LOCAL_HF_CHECKPOINT_DIR}"
rm -rf "${LOCAL_HF_CHECKPOINT_DIR}"
echo "Cleanup complete."

echo "Verification script finished. Please check the above generated text"
echo "All steps completed."
