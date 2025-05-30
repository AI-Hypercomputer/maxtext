#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
# set -e

# remeber to save your token here
export HF_AUTH_TOKEN=""

# Define variables for paths and arguments
MAXTEXT_PROJECT_DIR="/home/yixuannwang_google_com/projects/maxtext"
HF_CHECKPOINT_GCS_PATH="gs://yixuannwang-maxtext-logs/hf_llama31-8b_output" # (optional)GCS path for HF model
MAXTEST_CHECKPOINT_DIR="gs://maxtext-model-checkpoints/llama3.1-8b/2025-01-23-19-04/unscanned/checkpoints/0/items"
LOCAL_HF_CHECKPOINT_DIR="/tmp/hf_llama31-8b_output" # Local directory for HF model
GOLDEN_MODEL_ID="meta-llama/Llama-3.1-8B"

# This convert args are same as load llama3 in maxengine
CONVERT_MODULE="MaxText.ckpt_conversion.to_huggingface"
CONVERT_ARGS=(
    "MaxText/configs/base.yml"
    "model_name=llama3.1-8b"
    "tokenizer_path=assets/tokenizer_llama3.tiktoken"
    "tokenizer_type=tiktoken"
    "load_parameters_path=${MAXTEST_CHECKPOINT_DIR}"
    "per_device_batch_size=1"
    "max_prefill_predict_length=8"
    "max_target_length=16"
    "steps=1"
    "async_checkpointing=false"
    "scan_layers=false"
    "prompt='I love to'"
    "attention='dot_product'"
    "base_output_directory=${HF_CHECKPOINT_GCS_PATH}"
)

VERIFY_MODULE="MaxText.ckpt_conversion.test_hf"


VERIFY_ARGS=(
    "golden_model_id=${GOLDEN_MODEL_ID}"
    "hf_checkpoint_path=${HF_CHECKPOINT_GCS_PATH}"
)


# --- Step 1: Run the Hugging Face Conversion ---
# echo "Starting Hugging Face model conversion for llama3-8b..."
cd "$MAXTEXT_PROJECT_DIR"

# Construct the command
CONVERT_CMD=("python3" -m "$CONVERT_MODULE")
for arg in "${CONVERT_ARGS[@]}"; do
    CONVERT_CMD+=("$arg")
done

# Execute the command
"${CONVERT_CMD[@]}"

echo "Hugging Face model conversion finished."


# --- Step 2: Run the Verification Script ---
echo "Starting verification for the converted llama3-8b model..."

# Create local directory for checkpoints and download from GCS
echo "Creating local directory for HF checkpoints: ${LOCAL_HF_CHECKPOINT_DIR}"
mkdir -p "${LOCAL_HF_CHECKPOINT_DIR}"
echo "Downloading HF checkpoints from ${HF_CHECKPOINT_GCS_PATH} to ${LOCAL_HF_CHECKPOINT_DIR}..."
gsutil -m cp -r "${HF_CHECKPOINT_GCS_PATH}/*" "${LOCAL_HF_CHECKPOINT_DIR}/"
echo "Download complete."

# Construct the command
VERIFY_CMD=("python3" -m "$VERIFY_MODULE")
if [ ${#VERIFY_ARGS[@]} -ne 0 ]; then
    for arg in "${VERIFY_ARGS[@]}"; do
        VERIFY_CMD+=("$arg")
    done
fi

# Execute the command
"${VERIFY_CMD[@]}"

# Clean up the local checkpoint directory
echo "Cleaning up local HF checkpoint directory: ${LOCAL_HF_CHECKPOINT_DIR}"
rm -rf "${LOCAL_HF_CHECKPOINT_DIR}"
echo "Cleanup complete."

echo "Verification script finished. Please check the above generated text"
echo "All steps completed."
