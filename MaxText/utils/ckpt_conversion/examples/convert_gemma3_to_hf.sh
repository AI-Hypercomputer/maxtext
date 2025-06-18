#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

export HF_AUTH_TOKEN=""

DATE=$(date +%Y-%m-%d)
# Define variables for paths and arguments
HF_CHECKPOINT_GCS_PATH="gs://maxtext-model-checkpoints/HuggingFace/gemma3-4b/${DATE}" # (optional)GCS path for HF model
MAXTEXT_CHECKPOINT_DIR="gs://maxtext-model-checkpoints/gemma3-4b/2025-03-18-19-03/unscanned/checkpoints/0/items"
LOCAL_HF_CHECKPOINT_DIR="/tmp/hf_gemma3-4b_output" # HF requires a local dir
GOLDEN_MODEL_ID="google/gemma-3-4b-it"

CONVERT_MODULE="MaxText.utils.ckpt_conversion.to_huggingface"
CONVERT_ARGS=(
    "MaxText/configs/base.yml",
    "model_name=gemma3-4b",
    "tokenizer_path=assets/tokenizer.gemma3",
    "load_parameters_path=${MAXTEXT_CHECKPOINT_DIR}",
    "per_device_batch_size=1",
    "run_name=ht_test",
    "max_prefill_predict_length=8",
    "max_target_length=16",
    "steps=1",
    "async_checkpointing=false",
    "prompt='I love to'",
    "scan_layers=false",
    "attention='dot_product'",
    "base_output_directory=${HF_CHECKPOINT_GCS_PATH}"
)

VERIFY_MODULE="MaxText.tests.hf_ckpt_conversion_check"

VERIFY_ARGS=(
    "--golden_model_id=${GOLDEN_MODEL_ID}"
    "--hf_checkpoint_path=${LOCAL_HF_CHECKPOINT_DIR}" # Updated to local path
)


# --- Step 1: Run the Hugging Face Conversion ---
echo "Starting Hugging Face model conversion for gemma2-2b..."
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
echo "Starting verification for the converted gemma2-2b model..."

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

# Optional: Clean up the local checkpoint directory
echo "Cleaning up local HF checkpoint directory: ${LOCAL_HF_CHECKPOINT_DIR}"
rm -rf "${LOCAL_HF_CHECKPOINT_DIR}"
echo "Cleanup complete."

echo "Verification script finished. Please check the above generated text"
echo "All steps completed."
