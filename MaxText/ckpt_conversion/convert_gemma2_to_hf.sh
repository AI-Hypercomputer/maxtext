#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define variables for paths and arguments
MAXTEXT_PROJECT_DIR="/home/yixuannwang_google_com/projects/maxtext"

CONVERT_MODULE="MaxText.ckpt_conversion.to_huggingface"
CONVERT_ARGS=(
    "MaxText/configs/base.yml"
    "model_name=gemma2-2b"
    "tokenizer_path=assets/tokenizer.gemma"
    "load_parameters_path=gs://maxtext-model-checkpoints/gemma2-2b-it/2025-02-20-18-01/unscanned/checkpoints/0/items"
    "per_device_batch_size=1"
    "max_prefill_predict_length=8"
    "max_target_length=16"
    "steps=1"
    "async_checkpointing=false"
    "scan_layers=false"
    "prompt='I love to'"
    "attention='dot_product'"
    "base_output_directory=gs://yixuannwang-maxtext-logs/hf_gemma2-2b_output"
)

VERIFY_MODULE="MaxText.ckpt_conversion.test_hf"

VERIFY_ARGS=(
    "hf_checkpoint_path=gs://yixuannwang-maxtext-logs/hf_gemma2-2b_output"
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

# Construct the command
VERIFY_CMD=("python3" -m "$VERIFY_MODULE")
if [ ${#VERIFY_ARGS[@]} -ne 0 ]; then
    for arg in "${VERIFY_ARGS[@]}"; do
        VERIFY_CMD+=("$arg")
    done
fi

# Execute the command
"${VERIFY_CMD[@]}"

echo "Verification script finished. Please check the above generated text"

echo "All steps completed."
