#!/bin/bash

# This script converts a qwen3-30b-a3b-base model checkpoint from the MaxText
# format (orbax) to the Hugging Face format. It requires the MAXTEXT_CKPT_PATH and
# BASE_OUTPUT_DIRECTORY environment variables to be set.

set -e

# --- Environment Setup ---
if ! pip show maxtext &> /dev/null; then
    echo "maxtext not found in the environment. Please install it by running:"
    echo "uv pip install -e .[tpu] --resolution=lowest"
    exit 1
fi

# --- Environment Variables ---
export MAXTEXT_CKPT_PATH="${MAXTEXT_CKPT_PATH:-}" # GCS path to the MaxText checkpoint to convert
export BASE_OUTPUT_DIRECTORY="${BASE_OUTPUT_DIRECTORY:-}" # GCS bucket path for storing the converted HF checkpoint

# --- Variable Validation ---
if [ -z "$BASE_OUTPUT_DIRECTORY" ]; then
    echo "Error: BASE_OUTPUT_DIRECTORY is not set. Please set it in the script or as an environment variable."
    exit 1
fi

if [ -z "$MAXTEXT_CKPT_PATH" ]; then
    echo "Error: MAXTEXT_CKPT_PATH is not set. Please set it in the script or as an environment variable."
    exit 1
fi

# Install torch for conversion
echo "Installing torch for checkpoint conversion..."
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# Define the output path for the converted checkpoint
HF_CKPT_OUTPUT_DIR="${BASE_OUTPUT_DIRECTORY}/checkpoints/qwen3-30b-a3b-base-hf-converted"
echo "Converted Hugging Face checkpoint will be saved to: ${HF_CKPT_OUTPUT_DIR}"

# Run the conversion script
python -m maxtext.checkpoint_conversion.to_huggingface \
    src/maxtext/configs/base.yml \
    model_name=qwen3-30b-a3b-base \
    load_parameters_path="${MAXTEXT_CKPT_PATH}" \
    base_output_directory="${HF_CKPT_OUTPUT_DIR}" \
    scan_layers=True \
    weight_dtype=bfloat16 hardware=cpu skip_jax_distributed_system=True

echo "Conversion to Hugging Face format complete. Checkpoint saved to: ${HF_CKPT_OUTPUT_DIR}"
