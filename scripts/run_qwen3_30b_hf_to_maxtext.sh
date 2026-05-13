#!/bin/bash

# This script converts a Qwen3-30B-A3B model checkpoint from the Hugging Face
# format to the MaxText format. It requires the BASE_OUTPUT_DIRECTORY environment
# variable to be set to a GCS path where the converted checkpoint will be stored.

set -e

# --- Environment Setup ---
if ! pip show maxtext &> /dev/null; then
    echo "maxtext not found in the environment. Please install it by running:"
    echo "uv pip install -e .[tpu] --resolution=lowest"
    exit 1
fi

# --- Environment Variables ---
export BASE_OUTPUT_DIRECTORY="${BASE_OUTPUT_DIRECTORY:-}" # GCS bucket path for outputs (e.g., gs://my-bucket/outputs)

# --- Variable Validation ---
if [ -z "$BASE_OUTPUT_DIRECTORY" ]; then
    echo "Error: BASE_OUTPUT_DIRECTORY is not set. Please set it in the script or as an environment variable."
    exit 1
fi

# Install torch for conversion
echo "Installing torch for checkpoint conversion..."
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# Define the output path for the converted checkpoint
CONVERTED_CKPT_BASE_DIR="${BASE_OUTPUT_DIRECTORY}/checkpoints/qwen3-30b-a3b-converted"
echo "Converted checkpoint will be saved to: ${CONVERTED_CKPT_BASE_DIR}"

# Run the conversion script
python3 -m maxtext.checkpoint_conversion.to_maxtext \
    src/maxtext/configs/base.yml \
    model_name=qwen3-30b-a3b \
    base_output_directory="${CONVERTED_CKPT_BASE_DIR}" \
    scan_layers=True \
    weight_dtype=bfloat16 hardware=cpu skip_jax_distributed_system=True \
    checkpoint_storage_use_ocdbt=False checkpoint_storage_use_zarr3=False \
    --eager_load_method=safetensors

# Set MAXTEXT_CKPT_PATH to the newly created checkpoint path
export MAXTEXT_CKPT_PATH="${CONVERTED_CKPT_BASE_DIR}/0/items"
echo "Conversion complete. Using checkpoint at: ${MAXTEXT_CKPT_PATH}"
