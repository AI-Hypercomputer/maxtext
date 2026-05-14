#!/bin/bash
# Phase 0: Checkpoint Conversion
# Converts the HuggingFace Qwen2.5-1.5B-Instruct model to MaxText/Orbax format.

# Ensure we are using the venv
if [ -z "$VIRTUAL_ENV" ]; then
  echo "Error: Virtual environment not detected. Please run: source maxtext_venv/bin/activate"
  exit 1
fi

# Monitoring disk space
echo "Initial disk space:"
df -h .

export MODEL="qwen2.5-1.5b"
export BASE_OUTPUT_DIRECTORY="gs://igorts_europe/converted_checkpoints/qwen2.5-1.5b-instruct_mcjax_v2"

echo "Starting conversion for $MODEL..."

python3 -m maxtext.checkpoint_conversion.to_maxtext \
    model_name=${MODEL?} \
    base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    scan_layers=False \
    use_multimodal=false \
    hardware=cpu \
    skip_jax_distributed_system=true \
    checkpoint_storage_use_zarr3=true \
    checkpoint_storage_use_ocdbt=true \
    --lazy_load_tensors=true \
    --save_dtype=bfloat16

echo "Conversion complete."
echo "Final disk space:"
df -h .
