#!/bin/bash

# Example script to convert a MaxText Qwen3 MoE checkpoint back to HuggingFace format.
set -e

export HF_AUTH_TOKEN=""
DATE=$(date +%Y-%m-%d)

# Location of the MaxText checkpoint to convert.
MAXTEXT_CHECKPOINT_DIR="/path/to/maxtext/ckpt"

# Directory where the HuggingFace files will be written.
LOCAL_HF_CHECKPOINT_DIR="/tmp/hf_qwen3_moe_output"

CONVERT_MODULE="MaxText.ckpt_conversion.to_huggingface"
CONVERT_ARGS=(
    "MaxText/configs/base.yml"
    "model_name=qwen3-moe"
    "tokenizer_path=${MAXTEXT_CHECKPOINT_DIR}/hf-checkpoint"
    "load_parameters_path=${MAXTEXT_CHECKPOINT_DIR}"
    "per_device_batch_size=1"
    "steps=1"
    "async_checkpointing=false"
    "scan_layers=false"
    "prompt='Hello'"
    "attention='dot_product'"
    "base_output_directory=${LOCAL_HF_CHECKPOINT_DIR}"
)

# Run the conversion
python3 -m ${CONVERT_MODULE} ${CONVERT_ARGS[@]}

echo "Qwen3 MoE checkpoint saved to ${LOCAL_HF_CHECKPOINT_DIR}"
