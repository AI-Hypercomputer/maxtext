#!/bin/bash
# Phase 2.5: Prepare Inference Checkpoint
# This script uses the standard generate_param_only_checkpoint.py to convert
# the DPO training checkpoint into a clean inference checkpoint.

set -e

# Configuration
export MODEL_NAME="qwen2.5-1.5b"
export DPO_FULL_STATE="gs://igorts_europe/ttl=30d/dpo_quals/maxtext_run/dpo-verification-qwen-v3/checkpoints/100/model_params"
export INFERENCE_CKPT_DIR="gs://igorts_europe/ttl=30d/dpo_quals/maxtext_run/dpo-verification-qwen-v3/inference_ckpt"

# Ensure venv is active
if [ -z "$VIRTUAL_ENV" ]; then
  source ~/maxtext_venv/bin/activate
fi

export PYTHONPATH=src

echo "=== Generating Parameter-Only Checkpoint (NNX) ==="
python3 quals/generate_param_only_checkpoint_nnx.py \
    run_name="prepare-inference-nnx" \
    base_output_directory=${INFERENCE_CKPT_DIR} \
    model_name=${MODEL_NAME} \
    load_parameters_path=${DPO_FULL_STATE} \
    scan_layers=False \
    attention=dot_product \
    log_config=0

echo "=== Inference Checkpoint Generated at ${INFERENCE_CKPT_DIR}/0/items ==="
