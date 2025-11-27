#!/bin/bash

# This script validates a pre-converted MaxText checkpoint against its original
# HuggingFace counterpart to ensure numerical correctness.

# ---
# Example Usage:
#
# # (Required) Path to the converted MaxText checkpoint
# export MAXTEXT_CHECKPOINT_PATH=gs://path/to/converted_ckpt/0/items/
#
# # (Optional) Override the default HF model
# export HF_MODEL_PATH=MyCustom/Qwen3-variant
#
# bash end_to_end/tpu/qwen/moe/qwen3-30b-a3b/1_test_qwen3_30b_a3b.sh
# ---

set -ex

# --- Configuration & Input Validation ---

if [ -z "${MAXTEXT_CHECKPOINT_PATH}" ]; then
    echo "ERROR: The MAXTEXT_CHECKPOINT_PATH environment variable is not set."
    echo "Please set it to the full GCS path of the pre-converted MaxText checkpoint weights."
    exit 1
fi

# Set a default for the HF model path if it's not provided by the user
if [ -z "${HF_MODEL_PATH}" ]; then
    export HF_MODEL_PATH="Qwen/Qwen3-30B-A3B-Thinking-2507"
    echo "HF_MODEL_PATH is not set, using default: ${HF_MODEL_PATH}"
fi

# Install dependencies required for the logit checker.
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# --- Run the Forward Pass Logit Checker ---

echo "Validating MaxText checkpoint at ${MAXTEXT_CHECKPOINT_PATH}"
echo "Against original HF model: ${HF_MODEL_PATH}"

# This command runs the core validation logic.
JAX_PLATFORMS=cpu python3 -m MaxText.tests.forward_pass_logit_checker "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}"/configs/base.yml \
  tokenizer_type=huggingface \
  tokenizer_path="${MAXTEXT_ASSETS_ROOT:-${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText/assets}}"/qwen3-tokenizer \
  megablox=False \
  sparse_matmul=False \
  load_parameters_path=${MAXTEXT_CHECKPOINT_PATH} \
  model_name=qwen3-30b-a3b \
  checkpoint_storage_concurrent_gb=512 \
  skip_jax_distributed_system=True \
  --hf_model_path=${HF_MODEL_PATH} \
  --max_kl_div=0.15 \
  --run_hf_model=True

echo "Validation complete."
