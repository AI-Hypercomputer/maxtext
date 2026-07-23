#!/bin/bash
# Step 1 of 5 maxtext multimodal alignment proof of concept project.
#
# This file:
#   1. downloads and converts a vision-language model (e.g. gemma3-4b) from Hugging Face to MaxText format
#   2. downloads and converts an text-only model (e.g. qwen3-4b) from Hugging Face to MaxText format
#   3. stitches the vision component and LLM checkpoints into a single omni checkpoint
#   4. saves the stitched checkpoints to an output directory
#
# Example usage:
# HF_TOKEN="your_token" BASE_OUTPUT_DIRECTORY="gs://your_bucket/omni_checkpoints" ./prepare_checkpoint.sh

set -e

# Hugging Face Token & Login
HF_TOKEN="${HF_TOKEN:-}"

if [ -z "${BASE_OUTPUT_DIRECTORY}" ]; then
  echo "Error: BASE_OUTPUT_DIRECTORY is not set. Please set it as an environment variable."
  echo "Example: export BASE_OUTPUT_DIRECTORY=\"gs://your_bucket/omni_checkpoints\""
  exit 1
fi
BASE_OUTPUT_DIRECTORY="${BASE_OUTPUT_DIRECTORY%/}"

if [ -n "$HF_TOKEN" ]; then
  if command -v hf &> /dev/null; then
    echo "Logging into Hugging Face using hf..."
    hf auth login --token "$HF_TOKEN"
  elif command -v huggingface-cli &> /dev/null; then
    echo "Logging into Hugging Face using huggingface-cli..."
    huggingface-cli login --token "$HF_TOKEN"
  else
    echo "Neither hf nor huggingface-cli found. Skipping Hugging Face login."
  fi
fi

# Configuration & Paths
VISION_MAXTEXT_MODEL="gemma3-4b"
VISION_HF_REPO="google/gemma-3-4b-it"

LLM_MAXTEXT_MODEL="qwen3-4b"
LLM_HF_REPO="Qwen/Qwen3-4B"

# Automatically find maxtext package directory
MAXTEXT_PKG_DIR=$(python3 -c "import os, maxtext; print(os.path.dirname(maxtext.__file__))")
OMNI_CONFIG_PATH="${MAXTEXT_PKG_DIR}/experimental/omni_poc/omni-gemma3-qwen3.yml"

VISION_CKPT_DIR="${BASE_OUTPUT_DIRECTORY}/${VISION_MAXTEXT_MODEL}_converted"
LLM_CKPT_DIR="${BASE_OUTPUT_DIRECTORY}/${LLM_MAXTEXT_MODEL}_converted"
STITCHED_CKPT_DIR="${BASE_OUTPUT_DIRECTORY}/omni_stitched_${VISION_MAXTEXT_MODEL}_${LLM_MAXTEXT_MODEL}"

VISION_ITEMS_PATH="${VISION_CKPT_DIR}/0/items"
LLM_ITEMS_PATH="${LLM_CKPT_DIR}/0/items"
STITCHED_ITEMS_PATH="${STITCHED_CKPT_DIR}/0/items"

echo "Base Output Directory:  ${BASE_OUTPUT_DIRECTORY}"
echo "Vision Converted Path:  ${VISION_ITEMS_PATH}"
echo "LLM Converted Path:     ${LLM_ITEMS_PATH}"
echo "Stitched Target Path:   ${STITCHED_ITEMS_PATH}"
echo ""

export JAX_PLATFORMS=cpu

# Helper to check if local/GCS paths exist using python etils (same as python script)
path_exists() {
  python3 -c "from etils import epath; import sys; sys.exit(0 if epath.Path(sys.argv[1]).exists() else 1)" "$1"
}

# Step 1: Download & Convert Vision Model from Hugging Face -> MaxText
echo "============================================================"
if ! path_exists "$VISION_ITEMS_PATH"; then
  echo "Converting Vision Model (${VISION_MAXTEXT_MODEL}) from Hugging Face (${VISION_HF_REPO})..."
  python3 -m maxtext.checkpoint_conversion.to_maxtext \
    "${MAXTEXT_PKG_DIR}/configs/base.yml" \
    "model_name=${VISION_MAXTEXT_MODEL}" \
    "base_output_directory=${VISION_CKPT_DIR}" \
    "use_multimodal=True" \
    "scan_layers=True" \
    "skip_jax_distributed_system=True" \
    "--eager_load_method=transformers" \
    "--lazy_load_tensors=False" \
    "log_config=False"
  echo "Vision checkpoint conversion successful!"
  echo ""
else
  echo "Step 1: Vision checkpoint already exists at ${VISION_ITEMS_PATH}"
fi

# Step 2: Download & Convert Language Model from Hugging Face -> MaxText
echo "============================================================"
if ! path_exists "$LLM_ITEMS_PATH"; then
  echo "Converting Language Model (${LLM_MAXTEXT_MODEL}) from Hugging Face (${LLM_HF_REPO})..."
  python3 -m maxtext.checkpoint_conversion.to_maxtext \
    "${MAXTEXT_PKG_DIR}/configs/base.yml" \
    "model_name=${LLM_MAXTEXT_MODEL}" \
    "base_output_directory=${LLM_CKPT_DIR}" \
    "scan_layers=True" \
    "skip_jax_distributed_system=True" \
    "--eager_load_method=transformers" \
    "--lazy_load_tensors=False" \
    "log_config=False"
  echo "LLM checkpoint conversion successful!"
  echo ""
else
  echo "Step 2: LLM checkpoint already exists at ${LLM_ITEMS_PATH}"
fi

# Step 3: Checkpoint Stitching (Vision Tower + LLM Decoder + Fresh Projector)
echo "============================================================"
echo "Stitching Vision and LLM subtrees into unified Omni checkpoint..."
python3 -m maxtext.experimental.omni_poc.utils.stitch_checkpoint \
  "$OMNI_CONFIG_PATH" \
  "vision_load_path=${VISION_ITEMS_PATH}" \
  "llm_load_path=${LLM_ITEMS_PATH}" \
  "stitched_output_path=${STITCHED_ITEMS_PATH}"
