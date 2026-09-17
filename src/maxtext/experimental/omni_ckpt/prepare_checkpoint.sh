#!/bin/bash
# ==============================================================================
# Download, Convert, and Stitch Checkpoints for Omni Gemma3-4B + Qwen3-4B Variants
#
# Usage:
#   ./prepare_checkpoint.sh [LLM_HF_REPO]
#
# Examples:
#   # 1. Default base Qwen 3 4B:
#   ./prepare_checkpoint.sh Qwen/Qwen3-4B
#
#   # 2. Qwen 3 4B Thinking (Reasoning-focused variant):
#   ./prepare_checkpoint.sh Qwen/Qwen3-4B-Thinking-2507
#
#   # 3. Qwen 3 4B Instruct:
#   ./prepare_checkpoint.sh Qwen/Qwen3-4B-Instruct
#
#   # 4. Qwen 3 VL 4B (Instruct, scan_layers=false):
#   ./prepare_checkpoint.sh Qwen/Qwen3-VL-4B-Instruct
# ==============================================================================

set -e

# Temporary directories and HuggingFace cache in RAM disk
export TMPDIR=/dev/shm
export HF_HOME=/dev/shm/huggingface
export HF_DATASETS_CACHE=/dev/shm/huggingface/datasets
export TRANSFORMERS_CACHE=/dev/shm/huggingface/transformers
export HUGGINGFACE_HUB_CACHE=/dev/shm/huggingface/hub

# Hugging Face Token & Auth
: "${HF_TOKEN:?Error: HF_TOKEN is not set. Please run: export HF_TOKEN=hf_...}"
: "${GCS_BUCKET:?Error: GCS_BUCKET is not set. Please run: export GCS_BUCKET=gs://your-bucket}"
export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

# GCS Target Output Directory
BASE_OUTPUT_DIRECTORY="${BASE_OUTPUT_DIRECTORY:-${GCS_BUCKET}/omni_checkpoints_diff}"
BASE_OUTPUT_DIRECTORY="${BASE_OUTPUT_DIRECTORY%/}"

# Vision Model (Gemma 3 4B)
VISION_MAXTEXT_MODEL="gemma3-4b"
VISION_HF_REPO="google/gemma-3-4b-it"

# LLM Model & Hugging Face Repo (configurable via CLI argument or env var)
LLM_HF_REPO="${1:-${LLM_HF_REPO:-Qwen/Qwen3-4B}}"

# Derive an informative tag from the HF repo (e.g. qwen3-4b, qwen3-vl-4b-instruct)
LLM_TAG=$(echo "${LLM_HF_REPO}" | tr '[:upper:]' '[:lower:]' | sed 's|^.*/||' | tr '_' '-')

# Automatically configure conversion parameters and scan_layers based on whether the model is a VL model:
if [[ "${LLM_TAG}" == *"vl"* || "${LLM_HF_REPO}" == *"-VL-"* ]]; then
  if [[ "${LLM_TAG}" == *"2b"* ]]; then
    LLM_MAXTEXT_MODEL="qwen3-vl-2b"
  elif [[ "${LLM_TAG}" == *"30b"* ]]; then
    LLM_MAXTEXT_MODEL="qwen3-vl-30b-a3b"
  else
    LLM_MAXTEXT_MODEL="qwen3-vl-4b"
  fi
  LLM_USE_MULTIMODAL=true
  LLM_LOAD_METHOD="safetensors"
  SCAN_LAYERS="${SCAN_LAYERS:-false}"
else
  LLM_MAXTEXT_MODEL="${LLM_MAXTEXT_MODEL:-qwen3-4b}"
  LLM_USE_MULTIMODAL=false
  LLM_LOAD_METHOD="transformers"
  SCAN_LAYERS="${SCAN_LAYERS:-true}"
fi

SCAN_LAYERS=$(echo "${SCAN_LAYERS}" | tr '[:upper:]' '[:lower:]')
if [ "${SCAN_LAYERS}" = "false" ]; then
  SCAN_SUFFIX="_unscanned"
else
  SCAN_SUFFIX=""
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAXTEXT_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || (cd "${SCRIPT_DIR}/../../../../.." && pwd))"
OMNI_CONFIG_PATH="${SCRIPT_DIR}/maxtext-omni-gemma3-qwen3.yml"

VISION_CKPT_DIR="${BASE_OUTPUT_DIRECTORY}/${VISION_MAXTEXT_MODEL}${SCAN_SUFFIX}_converted"
LLM_CKPT_DIR="${BASE_OUTPUT_DIRECTORY}/${LLM_TAG}${SCAN_SUFFIX}_converted"
STITCHED_CKPT_DIR="${BASE_OUTPUT_DIRECTORY}/omni_stitched_${VISION_MAXTEXT_MODEL}_${LLM_TAG}${SCAN_SUFFIX}"

VISION_ITEMS_PATH="${VISION_CKPT_DIR}/0/items"
LLM_ITEMS_PATH="${LLM_CKPT_DIR}/0/items"
STITCHED_ITEMS_PATH="${STITCHED_CKPT_DIR}/0/items"

echo "============================================================"
echo ">>> [PREPARE CHECKPOINTS - omni_ckpt]"
echo ">>> Base Output Directory:  ${BASE_OUTPUT_DIRECTORY}"
echo ">>> Vision Source Model:    ${VISION_MAXTEXT_MODEL} (${VISION_HF_REPO})"
echo ">>> Vision Converted Path:  ${VISION_ITEMS_PATH}"
echo ">>> LLM Source Model:       ${LLM_MAXTEXT_MODEL} (${LLM_HF_REPO})"
echo ">>> LLM Variant Tag:        ${LLM_TAG}"
echo ">>> Scan Layers:            ${SCAN_LAYERS}"
echo ">>> LLM Converted Path:     ${LLM_ITEMS_PATH}"
echo ">>> Stitched Target Path:   ${STITCHED_ITEMS_PATH}"
echo "============================================================"

export JAX_PLATFORMS=cpu

path_exists() {
  local p="$1"
  python3 -c "from etils import epath; import sys; sys.exit(0 if epath.Path(sys.argv[1]).exists() else 1)" "$p" 2>/dev/null || \
  gcloud storage ls "$p" >/dev/null 2>&1 || \
  gsutil -q stat "$p" >/dev/null 2>&1
}

# 1. Convert Vision Encoder (Gemma 3 4B)
echo -e "\n=== [1/3] Converting Vision Encoder (${VISION_MAXTEXT_MODEL}, scan_layers=${SCAN_LAYERS}) ==="
if ! path_exists "${VISION_ITEMS_PATH}"; then
  (
    cd "${MAXTEXT_ROOT}"
    python3 -m maxtext.checkpoint_conversion.to_maxtext \
      src/maxtext/configs/base.yml \
      model_name="${VISION_MAXTEXT_MODEL}" \
      base_output_directory="${VISION_CKPT_DIR}" \
      hf_access_token="${HF_TOKEN}" \
      use_multimodal=true \
      scan_layers="${SCAN_LAYERS}" \
      skip_jax_distributed_system=True \
      --eager_load_method=transformers \
      --lazy_load_tensors=False \
      log_config=False
  )
else
  echo "--> Checkpoint already exists at ${VISION_ITEMS_PATH}. Skipping."
fi

# 2. Convert LLM Decoder (${LLM_HF_REPO})
echo -e "\n=== [2/3] Converting LLM Decoder (${LLM_HF_REPO} -> ${LLM_TAG}, scan_layers=${SCAN_LAYERS}) ==="
if ! path_exists "${LLM_ITEMS_PATH}"; then
  (
    cd "${MAXTEXT_ROOT}"
    python3 -m maxtext.checkpoint_conversion.to_maxtext \
      src/maxtext/configs/base.yml \
      model_name="${LLM_MAXTEXT_MODEL}" \
      base_output_directory="${LLM_CKPT_DIR}" \
      hf_access_token="${HF_TOKEN}" \
      --hf_model_path="${LLM_HF_REPO}" \
      use_multimodal="${LLM_USE_MULTIMODAL}" \
      scan_layers="${SCAN_LAYERS}" \
      skip_jax_distributed_system=True \
      --eager_load_method="${LLM_LOAD_METHOD}" \
      --lazy_load_tensors=False \
      log_config=False
  )
else
  echo "--> Checkpoint already exists at ${LLM_ITEMS_PATH}. Skipping."
fi

# 3. Stitch Subtrees into Unified Omni Checkpoint
echo -e "\n=== [3/3] Stitching Omni Checkpoint (${VISION_MAXTEXT_MODEL} + ${LLM_TAG}, scan_layers=${SCAN_LAYERS}) ==="
if ! path_exists "${STITCHED_ITEMS_PATH}"; then
  (
    cd "${MAXTEXT_ROOT}"
    python3 -m maxtext.experimental.omni_pipeline.utils.stitch_checkpoint \
      "${OMNI_CONFIG_PATH}" \
      "hf_access_token=${HF_TOKEN}" \
      "tokenizer_path=${LLM_HF_REPO}" \
      "scan_layers=${SCAN_LAYERS}" \
      "vision_load_path=${VISION_ITEMS_PATH}" \
      "llm_load_path=${LLM_ITEMS_PATH}" \
      "stitched_output_path=${STITCHED_ITEMS_PATH}"
  )
else
  echo "--> Stitched checkpoint already exists at ${STITCHED_ITEMS_PATH}. Skipping."
fi

echo -e "\n============================================================"
echo "Checkpoint preparation complete!"
echo "LLM Converted Checkpoint: ${LLM_ITEMS_PATH}"
echo "Stitched Omni Checkpoint: ${STITCHED_ITEMS_PATH}"
echo "============================================================"
