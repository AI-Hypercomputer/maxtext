#!/bin/bash
# ==============================================================================
# Download, Convert, and Stitch Checkpoints for Omni Gemma3-4B + Qwen3-14B
# ==============================================================================

set -e

# Temporary directories and HuggingFace cache in RAM disk
export TMPDIR=/dev/shm
export HF_HOME=/dev/shm/huggingface
export HF_DATASETS_CACHE=/dev/shm/huggingface/datasets
export TRANSFORMERS_CACHE=/dev/shm/huggingface/transformers
export HUGGINGFACE_HUB_CACHE=/dev/shm/huggingface/hub

: "${HF_TOKEN:?Error: HF_TOKEN is not set. Please run: export HF_TOKEN=hf_...}"
: "${GCS_BUCKET:?Error: GCS_BUCKET is not set. Please run: export GCS_BUCKET=gs://your-bucket}"
export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"


BASE_OUTPUT_DIRECTORY="${BASE_OUTPUT_DIRECTORY:-${GCS_BUCKET}/omni-gemma3-qwen3-14b/checkpoints}"
BASE_OUTPUT_DIRECTORY="${BASE_OUTPUT_DIRECTORY%/}"

# Models & Hugging Face Repos
VISION_MAXTEXT_MODEL="gemma3-4b"
VISION_HF_REPO="google/gemma-3-4b-it"

LLM_MAXTEXT_MODEL="qwen3-14b"
LLM_HF_REPO="Qwen/Qwen3-14B"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAXTEXT_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || (cd "${SCRIPT_DIR}/../../../../.." && pwd))"
OMNI_CONFIG_PATH="${SCRIPT_DIR}/maxtext-omni-gemma3-qwen3-14b.yml"

VISION_CKPT_DIR="${BASE_OUTPUT_DIRECTORY}/${VISION_MAXTEXT_MODEL}_converted"
LLM_CKPT_DIR="${BASE_OUTPUT_DIRECTORY}/${LLM_MAXTEXT_MODEL}_converted"
STITCHED_CKPT_DIR="${BASE_OUTPUT_DIRECTORY}/omni_stitched_${VISION_MAXTEXT_MODEL}_${LLM_MAXTEXT_MODEL}"

VISION_ITEMS_PATH="${VISION_CKPT_DIR}/0/items"
LLM_ITEMS_PATH="${LLM_CKPT_DIR}/0/items"
STITCHED_ITEMS_PATH="${STITCHED_CKPT_DIR}/0/items"

echo "============================================================"
echo ">>> [PREPARE CHECKPOINTS 14B]"
echo ">>> Base Output Directory:  ${BASE_OUTPUT_DIRECTORY}"
echo ">>> Vision Source Model:    ${VISION_MAXTEXT_MODEL} (${VISION_HF_REPO})"
echo ">>> Vision Converted Path:  ${VISION_ITEMS_PATH}"
echo ">>> LLM Source Model:       ${LLM_MAXTEXT_MODEL} (${LLM_HF_REPO})"
echo ">>> LLM Converted Path:     ${LLM_ITEMS_PATH}"
echo ">>> Stitched Target Path:   ${STITCHED_ITEMS_PATH}"
echo "============================================================"

export JAX_PLATFORMS=cpu

path_exists() {
  python3 -c "from etils import epath; import sys; sys.exit(0 if epath.Path(sys.argv[1]).exists() else 1)" "$1"
}

# Step 1: Download & Convert Vision Model (Gemma3-4B) from Hugging Face -> MaxText (scan_layers=True)
echo ""
echo "============================================================"
echo ">>> Step 1/3: Converting Vision Model (${VISION_MAXTEXT_MODEL}) with scan_layers=True..."
echo "============================================================"
(
  cd "${MAXTEXT_ROOT}"
  python3 -m maxtext.checkpoint_conversion.to_maxtext \
    src/maxtext/configs/base.yml \
    "model_name=${VISION_MAXTEXT_MODEL}" \
    "base_output_directory=${VISION_CKPT_DIR}" \
    "hf_access_token=${HF_TOKEN}" \
    "use_multimodal=True" \
    "scan_layers=True" \
    "skip_jax_distributed_system=True" \
    "--eager_load_method=transformers" \
    "--lazy_load_tensors=False" \
    "log_config=False"
)
echo ">>> Vision checkpoint conversion successful!"

# Step 2: Download & Convert Language Model (Qwen3-14B) from Hugging Face -> MaxText (scan_layers=True)
echo ""
echo "============================================================"
echo ">>> Step 2/3: Converting Language Model (${LLM_MAXTEXT_MODEL}) with scan_layers=True..."
echo "============================================================"
(
  cd "${MAXTEXT_ROOT}"
  python3 -m maxtext.checkpoint_conversion.to_maxtext \
    src/maxtext/configs/base.yml \
    "model_name=${LLM_MAXTEXT_MODEL}" \
    "base_output_directory=${LLM_CKPT_DIR}" \
    "hf_access_token=${HF_TOKEN}" \
    "scan_layers=True" \
    "skip_jax_distributed_system=True" \
    "--eager_load_method=transformers" \
    "--lazy_load_tensors=False" \
    "log_config=False"
)
echo ">>> LLM checkpoint conversion successful!"

# Step 3: Stitching (Vision Tower + LLM Decoder + Fresh 3-Layer Projector)
echo ""
echo "============================================================"
echo ">>> Step 3/3: Stitching Vision and LLM into unified Omni 14B checkpoint..."
echo "============================================================"
(
  cd "${MAXTEXT_ROOT}"
  python3 -m maxtext.experimental.omni_poc.utils.stitch_checkpoint \
    "${OMNI_CONFIG_PATH}" \
    "hf_access_token=${HF_TOKEN}" \
    "vision_load_path=${VISION_ITEMS_PATH}" \
    "llm_load_path=${LLM_ITEMS_PATH}" \
    "stitched_output_path=${STITCHED_ITEMS_PATH}"
)

echo ""
echo "============================================================"
echo ">>> Checkpoint preparation COMPLETE!"
echo ">>> Unified Stitched Checkpoint: ${STITCHED_ITEMS_PATH}"
echo "============================================================"
