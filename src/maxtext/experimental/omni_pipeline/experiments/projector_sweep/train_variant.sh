#!/bin/bash
# ==============================================================================
# XPK Single-Variant Execution Script (Pretrain / SFT / Pipeline)
# ==============================================================================
set -euo pipefail

STAGE="${1:-all}"                   # "pretrain", "sft", or "pipeline" / "all"
TAG="${2:-l2_h4096_silu}"           # architecture tag
LAYERS="${3:-2}"                    # num layers
HIDDEN="${4:-4096}"                 # hidden dimension
ACT="${5:-silu}"                    # activation

: "${HF_TOKEN:?Error: HF_TOKEN is not set. Please run: export HF_TOKEN=hf_...}"
: "${GCS_BUCKET:?Error: GCS_BUCKET is not set. Please run: export GCS_BUCKET=gs://your-bucket}"
export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
export TMPDIR=/dev/shm
export PYTHONPATH=src:${PYTHONPATH:-}

INIT_CKPT="${GCS_BUCKET}/omni_checkpoints/omni_stitched_gemma3-4b_qwen3-4b_${TAG}/0/items"
PRETRAIN_DIR="${GCS_BUCKET}/omni-gemma3-qwen3/multimodal/pretrain_chartnet_xpk/${TAG}"
PRETRAIN_RUN="pretrain_${TAG}"
PRETRAIN_CKPT_BASE="${PRETRAIN_DIR}/${PRETRAIN_RUN}/checkpoints"

SFT_DIR="${GCS_BUCKET}/omni-gemma3-qwen3/multimodal/sft_chartqa_xpk/${TAG}"
SFT_RUN="sft_${TAG}"

# Helper: dynamically find the latest numerical checkpoint step (e.g. 2171/items)
get_latest_ckpt() {
  local base_path="$1"
  local default_step="$2"
  python3 -c "
from etils import epath
import sys
try:
    base = epath.Path(sys.argv[1])
    if base.exists():
        ckpts = [p for p in base.iterdir() if p.name.isdigit()]
        if ckpts:
            latest = max(ckpts, key=lambda p: int(p.name))
            print(str(latest / 'items'))
            sys.exit(0)
except Exception:
    pass
print(f'{sys.argv[1]}/{sys.argv[2]}/items')
" "${base_path}" "${default_step}" 2>/dev/null || echo "${base_path}/${default_step}/items"
}

echo "=================================================================="
echo "Starting Execution: Stage=${STAGE}, Variant=${TAG} (L=${LAYERS}, H=${HIDDEN}, Act=${ACT})"
echo "=================================================================="

# ------------------------------------------------------------------------------
# Stage 1: ChartNet Pretraining
# ------------------------------------------------------------------------------
if [ "$STAGE" == "pretrain" ] || [ "$STAGE" == "pipeline" ] || [ "$STAGE" == "all" ]; then
  echo ""
  echo "=== [Stage 1] Pretraining on ChartNet (2,172 steps) ==="
  echo "Input Checkpoint:    ${INIT_CKPT}"
  echo "Output Directory:    ${PRETRAIN_DIR}"
  echo ""

  python3 -m maxtext.experimental.omni_pipeline.train_sft_omni \
    src/maxtext/experimental/omni_pipeline/pretrain-omni-gemma3-qwen3-chartnet-xpk-128.yml \
    vision_connector_num_layers=${LAYERS} \
    vision_connector_hidden_size=${HIDDEN} \
    vision_connector_activation=${ACT} \
    load_parameters_path=${INIT_CKPT} \
    base_output_directory=${PRETRAIN_DIR} \
    run_name=${PRETRAIN_RUN}

  echo "ChartNet Pretraining Complete for ${TAG}!"
fi

# ------------------------------------------------------------------------------
# Stage 2: ChartQA SFT
# ------------------------------------------------------------------------------
if [ "$STAGE" == "sft" ] || [ "$STAGE" == "pipeline" ] || [ "$STAGE" == "all" ]; then
  INPUT_SFT_CKPT=$(get_latest_ckpt "${PRETRAIN_CKPT_BASE}" "2171")
  echo ""
  echo "=== [Stage 2] Fine-tuning on ChartQA (1,105 steps) ==="
  echo "Input Checkpoint:    ${INPUT_SFT_CKPT}"
  echo "Output Directory:    ${SFT_DIR}"
  echo ""

  python3 -m maxtext.experimental.omni_pipeline.train_sft_omni \
    src/maxtext/experimental/omni_pipeline/sft-omni-gemma3-qwen3-xpk-128.yml \
    vision_connector_num_layers=${LAYERS} \
    vision_connector_hidden_size=${HIDDEN} \
    vision_connector_activation=${ACT} \
    load_parameters_path=${INPUT_SFT_CKPT} \
    base_output_directory=${SFT_DIR} \
    run_name=${SFT_RUN}

  echo "ChartQA SFT Complete for ${TAG}!"
fi

echo "=================================================================="
echo "All Done for Variant: ${TAG}"
echo "=================================================================="
