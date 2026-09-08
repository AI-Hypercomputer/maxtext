#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ==============================================================================
# End-to-End Omni Pipeline (Gemma 3 Vision + Qwen 3 LLM)
#
# Workflow:
#   1. Convert two HF checkpoints to MaxText format for checkpoint stitching (to_maxtext)
#   2. Stitch Vision Tower + Fresh MLP Projector + LLM Decoder (stitch_checkpoint)
#   3. Pretrain Projector for Multimodal Alignment on ChartNet (train_sft_omni)
#   4. Supervised Fine-Tuning Projector on ChartQA train split (train_sft_omni)
#   5. Multimodal Quality Evaluation on ChartQA test split (eval_sft_omni)
#
# Usage:
#   export BASE_OUTPUT_DIRECTORY="gs://YOUR_BUCKET/omni-gemma3-qwen3/multimodal"
#   export HF_TOKEN="<YOUR_HF_TOKEN>"
#   export TRAIN_STEPS=50           # Optional: overwrite steps for fast smoke test
#   export SCAN_LAYERS=true         # Optional: scan layers (default: true)
#   export EVAL_NUM_EXAMPLES=100    # Optional: -1 for full test split (default: 100)
#   export EVAL_SPLIT="test"        # Optional: evaluation split (default: test)
#   ./src/maxtext/experimental/omni_poc/maxtext_omni_pipeline_e2e.sh
# ==============================================================================

set -e

# ------------------------------------------------------------------------------
# 0. Environment & Configuration
# ------------------------------------------------------------------------------
export PYTHONPATH=src:${PYTHONPATH:-}

if [ -z "${BASE_OUTPUT_DIRECTORY}" ]; then
  echo "Error: BASE_OUTPUT_DIRECTORY is not set. Please set it as an environment variable."
  echo "Example: export BASE_OUTPUT_DIRECTORY=\"gs://YOUR_BUCKET/omni-gemma3-qwen3/multimodal\""
  exit 1
fi
BASE_OUTPUT_DIRECTORY="${BASE_OUTPUT_DIRECTORY%/}"

HF_TOKEN="${HF_TOKEN:-}"
if [ -n "$HF_TOKEN" ]; then
  export HF_TOKEN="$HF_TOKEN"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

# Model Names & Standard MaxText Variables
VISION_MODEL="gemma3-4b"
LLM_MODEL="qwen3-4b"

SCAN_LAYERS="${SCAN_LAYERS:-true}"

TRAIN_STEPS="${TRAIN_STEPS:-}"
PRETRAIN_STEPS="${PRETRAIN_STEPS:-${TRAIN_STEPS}}"
SFT_STEPS="${SFT_STEPS:-${TRAIN_STEPS}}"
EVAL_NUM_EXAMPLES="${EVAL_NUM_EXAMPLES:-100}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"

# Storage & Checkpoint Directories
CONVERTED_DIR="${BASE_OUTPUT_DIRECTORY}/converted"
VISION_ITEMS_PATH="${CONVERTED_DIR}/${VISION_MODEL}/0/items"
LLM_ITEMS_PATH="${CONVERTED_DIR}/${LLM_MODEL}/0/items"
STITCHED_ITEMS_PATH="${BASE_OUTPUT_DIRECTORY}/omni_checkpoints/omni_stitched_${VISION_MODEL}_${LLM_MODEL}/0/items"

PRETRAIN_DIR="${BASE_OUTPUT_DIRECTORY}/pretrain_chartnet"
PRETRAIN_RUN_NAME="pretrain_chartnet"

SFT_DIR="${BASE_OUTPUT_DIRECTORY}/sft_after_chartnet"
SFT_RUN_NAME="sft_chartqa"

EVAL_DIR="${BASE_OUTPUT_DIRECTORY}/sft_after_chartnet"
EVAL_RUN_NAME="eval"

# Helpers
path_exists() {
  python3 -c "
import sys
from etils import epath

sys.exit(0 if epath.Path(sys.argv[1]).exists() else 1)
" "$1"
}

get_latest_checkpoint() {
  python3 -c "
import sys
from etils import epath
p = epath.Path(sys.argv[1]) / 'checkpoints'
if p.exists():
  steps = [d.name for d in p.iterdir() if d.name.isdigit() and d.is_dir()]
  if steps:
    print(str(p / max(steps, key=int) / 'items'))
" "$1"
}

echo "=============================================================================="
echo "Starting Omni End-to-End Pipeline"
echo "  Base Output Directory:  ${BASE_OUTPUT_DIRECTORY}"
echo "  Vision Model:           ${VISION_MODEL}"
echo "  LLM Model:              ${LLM_MODEL}"
echo "  Stitched Target Path:   ${STITCHED_ITEMS_PATH}"
echo "  Scan Layers:            ${SCAN_LAYERS}"
[ -n "${PRETRAIN_STEPS}" ] && echo "  Pretrain Steps:         ${PRETRAIN_STEPS}"
[ -n "${SFT_STEPS}" ]      && echo "  SFT Steps:              ${SFT_STEPS}"
echo "  Eval Examples:          ${EVAL_NUM_EXAMPLES} | Split: ${EVAL_SPLIT}"
echo "=============================================================================="

# ------------------------------------------------------------------------------
# STEP 1: Convert Hugging Face Checkpoints -> MaxText Format
# ------------------------------------------------------------------------------
echo -e "\n=== [Step 1a/5] Converting Vision Model (${VISION_MODEL}) ==="
if ! path_exists "$VISION_ITEMS_PATH"; then
  JAX_PLATFORMS=cpu python3 -m maxtext.checkpoint_conversion.to_maxtext \
    src/maxtext/configs/base.yml \
    model_name=${VISION_MODEL} \
    base_output_directory=${CONVERTED_DIR}/${VISION_MODEL} \
    hf_access_token=${HF_TOKEN} \
    use_multimodal=true \
    scan_layers=${SCAN_LAYERS} \
    skip_jax_distributed_system=True \
    --eager_load_method=transformers \
    --lazy_load_tensors=False \
    log_config=False
else
  echo "--> Vision checkpoint already exists at ${VISION_ITEMS_PATH}. Skipping conversion."
fi

echo -e "\n=== [Step 1b/5] Converting LLM (${LLM_MODEL}) ==="
if ! path_exists "$LLM_ITEMS_PATH"; then
  JAX_PLATFORMS=cpu python3 -m maxtext.checkpoint_conversion.to_maxtext \
    src/maxtext/configs/base.yml \
    model_name=${LLM_MODEL} \
    base_output_directory=${CONVERTED_DIR}/${LLM_MODEL} \
    hf_access_token=${HF_TOKEN} \
    use_multimodal=false \
    scan_layers=${SCAN_LAYERS} \
    skip_jax_distributed_system=True \
    --eager_load_method=transformers \
    --lazy_load_tensors=False \
    log_config=False
else
  echo "--> LLM checkpoint already exists at ${LLM_ITEMS_PATH}. Skipping conversion."
fi

# ------------------------------------------------------------------------------
# STEP 2: Stitch Checkpoints (Vision + LLM + Fresh Projector)
# ------------------------------------------------------------------------------
echo -e "\n=== [Step 2/5] Stitching Omni Checkpoint ==="
if ! path_exists "$STITCHED_ITEMS_PATH"; then
  JAX_PLATFORMS=cpu python3 -m maxtext.experimental.omni_poc.utils.stitch_checkpoint \
    src/maxtext/experimental/omni_poc/maxtext-omni-gemma3-qwen3.yml \
    hf_access_token=${HF_TOKEN} \
    vision_load_path=${VISION_ITEMS_PATH} \
    llm_load_path=${LLM_ITEMS_PATH} \
    stitched_output_path=${STITCHED_ITEMS_PATH}
else
  echo "--> Stitched checkpoint already exists at ${STITCHED_ITEMS_PATH}. Skipping stitching."
fi

# ------------------------------------------------------------------------------
# STEP 3: Pretrain / Align Vision Projector (ChartNet)
# ------------------------------------------------------------------------------
echo -e "\n=== [Step 3/5] Pretraining Vision Projector (ChartNet) ==="
if [ -n "${PRETRAIN_STEPS}" ]; then
  python3 -m maxtext.experimental.omni_poc.train_sft_omni \
    src/maxtext/experimental/omni_poc/configs/pretrain-maxtext-omni-gemma3-qwen3-chartnet.yml \
    load_parameters_path=${STITCHED_ITEMS_PATH} \
    base_output_directory=${PRETRAIN_DIR} \
    run_name=${PRETRAIN_RUN_NAME} \
    hf_access_token=${HF_TOKEN} \
    steps=${PRETRAIN_STEPS} \
    checkpoint_period=${PRETRAIN_STEPS} \
    save_checkpoint_on_completion=true
else
  python3 -m maxtext.experimental.omni_poc.train_sft_omni \
    src/maxtext/experimental/omni_poc/configs/pretrain-maxtext-omni-gemma3-qwen3-chartnet.yml \
    load_parameters_path=${STITCHED_ITEMS_PATH} \
    base_output_directory=${PRETRAIN_DIR} \
    run_name=${PRETRAIN_RUN_NAME} \
    hf_access_token=${HF_TOKEN}
fi

PRETRAIN_FINAL_CKPT=$(get_latest_checkpoint "${PRETRAIN_DIR}/${PRETRAIN_RUN_NAME}")
if [ -z "$PRETRAIN_FINAL_CKPT" ]; then
  echo "Error: Pretrain checkpoint not found in ${PRETRAIN_DIR}/${PRETRAIN_RUN_NAME}/checkpoints."
  exit 1
fi
echo "--> Pretrained Checkpoint: ${PRETRAIN_FINAL_CKPT}"

# ------------------------------------------------------------------------------
# STEP 4: Supervised Fine-Tuning (ChartQA)
# ------------------------------------------------------------------------------
echo -e "\n=== [Step 4/5] Supervised Fine-Tuning (ChartQA) ==="
if [ -n "${SFT_STEPS}" ]; then
  python3 -m maxtext.experimental.omni_poc.train_sft_omni \
    src/maxtext/experimental/omni_poc/configs/sft-maxtext-omni-gemma3-qwen3.yml \
    load_parameters_path=${PRETRAIN_FINAL_CKPT} \
    base_output_directory=${SFT_DIR} \
    run_name=${SFT_RUN_NAME} \
    hf_access_token=${HF_TOKEN} \
    steps=${SFT_STEPS} \
    checkpoint_period=${SFT_STEPS} \
    save_checkpoint_on_completion=true
else
  python3 -m maxtext.experimental.omni_poc.train_sft_omni \
    src/maxtext/experimental/omni_poc/configs/sft-maxtext-omni-gemma3-qwen3.yml \
    load_parameters_path=${PRETRAIN_FINAL_CKPT} \
    base_output_directory=${SFT_DIR} \
    run_name=${SFT_RUN_NAME} \
    hf_access_token=${HF_TOKEN}
fi

SFT_FINAL_CKPT=$(get_latest_checkpoint "${SFT_DIR}/${SFT_RUN_NAME}")
if [ -z "$SFT_FINAL_CKPT" ]; then
  echo "Error: SFT checkpoint not found in ${SFT_DIR}/${SFT_RUN_NAME}/checkpoints."
  exit 1
fi
echo "--> SFT Final Checkpoint: ${SFT_FINAL_CKPT}"

# ------------------------------------------------------------------------------
# STEP 5: Multimodal Quality Evaluation (ChartQA Benchmark)
# ------------------------------------------------------------------------------
echo -e "\n=== [Step 5/5] Evaluating SFT Omni Model on ChartQA ==="
python3 -m maxtext.experimental.omni_poc.eval_sft_omni \
  src/maxtext/experimental/omni_poc/configs/sft-maxtext-omni-gemma3-qwen3.yml \
  load_parameters_path=${SFT_FINAL_CKPT} \
  base_output_directory=${EVAL_DIR} \
  run_name=${EVAL_RUN_NAME} \
  hf_access_token=${HF_TOKEN} \
  --ckpt_type=sft \
  --num_examples=${EVAL_NUM_EXAMPLES} \
  --hf_eval_split=${EVAL_SPLIT}

echo -e "\n=============================================================================="
echo "Omni End-to-End Pipeline Complete!"
echo "  Stitched Base:       ${STITCHED_ITEMS_PATH}"
echo "  Pretrain Checkpoint: ${PRETRAIN_FINAL_CKPT}"
echo "  SFT Final Checkpoint: ${SFT_FINAL_CKPT}"
echo "  Eval Results Dir:    ${EVAL_DIR}/${EVAL_RUN_NAME}"
echo "=============================================================================="
