#!/bin/bash
# ==============================================================================
# Omni Gemma3-4B Vision + Qwen3-4B LLM Pipeline with Normalized Projector (omni_norm)
#
# Architecture:
#   - Vision Tower: Gemma 3 4B ViT (896x896 -> 256 tokens)
#   - Projector: RMSNorm (custom_linear_norm, 1152-dim) + 3-Layer GELU MLP (1152 -> 4096 -> 2560)
#   - Text Decoder: Qwen 3 4B LLM (36 layers, 2560 hidden dim)
#
# Stages:
#   Stage 1: ChartNet Core CSV Table Grounding (1.7M examples, 2 Epochs = 26,533 steps)
#   Stage 2: ChartQA Visual QA Full-Decoder SFT (5 Epochs = 1,105 steps, lr=1.5e-5)
# ==============================================================================

set -e

ACTION="${1:-help}"
ARG2="${2:-}"

# Temporary directories and memory cache in RAM disk
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
USER_PREFIX="${USER_PREFIX:-user}"

# GCS Target Output Directory Root
BASE_CKPT_DIR="${BASE_CKPT_DIR:-${GCS_BUCKET}/omni_norm_checkpoints}"
BASE_CKPT_DIR="${BASE_CKPT_DIR%/}"

# Vision Model Fixed: Gemma 3 4B
VISION_MAXTEXT_MODEL="gemma3-4b"
VISION_HF_REPO="google/gemma-3-4b-it"

# Parse Target LLM Checkpoint / Tag
LLM_MAXTEXT_MODEL="qwen3-4b"
LLM_HF_REPO="${LLM_HF_REPO:-Qwen/Qwen3-4B}"
LLM_TAG="${LLM_TAG:-}"

if [[ -n "${ARG2}" && ( "${ACTION}" == "stitch" || "${ACTION}" == "prepare" || "${ACTION}" == pipeline* || "${ACTION}" == stage1* || "${ACTION}" == stage2* ) ]]; then
  if [[ "${ARG2}" == gs://* || "${ARG2}" == */checkpoints/* || "${ARG2}" == "${USER_PREFIX}-"* ]]; then
    if [[ "${ARG2}" =~ omni_([a-zA-Z0-9_-]+)/stage || "${ARG2}" =~ omni_([a-zA-Z0-9_-]+) || "${ARG2}" =~ (qwen3-[a-zA-Z0-9_-]+) ]]; then
      detected_tag="${BASH_REMATCH[1]}"
      detected_tag="${detected_tag%_unscanned}"
      detected_tag="${detected_tag#norm_}"
      LLM_TAG="${LLM_TAG:-${detected_tag}}"
      case "${LLM_TAG}" in
        qwen3-4b-thinking-2507|qwen3-4b-thinking) LLM_HF_REPO="Qwen/Qwen3-4B-Thinking-2507" ;;
        qwen3-4b-instruct-2507|qwen3-4b-instruct) LLM_HF_REPO="Qwen/Qwen3-4B-Instruct-2507" ;;
        qwen3-vl-4b-instruct|qwen3-vl-4b)          LLM_HF_REPO="Qwen/Qwen3-VL-4B-Instruct" ;;
        qwen3-vl-2b-instruct|qwen3-vl-2b)          LLM_HF_REPO="Qwen/Qwen3-VL-2B-Instruct" ;;
        qwen3-4b-base)                            LLM_HF_REPO="Qwen/Qwen3-4B-Base" ;;
        qwen3-4b)                                 LLM_HF_REPO="Qwen/Qwen3-4B" ;;
      esac
    fi
  elif [[ "${ARG2}" == *"/"* ]]; then
    LLM_HF_REPO="${ARG2}"
    LLM_TAG=$(echo "${LLM_HF_REPO}" | tr '[:upper:]' '[:lower:]' | sed 's|^.*/||' | tr '_' '-')
  elif [[ "${ARG2}" == qwen3* || "${ARG2}" == Qwen* ]]; then
    LLM_TAG=$(echo "${ARG2}" | tr '[:upper:]' '[:lower:]' | sed 's|^.*/||' | tr '_' '-')
    case "${LLM_TAG}" in
      qwen3-4b-thinking-2507|qwen3-4b-thinking) LLM_HF_REPO="Qwen/Qwen3-4B-Thinking-2507" ;;
      qwen3-4b-instruct-2507|qwen3-4b-instruct) LLM_HF_REPO="Qwen/Qwen3-4B-Instruct-2507" ;;
      qwen3-vl-4b-instruct|qwen3-vl-4b)          LLM_HF_REPO="Qwen/Qwen3-VL-4B-Instruct" ;;
      qwen3-vl-2b-instruct|qwen3-vl-2b)          LLM_HF_REPO="Qwen/Qwen3-VL-2B-Instruct" ;;
      qwen3-4b-base)                            LLM_HF_REPO="Qwen/Qwen3-4B-Base" ;;
      qwen3-4b)                                 LLM_HF_REPO="Qwen/Qwen3-4B" ;;
      *)                                        LLM_HF_REPO="Qwen/${ARG2}" ;;
    esac
  fi
elif [[ ( "${ACTION}" == "eval" || "${ACTION}" == "eval_xpk" || "${ACTION}" == "eval-xpk" ) && -n "${ARG2}" ]]; then
  if [[ "${ARG2}" =~ omni_([a-zA-Z0-9_-]+)/stage || "${ARG2}" =~ omni_([a-zA-Z0-9_-]+) || "${ARG2}" =~ (qwen3-[a-zA-Z0-9_-]+) ]]; then
    detected_tag="${BASH_REMATCH[1]}"
    detected_tag="${detected_tag%_unscanned}"
    detected_tag="${detected_tag#norm_}"
    LLM_TAG="${LLM_TAG:-${detected_tag}}"
    case "${LLM_TAG}" in
      qwen3-4b-thinking-2507|qwen3-4b-thinking) LLM_HF_REPO="Qwen/Qwen3-4B-Thinking-2507" ;;
      qwen3-4b-instruct-2507|qwen3-4b-instruct) LLM_HF_REPO="Qwen/Qwen3-4B-Instruct-2507" ;;
      qwen3-vl-4b-instruct|qwen3-vl-4b)          LLM_HF_REPO="Qwen/Qwen3-VL-4B-Instruct" ;;
      qwen3-vl-2b-instruct|qwen3-vl-2b)          LLM_HF_REPO="Qwen/Qwen3-VL-2B-Instruct" ;;
      qwen3-4b-base)                            LLM_HF_REPO="Qwen/Qwen3-4B-Base" ;;
      qwen3-4b)                                 LLM_HF_REPO="Qwen/Qwen3-4B" ;;
    esac
  fi
fi

if [[ -z "${LLM_TAG}" ]]; then
  LLM_TAG=$(echo "${LLM_HF_REPO}" | tr '[:upper:]' '[:lower:]' | sed 's|^.*/||' | tr '_' '-')
fi
[ -z "${LLM_TAG}" ] && LLM_TAG="qwen3-4b"

# Automatically configure conversion parameters and scan_layers
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
  LLM_MAXTEXT_MODEL="qwen3-4b"
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

TAG_SHORT=$(echo "${LLM_TAG}" | sed 's/qwen3-4b-//; s/qwen3-//; s/-instruct//' | tr -cd '[:alnum:]-')
[ -z "${TAG_SHORT}" ] && TAG_SHORT="4b"

# Directories & Checkpoints
EXP_NAME="${EXP_NAME:-omni_norm_${LLM_TAG}${SCAN_SUFFIX}}"
WORKING_DIR="${WORKING_DIR:-${GCS_BUCKET}/omni_norm/${EXP_NAME}}"

VISION_CKPT_DIR="${BASE_CKPT_DIR}/${VISION_MAXTEXT_MODEL}${SCAN_SUFFIX}_converted"
LLM_CKPT_DIR="${BASE_CKPT_DIR}/${LLM_TAG}${SCAN_SUFFIX}_converted"
STITCHED_CKPT_DIR="${BASE_CKPT_DIR}/omni_stitched_norm_${VISION_MAXTEXT_MODEL}_${LLM_TAG}${SCAN_SUFFIX}"

VISION_SOURCE_CKPT="${VISION_SOURCE_CKPT:-${VISION_CKPT_DIR}/0/items}"
LLM_SOURCE_CKPT="${LLM_SOURCE_CKPT:-${LLM_CKPT_DIR}/0/items}"
STITCHED_CKPT="${STITCHED_CKPT:-${STITCHED_CKPT_DIR}/0/items}"

# Datasets
CHARTNET_DATASET_DIR="${CHARTNET_DATASET_DIR:-${GCS_BUCKET}/datasets/chartnet_core_sharded}"
CHARTQA_DATASET_DIR="${CHARTQA_DATASET_DIR:-${GCS_BUCKET}/datasets/chartqa_shuffled}"

# Locate Configs from Current Directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAXTEXT_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"
OMNI_CONFIG_PATH="${OMNI_CONFIG_PATH:-src/maxtext/experimental/omni_norm/maxtext-omni-gemma3-qwen3.yml}"

# Stage 1: ChartNet Core CSV Table Grounding (2 Epochs = 26,533 steps on 128 devices)
STAGE1_CONFIG="${STAGE1_CONFIG:-src/maxtext/experimental/omni_norm/pretrain-omni-gemma3-qwen3-chartnet-core-csv-xpk-128.yml}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-2}"
STAGE1_STEPS="${STAGE1_STEPS:-26533}"
STAGE1_CKPT_PERIOD="${STAGE1_CKPT_PERIOD:-1000}"
STAGE1_OUTPUT_DIR="${WORKING_DIR}/stage1_chartnet_csv"
STAGE1_RUN_NAME="${EXP_NAME}_stage1_csv"
STAGE1_FINAL_CKPT="${STAGE1_OUTPUT_DIR}/${STAGE1_RUN_NAME}/checkpoints/$((STAGE1_STEPS - 1))/items"

# Stage 1 (Alternative 6-Epoch): ChartNet CSV Table Grounding (6 Epochs = 4,343 steps = 2,172*2 on 128 devices)
CHARTNET_6EP_DATASET_DIR="${CHARTNET_6EP_DATASET_DIR:-${GCS_BUCKET}/datasets/chartnet_sharded}"
STAGE1_6EP_CONFIG="${STAGE1_6EP_CONFIG:-src/maxtext/experimental/omni_norm/pretrain-omni-gemma3-qwen3-chartnet-xpk-128-csv-6ep.yml}"
STAGE1_6EP_EPOCHS="${STAGE1_6EP_EPOCHS:-6}"
STAGE1_6EP_STEPS="${STAGE1_6EP_STEPS:-4343}"
STAGE1_6EP_CKPT_PERIOD="${STAGE1_6EP_CKPT_PERIOD:-500}"
STAGE1_6EP_OUTPUT_DIR="${WORKING_DIR}/stage1_csv_6ep"
STAGE1_6EP_RUN_NAME="${EXP_NAME}_stage1_csv_6ep"
STAGE1_6EP_FINAL_CKPT="${STAGE1_6EP_OUTPUT_DIR}/${STAGE1_6EP_RUN_NAME}/checkpoints/$((STAGE1_6EP_STEPS - 1))/items"

# Stage 2: ChartQA Visual QA Full-Decoder SFT (5 Epochs = 1,105 steps on 128 devices, lr=1.5e-5)
STAGE2_CONFIG="${STAGE2_CONFIG:-src/maxtext/experimental/omni_norm/sft-omni-gemma3-qwen3-full-decoder-xpk-128.yml}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-5}"
STAGE2_STEPS="${STAGE2_STEPS:-1105}"
STAGE2_OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-${WORKING_DIR}/stage2_full_decoder}"
STAGE2_RUN_NAME="${STAGE2_RUN_NAME:-${EXP_NAME}_stage2_full_decoder}"
STAGE2_FINAL_CKPT="${STAGE2_OUTPUT_DIR}/${STAGE2_RUN_NAME}/checkpoints/$((STAGE2_STEPS - 1))/items"

# Stage 2 (Full-Decoder SFT after 6-Epoch): ChartQA Visual QA Full-Decoder SFT (5 Epochs = 1,105 steps, lr=1.5e-5)
STAGE2_6EP_OUTPUT_DIR="${STAGE2_6EP_OUTPUT_DIR:-${WORKING_DIR}/stage2_full_decoder_after_6eps}"
STAGE2_6EP_RUN_NAME="${STAGE2_6EP_RUN_NAME:-${EXP_NAME}_stage2_full_decoder_after_6eps}"
STAGE2_6EP_FINAL_CKPT="${STAGE2_6EP_OUTPUT_DIR}/${STAGE2_6EP_RUN_NAME}/checkpoints/$((STAGE2_STEPS - 1))/items"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${STAGE2_OUTPUT_DIR}/eval}"

# Stage 2 (Alternative Projector-Only MLP): ChartQA Visual QA SFT (5 Epochs = 1,105 steps on 128 devices, lr=3.e-4)
STAGE2_MLP_CONFIG="${STAGE2_MLP_CONFIG:-src/maxtext/experimental/omni_norm/sft-omni-gemma3-qwen3-mlp-xpk-128.yml}"
STAGE2_MLP_EPOCHS="${STAGE2_MLP_EPOCHS:-5}"
STAGE2_MLP_STEPS="${STAGE2_MLP_STEPS:-1105}"
STAGE2_MLP_OUTPUT_DIR="${STAGE2_MLP_OUTPUT_DIR:-${WORKING_DIR}/stage2_mlp}"
STAGE2_MLP_RUN_NAME="${STAGE2_MLP_RUN_NAME:-${EXP_NAME}_stage2_mlp}"
STAGE2_MLP_FINAL_CKPT="${STAGE2_MLP_OUTPUT_DIR}/${STAGE2_MLP_RUN_NAME}/checkpoints/$((STAGE2_MLP_STEPS - 1))/items"

# Stage 2 (Alternative Projector-Only MLP after 6-Epoch): ChartQA Visual QA SFT (5 Epochs = 1,105 steps on 128 devices, lr=3.e-4)
STAGE2_MLP_6EP_OUTPUT_DIR="${STAGE2_MLP_6EP_OUTPUT_DIR:-${WORKING_DIR}/stage2_mlp_after_6eps}"
STAGE2_MLP_6EP_RUN_NAME="${STAGE2_MLP_6EP_RUN_NAME:-${EXP_NAME}_stage2_mlp_after_6eps}"
STAGE2_MLP_6EP_FINAL_CKPT="${STAGE2_MLP_6EP_OUTPUT_DIR}/${STAGE2_MLP_6EP_RUN_NAME}/checkpoints/$((STAGE2_MLP_STEPS - 1))/items"

# Cluster Config
XPK_CLUSTER="${XPK_CLUSTER:-v4-128-bodaborg-us-central2-b}"
XPK_PROJECT="${XPK_PROJECT:-cloud-tpu-multipod-dev}"
XPK_ZONE="${XPK_ZONE:-us-central2-b}"
XPK_DEVICE_TYPE="${XPK_DEVICE_TYPE:-v4-128}"
XPK_NUM_SLICES="${XPK_NUM_SLICES:-1}"
XPK_BASE_DOCKER_IMAGE="${XPK_BASE_DOCKER_IMAGE:-gcr.io/tpu-prod-env-multipod/maxtext_base_image:latest}"

USER_PREFIX="${USER_PREFIX:-user}"
TIMESTAMP="$(date +%m%d-%H%M)"

path_exists() {
  local p="$1"
  [ "${SKIP_PATH_EXISTS:-false}" = "true" ] && return 0
  [ "${FORCE:-false}" = "true" ] && return 0
  if [[ "$p" == gs://* ]]; then
    gcloud storage ls "$p" >/dev/null 2>&1 || \
    gcloud storage ls "${p%/}/" >/dev/null 2>&1 || \
    gsutil -q stat "$p" >/dev/null 2>&1 || \
    python3 -c "import sys; from etils import epath; sys.exit(0 if epath.Path(sys.argv[1]).exists() else 1)" "$p" >/dev/null 2>&1
  else
    [ -e "$p" ]
  fi
}

# ------------------------------------------------------------------------------
# 1. Download, Convert, and Stitch Checkpoint with Normalized Projector
# ------------------------------------------------------------------------------
stitch_ckpt() {
  echo "=================================================================="
  echo ">>> [STITCH & PREPARE CHECKPOINT (omni_norm)]"
  echo ">>> Vision Model:      ${VISION_MAXTEXT_MODEL} (${VISION_HF_REPO})"
  echo ">>> LLM Model:         ${LLM_MAXTEXT_MODEL} (${LLM_HF_REPO})"
  echo ">>> LLM Tag:           ${LLM_TAG}"
  echo ">>> Scan Layers:       ${SCAN_LAYERS}"
  echo ">>> Vision Checkpoint: ${VISION_SOURCE_CKPT}"
  echo ">>> LLM Checkpoint:    ${LLM_SOURCE_CKPT}"
  echo ">>> Stitched Output:   ${STITCHED_CKPT}"
  echo "=================================================================="

  export JAX_PLATFORMS=cpu

  # Fallback to pre-existing converted vision / LLM checkpoints if available
  if ! path_exists "${VISION_SOURCE_CKPT}" && path_exists "${GCS_BUCKET}/omni_checkpoints/${VISION_MAXTEXT_MODEL}${SCAN_SUFFIX}_converted/0/items"; then
    VISION_SOURCE_CKPT="${GCS_BUCKET}/omni_checkpoints/${VISION_MAXTEXT_MODEL}${SCAN_SUFFIX}_converted/0/items"
    echo "--> Reusing pre-converted vision checkpoint from: ${VISION_SOURCE_CKPT}"
  fi
  if ! path_exists "${LLM_SOURCE_CKPT}" && path_exists "${GCS_BUCKET}/omni_checkpoints/${LLM_TAG}${SCAN_SUFFIX}_converted/0/items"; then
    LLM_SOURCE_CKPT="${GCS_BUCKET}/omni_checkpoints/${LLM_TAG}${SCAN_SUFFIX}_converted/0/items"
    echo "--> Reusing pre-converted LLM checkpoint from: ${LLM_SOURCE_CKPT}"
  fi

  # Step 1: Convert Vision Encoder
  echo -e "\n=== [1/3] Converting Vision Encoder (${VISION_MAXTEXT_MODEL}, scan_layers=${SCAN_LAYERS}) ==="
  if ! path_exists "${VISION_SOURCE_CKPT}"; then
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
    echo "--> Vision checkpoint already exists at ${VISION_SOURCE_CKPT}. Skipping."
  fi

  # Step 2: Convert LLM Decoder
  echo -e "\n=== [2/3] Converting LLM Decoder (${LLM_HF_REPO} -> ${LLM_TAG}, scan_layers=${SCAN_LAYERS}) ==="
  if ! path_exists "${LLM_SOURCE_CKPT}"; then
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
    echo "--> LLM checkpoint already exists at ${LLM_SOURCE_CKPT}. Skipping."
  fi

  # Step 3: Stitch into Unified Multimodal Checkpoint with Normalized Projector
  echo -e "\n=== [3/3] Stitching Subtrees into Unified Omni Checkpoint with Normalized Projector ==="
  if ! path_exists "${STITCHED_CKPT}" || [ "${FORCE:-false}" = "true" ] || [ "${FORCE:-false}" = "1" ]; then
    if path_exists "${STITCHED_CKPT}"; then
      echo "--> Overwriting existing stitched checkpoint at ${STITCHED_CKPT} (FORCE=true)..."
      if [[ "${STITCHED_CKPT_DIR}" == gs://* ]]; then
        gcloud storage rm -r "${STITCHED_CKPT_DIR}" >/dev/null 2>&1 || true
      else
        rm -rf "${STITCHED_CKPT_DIR}"
      fi
    fi
    (
      cd "${MAXTEXT_ROOT}"
      python3 -m maxtext.experimental.omni_pipeline.utils.stitch_checkpoint \
        "${OMNI_CONFIG_PATH}" \
        "hf_access_token=${HF_TOKEN}" \
        "tokenizer_path=${LLM_HF_REPO}" \
        "scan_layers=${SCAN_LAYERS}" \
        "vision_load_path=${VISION_SOURCE_CKPT}" \
        "llm_load_path=${LLM_SOURCE_CKPT}" \
        "stitched_output_path=${STITCHED_CKPT}"
    )
  else
    echo "--> Stitched checkpoint already exists at ${STITCHED_CKPT}."
    echo "    To re-stitch and overwrite, run: '$0 restitch' or 'FORCE=1 $0 stitch'"
  fi

  echo -e "\n=================================================================="
  echo ">>> Checkpoint preparation and stitching complete!"
  echo ">>> Stitched Path: ${STITCHED_CKPT}"
  echo "=================================================================="
}

# ------------------------------------------------------------------------------
# 2. Stage 1: ChartNet Core CSV Table Grounding on XPK
# ------------------------------------------------------------------------------
stage1_xpk() {
  local input_ckpt="${1:-${STITCHED_CKPT}}"

  if path_exists "${input_ckpt}" && [ "${FORCE:-false}" != "true" ] && [ "${FORCE:-false}" != "1" ]; then
    echo ">>> Stitched checkpoint already exists at ${input_ckpt}. Skipping stitching."
  else
    echo ">>> Warning: Stitched checkpoint not found at ${input_ckpt}. Running stitch first..."
    stitch_ckpt
  fi

  local workload_name
  if [ "${LLM_TAG}" = "qwen3-4b" ]; then
    workload_name="${USER_PREFIX}-norm-s1-core-${TIMESTAMP}"
  else
    workload_name="${USER_PREFIX}-${TAG_SHORT}-norm-s1-${TIMESTAMP}"
  fi
  workload_name="${workload_name:0:39}"
  workload_name="${workload_name%-}"

  echo "=================================================================="
  echo ">>> [XPK STAGE 1 - ChartNet Core CSV Grounding (Normalized Projector)]"
  echo ">>> Submitting Workload: ${workload_name}"
  echo ">>> LLM Tag:          ${LLM_TAG}"
  echo ">>> Config:           ${STAGE1_CONFIG}"
  echo ">>> Input Checkpoint: ${input_ckpt}"
  echo ">>> Working Dir:      ${STAGE1_OUTPUT_DIR}/${STAGE1_RUN_NAME}"
  echo ">>> Schedule:         ${STAGE1_EPOCHS} Epochs = ${STAGE1_STEPS} Steps"
  echo "=================================================================="

  (
    cd "${MAXTEXT_ROOT}"
    xpk workload create \
      --cluster "${XPK_CLUSTER}" \
      --project "${XPK_PROJECT}" \
      --zone "${XPK_ZONE}" \
      --workload "${workload_name}" \
      --tpu-type "${XPK_DEVICE_TYPE}" \
      --num-slices "${XPK_NUM_SLICES}" \
      --base-docker-image "${XPK_BASE_DOCKER_IMAGE}" \
      --script-dir . \
      --command "export MEGASCALE_NUM_SLICES=${XPK_NUM_SLICES} && export TMPDIR=/dev/shm && export PYTHONPATH=src:\${PYTHONPATH:-} && export HF_HOME=/dev/shm/huggingface && export HF_TOKEN=${HF_TOKEN} && export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} && python3 -m pip install --no-cache-dir -U 'orbax-checkpoint>=0.12.4' && gcloud storage cp -r ${CHARTNET_DATASET_DIR} /dev/shm/ && python3 -m maxtext.experimental.omni_pipeline.train_sft_omni ${STAGE1_CONFIG} load_parameters_path=${input_ckpt} base_output_directory=${STAGE1_OUTPUT_DIR} run_name=${STAGE1_RUN_NAME} num_epoch=${STAGE1_EPOCHS} steps=${STAGE1_STEPS} eval_interval=-1 eval_steps=0 checkpoint_period=${STAGE1_CKPT_PERIOD} tokenizer_path=${LLM_HF_REPO} hf_access_token=${HF_TOKEN} scan_layers=${SCAN_LAYERS} grain_worker_count=0"
  )
}

# ------------------------------------------------------------------------------
# 3. Stage 2: ChartQA Visual QA Full-Decoder SFT on XPK
# ------------------------------------------------------------------------------
stage2_xpk() {
  local input_ckpt="${1:-}"
  local output_dir="${2:-}"
  local run_name="${3:-}"

  if [ -n "${input_ckpt}" ]; then
    input_ckpt="${input_ckpt%/_CHECKPOINT_METADATA}"
    input_ckpt="${input_ckpt/#https:\/\/storage.googleapis.com\//gs:\/\/}"
    [[ "${input_ckpt}" != gs://* && "${input_ckpt}" == "${GCS_BUCKET#gs://}"* ]] && input_ckpt="gs://${input_ckpt}"
    if [[ "${input_ckpt}" != */items ]]; then
      input_ckpt="${input_ckpt%/}/items"
    fi
  fi

  if [ -z "${input_ckpt}" ]; then
    if path_exists "${STAGE1_FINAL_CKPT}"; then
      input_ckpt="${STAGE1_FINAL_CKPT}"
    elif path_exists "${STAGE1_6EP_FINAL_CKPT}"; then
      input_ckpt="${STAGE1_6EP_FINAL_CKPT}"
    else
      input_ckpt="${STAGE1_FINAL_CKPT}"
    fi
  fi

  if ! path_exists "${input_ckpt}" && ! path_exists "${input_ckpt%/items}"; then
    echo ">>> Error: Input checkpoint not found at: ${input_ckpt}"
    echo ">>> Please run Stage 1 first, or provide an explicit checkpoint path: $0 stage2 <ckpt_path>"
    exit 1
  fi

  if [ -z "${output_dir}" ]; then
    if [[ "${input_ckpt}" == *"stage1_csv_6ep"* || "${input_ckpt}" == *"6ep"* ]]; then
      output_dir="${STAGE2_6EP_OUTPUT_DIR}"
      run_name="${run_name:-${STAGE2_6EP_RUN_NAME}}"
    else
      output_dir="${STAGE2_OUTPUT_DIR}"
      run_name="${run_name:-${STAGE2_RUN_NAME}}"
    fi
  else
    [ -z "${run_name}" ] && run_name="$(basename "${output_dir}")"
  fi

  local workload_name
  if [[ "${output_dir}" == *"6ep"* || "${input_ckpt}" == *"6ep"* ]]; then
    if [ "${LLM_TAG}" = "qwen3-4b" ]; then
      workload_name="${USER_PREFIX}-norm-s2-fdec-6ep-${TIMESTAMP}"
    else
      workload_name="${USER_PREFIX}-${TAG_SHORT}-norm-s2-6ep-${TIMESTAMP}"
    fi
  else
    if [ "${LLM_TAG}" = "qwen3-4b" ]; then
      workload_name="${USER_PREFIX}-norm-s2-fdec-${TIMESTAMP}"
    else
      workload_name="${USER_PREFIX}-${TAG_SHORT}-norm-s2-${TIMESTAMP}"
    fi
  fi
  workload_name="${workload_name:0:39}"
  workload_name="${workload_name%-}"

  echo "=================================================================="
  echo ">>> [XPK STAGE 2 - ChartQA Full-Decoder SFT (Normalized Projector)]"
  echo ">>> Submitting Workload: ${workload_name}"
  echo ">>> LLM Tag:          ${LLM_TAG}"
  echo ">>> Tokenizer Path:   ${LLM_HF_REPO}"
  echo ">>> Config:           ${STAGE2_CONFIG}"
  echo ">>> Input Checkpoint: ${input_ckpt}"
  echo ">>> Working Dir:      ${output_dir}/${run_name}"
  echo ">>> Schedule:         ${STAGE2_EPOCHS} Epochs = ${STAGE2_STEPS} Steps"
  echo "=================================================================="

  (
    cd "${MAXTEXT_ROOT}"
    xpk workload create \
      --cluster "${XPK_CLUSTER}" \
      --project "${XPK_PROJECT}" \
      --zone "${XPK_ZONE}" \
      --workload "${workload_name}" \
      --tpu-type "${XPK_DEVICE_TYPE}" \
      --num-slices "${XPK_NUM_SLICES}" \
      --base-docker-image "${XPK_BASE_DOCKER_IMAGE}" \
      --script-dir . \
      --command "export MEGASCALE_NUM_SLICES=${XPK_NUM_SLICES} && export TMPDIR=/dev/shm && export PYTHONPATH=src:\${PYTHONPATH:-} && export HF_HOME=/dev/shm/huggingface && export HF_TOKEN=${HF_TOKEN} && export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} && python3 -m pip install --no-cache-dir -U 'orbax-checkpoint>=0.12.4' && gcloud storage cp -r ${CHARTQA_DATASET_DIR} /dev/shm/ && python3 -m maxtext.experimental.omni_pipeline.train_sft_omni ${STAGE2_CONFIG} load_parameters_path=${input_ckpt} base_output_directory=${output_dir} run_name=${run_name} steps=${STAGE2_STEPS} tokenizer_path=${LLM_HF_REPO} hf_access_token=${HF_TOKEN} scan_layers=${SCAN_LAYERS} grain_worker_count=0"
  )
}

# ------------------------------------------------------------------------------
# Stage 2 (Full-Decoder SFT after 6-Epoch ChartNet Grounding) on XPK
# ------------------------------------------------------------------------------
stage2_after_6eps_xpk() {
  local input_ckpt="${1:-${STAGE1_6EP_FINAL_CKPT}}"
  local output_dir="${2:-${STAGE2_6EP_OUTPUT_DIR}}"
  local run_name="${3:-${STAGE2_6EP_RUN_NAME}}"
  stage2_xpk "${input_ckpt}" "${output_dir}" "${run_name}"
}

# ------------------------------------------------------------------------------
# Stage 1 (6-Epoch ChartNet CSV Grounding, 4,343 steps = 2,172*2) on XPK
# ------------------------------------------------------------------------------
stage1_6ep_xpk() {
  local input_ckpt="${1:-${STITCHED_CKPT}}"

  if path_exists "${input_ckpt}" && [ "${FORCE:-false}" != "true" ] && [ "${FORCE:-false}" != "1" ]; then
    echo ">>> Stitched checkpoint already exists at ${input_ckpt}. Skipping stitching."
  else
    echo ">>> Warning: Stitched checkpoint not found at ${input_ckpt}. Running stitch first..."
    stitch_ckpt
  fi

  local workload_name
  if [ "${LLM_TAG}" = "qwen3-4b" ]; then
    workload_name="${USER_PREFIX}-norm-s1-csv6ep-${TIMESTAMP}"
  else
    workload_name="${USER_PREFIX}-${TAG_SHORT}-norm-s1-csv6ep-${TIMESTAMP}"
  fi
  workload_name="${workload_name:0:39}"
  workload_name="${workload_name%-}"

  echo "=================================================================="
  echo ">>> [XPK STAGE 1 - ChartNet CSV 6-Epoch Grounding (Projector Only)]"
  echo ">>> Submitting Workload: ${workload_name}"
  echo ">>> LLM Tag:          ${LLM_TAG}"
  echo ">>> Config:           ${STAGE1_6EP_CONFIG}"
  echo ">>> Input Checkpoint: ${input_ckpt}"
  echo ">>> Working Dir:      ${STAGE1_6EP_OUTPUT_DIR}/${STAGE1_6EP_RUN_NAME}"
  echo ">>> Schedule:         ${STAGE1_6EP_EPOCHS} Epochs = ${STAGE1_6EP_STEPS} Steps"
  echo "=================================================================="

  (
    cd "${MAXTEXT_ROOT}"
    xpk workload create \
      --cluster "${XPK_CLUSTER}" \
      --project "${XPK_PROJECT}" \
      --zone "${XPK_ZONE}" \
      --workload "${workload_name}" \
      --tpu-type "${XPK_DEVICE_TYPE}" \
      --num-slices "${XPK_NUM_SLICES}" \
      --base-docker-image "${XPK_BASE_DOCKER_IMAGE}" \
      --script-dir . \
      --command "export MEGASCALE_NUM_SLICES=${XPK_NUM_SLICES} && export TMPDIR=/dev/shm && export PYTHONPATH=src:\${PYTHONPATH:-} && export HF_HOME=/dev/shm/huggingface && export HF_TOKEN=${HF_TOKEN} && export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} && python3 -m pip install --no-cache-dir -U 'orbax-checkpoint>=0.12.4' && gcloud storage cp -r ${CHARTNET_6EP_DATASET_DIR} /dev/shm/ && python3 -m maxtext.experimental.omni_pipeline.train_sft_omni ${STAGE1_6EP_CONFIG} load_parameters_path=${input_ckpt} base_output_directory=${STAGE1_6EP_OUTPUT_DIR} run_name=${STAGE1_6EP_RUN_NAME} num_epoch=${STAGE1_6EP_EPOCHS} steps=${STAGE1_6EP_STEPS} eval_interval=-1 eval_steps=0 checkpoint_period=${STAGE1_6EP_CKPT_PERIOD} tokenizer_path=${LLM_HF_REPO} hf_access_token=${HF_TOKEN} scan_layers=${SCAN_LAYERS} grain_worker_count=0"
  )
}

# ------------------------------------------------------------------------------
# Stage 2 (Projector-Only MLP SFT, 1,105 steps) on XPK
# ------------------------------------------------------------------------------
stage2_mlp_xpk() {
  local input_ckpt="${1:-}"
  local output_dir="${2:-}"
  local run_name="${3:-}"

  if [ -n "${input_ckpt}" ]; then
    input_ckpt="${input_ckpt%/_CHECKPOINT_METADATA}"
    input_ckpt="${input_ckpt/#https:\/\/storage.googleapis.com\//gs:\/\/}"
    [[ "${input_ckpt}" != gs://* && "${input_ckpt}" == "${GCS_BUCKET#gs://}"* ]] && input_ckpt="gs://${input_ckpt}"
    if [[ "${input_ckpt}" != */items ]]; then
      input_ckpt="${input_ckpt%/}/items"
    fi
  fi

  if [ -z "${input_ckpt}" ]; then
    if path_exists "${STAGE1_FINAL_CKPT}"; then
      input_ckpt="${STAGE1_FINAL_CKPT}"
    elif path_exists "${STAGE1_6EP_FINAL_CKPT}"; then
      input_ckpt="${STAGE1_6EP_FINAL_CKPT}"
    else
      input_ckpt="${STAGE1_FINAL_CKPT}"
    fi
  fi

  if ! path_exists "${input_ckpt}" && ! path_exists "${input_ckpt%/items}"; then
    echo ">>> Error: Input checkpoint not found at: ${input_ckpt}"
    echo ">>> Please run Stage 1 first, or provide an explicit checkpoint path: $0 stage2_mlp <ckpt_path>"
    exit 1
  fi

  if [ -z "${output_dir}" ]; then
    if [[ "${input_ckpt}" == *"stage1_csv_6ep"* || "${input_ckpt}" == *"6ep"* ]]; then
      output_dir="${STAGE2_MLP_6EP_OUTPUT_DIR}"
      run_name="${run_name:-${STAGE2_MLP_6EP_RUN_NAME}}"
    else
      output_dir="${STAGE2_MLP_OUTPUT_DIR}"
      run_name="${run_name:-${STAGE2_MLP_RUN_NAME}}"
    fi
  else
    [ -z "${run_name}" ] && run_name="$(basename "${output_dir}")"
  fi

  local workload_name
  if [[ "${output_dir}" == *"6ep"* || "${input_ckpt}" == *"6ep"* ]]; then
    if [ "${LLM_TAG}" = "qwen3-4b" ]; then
      workload_name="${USER_PREFIX}-norm-s2-mlp-6ep-${TIMESTAMP}"
    else
      workload_name="${USER_PREFIX}-${TAG_SHORT}-norm-s2-mlp-6ep-${TIMESTAMP}"
    fi
  else
    if [ "${LLM_TAG}" = "qwen3-4b" ]; then
      workload_name="${USER_PREFIX}-norm-s2-mlp-${TIMESTAMP}"
    else
      workload_name="${USER_PREFIX}-${TAG_SHORT}-norm-s2-mlp-${TIMESTAMP}"
    fi
  fi
  workload_name="${workload_name:0:39}"
  workload_name="${workload_name%-}"

  echo "=================================================================="
  echo ">>> [XPK STAGE 2 - ChartQA Projector-Only (MLP) SFT (Normalized Projector)]"
  echo ">>> Submitting Workload: ${workload_name}"
  echo ">>> LLM Tag:          ${LLM_TAG}"
  echo ">>> Tokenizer Path:   ${LLM_HF_REPO}"
  echo ">>> Config:           ${STAGE2_MLP_CONFIG}"
  echo ">>> Input Checkpoint: ${input_ckpt}"
  echo ">>> Working Dir:      ${output_dir}/${run_name}"
  echo ">>> Schedule:         ${STAGE2_MLP_EPOCHS} Epochs = ${STAGE2_MLP_STEPS} Steps"
  echo "=================================================================="

  (
    cd "${MAXTEXT_ROOT}"
    xpk workload create \
      --cluster "${XPK_CLUSTER}" \
      --project "${XPK_PROJECT}" \
      --zone "${XPK_ZONE}" \
      --workload "${workload_name}" \
      --tpu-type "${XPK_DEVICE_TYPE}" \
      --num-slices "${XPK_NUM_SLICES}" \
      --base-docker-image "${XPK_BASE_DOCKER_IMAGE}" \
      --script-dir . \
      --command "export MEGASCALE_NUM_SLICES=${XPK_NUM_SLICES} && export TMPDIR=/dev/shm && export PYTHONPATH=src:\${PYTHONPATH:-} && export HF_HOME=/dev/shm/huggingface && export HF_TOKEN=${HF_TOKEN} && export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} && python3 -m pip install --no-cache-dir -U 'orbax-checkpoint>=0.12.4' && gcloud storage cp -r ${CHARTQA_DATASET_DIR} /dev/shm/ && python3 -m maxtext.experimental.omni_pipeline.train_sft_omni ${STAGE2_MLP_CONFIG} load_parameters_path=${input_ckpt} base_output_directory=${output_dir} run_name=${run_name} steps=${STAGE2_MLP_STEPS} tokenizer_path=${LLM_HF_REPO} hf_access_token=${HF_TOKEN} scan_layers=${SCAN_LAYERS} grain_worker_count=0"
  )
}

# ------------------------------------------------------------------------------
# Stage 2 (Projector-Only MLP SFT after 6-Epoch ChartNet Grounding) on XPK
# ------------------------------------------------------------------------------
stage2_mlp_after_6eps_xpk() {
  local input_ckpt="${1:-${STAGE1_6EP_FINAL_CKPT}}"
  local output_dir="${2:-${STAGE2_MLP_6EP_OUTPUT_DIR}}"
  local run_name="${3:-${STAGE2_MLP_6EP_RUN_NAME}}"
  stage2_mlp_xpk "${input_ckpt}" "${output_dir}" "${run_name}"
}

# ------------------------------------------------------------------------------
# Full 2-Stage Pipeline (Both Projector-Only) in a Single XPK Workload
# ------------------------------------------------------------------------------
pipeline_mlp_xpk() {
  local input_ckpt="${1:-${STITCHED_CKPT}}"

  if path_exists "${input_ckpt}" && [ "${FORCE:-false}" != "true" ] && [ "${FORCE:-false}" != "1" ]; then
    echo "=================================================================="
    echo ">>> [CHECKPOINT CHECK] Stitched checkpoint already exists at:"
    echo ">>>   ${input_ckpt}"
    echo ">>> Skipping stitching."
    echo "=================================================================="
  else
    echo ">>> Stitched checkpoint not found at ${input_ckpt}. Running stitch first..."
    stitch_ckpt
  fi

  local workload_name
  if [ "${LLM_TAG}" = "qwen3-4b" ]; then
    workload_name="${USER_PREFIX}-norm-2s-mlp-${TIMESTAMP}"
  else
    workload_name="${USER_PREFIX}-${TAG_SHORT}-norm-2s-mlp-${TIMESTAMP}"
  fi
  workload_name="${workload_name:0:39}"
  workload_name="${workload_name%-}"

  echo "=================================================================="
  echo ">>> [XPK FULL 2-STAGE PIPELINE - Both Projector-Only (Normalized)]"
  echo ">>> Submitting Workload: ${workload_name}"
  echo ">>> LLM Tag:             ${LLM_TAG}"
  echo ">>> Input Checkpoint:    ${input_ckpt}"
  echo ">>> Stage 1: ChartNet CSV 6-Epoch (${STAGE1_6EP_STEPS} steps, No Eval)"
  echo ">>> Stage 2: ChartQA SFT MLP     (${STAGE2_MLP_STEPS} steps)"
  echo "=================================================================="

  (
    cd "${MAXTEXT_ROOT}"
    xpk workload create \
      --cluster "${XPK_CLUSTER}" \
      --project "${XPK_PROJECT}" \
      --zone "${XPK_ZONE}" \
      --workload "${workload_name}" \
      --tpu-type "${XPK_DEVICE_TYPE}" \
      --num-slices "${XPK_NUM_SLICES}" \
      --base-docker-image "${XPK_BASE_DOCKER_IMAGE}" \
      --script-dir . \
      --command "export MEGASCALE_NUM_SLICES=${XPK_NUM_SLICES} && export TMPDIR=/dev/shm && export PYTHONPATH=src:\${PYTHONPATH:-} && export HF_HOME=/dev/shm/huggingface && export HF_TOKEN=${HF_TOKEN} && export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} && python3 -m pip install --no-cache-dir -U 'orbax-checkpoint>=0.12.4' && gcloud storage cp -r ${CHARTNET_6EP_DATASET_DIR} /dev/shm/ && gcloud storage cp -r ${CHARTQA_DATASET_DIR} /dev/shm/ && echo '=== Stage 1: ChartNet CSV 6-Epoch (${STAGE1_6EP_STEPS} steps, No Eval) ===' && python3 -m maxtext.experimental.omni_pipeline.train_sft_omni ${STAGE1_6EP_CONFIG} load_parameters_path=${input_ckpt} base_output_directory=${STAGE1_6EP_OUTPUT_DIR} run_name=${STAGE1_6EP_RUN_NAME} num_epoch=${STAGE1_6EP_EPOCHS} steps=${STAGE1_6EP_STEPS} eval_interval=-1 eval_steps=0 checkpoint_period=${STAGE1_6EP_CKPT_PERIOD} tokenizer_path=${LLM_HF_REPO} hf_access_token=${HF_TOKEN} scan_layers=${SCAN_LAYERS} grain_worker_count=0 && echo '=== Stage 2: ChartQA Visual QA SFT MLP (${STAGE2_MLP_STEPS} steps) ===' && python3 -m maxtext.experimental.omni_pipeline.train_sft_omni ${STAGE2_MLP_CONFIG} load_parameters_path=${STAGE1_6EP_FINAL_CKPT} base_output_directory=${STAGE2_MLP_6EP_OUTPUT_DIR} run_name=${STAGE2_MLP_6EP_RUN_NAME} steps=${STAGE2_MLP_STEPS} tokenizer_path=${LLM_HF_REPO} hf_access_token=${HF_TOKEN} scan_layers=${SCAN_LAYERS} grain_worker_count=0"
  )
}

# ------------------------------------------------------------------------------
# Full 2-Stage Pipeline: ChartNet Core CSV 2-Epoch (26,533 steps) + ChartQA SFT MLP (1,105 steps)
# Both Projector-Only (Normalized Projector) in a Single XPK Workload
# ------------------------------------------------------------------------------
pipeline_core_mlp_xpk() {
  local input_ckpt="${1:-${STITCHED_CKPT}}"

  if path_exists "${input_ckpt}" && [ "${FORCE:-false}" != "true" ] && [ "${FORCE:-false}" != "1" ]; then
    echo "=================================================================="
    echo ">>> [CHECKPOINT CHECK] Stitched checkpoint already exists at:"
    echo ">>>   ${input_ckpt}"
    echo ">>> Skipping stitching."
    echo "=================================================================="
  else
    echo ">>> Stitched checkpoint not found at ${input_ckpt}. Running stitch first..."
    stitch_ckpt
  fi

  local workload_name
  if [ "${LLM_TAG}" = "qwen3-4b" ]; then
    workload_name="${USER_PREFIX}-norm-2s-c-mlp-${TIMESTAMP}"
  else
    workload_name="${USER_PREFIX}-${TAG_SHORT}-norm-2s-c-mlp-${TIMESTAMP}"
  fi
  workload_name="${workload_name:0:39}"
  workload_name="${workload_name%-}"

  echo "=================================================================="
  echo ">>> [XPK FULL 2-STAGE PIPELINE - Core CSV 2ep + ChartQA MLP (Both Projector-Only)]"
  echo ">>> Submitting Workload: ${workload_name}"
  echo ">>> LLM Tag:             ${LLM_TAG}"
  echo ">>> Input Checkpoint:    ${input_ckpt}"
  echo ">>> Stage 1: ChartNet Core CSV 2-Epoch (${STAGE1_STEPS} steps, No Eval)"
  echo ">>> Stage 2: ChartQA SFT MLP           (${STAGE2_MLP_STEPS} steps)"
  echo "=================================================================="

  (
    cd "${MAXTEXT_ROOT}"
    xpk workload create \
      --cluster "${XPK_CLUSTER}" \
      --project "${XPK_PROJECT}" \
      --zone "${XPK_ZONE}" \
      --workload "${workload_name}" \
      --tpu-type "${XPK_DEVICE_TYPE}" \
      --num-slices "${XPK_NUM_SLICES}" \
      --base-docker-image "${XPK_BASE_DOCKER_IMAGE}" \
      --script-dir . \
      --command "export MEGASCALE_NUM_SLICES=${XPK_NUM_SLICES} && export TMPDIR=/dev/shm && export PYTHONPATH=src:\${PYTHONPATH:-} && export HF_HOME=/dev/shm/huggingface && export HF_TOKEN=${HF_TOKEN} && export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} && python3 -m pip install --no-cache-dir -U 'orbax-checkpoint>=0.12.4' && gcloud storage cp -r ${CHARTNET_DATASET_DIR} /dev/shm/ && gcloud storage cp -r ${CHARTQA_DATASET_DIR} /dev/shm/ && echo '=== Stage 1: ChartNet Core CSV 2-Epoch (${STAGE1_STEPS} steps, No Eval) ===' && python3 -m maxtext.experimental.omni_pipeline.train_sft_omni ${STAGE1_CONFIG} load_parameters_path=${input_ckpt} base_output_directory=${STAGE1_OUTPUT_DIR} run_name=${STAGE1_RUN_NAME} num_epoch=${STAGE1_EPOCHS} steps=${STAGE1_STEPS} eval_interval=-1 eval_steps=0 checkpoint_period=${STAGE1_CKPT_PERIOD} tokenizer_path=${LLM_HF_REPO} hf_access_token=${HF_TOKEN} scan_layers=${SCAN_LAYERS} grain_worker_count=0 && echo '=== Stage 2: ChartQA Visual QA SFT MLP (${STAGE2_MLP_STEPS} steps) ===' && python3 -m maxtext.experimental.omni_pipeline.train_sft_omni ${STAGE2_MLP_CONFIG} load_parameters_path=${STAGE1_FINAL_CKPT} base_output_directory=${STAGE2_MLP_OUTPUT_DIR} run_name=${STAGE2_MLP_RUN_NAME} steps=${STAGE2_MLP_STEPS} tokenizer_path=${LLM_HF_REPO} hf_access_token=${HF_TOKEN} scan_layers=${SCAN_LAYERS} grain_worker_count=0"
  )
}

# ------------------------------------------------------------------------------
# 4. Evaluation
# ------------------------------------------------------------------------------
eval_sft() {
  local ckpt_path="${1:-}"
  if [ -z "${ckpt_path}" ]; then
    if path_exists "${STAGE2_MLP_6EP_FINAL_CKPT}"; then
      ckpt_path="${STAGE2_MLP_6EP_FINAL_CKPT}"
    else
      ckpt_path="${STAGE2_FINAL_CKPT}"
    fi
  fi
  ckpt_path="${ckpt_path%/_CHECKPOINT_METADATA}"
  ckpt_path="${ckpt_path/#https:\/\/storage.googleapis.com\//gs:\/\/}"
  [[ "${ckpt_path}" != gs://* && "${ckpt_path}" == "${GCS_BUCKET#gs://}"* ]] && ckpt_path="gs://${ckpt_path}"
  if [[ "${ckpt_path}" != */items ]]; then
    ckpt_path="${ckpt_path%/}/items"
  fi
  local eval_dir="${2:-}"
  local num_examples="${3:-2500}"

  if [ -z "${eval_dir}" ]; then
    if [[ "${ckpt_path}" =~ (.*)/checkpoints/.* ]]; then
      eval_dir="${BASH_REMATCH[1]}/eval"
    else
      eval_dir="${EVAL_OUTPUT_DIR}"
    fi
  fi

  local step_tag="eval"
  if [[ "${ckpt_path}" =~ /checkpoints/([0-9]+)/items ]]; then
    step_tag="step_${BASH_REMATCH[1]}"
  fi

  local eval_config="${STAGE2_CONFIG}"
  if [[ "${ckpt_path}" == *"stage2_mlp"* || "${ckpt_path}" == *"mlp"* ]]; then
    eval_config="${STAGE2_MLP_CONFIG}"
  fi

  echo "=================================================================="
  echo ">>> [EVAL] Evaluating Normalized Checkpoint: ${ckpt_path}"
  echo ">>> Config:          ${eval_config}"
  echo ">>> Destination Dir: ${eval_dir}"
  echo ">>> Examples:        ${num_examples}"
  echo ">>> Run Name:        eval_${step_tag}"
  echo ">>> Tokenizer Path:  ${LLM_HF_REPO}"
  echo ">>> Scan Layers:     ${SCAN_LAYERS}"
  echo "=================================================================="

  MEGASCALE_NUM_SLICES=1 HF_TOKEN="${HF_TOKEN}" HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}" python3 -m maxtext.experimental.omni_pipeline.eval_sft_omni \
    "${eval_config}" \
    "load_parameters_path=${ckpt_path}" \
    "base_output_directory=${eval_dir}" \
    "run_name=eval_${step_tag}" \
    "tokenizer_path=${LLM_HF_REPO}" \
    "hf_access_token=${HF_TOKEN}" \
    "hf_path=HuggingFaceM4/ChartQA" \
    "scan_layers=${SCAN_LAYERS}" \
    --ckpt_type=sft \
    --num_examples="${num_examples}" \
    --hf_eval_split=test \
    --tmp_results_file="omni_norm_eval_${step_tag}.csv"
}

# ------------------------------------------------------------------------------
# 5. Evaluation on XPK Cluster
# ------------------------------------------------------------------------------
eval_xpk() {
  local ckpt_path="${1:-}"
  if [ -z "${ckpt_path}" ]; then
    if path_exists "${STAGE2_MLP_6EP_FINAL_CKPT}"; then
      ckpt_path="${STAGE2_MLP_6EP_FINAL_CKPT}"
    else
      ckpt_path="${STAGE2_MLP_FINAL_CKPT}"
    fi
  fi
  ckpt_path="${ckpt_path%/_CHECKPOINT_METADATA}"
  ckpt_path="${ckpt_path/#https:\/\/storage.googleapis.com\//gs:\/\/}"
  [[ "${ckpt_path}" != gs://* && "${ckpt_path}" == "${GCS_BUCKET#gs://}"* ]] && ckpt_path="gs://${ckpt_path}"
  if [[ "${ckpt_path}" != */items ]]; then
    ckpt_path="${ckpt_path%/}/items"
  fi
  local eval_dir="${2:-}"
  local num_examples="${3:-2500}"

  if [ -z "${eval_dir}" ]; then
    if [[ "${ckpt_path}" =~ (.*)/checkpoints/.* ]]; then
      eval_dir="${BASH_REMATCH[1]}/eval"
    else
      eval_dir="${EVAL_OUTPUT_DIR}"
    fi
  fi

  local step_tag="eval"
  if [[ "${ckpt_path}" =~ /checkpoints/([0-9]+)/items ]]; then
    step_tag="step_${BASH_REMATCH[1]}"
  fi

  local eval_config="${STAGE2_CONFIG}"
  if [[ "${ckpt_path}" == *"stage2_mlp"* || "${ckpt_path}" == *"mlp"* ]]; then
    eval_config="${STAGE2_MLP_CONFIG}"
  fi

  local workload_name
  if [ "${LLM_TAG}" = "qwen3-4b" ]; then
    workload_name="${USER_PREFIX}-norm-eval-${step_tag}-${TIMESTAMP}"
  else
    workload_name="${USER_PREFIX}-${TAG_SHORT}-norm-eval-${step_tag}-${TIMESTAMP}"
  fi
  workload_name="${workload_name:0:39}"
  workload_name="${workload_name%-}"

  echo "=================================================================="
  echo ">>> [XPK EVAL - ChartQA Test Split (Normalized Projector)]"
  echo ">>> Submitting Workload: ${workload_name}"
  echo ">>> Config:              ${eval_config}"
  echo ">>> Input Checkpoint:    ${ckpt_path}"
  echo ">>> Destination Dir:     ${eval_dir}"
  echo ">>> Run Name:            eval_${step_tag}"
  echo ">>> Tokenizer Path:      ${LLM_HF_REPO}"
  echo ">>> Examples:            ${num_examples}"
  echo "=================================================================="

  (
    cd "${MAXTEXT_ROOT}"
    xpk workload create \
      --cluster "${XPK_CLUSTER}" \
      --project "${XPK_PROJECT}" \
      --zone "${XPK_ZONE}" \
      --workload "${workload_name}" \
      --tpu-type "${XPK_DEVICE_TYPE}" \
      --num-slices "${XPK_NUM_SLICES}" \
      --base-docker-image "${XPK_BASE_DOCKER_IMAGE}" \
      --script-dir . \
      --command "export MEGASCALE_NUM_SLICES=${XPK_NUM_SLICES} && export TMPDIR=/dev/shm && export PYTHONPATH=src:\${PYTHONPATH:-} && export HF_HOME=/dev/shm/huggingface && export HF_TOKEN=${HF_TOKEN} && export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} && python3 -m pip install --no-cache-dir -U 'orbax-checkpoint>=0.12.4' && python3 -m maxtext.experimental.omni_pipeline.eval_sft_omni ${eval_config} load_parameters_path=${ckpt_path} base_output_directory=${eval_dir} run_name=eval_${step_tag} tokenizer_path=${LLM_HF_REPO} hf_access_token=${HF_TOKEN} hf_path=HuggingFaceM4/ChartQA scan_layers=${SCAN_LAYERS} --ckpt_type=sft --num_examples=${num_examples} --hf_eval_split=test --tmp_results_file=omni_norm_eval_${step_tag}.csv"
  )
}

# ------------------------------------------------------------------------------
# Dispatcher
# ------------------------------------------------------------------------------
case "$ACTION" in
  stitch|prepare)
    stitch_ckpt
    ;;
  restitch)
    FORCE=true stitch_ckpt
    ;;
  stage1)
    stage1_xpk "${ARG2}"
    ;;
  stage1_6ep|stage1_csv)
    stage1_6ep_xpk "${ARG2}"
    ;;
  stage2)
    stage2_xpk "${ARG2}" "${3:-}" "${4:-}"
    ;;
  stage2_after_6eps|stage2_6ep|stage2_fdec_after_6eps)
    stage2_after_6eps_xpk "${ARG2}" "${3:-}" "${4:-}"
    ;;
  stage2_mlp)
    stage2_mlp_xpk "${ARG2}" "${3:-}" "${4:-}"
    ;;
  stage2_mlp_after_6eps|stage2_mlp_6ep)
    stage2_mlp_after_6eps_xpk "${ARG2}" "${3:-}" "${4:-}"
    ;;
  pipeline_core_mlp|pipeline_core)
    pipeline_core_mlp_xpk "${ARG2}"
    ;;
  pipeline_mlp|pipeline)
    pipeline_mlp_xpk "${ARG2}"
    ;;
  eval)
    eval_sft "${ARG2}" "${3:-}" "${4:-2500}"
    ;;
  eval_xpk|eval-xpk)
    eval_xpk "${ARG2}" "${3:-}" "${4:-2500}"
    ;;
  list)
    xpk workload list --cluster "${XPK_CLUSTER}" --project "${XPK_PROJECT}" --zone "${XPK_ZONE}"
    ;;
  status)
    kubectl get pods -l jobset.sigs.k8s.io/jobset-name="${ARG2}" -w
    ;;
  logs)
    pod_name=$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name="${ARG2}" -o jsonpath='{.items[0].metadata.name}')
    kubectl logs "${pod_name}" -c jax-tpu -f
    ;;
  delete)
    xpk workload delete --workload "${ARG2}" --cluster "${XPK_CLUSTER}" --project "${XPK_PROJECT}" --zone "${XPK_ZONE}"
    ;;
  help|*)
    echo "Usage: $0 <stitch|stage1|stage1_6ep|stage2|stage2_mlp|stage2_mlp_after_6eps|pipeline_core_mlp|pipeline_mlp|eval|eval_xpk|list|status|logs|delete> [hf_repo|checkpoint_path|workload_name] [eval_output_dir] [num_examples]"
    echo ""
    echo "Examples:"
    echo "  # Stitch new normalized checkpoint with Qwen3-4B:"
    echo "  $0 stitch Qwen/Qwen3-4B"
    echo ""
    echo "  # --- 2-Stage Pipeline (Both Projector-Only) ---"
    echo "  # Run ChartNet Core CSV 2-Epoch (26,533 steps) -> ChartQA SFT MLP (1,105 steps) in 1 XPK job:"
    echo "  $0 pipeline_core_mlp"
    echo ""
    echo "  # Run ChartNet CSV 6-Epoch (4,343 steps) -> ChartQA SFT MLP (1,105 steps) in 1 XPK job:"
    echo "  $0 pipeline_mlp"
    echo ""
    echo "  # --- Run Stages Individually (Projector-Only) ---"
    echo "  # Run Stage 1 (ChartNet Core CSV 2-Epoch = 26,533 steps):"
    echo "  $0 stage1"
    echo ""
    echo "  # Run Stage 2 (ChartQA SFT MLP-only = 1,105 steps):"
    echo "  $0 stage2_mlp [stage1_checkpoint_path]"
    echo ""
    echo "  # Run Stage 2 after 6-Epoch Stage 1 (saves to stage2_mlp_after_6eps):"
    echo "  $0 stage2_mlp_after_6eps [stage1_checkpoint_path]"
    echo ""
    echo "  # --- Standard Core Pipeline (Stage 1 Core 26k steps -> Stage 2 Full-Decoder) ---"
    echo "  $0 stage1"
    echo "  $0 stage2 <stage1_checkpoint_path>"
    echo ""
    echo "  # Run Stage 2 Full-Decoder after 6-Epoch Stage 1 (saves to stage2_full_decoder_after_6eps):"
    echo "  $0 stage2_after_6eps [stage1_checkpoint_path]"
    echo ""
    echo "  # Evaluate SFT Checkpoint:"
    echo "  $0 eval <stage2_checkpoint_path>"
    ;;
esac
