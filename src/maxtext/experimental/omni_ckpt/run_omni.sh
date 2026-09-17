#!/bin/bash
# ==============================================================================
# Omni Gemma3-4B Vision + Qwen3-4B LLM 2-Stage Training Pipeline on XPK (omni_ckpt)
#
# Stage 1: ChartNet CSV Table Grounding (ibm-granite/ChartNet, 6 Epochs = 4,343 steps = 2,172*2, No Eval)
# Stage 2: ChartQA Visual QA SFT        (HuggingFaceM4/ChartQA, 5 Epochs = 1,105 steps)
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
BASE_CKPT_DIR="${BASE_CKPT_DIR:-${GCS_BUCKET}/omni_checkpoints_diff}"
BASE_CKPT_DIR="${BASE_CKPT_DIR%/}"

# Vision Model Fixed: Gemma 3 4B
VISION_MAXTEXT_MODEL="gemma3-4b"
VISION_HF_REPO="google/gemma-3-4b-it"

# Parse Target LLM Checkpoint / Tag
# Can be passed via $2, env var LLM_HF_REPO, or env var LLM_TAG
LLM_MAXTEXT_MODEL="qwen3-4b"
LLM_HF_REPO="${LLM_HF_REPO:-Qwen/Qwen3-4B}"
LLM_TAG="${LLM_TAG:-}"

if [[ -n "${ARG2}" && ( "${ACTION}" == "stitch" || "${ACTION}" == "prepare" || "${ACTION}" == "pipeline" || "${ACTION}" == "stage1" || "${ACTION}" == "stage2" ) ]]; then
  if [[ "${ARG2}" == gs://* || "${ARG2}" == */checkpoints/* || "${ARG2}" == "${USER_PREFIX}-"* ]]; then
    if [[ "${ARG2}" =~ omni_([a-zA-Z0-9_-]+)/stage || "${ARG2}" =~ omni_([a-zA-Z0-9_-]+) || "${ARG2}" =~ (qwen3-[a-zA-Z0-9_-]+) ]]; then
      detected_tag="${BASH_REMATCH[1]}"
      detected_tag="${detected_tag%_unscanned}"
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
elif [[ "${ACTION}" == "eval" && -n "${ARG2}" ]]; then
  if [[ "${ARG2}" =~ omni_([a-zA-Z0-9_-]+)/stage || "${ARG2}" =~ omni_([a-zA-Z0-9_-]+) || "${ARG2}" =~ (qwen3-[a-zA-Z0-9_-]+) ]]; then
    detected_tag="${BASH_REMATCH[1]}"
    detected_tag="${detected_tag%_unscanned}"
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

# Compute Short Tag for workload names
TAG_SHORT=$(echo "${LLM_TAG}" | sed 's/qwen3-4b-//; s/qwen3-//; s/-instruct//' | tr -cd '[:alnum:]-')
[ -z "${TAG_SHORT}" ] && TAG_SHORT="4b"

# Directories & Checkpoints (isolated by SCAN_SUFFIX)
EXP_NAME="${EXP_NAME:-omni_${LLM_TAG}${SCAN_SUFFIX}}"
WORKING_DIR="${WORKING_DIR:-${GCS_BUCKET}/omni_ckpt/${EXP_NAME}}"

VISION_CKPT_DIR="${BASE_CKPT_DIR}/${VISION_MAXTEXT_MODEL}${SCAN_SUFFIX}_converted"
LLM_CKPT_DIR="${BASE_CKPT_DIR}/${LLM_TAG}${SCAN_SUFFIX}_converted"
STITCHED_CKPT_DIR="${BASE_CKPT_DIR}/omni_stitched_${VISION_MAXTEXT_MODEL}_${LLM_TAG}${SCAN_SUFFIX}"

VISION_SOURCE_CKPT="${VISION_SOURCE_CKPT:-${VISION_CKPT_DIR}/0/items}"
LLM_SOURCE_CKPT="${LLM_SOURCE_CKPT:-${LLM_CKPT_DIR}/0/items}"
STITCHED_CKPT="${STITCHED_CKPT:-${STITCHED_CKPT_DIR}/0/items}"

# Pre-sharded Dataset Storage Paths (GCS)
CHARTNET_DATASET_DIR="${CHARTNET_DATASET_DIR:-${GCS_BUCKET}/datasets/chartnet_sharded}"
CHARTQA_DATASET_DIR="${CHARTQA_DATASET_DIR:-${GCS_BUCKET}/datasets/chartqa_shuffled}"

# Stage 1: ChartNet CSV Table Grounding (6 Epochs = 4,343 steps on 128 devices, No Eval)
STAGE1_CONFIG="${STAGE1_CONFIG:-src/maxtext/experimental/omni_ckpt/pretrain-omni-gemma3-qwen3-chartnet-xpk-128-csv-6ep.yml}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-6}"
STAGE1_STEPS="${STAGE1_STEPS:-4343}"
STAGE1_CKPT_PERIOD="${STAGE1_CKPT_PERIOD:-500}"
STAGE1_OUTPUT_DIR="${WORKING_DIR}/stage1_chartnet_csv_6ep"
STAGE1_RUN_NAME="${EXP_NAME}_stage1_csv_6ep"
STAGE1_FINAL_CKPT="${STAGE1_OUTPUT_DIR}/${STAGE1_RUN_NAME}/checkpoints/$((STAGE1_STEPS - 1))/items"

# Stage 2: ChartQA Visual QA SFT (5 Epochs = 1,105 steps on 128 devices)
# Defaults to Projector-only (MLP) SFT (custom_linear fine-tuning, decoder frozen)
STAGE2_CONFIG="${STAGE2_CONFIG:-src/maxtext/experimental/omni_ckpt/sft-omni-gemma3-qwen3-mlp-xpk-128.yml}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-5}"
STAGE2_STEPS="${STAGE2_STEPS:-1105}"
STAGE2_OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-${WORKING_DIR}/stage2_mlp}"
STAGE2_RUN_NAME="${STAGE2_RUN_NAME:-${EXP_NAME}_stage2_mlp}"
STAGE2_FINAL_CKPT="${STAGE2_OUTPUT_DIR}/${STAGE2_RUN_NAME}/checkpoints/$((STAGE2_STEPS - 1))/items"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${STAGE2_OUTPUT_DIR}/eval}"

# XPK Cluster Configuration
XPK_CLUSTER="${XPK_CLUSTER:-v4-128-bodaborg-us-central2-b}"
XPK_PROJECT="${XPK_PROJECT:-cloud-tpu-multipod-dev}"
XPK_ZONE="${XPK_ZONE:-us-central2-b}"
XPK_DEVICE_TYPE="${XPK_DEVICE_TYPE:-v4-128}"
XPK_NUM_SLICES="${XPK_NUM_SLICES:-1}"
XPK_BASE_DOCKER_IMAGE="${XPK_BASE_DOCKER_IMAGE:-gcr.io/tpu-prod-env-multipod/maxtext_base_image:latest}"

USER_PREFIX="${USER_PREFIX:-user}"
TIMESTAMP="${TIMESTAMP:-$(date +%m%d)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAXTEXT_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || (cd "${SCRIPT_DIR}/../../../../.." && pwd))"
OMNI_CONFIG_PATH="${SCRIPT_DIR}/maxtext-omni-gemma3-qwen3.yml"

path_exists() {
  local p="$1"
  python3 -c "from etils import epath; import sys; sys.exit(0 if epath.Path(sys.argv[1]).exists() else 1)" "$p" 2>/dev/null || \
  gcloud storage ls "$p" >/dev/null 2>&1 || \
  gsutil -q stat "$p" >/dev/null 2>&1
}

# ------------------------------------------------------------------------------
# 0. Check Dataset Status
# ------------------------------------------------------------------------------
check_data_status() {
  echo "=================================================================="
  echo ">>> [DATA CHECK] Checking ChartNet and ChartQA Dataset Paths in GCS"
  echo "=================================================================="
  echo "1. Checking ChartNet Sharded Path in GCS: ${CHARTNET_DATASET_DIR}"
  if gcloud storage ls "${CHARTNET_DATASET_DIR}/*.parquet" >/dev/null 2>&1; then
    count=$(gcloud storage ls "${CHARTNET_DATASET_DIR}/*.parquet" | wc -l)
    echo "   [FOUND] ChartNet dataset exists with ${count} parquet shards."
  else
    echo "   [NOT FOUND] ChartNet dataset not found in GCS."
  fi
  echo ""
  echo "2. Checking ChartQA Shuffled Path in GCS: ${CHARTQA_DATASET_DIR}"
  if gcloud storage ls "${CHARTQA_DATASET_DIR}/*.parquet" >/dev/null 2>&1; then
    count=$(gcloud storage ls "${CHARTQA_DATASET_DIR}/*.parquet" | wc -l)
    echo "   [FOUND] ChartQA dataset exists with ${count} parquet shards."
  else
    echo "   [NOT FOUND] ChartQA dataset not found in GCS."
  fi
  echo "=================================================================="
}

# ------------------------------------------------------------------------------
# 1. Download, Convert, and Stitch Checkpoints (Self-Contained)
# ------------------------------------------------------------------------------
stitch_ckpt() {
  echo "=================================================================="
  echo ">>> [STITCH & PREPARE CHECKPOINT]"
  echo ">>> Vision Model:      ${VISION_MAXTEXT_MODEL} (${VISION_HF_REPO})"
  echo ">>> LLM Model:         ${LLM_MAXTEXT_MODEL} (${LLM_HF_REPO})"
  echo ">>> LLM Tag:           ${LLM_TAG}"
  echo ">>> Scan Layers:       ${SCAN_LAYERS}"
  echo ">>> Vision Checkpoint: ${VISION_SOURCE_CKPT}"
  echo ">>> LLM Checkpoint:    ${LLM_SOURCE_CKPT}"
  echo ">>> Stitched Output:   ${STITCHED_CKPT}"
  echo "=================================================================="

  export JAX_PLATFORMS=cpu

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

  # Step 3: Stitch into Unified Multimodal Checkpoint
  echo -e "\n=== [3/3] Stitching Subtrees into Unified Omni Checkpoint (scan_layers=${SCAN_LAYERS}) ==="
  if ! path_exists "${STITCHED_CKPT}"; then
    (
      cd "${MAXTEXT_ROOT}"
      python3 -m maxtext.experimental.omni_poc.utils.stitch_checkpoint \
        "${OMNI_CONFIG_PATH}" \
        "hf_access_token=${HF_TOKEN}" \
        "tokenizer_path=${LLM_HF_REPO}" \
        "scan_layers=${SCAN_LAYERS}" \
        "vision_load_path=${VISION_SOURCE_CKPT}" \
        "llm_load_path=${LLM_SOURCE_CKPT}" \
        "stitched_output_path=${STITCHED_CKPT}"
    )
  else
    echo "--> Stitched checkpoint already exists at ${STITCHED_CKPT}. Skipping."
  fi

  echo -e "\n=================================================================="
  echo ">>> Checkpoint preparation and stitching complete!"
  echo ">>> Stitched Path: ${STITCHED_CKPT}"
  echo "=================================================================="
}

# ------------------------------------------------------------------------------
# 2. Stage 1: ChartNet CSV Table Grounding on XPK (6 Epochs = 4,343 steps)
# ------------------------------------------------------------------------------
stage1_xpk() {
  local input_ckpt="${1:-${STITCHED_CKPT}}"

  if ! path_exists "${input_ckpt}"; then
    echo ">>> Warning: Stitched checkpoint not found at ${input_ckpt}. Running stitch first..."
    stitch_ckpt
  fi

  local workload_name
  if [ "${LLM_TAG}" = "qwen3-4b" ]; then
    workload_name="${USER_PREFIX}-omni-s1-csv6ep-${TIMESTAMP}"
  else
    workload_name="${USER_PREFIX}-${TAG_SHORT}-s1-csv6ep-${TIMESTAMP}"
  fi
  workload_name="${workload_name:0:39}"
  workload_name="${workload_name%-}"

  echo "=================================================================="
  echo ">>> [XPK STAGE 1 - ChartNet CSV (6 Epochs)] Submitting Workload: ${workload_name}"
  echo ">>> LLM Tag:          ${LLM_TAG}"
  echo ">>> Config:           ${STAGE1_CONFIG}"
  echo ">>> Input Checkpoint: ${input_ckpt}"
  echo ">>> Working Dir:      ${STAGE1_OUTPUT_DIR}/${STAGE1_RUN_NAME}"
  echo ">>> Schedule:         ${STAGE1_EPOCHS} Epochs = ${STAGE1_STEPS} Steps (No Eval)"
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
      --command "export MEGASCALE_NUM_SLICES=${XPK_NUM_SLICES} && export TMPDIR=/dev/shm && export PYTHONPATH=src:\${PYTHONPATH:-} && export HF_HOME=/dev/shm/huggingface && export HF_TOKEN=${HF_TOKEN} && export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} && python3 -m pip install --no-cache-dir -U 'orbax-checkpoint>=0.12.4' && gcloud storage cp -r ${CHARTNET_DATASET_DIR} /dev/shm/ && python3 -m maxtext.experimental.omni_poc.train_sft_omni ${STAGE1_CONFIG} load_parameters_path=${input_ckpt} base_output_directory=${STAGE1_OUTPUT_DIR} run_name=${STAGE1_RUN_NAME} num_epoch=${STAGE1_EPOCHS} steps=${STAGE1_STEPS} eval_interval=-1 eval_steps=0 checkpoint_period=${STAGE1_CKPT_PERIOD} hf_access_token=${HF_TOKEN} scan_layers=${SCAN_LAYERS} grain_worker_count=0"
  )
}

# ------------------------------------------------------------------------------
# 3. Stage 2: ChartQA Visual QA SFT on XPK (5 Epochs = 1,105 steps)
# ------------------------------------------------------------------------------
stage2_xpk() {
  local input_ckpt="${1:-${STAGE1_FINAL_CKPT}}"
  input_ckpt="${input_ckpt%/_CHECKPOINT_METADATA}"
  input_ckpt="${input_ckpt/#https:\/\/storage.googleapis.com\//gs:\/\/}"
  [[ "${input_ckpt}" != gs://* && "${input_ckpt}" == "${GCS_BUCKET#gs://}"* ]] && input_ckpt="gs://${input_ckpt}"
  if [[ "${input_ckpt}" != */items ]]; then
    input_ckpt="${input_ckpt%/}/items"
  fi

  local workload_name
  if [ "${LLM_TAG}" = "qwen3-4b" ]; then
    workload_name="${USER_PREFIX}-omni-s2-mlp-${TIMESTAMP}"
  else
    workload_name="${USER_PREFIX}-${TAG_SHORT}-s2-mlp-${TIMESTAMP}"
  fi
  workload_name="${workload_name:0:39}"
  workload_name="${workload_name%-}"

  echo "=================================================================="
  echo ">>> [XPK STAGE 2 - ChartQA SFT (MLP)] Submitting Workload: ${workload_name}"
  echo ">>> LLM Tag:          ${LLM_TAG}"
  echo ">>> Tokenizer Path:   ${LLM_HF_REPO}"
  echo ">>> Config:           ${STAGE2_CONFIG}"
  echo ">>> Input Checkpoint: ${input_ckpt}"
  echo ">>> Working Dir:      ${STAGE2_OUTPUT_DIR}/${STAGE2_RUN_NAME}"
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
      --command "export MEGASCALE_NUM_SLICES=${XPK_NUM_SLICES} && export TMPDIR=/dev/shm && export PYTHONPATH=src:\${PYTHONPATH:-} && export HF_HOME=/dev/shm/huggingface && export HF_TOKEN=${HF_TOKEN} && export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} && python3 -m pip install --no-cache-dir -U 'orbax-checkpoint>=0.12.4' && gcloud storage cp -r ${CHARTQA_DATASET_DIR} /dev/shm/ && python3 -m maxtext.experimental.omni_poc.train_sft_omni ${STAGE2_CONFIG} load_parameters_path=${input_ckpt} base_output_directory=${STAGE2_OUTPUT_DIR} run_name=${STAGE2_RUN_NAME} steps=${STAGE2_STEPS} tokenizer_path=${LLM_HF_REPO} hf_access_token=${HF_TOKEN} scan_layers=${SCAN_LAYERS} grain_worker_count=0"
  )
}

# ------------------------------------------------------------------------------
# 4. Full 2-Stage Pipeline on XPK (Stage 1 CSV 6ep -> Stage 2 ChartQA SFT)
# ------------------------------------------------------------------------------
pipeline_xpk() {
  if ! path_exists "${STITCHED_CKPT}"; then
    echo ">>> Stitched checkpoint not found at ${STITCHED_CKPT}."
    echo ">>> Running preparation and stitching first..."
    stitch_ckpt
  fi

  local workload_name
  if [ "${LLM_TAG}" = "qwen3-4b" ]; then
    workload_name="${USER_PREFIX}-omni-2stage-csv6ep-${TIMESTAMP}"
  else
    workload_name="${USER_PREFIX}-${TAG_SHORT}-2s-csv6ep-${TIMESTAMP}"
  fi
  workload_name="${workload_name:0:39}"
  workload_name="${workload_name%-}"

  echo "=================================================================="
  echo ">>> [XPK FULL 2-STAGE PIPELINE: ChartNet CSV (6 Epochs) -> ChartQA SFT (MLP)]"
  echo ">>> Submitting Workload: ${workload_name}"
  echo ">>> LLM Model:           ${LLM_HF_REPO} (${LLM_TAG})"
  echo ">>> Base Working Dir:    ${WORKING_DIR}"
  echo ">>> Stage 1: ChartNet CSV Grounding (6 Epochs = ${STAGE1_STEPS} steps -> ${STAGE1_FINAL_CKPT})"
  echo ">>> Stage 2: ChartQA Visual QA SFT  (5 Epochs = ${STAGE2_STEPS} steps  -> ${STAGE2_FINAL_CKPT})"
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
      --command "export MEGASCALE_NUM_SLICES=${XPK_NUM_SLICES} && export TMPDIR=/dev/shm && export PYTHONPATH=src:\${PYTHONPATH:-} && export HF_HOME=/dev/shm/huggingface && export HF_TOKEN=${HF_TOKEN} && export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} && python3 -m pip install --no-cache-dir -U 'orbax-checkpoint>=0.12.4' && gcloud storage cp -r ${CHARTNET_DATASET_DIR} /dev/shm/ && gcloud storage cp -r ${CHARTQA_DATASET_DIR} /dev/shm/ && echo '=== Stage 1: ChartNet CSV Table Grounding (6 Epochs = ${STAGE1_STEPS} steps, No Eval) ===' && python3 -m maxtext.experimental.omni_poc.train_sft_omni ${STAGE1_CONFIG} load_parameters_path=${STITCHED_CKPT} base_output_directory=${STAGE1_OUTPUT_DIR} run_name=${STAGE1_RUN_NAME} num_epoch=${STAGE1_EPOCHS} steps=${STAGE1_STEPS} eval_interval=-1 eval_steps=0 checkpoint_period=${STAGE1_CKPT_PERIOD} tokenizer_path=${LLM_HF_REPO} hf_access_token=${HF_TOKEN} scan_layers=${SCAN_LAYERS} grain_worker_count=0 && echo '=== Stage 2: ChartQA Visual QA SFT (${STAGE2_STEPS} steps) ===' && python3 -m maxtext.experimental.omni_poc.train_sft_omni ${STAGE2_CONFIG} load_parameters_path=${STAGE1_FINAL_CKPT} base_output_directory=${STAGE2_OUTPUT_DIR} run_name=${STAGE2_RUN_NAME} steps=${STAGE2_STEPS} tokenizer_path=${LLM_HF_REPO} hf_access_token=${HF_TOKEN} scan_layers=${SCAN_LAYERS} grain_worker_count=0"
  )

  echo ""
  echo ">>> 2-Stage Pipeline Workload submitted: ${workload_name}"
  echo "=================================================================="
}

# ------------------------------------------------------------------------------
# 5. Evaluation
# ------------------------------------------------------------------------------
eval_sft() {
  local ckpt_path="${1:-${STAGE2_FINAL_CKPT}}"
  ckpt_path="${ckpt_path%/_CHECKPOINT_METADATA}"
  ckpt_path="${ckpt_path/#https:\/\/storage.googleapis.com\//gs:\/\/}"
  [[ "${ckpt_path}" != gs://* && "${ckpt_path}" == "${GCS_BUCKET#gs://}"* ]] && ckpt_path="gs://${ckpt_path}"
  if [[ "${ckpt_path}" != */items ]]; then
    ckpt_path="${ckpt_path%/}/items"
  fi
  local eval_dir="${2:-${EVAL_OUTPUT_DIR}}"
  local num_examples="${3:-2500}"

  # Extract step number from path (e.g. .../checkpoints/1104/items -> 1104)
  local step_tag="eval"
  if [[ "${ckpt_path}" =~ /checkpoints/([0-9]+)/items ]]; then
    step_tag="step_${BASH_REMATCH[1]}"
  fi

  echo "=================================================================="
  echo ">>> [EVAL] Evaluating SFT Checkpoint: ${ckpt_path}"
  echo ">>> Destination Dir: ${eval_dir}"
  echo ">>> Examples:        ${num_examples}"
  echo ">>> Run Name:        eval_${step_tag}"
  echo ">>> Tokenizer Path:  ${LLM_HF_REPO}"
  echo "=================================================================="

  MEGASCALE_NUM_SLICES=1 HF_TOKEN="${HF_TOKEN}" HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}" python3 -m maxtext.experimental.omni_poc.eval_sft_omni \
    "${STAGE2_CONFIG}" \
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
    --tmp_results_file="omni_eval_${step_tag}.csv"
}

# ------------------------------------------------------------------------------
# Dispatcher
# ------------------------------------------------------------------------------
case "$ACTION" in
  check_data|data_status)
    check_data_status
    ;;
  prepare|stitch)
    stitch_ckpt
    ;;
  stage1)
    stage1_xpk "${ARG2}"
    ;;
  stage2|stage3)
    stage2_xpk "${ARG2}"
    ;;
  pipeline)
    pipeline_xpk
    ;;
  eval)
    eval_sft "${ARG2}" "${3:-}" "${4:-2500}"
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
    echo "Usage: $0 <check_data|stitch|stage1|stage2|pipeline|eval|list|status|logs|delete> [hf_repo|checkpoint_path|workload_name] [eval_output_dir] [num_examples]"
    echo ""
    echo "Examples:"
    echo "  # Stitch new checkpoint from Hugging Face:"
    echo "  $0 stitch Qwen/Qwen3-4B-Thinking-2507"
    echo ""
    echo "  # Launch 2-Stage Pipeline with default Qwen 3 4B:"
    echo "  $0 pipeline"
    echo ""
    echo "  # Stitch Qwen3-VL 4B with Gemma 3 4B (scan_layers=false):"
    echo "  $0 stitch Qwen/Qwen3-VL-4B-Instruct"
    echo ""
    echo "  # Launch 2-Stage Pipeline with Qwen3-VL 4B:"
    echo "  $0 pipeline Qwen/Qwen3-VL-4B-Instruct"
    echo ""
    echo "  # Launch 2-Stage Pipeline with custom Qwen 3 variant:"
    echo "  $0 pipeline Qwen/Qwen3-4B-Thinking-2507"
    echo "  # or:"
    echo "  LLM_TAG=qwen3-4b-thinking-2507 $0 pipeline"
    ;;
esac
