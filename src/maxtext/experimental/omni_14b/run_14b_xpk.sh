#!/bin/bash
# ==============================================================================
# Omni Gemma3-4B Vision + Qwen3-14B LLM 3-Stage Training Pipeline on XPK
# Stage 1: ChartNet Dense Summary Alignment (2,172 steps)
# Stage 2: ChartNet CSV Table Grounding     (2,172 steps)
# Stage 3: ChartQA Visual QA SFT            (1,105 steps)
# ==============================================================================

set -e

ACTION="${1:-help}"

# Temporary directories and memory cache
export TMPDIR=/dev/shm
export HF_HOME=/dev/shm/huggingface
export HF_DATASETS_CACHE=/dev/shm/huggingface/datasets
export TRANSFORMERS_CACHE=/dev/shm/huggingface/transformers
export HUGGINGFACE_HUB_CACHE=/dev/shm/huggingface/hub


: "${HF_TOKEN:?Error: HF_TOKEN is not set. Please run: export HF_TOKEN=hf_...}"
: "${GCS_BUCKET:?Error: GCS_BUCKET is not set. Please run: export GCS_BUCKET=gs://your-bucket}"
export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"


CKPT_DIR="${CKPT_DIR:-${GCS_BUCKET}/omni-gemma3-qwen3-14b/checkpoints}"
WORKING_DIR="${WORKING_DIR:-${GCS_BUCKET}/omni-gemma3-qwen3-14b/multimodal/3stage_pipeline/${EXP_NAME}}"

# Base Storage Buckets & Initial Checkpoints
VISION_SOURCE_CKPT="${VISION_SOURCE_CKPT:-${CKPT_DIR}/gemma3-4b_converted/0/items}"
LLM_SOURCE_CKPT="${LLM_SOURCE_CKPT:-${CKPT_DIR}/qwen3-14b_converted/0/items}"
STITCHED_CKPT="${STITCHED_CKPT:-${CKPT_DIR}/omni_stitched_gemma3-4b_qwen3-14b/0/items}"

# Pre-sharded Dataset Storage Paths (GCS)
CHARTNET_DATASET_DIR="${CHARTNET_DATASET_DIR:-${GCS_BUCKET}/datasets/chartnet_sharded}"
CHARTQA_DATASET_DIR="${CHARTQA_DATASET_DIR:-${GCS_BUCKET}/datasets/chartqa_shuffled}"

# Stage 1: ChartNet Summary Alignment
STAGE1_OUTPUT_DIR="${WORKING_DIR}/stage1_chartnet_summary"
STAGE1_RUN_NAME="${EXP_NAME}_stage1_summary"
STAGE1_FINAL_CKPT="${STAGE1_OUTPUT_DIR}/${STAGE1_RUN_NAME}/checkpoints/2171/items"

# Stage 2: ChartNet CSV Table Alignment
STAGE2_OUTPUT_DIR="${WORKING_DIR}/stage2_chartnet_csv"
STAGE2_RUN_NAME="${EXP_NAME}_stage2_csv"
STAGE2_FINAL_CKPT="${STAGE2_OUTPUT_DIR}/${STAGE2_RUN_NAME}/checkpoints/2171/items"

# Stage 3: ChartQA Visual QA SFT
STAGE3_OUTPUT_DIR="${WORKING_DIR}/stage3_chartqa_sft"
STAGE3_RUN_NAME="${EXP_NAME}_stage3_sft"
STAGE3_FINAL_CKPT="${STAGE3_OUTPUT_DIR}/${STAGE3_RUN_NAME}/checkpoints/1104/items"

# XPK Cluster Configuration
XPK_CLUSTER="${XPK_CLUSTER:-v4-128-bodaborg-us-central2-b}"
XPK_PROJECT="${XPK_PROJECT:-cloud-tpu-multipod-dev}"
XPK_ZONE="${XPK_ZONE:-us-central2-b}"
XPK_DEVICE_TYPE="${XPK_DEVICE_TYPE:-v4-128}"
XPK_NUM_SLICES="${XPK_NUM_SLICES:-1}"
XPK_BASE_DOCKER_IMAGE="${XPK_BASE_DOCKER_IMAGE:-gcr.io/tpu-prod-env-multipod/maxtext_base_image:latest}"

USER_PREFIX="${USER_PREFIX:-user}"
TIMESTAMP=$(date +%m%d-%H%M)


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAXTEXT_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || (cd "${SCRIPT_DIR}/../../../../.." && pwd))"

# ------------------------------------------------------------------------------
# 1. Prepare Checkpoints: Download, Convert (scan_layers=true), and Stitch
# ------------------------------------------------------------------------------
prepare_ckpt() {
  echo "=================================================================="
  echo ">>> [PREPARE] Downloading & Converting Gemma3-4B + Qwen3-14B Checkpoints"
  echo "=================================================================="
  HF_TOKEN="${HF_TOKEN}" BASE_OUTPUT_DIRECTORY="${CKPT_DIR}" "${SCRIPT_DIR}/prepare_checkpoint.sh"
}

stitch_ckpt() {
  echo "=================================================================="
  echo ">>> [STITCH] Stitching Gemma3-4B Vision + Qwen3-14B LLM"
  echo ">>> Vision Source: ${VISION_SOURCE_CKPT}"
  echo ">>> LLM Source:    ${LLM_SOURCE_CKPT}"
  echo ">>> Output Path:   ${STITCHED_CKPT}"
  echo "=================================================================="

  (
    cd "${MAXTEXT_ROOT}"
    HF_TOKEN="${HF_TOKEN}" HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}" JAX_PLATFORMS=cpu python3 -m maxtext.experimental.omni_poc.utils.stitch_checkpoint \
      "${SCRIPT_DIR}/maxtext-omni-gemma3-qwen3-14b.yml" \
      "hf_access_token=${HF_TOKEN}" \
      "vision_load_path=${VISION_SOURCE_CKPT}" \
      "llm_load_path=${LLM_SOURCE_CKPT}" \
      "stitched_output_path=${STITCHED_CKPT}"
  )
}

# ------------------------------------------------------------------------------
# 2. Stage 1: ChartNet Summary Pretraining on XPK
# ------------------------------------------------------------------------------
stage1_xpk() {
  local input_ckpt="${1:-${STITCHED_CKPT}}"
  local workload_name="${USER_PREFIX}-14b-s1-${TIMESTAMP}"

  echo "=================================================================="
  echo ">>> [XPK STAGE 1 - ChartNet Summary 14B] Submitting Workload: ${workload_name}"
  echo ">>> Input Checkpoint: ${input_ckpt}"
  echo ">>> Working Dir:      ${STAGE1_OUTPUT_DIR}/${STAGE1_RUN_NAME}"
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
      --command "export MEGASCALE_NUM_SLICES=${XPK_NUM_SLICES} && export TMPDIR=/dev/shm && export PYTHONPATH=src:\${PYTHONPATH:-} && export HF_HOME=/dev/shm/huggingface && export HF_TOKEN=${HF_TOKEN} && export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} && python3 -m pip install --no-cache-dir -U 'orbax-checkpoint>=0.12.4' && gcloud storage cp -r ${CHARTNET_DATASET_DIR} /dev/shm/ && python3 -m maxtext.experimental.omni_poc.train_sft_omni src/maxtext/experimental/omni_14b/pretrain-omni-gemma3-qwen3-14b-chartnet-xpk-128.yml load_parameters_path=${input_ckpt} base_output_directory=${STAGE1_OUTPUT_DIR} run_name=${STAGE1_RUN_NAME} hf_access_token=${HF_TOKEN} scan_layers=true grain_worker_count=0"
  )
}

# ------------------------------------------------------------------------------
# 3. Stage 2: ChartNet CSV Pretraining on XPK
# ------------------------------------------------------------------------------
stage2_xpk() {
  local input_ckpt="${1:-${STAGE1_FINAL_CKPT}}"
  local workload_name="${USER_PREFIX}-14b-s2-${TIMESTAMP}"

  echo "=================================================================="
  echo ">>> [XPK STAGE 2 - ChartNet CSV 14B] Submitting Workload: ${workload_name}"
  echo ">>> Input Checkpoint: ${input_ckpt}"
  echo ">>> Working Dir:      ${STAGE2_OUTPUT_DIR}/${STAGE2_RUN_NAME}"
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
      --command "export MEGASCALE_NUM_SLICES=${XPK_NUM_SLICES} && export TMPDIR=/dev/shm && export PYTHONPATH=src:\${PYTHONPATH:-} && export HF_HOME=/dev/shm/huggingface && export HF_TOKEN=${HF_TOKEN} && export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} && python3 -m pip install --no-cache-dir -U 'orbax-checkpoint>=0.12.4' && gcloud storage cp -r ${CHARTNET_DATASET_DIR} /dev/shm/ && python3 -m maxtext.experimental.omni_poc.train_sft_omni src/maxtext/experimental/omni_14b/pretrain-omni-gemma3-qwen3-14b-chartnet-xpk-128-csv.yml load_parameters_path=${input_ckpt} base_output_directory=${STAGE2_OUTPUT_DIR} run_name=${STAGE2_RUN_NAME} hf_access_token=${HF_TOKEN} scan_layers=true grain_worker_count=0"
  )
}

# ------------------------------------------------------------------------------
# 4. Stage 3: ChartQA SFT on XPK
# ------------------------------------------------------------------------------
stage3_xpk() {
  local input_ckpt="${1:-${STAGE2_FINAL_CKPT}}"
  local workload_name="${USER_PREFIX}-14b-s3-${TIMESTAMP}"

  echo "=================================================================="
  echo ">>> [XPK STAGE 3 - ChartQA SFT 14B (Projector Only)] Submitting Workload: ${workload_name}"
  echo ">>> Input Checkpoint: ${input_ckpt}"
  echo ">>> Working Dir:      ${STAGE3_OUTPUT_DIR}/${STAGE3_RUN_NAME}"
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
      --command "export MEGASCALE_NUM_SLICES=${XPK_NUM_SLICES} && export TMPDIR=/dev/shm && export PYTHONPATH=src:\${PYTHONPATH:-} && export HF_HOME=/dev/shm/huggingface && export HF_TOKEN=${HF_TOKEN} && export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} && python3 -m pip install --no-cache-dir -U 'orbax-checkpoint>=0.12.4' && gcloud storage cp -r ${CHARTQA_DATASET_DIR} /dev/shm/ && python3 -m maxtext.experimental.omni_poc.train_sft_omni src/maxtext/experimental/omni_14b/sft-omni-gemma3-qwen3-14b-xpk-128.yml load_parameters_path=${input_ckpt} base_output_directory=${STAGE3_OUTPUT_DIR} run_name=${STAGE3_RUN_NAME} hf_access_token=${HF_TOKEN} scan_layers=true grain_worker_count=0"
  )
}

# ------------------------------------------------------------------------------
# 5. Full 3-Stage Pipeline on XPK (Stage 1 -> Stage 2 -> Stage 3 in ONE Job)
# ------------------------------------------------------------------------------
pipeline_xpk() {
  local workload_name="${USER_PREFIX}-14b-3stage-${TIMESTAMP}"

  echo "=================================================================="
  echo ">>> [XPK FULL 3-STAGE PIPELINE 14B] Submitting Workload: ${workload_name}"
  echo ">>> Base Working Dir: ${WORKING_DIR}"
  echo ">>> Stage 1: ChartNet Summary Alignment (2,172 steps -> ${STAGE1_FINAL_CKPT})"
  echo ">>> Stage 2: ChartNet CSV Table Grounding (2,172 steps -> ${STAGE2_FINAL_CKPT})"
  echo ">>> Stage 3: ChartQA SFT                (1,105 steps -> ${STAGE3_FINAL_CKPT})"
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
      --command "export MEGASCALE_NUM_SLICES=${XPK_NUM_SLICES} && export TMPDIR=/dev/shm && export PYTHONPATH=src:\${PYTHONPATH:-} && export HF_HOME=/dev/shm/huggingface && export HF_TOKEN=${HF_TOKEN} && export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} && python3 -m pip install --no-cache-dir -U 'orbax-checkpoint>=0.12.4' && gcloud storage cp -r ${CHARTNET_DATASET_DIR} /dev/shm/ && gcloud storage cp -r ${CHARTQA_DATASET_DIR} /dev/shm/ && echo '=== Stage 1: ChartNet Summary (14B) ===' && python3 -m maxtext.experimental.omni_poc.train_sft_omni src/maxtext/experimental/omni_14b/pretrain-omni-gemma3-qwen3-14b-chartnet-xpk-128.yml load_parameters_path=${STITCHED_CKPT} base_output_directory=${STAGE1_OUTPUT_DIR} run_name=${STAGE1_RUN_NAME} hf_access_token=${HF_TOKEN} scan_layers=true grain_worker_count=0 && echo '=== Stage 2: ChartNet CSV (14B) ===' && python3 -m maxtext.experimental.omni_poc.train_sft_omni src/maxtext/experimental/omni_14b/pretrain-omni-gemma3-qwen3-14b-chartnet-xpk-128-csv.yml load_parameters_path=${STAGE1_FINAL_CKPT} base_output_directory=${STAGE2_OUTPUT_DIR} run_name=${STAGE2_RUN_NAME} hf_access_token=${HF_TOKEN} scan_layers=true grain_worker_count=0 && echo '=== Stage 3: ChartQA SFT (14B) ===' && python3 -m maxtext.experimental.omni_poc.train_sft_omni src/maxtext/experimental/omni_14b/sft-omni-gemma3-qwen3-14b-xpk-128.yml load_parameters_path=${STAGE2_FINAL_CKPT} base_output_directory=${STAGE3_OUTPUT_DIR} run_name=${STAGE3_RUN_NAME} hf_access_token=${HF_TOKEN} scan_layers=true grain_worker_count=0"
  )

  echo ""
  echo ">>> 3-Stage Pipeline 14B Workload submitted: ${workload_name}"
  echo "=================================================================="
}

# ------------------------------------------------------------------------------
# 6. Evaluation
# ------------------------------------------------------------------------------
eval_sft() {
  local ckpt_path="${1:-${STAGE3_FINAL_CKPT}}"
  echo "=================================================================="
  echo ">>> [EVAL] Evaluating Stage 3 SFT 14B Checkpoint: ${ckpt_path}"
  echo "=================================================================="

  MEGASCALE_NUM_SLICES=1 HF_TOKEN="${HF_TOKEN}" HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}" python3 -m maxtext.experimental.omni_poc.eval_sft_omni \
    "${SCRIPT_DIR}/sft-omni-gemma3-qwen3-14b-xpk-128.yml" \
    "load_parameters_path=${ckpt_path}" \
    "hf_access_token=${HF_TOKEN}" \
    "hf_path=HuggingFaceM4/ChartQA" \
    --ckpt_type=sft \
    --hf_eval_split=test \
    --num_examples=100
}

# ------------------------------------------------------------------------------
# Dispatcher
# ------------------------------------------------------------------------------
case "$ACTION" in
  prepare|convert|prepare_ckpt)
    prepare_ckpt
    ;;
  stitch)
    stitch_ckpt
    ;;
  stage1)
    stage1_xpk "${2:-}"
    ;;
  stage2)
    stage2_xpk "${2:-}"
    ;;
  stage3)
    stage3_xpk "${2:-}"
    ;;
  pipeline)
    pipeline_xpk
    ;;
  eval)
    eval_sft "${2:-}"
    ;;
  list)
    xpk workload list --cluster "${XPK_CLUSTER}" --project "${XPK_PROJECT}" --zone "${XPK_ZONE}"
    ;;
  status)
    kubectl get pods -l jobset.sigs.k8s.io/jobset-name="${2}" -w
    ;;
  logs)
    pod_name=$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name="${2}" -o jsonpath='{.items[0].metadata.name}')
    kubectl logs "${pod_name}" -c jax-tpu -f
    ;;
  delete)
    xpk workload delete --workload "${2}" --cluster "${XPK_CLUSTER}" --project "${XPK_PROJECT}" --zone "${XPK_ZONE}"
    ;;
  help|*)
    echo "Usage: $0 <prepare|stitch|stage1|stage2|stage3|pipeline|eval|list|status|logs|delete> [checkpoint_path|workload_name]"
    ;;
esac
