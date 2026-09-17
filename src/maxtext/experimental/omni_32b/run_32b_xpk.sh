#!/bin/bash
# ==============================================================================
# Omni-Gemma3-Qwen3-32B 3-Stage Training & Pipeline Orchestration Script
# Model: Gemma 3 4B Vision Tower + Qwen 3 32B LLM Decoder + Custom 3-Layer MLP Projector
# Hardware: Cloud TPU v4-128 (128 chips, 1 slice) via XPK
# ==============================================================================

set -euo pipefail

# ------------------------------------------------------------------------------
# 1. Environment & Global Variables
# ------------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAXTEXT_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || (cd "${SCRIPT_DIR}/../../../../.." && pwd))"

ACTION="${1:-help}"
TIMESTAMP="$(date +%m%d-%H%M%S)"
USER_PREFIX="${USER_PREFIX:-${USER:-user}}"
USER_PREFIX="$(echo "${USER_PREFIX}" | tr '[:upper:]' '[:lower:]' | tr '_' '-' | sed 's/-google-com$//' | tr -cd 'a-z0-9-')"
USER_PREFIX="${USER_PREFIX:0:10}"

: "${HF_TOKEN:?Error: HF_TOKEN is not set. Please run: export HF_TOKEN=hf_...}"
: "${GCS_BUCKET:?Error: GCS_BUCKET is not set. Please run: export GCS_BUCKET=gs://your-bucket}"
export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

# GCS Storage Directories
EXP_NAME="${EXP_NAME:-omni_gemma3_qwen3_32b_3stage}"
CKPT_DIR="${CKPT_DIR:-${GCS_BUCKET}/omni-gemma3-qwen3-32b/checkpoints}"
WORKING_DIR="${WORKING_DIR:-${GCS_BUCKET}/omni-gemma3-qwen3-32b/multimodal/3stage_pipeline/${EXP_NAME}}"

# Base Storage Buckets & Initial Checkpoints
VISION_SOURCE_CKPT="${VISION_SOURCE_CKPT:-${CKPT_DIR}/gemma3-4b_converted/0/items}"
LLM_SOURCE_CKPT="${LLM_SOURCE_CKPT:-${CKPT_DIR}/qwen3-32b_converted/0/items}"
STITCHED_CKPT="${STITCHED_CKPT:-${CKPT_DIR}/omni_stitched_gemma3-4b_qwen3-32b/0/items}"

# Pre-sharded Dataset Paths on GCS
CHARTNET_DATASET_DIR="gs://${GCS_BUCKET}/datasets/chartnet_sharded"
CHARTQA_DATASET_DIR="gs://${GCS_BUCKET}/datasets/chartqa_shuffled"

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
STAGE3_FINAL_CKPT="${STAGE3_OUTPUT_DIR}/${STAGE3_RUN_NAME}/checkpoints/1767/items"

# XPK Cluster Configuration
XPK_CLUSTER="${XPK_CLUSTER:-v4-128-bodaborg-us-central2-b}"
XPK_ZONE="${XPK_ZONE:-us-central2-b}"
XPK_PROJECT="${XPK_PROJECT:-tpu-prod-env-multipod}"
XPK_DEVICE_TYPE="${XPK_DEVICE_TYPE:-v4-128}"
XPK_NUM_SLICES="${XPK_NUM_SLICES:-1}"
XPK_BASE_DOCKER_IMAGE="${XPK_BASE_DOCKER_IMAGE:-gcr.io/tpu-prod-env-multipod/maxtext_base_image:latest}"

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
log_header() {
  echo "=================================================================="
  echo ">>> $1"
  echo "=================================================================="
}

prepare_ckpt() {
  log_header "Running Full Checkpoint Preparation (Download -> Convert -> Stitch)"
  bash "${SCRIPT_DIR}/prepare_checkpoint.sh"
}

stitch_only() {
  local vision_in="${VISION_SOURCE_CKPT}"
  local llm_in="${LLM_SOURCE_CKPT}"
  local stitched_out="${STITCHED_CKPT}"

  log_header "Stitching Vision + LLM into Omni 32B Checkpoint"
  echo ">>> Vision Checkpoint:  ${vision_in}"
  echo ">>> LLM Checkpoint:     ${llm_in}"
  echo ">>> Stitched Output:    ${stitched_out}"

  (
    cd "${MAXTEXT_ROOT}"
    JAX_PLATFORMS=cpu python3 -m maxtext.experimental.omni_pipeline.utils.stitch_checkpoint \
      "${SCRIPT_DIR}/maxtext-omni-gemma3-qwen3-32b.yml" \
      "hf_access_token=${HF_TOKEN}" \
      "vision_load_path=${vision_in}" \
      "llm_load_path=${llm_in}" \
      "stitched_output_path=${stitched_out}" \
      "checkpoint_storage_concurrent_gb=64"
  )
}

# ------------------------------------------------------------------------------
# 2. Stage 1: ChartNet Summary Alignment on XPK
# ------------------------------------------------------------------------------
stage1_xpk() {
  local input_ckpt="${1:-${STITCHED_CKPT}}"
  local workload_name="${USER_PREFIX}-32b-s1-${TIMESTAMP}"

  echo "=================================================================="
  echo ">>> [XPK STAGE 1 - ChartNet Summary 32B] Submitting Workload: ${workload_name}"
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
      --command "export MEGASCALE_NUM_SLICES=${XPK_NUM_SLICES} && export TMPDIR=/dev/shm && export PYTHONPATH=src:\${PYTHONPATH:-} && export HF_HOME=/dev/shm/huggingface && export HF_TOKEN=${HF_TOKEN} && export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} && python3 -m pip install --no-cache-dir -U 'orbax-checkpoint>=0.12.4' && gcloud storage cp -r ${CHARTNET_DATASET_DIR} /dev/shm/ && python3 -m maxtext.experimental.omni_pipeline.train_sft_omni src/maxtext/experimental/omni_32b/pretrain-omni-gemma3-qwen3-32b-chartnet-xpk-128.yml load_parameters_path=${input_ckpt} base_output_directory=${STAGE1_OUTPUT_DIR} run_name=${STAGE1_RUN_NAME} hf_access_token=${HF_TOKEN} scan_layers=true grain_worker_count=0 allow_split_physical_axes=true"
  )
}

# ------------------------------------------------------------------------------
# 3. Stage 2: ChartNet CSV Pretraining on XPK
# ------------------------------------------------------------------------------
stage2_xpk() {
  local input_ckpt="${1:-${STAGE1_FINAL_CKPT}}"
  local workload_name="${USER_PREFIX}-32b-s2-${TIMESTAMP}"

  echo "=================================================================="
  echo ">>> [XPK STAGE 2 - ChartNet CSV 32B] Submitting Workload: ${workload_name}"
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
      --command "export MEGASCALE_NUM_SLICES=${XPK_NUM_SLICES} && export TMPDIR=/dev/shm && export PYTHONPATH=src:\${PYTHONPATH:-} && export HF_HOME=/dev/shm/huggingface && export HF_TOKEN=${HF_TOKEN} && export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} && python3 -m pip install --no-cache-dir -U 'orbax-checkpoint>=0.12.4' && gcloud storage cp -r ${CHARTNET_DATASET_DIR} /dev/shm/ && python3 -m maxtext.experimental.omni_pipeline.train_sft_omni src/maxtext/experimental/omni_32b/pretrain-omni-gemma3-qwen3-32b-chartnet-xpk-128-csv.yml load_parameters_path=${input_ckpt} base_output_directory=${STAGE2_OUTPUT_DIR} run_name=${STAGE2_RUN_NAME} hf_access_token=${HF_TOKEN} scan_layers=true grain_worker_count=0 allow_split_physical_axes=true"
  )
}

# ------------------------------------------------------------------------------
# 4. Stage 3: ChartQA SFT on XPK
# ------------------------------------------------------------------------------
stage3_xpk() {
  local input_ckpt="${1:-${STAGE2_FINAL_CKPT}}"
  local workload_name="${USER_PREFIX}-32b-s3-${TIMESTAMP}"

  echo "=================================================================="
  echo ">>> [XPK STAGE 3 - ChartQA SFT 32B (Projector Only)] Submitting Workload: ${workload_name}"
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
      --command "export MEGASCALE_NUM_SLICES=${XPK_NUM_SLICES} && export TMPDIR=/dev/shm && export PYTHONPATH=src:\${PYTHONPATH:-} && export HF_HOME=/dev/shm/huggingface && export HF_TOKEN=${HF_TOKEN} && export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} && python3 -m pip install --no-cache-dir -U 'orbax-checkpoint>=0.12.4' && gcloud storage cp -r ${CHARTQA_DATASET_DIR} /dev/shm/ && python3 -m maxtext.experimental.omni_pipeline.train_sft_omni src/maxtext/experimental/omni_32b/sft-omni-gemma3-qwen3-32b-xpk-128.yml load_parameters_path=${input_ckpt} base_output_directory=${STAGE3_OUTPUT_DIR} run_name=${STAGE3_RUN_NAME} hf_access_token=${HF_TOKEN} scan_layers=true grain_worker_count=0 allow_split_physical_axes=true"
  )
}

# ------------------------------------------------------------------------------
# 5. Full 3-Stage Pipeline on XPK (Stage 1 -> Stage 2 -> Stage 3 in ONE Job)
# ------------------------------------------------------------------------------
pipeline_xpk() {
  local workload_name="${USER_PREFIX}-32b-3stage-${TIMESTAMP}"

  echo "=================================================================="
  echo ">>> [XPK FULL 3-STAGE PIPELINE 32B] Submitting Workload: ${workload_name}"
  echo ">>> Base Working Dir: ${WORKING_DIR}"
  echo ">>> Stage 1: ChartNet Summary Alignment (2,172 steps -> ${STAGE1_FINAL_CKPT})"
  echo ">>> Stage 2: ChartNet CSV Table Grounding (2,172 steps -> ${STAGE2_FINAL_CKPT})"
  echo ">>> Stage 3: ChartQA SFT                (1,768 steps -> ${STAGE3_FINAL_CKPT})"
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
      --command "export MEGASCALE_NUM_SLICES=${XPK_NUM_SLICES} && export TMPDIR=/dev/shm && export PYTHONPATH=src:\${PYTHONPATH:-} && export HF_HOME=/dev/shm/huggingface && export HF_TOKEN=${HF_TOKEN} && export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} && python3 -m pip install --no-cache-dir -U 'orbax-checkpoint>=0.12.4' && gcloud storage cp -r ${CHARTNET_DATASET_DIR} /dev/shm/ && gcloud storage cp -r ${CHARTQA_DATASET_DIR} /dev/shm/ && echo '=== Stage 1: ChartNet Summary (32B) ===' && python3 -m maxtext.experimental.omni_pipeline.train_sft_omni src/maxtext/experimental/omni_32b/pretrain-omni-gemma3-qwen3-32b-chartnet-xpk-128.yml load_parameters_path=${STITCHED_CKPT} base_output_directory=${STAGE1_OUTPUT_DIR} run_name=${STAGE1_RUN_NAME} hf_access_token=${HF_TOKEN} scan_layers=true grain_worker_count=0 allow_split_physical_axes=true && echo '=== Stage 2: ChartNet CSV (32B) ===' && python3 -m maxtext.experimental.omni_pipeline.train_sft_omni src/maxtext/experimental/omni_32b/pretrain-omni-gemma3-qwen3-32b-chartnet-xpk-128-csv.yml load_parameters_path=${STAGE1_FINAL_CKPT} base_output_directory=${STAGE2_OUTPUT_DIR} run_name=${STAGE2_RUN_NAME} hf_access_token=${HF_TOKEN} scan_layers=true grain_worker_count=0 allow_split_physical_axes=true && echo '=== Stage 3: ChartQA SFT (32B) ===' && python3 -m maxtext.experimental.omni_pipeline.train_sft_omni src/maxtext/experimental/omni_32b/sft-omni-gemma3-qwen3-32b-xpk-128.yml load_parameters_path=${STAGE2_FINAL_CKPT} base_output_directory=${STAGE3_OUTPUT_DIR} run_name=${STAGE3_RUN_NAME} hf_access_token=${HF_TOKEN} scan_layers=true grain_worker_count=0 allow_split_physical_axes=true"
  )

  echo ""
  echo ">>> 3-Stage Pipeline 32B Workload submitted: ${workload_name}"
  echo "=================================================================="
}

# ------------------------------------------------------------------------------
# 6. Evaluation
# ------------------------------------------------------------------------------
eval_sft() {
  local ckpt_path="${1:-${STAGE3_FINAL_CKPT}}"
  ckpt_path="${ckpt_path%/_CHECKPOINT_METADATA}"
  ckpt_path="${ckpt_path/#https:\/\/storage.googleapis.com\//gs:\/\/}"
  if [[ "${ckpt_path}" != */items ]]; then
    ckpt_path="${ckpt_path%/}/items"
  fi
  local ckpt_step="$(basename "$(dirname "${ckpt_path}")")"
  echo "=================================================================="
  echo ">>> [EVAL] Evaluating Stage 3 SFT 32B Checkpoint: ${ckpt_path}"
  echo "=================================================================="

  MEGASCALE_NUM_SLICES=1 HF_TOKEN="${HF_TOKEN}" HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}" python3 -m maxtext.experimental.omni_pipeline.eval_sft_omni \
    "${SCRIPT_DIR}/sft-omni-gemma3-qwen3-32b-xpk-128.yml" \
    "load_parameters_path=${ckpt_path}" \
    "hf_access_token=${HF_TOKEN}" \
    "hf_path=HuggingFaceM4/ChartQA" \
    "run_name=eval_${ckpt_step}" \
    --ckpt_type=sft \
    --hf_eval_split=test \
    --num_examples=-1
}

eval_xpk() {
  local input_ckpt="${1:-${STAGE3_FINAL_CKPT}}"
  input_ckpt="${input_ckpt%/_CHECKPOINT_METADATA}"
  input_ckpt="${input_ckpt/#https:\/\/storage.googleapis.com\//gs:\/\/}"
  if [[ "${input_ckpt}" != */items ]]; then
    input_ckpt="${input_ckpt%/}/items"
  fi
  local ckpt_step="$(basename "$(dirname "${input_ckpt}")")"
  local workload_name="${USER_PREFIX}-32b-eval-${ckpt_step}-${TIMESTAMP}"

  echo "=================================================================="
  echo ">>> [XPK EVAL - ChartQA Test Split 32B] Submitting Workload: ${workload_name}"
  echo ">>> Input Checkpoint: ${input_ckpt}"
  echo ">>> Output Dir:       ${STAGE3_OUTPUT_DIR}"
  echo ">>> Run Name:         eval_${ckpt_step}"
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
      --command "export MEGASCALE_NUM_SLICES=${XPK_NUM_SLICES} && export TMPDIR=/dev/shm && export PYTHONPATH=src:\${PYTHONPATH:-} && export HF_HOME=/dev/shm/huggingface && export HF_TOKEN=${HF_TOKEN} && export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} && python3 -m pip install --no-cache-dir -U 'orbax-checkpoint>=0.12.4' && python3 -m maxtext.experimental.omni_pipeline.eval_sft_omni src/maxtext/experimental/omni_32b/sft-omni-gemma3-qwen3-32b-xpk-128.yml load_parameters_path=${input_ckpt} base_output_directory=${STAGE3_OUTPUT_DIR} run_name=eval_${ckpt_step} hf_access_token=${HF_TOKEN} scan_layers=true allow_split_physical_axes=true max_prefill_predict_length=384 max_target_length=448 --ckpt_type=sft --hf_eval_split=test --num_examples=-1"
  )
}

# ------------------------------------------------------------------------------
# Dispatcher
# ------------------------------------------------------------------------------
case "$ACTION" in
  prepare|convert|prepare_ckpt)
    prepare_ckpt
    ;;
  stitch)
    stitch_only
    ;;
  stage1|s1)
    stage1_xpk "${2:-}"
    ;;
  stage2|s2)
    stage2_xpk "${2:-}"
    ;;
  stage3|s3)
    stage3_xpk "${2:-}"
    ;;
  pipeline|all)
    pipeline_xpk
    ;;
  eval)
    eval_sft "${2:-}"
    ;;
  eval_xpk|eval-xpk)
    eval_xpk "${2:-}"
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
  *)
    echo "Usage: $0 {prepare|stitch|stage1|stage2|stage3|pipeline|eval|eval-xpk|list|status|logs|delete}"
    echo ""
    echo "Commands:"
    echo "  prepare             - Download HF weights, convert (scan_layers=True), and stitch"
    echo "  stitch              - Stitch existing converted Gemma3-4B and Qwen3-32B checkpoints"
    echo "  stage1 [ckpt]       - Launch Stage 1 ChartNet Summary pretraining on XPK"
    echo "  stage2 [ckpt]       - Launch Stage 2 ChartNet CSV pretraining on XPK"
    echo "  stage3 [ckpt]       - Launch Stage 3 ChartQA SFT on XPK"
    echo "  pipeline            - Launch full 3-Stage Pipeline sequentially in a single XPK job"
    echo "  eval [ckpt]         - Run local evaluation with multimodal_eval on ChartQA"
    echo "  eval-xpk [ckpt]     - Launch ChartQA evaluation workload on XPK cluster"
    echo "  list                - List all workloads on the target XPK cluster"
    echo "  status <workload>   - Watch pod status for a workload"
    echo "  logs <workload>     - Tail logs for a workload"
    echo "  delete <workload>   - Delete a workload from XPK"
    exit 1
    ;;
esac
