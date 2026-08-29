#!/bin/bash
# ==============================================================================
# Omni Qwen3-VL-4B Vision + Qwen3-14B LLM 3-Stage Training Pipeline on XPK
# Stage 1: ChartNet CSV Table Alignment (2,172 steps)
# Stage 2: ChartNet Dense Summary Alignment (2,172 steps)
# Stage 3: ChartQA Visual QA SFT (1,105 steps)
# ==============================================================================

set -e

ACTION="${1:-help}"

# Global Experiment / Model Name
EXP_NAME="${EXP_NAME:-omni_qwen3_vl_14b_3stage}"

# Base Storage Buckets & Checkpoints
GCS_BUCKET="${GCS_BUCKET:-gs://yuchenhou-maxtext-logs}"
VISION_SOURCE_CKPT="${VISION_SOURCE_CKPT:-${GCS_BUCKET}/checkpoints/qwen3-vl-4b-processor/0/items}"
LLM_SOURCE_CKPT="${LLM_SOURCE_CKPT:-${GCS_BUCKET}/omni_checkpoints/qwen3-14b_unscanned/0/items}"
STITCHED_CKPT="${STITCHED_CKPT:-${GCS_BUCKET}/omni_checkpoints/${EXP_NAME}/0/items}"

# Stage 1: ChartNet Summary
STAGE1_OUTPUT_DIR="${GCS_BUCKET}/experimental/${EXP_NAME}/stage1_chartnet_summary"
STAGE1_RUN_NAME="${EXP_NAME}_stage1_summary"
STAGE1_FINAL_CKPT="${STAGE1_OUTPUT_DIR}/${STAGE1_RUN_NAME}/checkpoints/2171/items"

# Stage 2: ChartNet CSV
STAGE2_OUTPUT_DIR="${GCS_BUCKET}/experimental/${EXP_NAME}/stage2_chartnet_csv"
STAGE2_RUN_NAME="${EXP_NAME}_stage2_csv"
STAGE2_FINAL_CKPT="${STAGE2_OUTPUT_DIR}/${STAGE2_RUN_NAME}/checkpoints/2171/items"

# Stage 3: ChartQA SFT
STAGE3_OUTPUT_DIR="${GCS_BUCKET}/experimental/${EXP_NAME}/stage3_chartqa_sft"
STAGE3_RUN_NAME="${EXP_NAME}_stage3_sft"
STAGE3_FINAL_CKPT="${STAGE3_OUTPUT_DIR}/${STAGE3_RUN_NAME}/checkpoints/1104/items"

# XPK Cluster Configuration
XPK_CLUSTER="${XPK_CLUSTER:-v4-128-bodaborg-us-central2-b}"
XPK_PROJECT="${XPK_PROJECT:-cloud-tpu-multipod-dev}"
XPK_ZONE="${XPK_ZONE:-us-central2-b}"
XPK_DEVICE_TYPE="${XPK_DEVICE_TYPE:-v4-128}"
XPK_NUM_SLICES="${XPK_NUM_SLICES:-1}"
XPK_BASE_DOCKER_IMAGE="${XPK_BASE_DOCKER_IMAGE:-gcr.io/tpu-prod-env-multipod/maxtext_base_image:latest}"

USER_PREFIX="${USER_PREFIX:-yuchenhou}"
HF_TOKEN="${HF_TOKEN:-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}"
TIMESTAMP=$(date +%m%d-%H%M)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAXTEXT_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || (cd "${SCRIPT_DIR}/../../../../.." && pwd))"

# ------------------------------------------------------------------------------
# 1. Stitch Checkpoint (CPU-bound: Qwen3-VL-4B ViT + Qwen3-14B LLM Backbone)
# ------------------------------------------------------------------------------
stitch_ckpt() {
  echo "=================================================================="
  echo ">>> [STITCH] Stitching Qwen3-VL-4B Vision + Qwen3-14B LLM (Unscanned)"
  echo ">>> Vision Source: ${VISION_SOURCE_CKPT}"
  echo ">>> LLM Source:    ${LLM_SOURCE_CKPT}"
  echo ">>> Output Path:   ${STITCHED_CKPT}"
  echo "=================================================================="

  (
    cd "${MAXTEXT_ROOT}"
    JAX_PLATFORMS=cpu python3 -m maxtext.experimental.omni_poc.utils.stitch_checkpoint \
      "${SCRIPT_DIR}/maxtext-omni-qwen3-vl-14b.yml" \
      "scan_layers=false" \
      "vision_load_path=${VISION_SOURCE_CKPT}" \
      "llm_load_path=${LLM_SOURCE_CKPT}" \
      "stitched_output_path=${STITCHED_CKPT}"
  )
}

# ------------------------------------------------------------------------------
# 2. Stage 1: ChartNet Summary Pretraining on XPK
# ------------------------------------------------------------------------------
stage1_xpk() {
  local workload_name="${USER_PREFIX}-qwen14b-s1-${TIMESTAMP}"

  echo "=================================================================="
  echo ">>> [XPK STAGE 1 - ChartNet Summary] Submitting Workload: ${workload_name}"
  echo ">>> Input Checkpoint: ${STITCHED_CKPT}"
  echo ">>> Output Dir:       ${STAGE1_OUTPUT_DIR}/${STAGE1_RUN_NAME}"
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
      --command "export TMPDIR=/dev/shm && export PYTHONPATH=src:\${PYTHONPATH:-} && export HF_HOME=/dev/shm/huggingface && export HF_TOKEN=${HF_TOKEN} && python3 -m maxtext.experimental.omni_poc.train_sft_omni src/maxtext/experimental/omni_qwen_14b_3stage/stage1-pretrain-chartnet-summary.yml load_parameters_path=${STITCHED_CKPT} base_output_directory=${STAGE1_OUTPUT_DIR} run_name=${STAGE1_RUN_NAME} scan_layers=false ici_fsdp_parallelism=-1 ici_tensor_parallelism=4 grain_worker_count=0"
  )
}

# ------------------------------------------------------------------------------
# 3. Stage 2: ChartNet CSV Pretraining on XPK
# ------------------------------------------------------------------------------
stage2_xpk() {
  local input_ckpt="${1:-${STAGE1_FINAL_CKPT}}"
  local workload_name="${USER_PREFIX}-qwen14b-s2-${TIMESTAMP}"

  echo "=================================================================="
  echo ">>> [XPK STAGE 2 - ChartNet CSV] Submitting Workload: ${workload_name}"
  echo ">>> Input Checkpoint: ${input_ckpt}"
  echo ">>> Output Dir:       ${STAGE2_OUTPUT_DIR}/${STAGE2_RUN_NAME}"
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
      --command "export TMPDIR=/dev/shm && export PYTHONPATH=src:\${PYTHONPATH:-} && export HF_HOME=/dev/shm/huggingface && export HF_TOKEN=${HF_TOKEN} && python3 -m maxtext.experimental.omni_poc.train_sft_omni src/maxtext/experimental/omni_qwen_14b_3stage/stage2-pretrain-chartnet-csv.yml load_parameters_path=${input_ckpt} base_output_directory=${STAGE2_OUTPUT_DIR} run_name=${STAGE2_RUN_NAME} scan_layers=false ici_fsdp_parallelism=-1 ici_tensor_parallelism=4 grain_worker_count=0"
  )
}

# ------------------------------------------------------------------------------
# 4. Stage 3: ChartQA SFT on XPK
# ------------------------------------------------------------------------------
stage3_xpk() {
  local input_ckpt="${1:-${STAGE2_FINAL_CKPT}}"
  local workload_name="${USER_PREFIX}-qwen14b-s3-${TIMESTAMP}"

  echo "=================================================================="
  echo ">>> [XPK STAGE 3 - ChartQA SFT (Projector Only)] Submitting Workload: ${workload_name}"
  echo ">>> Input Checkpoint: ${input_ckpt}"
  echo ">>> Output Dir:       ${STAGE3_OUTPUT_DIR}/${STAGE3_RUN_NAME}"
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
      --command "export TMPDIR=/dev/shm && export PYTHONPATH=src:\${PYTHONPATH:-} && export HF_HOME=/dev/shm/huggingface && export HF_TOKEN=${HF_TOKEN} && python3 -m maxtext.experimental.omni_poc.train_sft_omni src/maxtext/experimental/omni_qwen_14b_3stage/stage3-sft-chartqa.yml load_parameters_path=${input_ckpt} base_output_directory=${STAGE3_OUTPUT_DIR} run_name=${STAGE3_RUN_NAME} scan_layers=false ici_fsdp_parallelism=-1 ici_tensor_parallelism=4 grain_worker_count=0"
  )
}

stage3_vit_xpk() {
  local input_ckpt="${1:-${STAGE2_FINAL_CKPT}}"
  local run_name="${EXP_NAME}_stage3_vit_tune"
  local workload_name="${USER_PREFIX}-qwen14b-s3vit-${TIMESTAMP}"

  echo "=================================================================="
  echo ">>> [XPK STAGE 3 - ChartQA SFT (ViT + Projector Tuned, 8 Epochs)] Submitting Workload: ${workload_name}"
  echo ">>> Input Checkpoint: ${input_ckpt}"
  echo ">>> Output Dir:       ${STAGE3_OUTPUT_DIR}/${run_name}"
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
      --command "export TMPDIR=/dev/shm && export PYTHONPATH=src:\${PYTHONPATH:-} && export HF_HOME=/dev/shm/huggingface && export HF_TOKEN=${HF_TOKEN} && python3 -m maxtext.experimental.omni_poc.train_sft_omni src/maxtext/experimental/omni_qwen_14b_3stage/stage3-sft-chartqa-vit-tune.yml load_parameters_path=${input_ckpt} base_output_directory=${STAGE3_OUTPUT_DIR} run_name=${run_name} scan_layers=false ici_fsdp_parallelism=-1 ici_tensor_parallelism=4 grain_worker_count=0"
  )
}

# ------------------------------------------------------------------------------
# 5. Full 3-Stage Pipeline on XPK (Stage 1 -> Stage 2 -> Stage 3 in ONE Job)
# ------------------------------------------------------------------------------
pipeline_xpk() {
  local workload_name="${USER_PREFIX}-qwen14b-3stage-${TIMESTAMP}"

  echo "=================================================================="
  echo ">>> [XPK FULL 3-STAGE PIPELINE] Submitting Workload: ${workload_name}"
  echo ">>> Stage 1: ChartNet Summary Alignment (2,172 steps -> checkpoint 2171)"
  echo ">>> Stage 2: ChartNet CSV Table Grounding (2,172 steps -> checkpoint 2171)"
  echo ">>> Stage 3: ChartQA SFT                (1,105 steps -> checkpoint 1104)"
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
      --command "export TMPDIR=/dev/shm && export PYTHONPATH=src:\${PYTHONPATH:-} && export HF_HOME=/dev/shm/huggingface && export HF_TOKEN=${HF_TOKEN} && echo '=== Stage 1: ChartNet Summary ===' && python3 -m maxtext.experimental.omni_poc.train_sft_omni src/maxtext/experimental/omni_qwen_14b_3stage/stage1-pretrain-chartnet-summary.yml load_parameters_path=${STITCHED_CKPT} base_output_directory=${STAGE1_OUTPUT_DIR} run_name=${STAGE1_RUN_NAME} scan_layers=false ici_fsdp_parallelism=-1 ici_tensor_parallelism=4 grain_worker_count=0 && echo '=== Stage 2: ChartNet CSV ===' && python3 -m maxtext.experimental.omni_poc.train_sft_omni src/maxtext/experimental/omni_qwen_14b_3stage/stage2-pretrain-chartnet-csv.yml load_parameters_path=${STAGE1_FINAL_CKPT} base_output_directory=${STAGE2_OUTPUT_DIR} run_name=${STAGE2_RUN_NAME} scan_layers=false ici_fsdp_parallelism=-1 ici_tensor_parallelism=4 grain_worker_count=0 && echo '=== Stage 3: ChartQA SFT ===' && python3 -m maxtext.experimental.omni_poc.train_sft_omni src/maxtext/experimental/omni_qwen_14b_3stage/stage3-sft-chartqa.yml load_parameters_path=${STAGE2_FINAL_CKPT} base_output_directory=${STAGE3_OUTPUT_DIR} run_name=${STAGE3_RUN_NAME} scan_layers=false ici_fsdp_parallelism=-1 ici_tensor_parallelism=4 grain_worker_count=0"
  )

  echo ""
  echo ">>> 3-Stage Pipeline Workload submitted: ${workload_name}"
  echo "=================================================================="
}

# ------------------------------------------------------------------------------
# 6. Local TPU VM Evaluation & Verification
# ------------------------------------------------------------------------------
eval_sft() {
  local ckpt_path="${1:-${STAGE3_OUTPUT_DIR}/${STAGE3_RUN_NAME}/checkpoints/1104/items}"
  echo "=================================================================="
  echo ">>> [EVAL] Evaluating Final Stage 3 SFT Checkpoint: ${ckpt_path}"
  echo "=================================================================="

  python3 -m maxtext.experimental.omni_poc.eval_sft_omni \
    "${SCRIPT_DIR}/stage3-sft-chartqa.yml" \
    "load_parameters_path=${ckpt_path}" \
    --ckpt_type=sft \
    --num_examples=100
}

test_stitched_model() {
  echo "=================================================================="
  echo ">>> [TEST STITCH] Verifying 3-Stage Stitched Model Initializations"
  echo ">>> Stitched Path: ${STITCHED_CKPT}"
  echo "=================================================================="

  (
    cd "${MAXTEXT_ROOT}"
    python3 "${SCRIPT_DIR}/test_stitch_qwen_14b.py" \
      --config_path="${SCRIPT_DIR}/maxtext-omni-qwen3-vl-14b.yml" \
      --vision_checkpoint="${VISION_SOURCE_CKPT}" \
      --llm_checkpoint="${LLM_SOURCE_CKPT}" \
      --output_checkpoint="${STITCHED_CKPT}"
  )
}

# ------------------------------------------------------------------------------
# Dispatcher
# ------------------------------------------------------------------------------
case "$ACTION" in
  stitch)
    stitch_ckpt
    ;;
  test_stitch|test)
    test_stitched_model
    ;;
  stage1)
    stage1_xpk
    ;;
  stage2)
    stage2_xpk "${2:-}"
    ;;
  stage3)
    stage3_xpk "${2:-}"
    ;;
  stage3_vit|s3vit)
    stage3_vit_xpk "${2:-}"
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
    echo "Usage: $0 <stitch|test_stitch|stage1|stage2|stage3|pipeline|eval|list|status|logs|delete> [checkpoint_path|workload_name]"
    ;;
esac
