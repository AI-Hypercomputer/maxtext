#!/bin/bash
# ==============================================================================
# Omni Qwen3-VL-4B Vision + Qwen3-14B LLM Pretrain & SFT Pipeline on XPK (v4-128)
# ==============================================================================

set -e

ACTION="${1:-help}"

# Global Experiment / Model Name
EXP_NAME="${EXP_NAME:-qwen3-vl-14b-3mlp-layernorm}"

# Base Storage Buckets & Checkpoints
GCS_BUCKET="${GCS_BUCKET:-gs://yuchenhou-maxtext-logs}"
VISION_SOURCE_CKPT="${VISION_SOURCE_CKPT:-${GCS_BUCKET}/checkpoints/qwen3-vl-4b-processor/0/items}"
LLM_SOURCE_CKPT="${LLM_SOURCE_CKPT:-${GCS_BUCKET}/omni_checkpoints/qwen3-14b_unscanned/0/items}"
STITCHED_CKPT="${STITCHED_CKPT:-${GCS_BUCKET}/omni_checkpoints/${EXP_NAME}/0/items}"

# Experimental Output Directories
PRETRAIN_OUTPUT_DIR="${PRETRAIN_OUTPUT_DIR:-${GCS_BUCKET}/experimental/${EXP_NAME}/pretrain_chartnet_csv}"
PRETRAIN_RUN_NAME="${PRETRAIN_RUN_NAME:-${EXP_NAME}_pretrain_xpk}"
PRETRAIN_FINAL_CKPT="${PRETRAIN_OUTPUT_DIR}/${PRETRAIN_RUN_NAME}/checkpoints/2171/items"

SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-${GCS_BUCKET}/experimental/${EXP_NAME}/sft_chartqa}"
SFT_RUN_NAME="${SFT_RUN_NAME:-${EXP_NAME}_sft_xpk}"

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
# 2. Stage 1: ChartNet CSV Pretraining on XPK
# ------------------------------------------------------------------------------
pretrain_xpk() {
  local workload_name="${USER_PREFIX}-qwen14b-pretrain-csv-${TIMESTAMP}"

  echo "=================================================================="
  echo ">>> [XPK PRETRAIN - ChartNet CSV] Submitting Workload: ${workload_name}"
  echo ">>> Model:            Qwen 3 VL 4B + Qwen 3 14B"
  echo ">>> Cluster:          ${XPK_CLUSTER} (${XPK_DEVICE_TYPE}, ${XPK_ZONE})"
  echo ">>> Input Checkpoint: ${STITCHED_CKPT}"
  echo ">>> Output Dir:       ${PRETRAIN_OUTPUT_DIR}/${PRETRAIN_RUN_NAME}"
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
      --command "export TMPDIR=/dev/shm && export PYTHONPATH=src:\${PYTHONPATH:-} && export HF_HOME=/dev/shm/huggingface && export HF_TOKEN=${HF_TOKEN} && python3 -m maxtext.experimental.omni_poc.train_sft_omni src/maxtext/experimental/omni_qwen_14b/pretrain-omni-qwen3-vl-14b-chartnet.yml load_parameters_path=${STITCHED_CKPT} base_output_directory=${PRETRAIN_OUTPUT_DIR} run_name=${PRETRAIN_RUN_NAME} scan_layers=false ici_fsdp_parallelism=-1 ici_tensor_parallelism=4 grain_worker_count=0"
  )

  echo ""
  echo ">>> Workload submitted: ${workload_name}"
  echo ">>> To check pod status:"
  echo "    kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${workload_name} -w"
  echo ">>> To stream logs:"
  echo "    kubectl logs \$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${workload_name} -o jsonpath='{.items[0].metadata.name}') -c jax-tpu -f"
  echo "=================================================================="
}

# ------------------------------------------------------------------------------
# 3. Stage 2: ChartQA SFT on XPK
# ------------------------------------------------------------------------------
sft_xpk() {
  local input_ckpt="${1:-${PRETRAIN_FINAL_CKPT}}"
  local workload_name="${USER_PREFIX}-qwen14b-sft-chartqa-${TIMESTAMP}"

  echo "=================================================================="
  echo ">>> [XPK SFT - ChartQA] Submitting Workload: ${workload_name}"
  echo ">>> Model:            Qwen 3 VL 4B + Qwen 3 14B"
  echo ">>> Cluster:          ${XPK_CLUSTER} (${XPK_DEVICE_TYPE}, ${XPK_ZONE})"
  echo ">>> Input Checkpoint: ${input_ckpt}"
  echo ">>> Output Dir:       ${SFT_OUTPUT_DIR}/${SFT_RUN_NAME}"
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
      --command "export TMPDIR=/dev/shm && export PYTHONPATH=src:\${PYTHONPATH:-} && export HF_HOME=/dev/shm/huggingface && export HF_TOKEN=${HF_TOKEN} && python3 -m maxtext.experimental.omni_poc.train_sft_omni src/maxtext/experimental/omni_qwen_14b/sft-omni-qwen3-vl-14b-chartqa.yml load_parameters_path=${input_ckpt} base_output_directory=${SFT_OUTPUT_DIR} run_name=${SFT_RUN_NAME} scan_layers=false ici_fsdp_parallelism=-1 ici_tensor_parallelism=4 grain_worker_count=0"
  )

  echo ""
  echo ">>> Workload submitted: ${workload_name}"
  echo ">>> To check pod status:"
  echo "    kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${workload_name} -w"
  echo ">>> To stream logs:"
  echo "    kubectl logs \$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${workload_name} -o jsonpath='{.items[0].metadata.name}') -c jax-tpu -f"
  echo "=================================================================="
}

# ------------------------------------------------------------------------------
# 4. Pipeline on XPK (Pretrain -> SFT in ONE Job)
# ------------------------------------------------------------------------------
pipeline_xpk() {
  local workload_name="${USER_PREFIX}-qwen14b-pipe-${TIMESTAMP}"

  echo "=================================================================="
  echo ">>> [XPK FULL PIPELINE Qwen3-VL + Qwen3-14B] Submitting Workload: ${workload_name}"
  echo ">>> Stage 1: ChartNet CSV Pretraining (2,172 steps -> saves to checkpoint 2171)"
  echo ">>> Stage 2: ChartQA SFT (1,105 steps -> loads checkpoint 2171)"
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
      --command "export TMPDIR=/dev/shm && export PYTHONPATH=src:\${PYTHONPATH:-} && export HF_HOME=/dev/shm/huggingface && export HF_TOKEN=${HF_TOKEN} && echo '=== Stage 1: Pretrain ===' && python3 -m maxtext.experimental.omni_poc.train_sft_omni src/maxtext/experimental/omni_qwen_14b/pretrain-omni-qwen3-vl-14b-chartnet.yml load_parameters_path=${STITCHED_CKPT} base_output_directory=${PRETRAIN_OUTPUT_DIR} run_name=${PRETRAIN_RUN_NAME} scan_layers=false ici_fsdp_parallelism=-1 ici_tensor_parallelism=4 grain_worker_count=0 && echo '=== Stage 2: SFT ===' && python3 -m maxtext.experimental.omni_poc.train_sft_omni src/maxtext/experimental/omni_qwen_14b/sft-omni-qwen3-vl-14b-chartqa.yml load_parameters_path=${PRETRAIN_FINAL_CKPT} base_output_directory=${SFT_OUTPUT_DIR} run_name=${SFT_RUN_NAME} scan_layers=false ici_fsdp_parallelism=-1 ici_tensor_parallelism=4 grain_worker_count=0"
  )

  echo ""
  echo ">>> Pipeline Workload submitted: ${workload_name}"
  echo "=================================================================="
}

# ------------------------------------------------------------------------------
# 5. Local TPU VM Evaluation & Decode
# ------------------------------------------------------------------------------
eval_sft() {
  local ckpt_path="${1:-${SFT_OUTPUT_DIR}/${SFT_RUN_NAME}/checkpoints/1104/items}"
  echo "=================================================================="
  echo ">>> [EVAL] Evaluating SFT Checkpoint: ${ckpt_path}"
  echo "=================================================================="

  python3 -m maxtext.experimental.omni_poc.eval_sft_omni \
    "${SCRIPT_DIR}/sft-omni-qwen3-vl-14b-chartqa.yml" \
    "load_parameters_path=${ckpt_path}" \
    --ckpt_type=sft \
    --num_examples=100
}

# ------------------------------------------------------------------------------
# 5. Test Stitched Model (Shapes, Layers & Forward Pass Verification)
# ------------------------------------------------------------------------------
test_stitched_model() {
  echo "=================================================================="
  echo ">>> [TEST STITCH] Verifying Stitched Omni Qwen3-VL-4B + Qwen3-14B"
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

decode_samples() {
  local ckpt_path="${1:-${STITCHED_CKPT}}"
  echo "=================================================================="
  echo ">>> [DECODE] Running Autoregressive Inference: ${ckpt_path}"
  echo "=================================================================="

  (
    cd "${MAXTEXT_ROOT}"
    python3 "${SCRIPT_DIR}/decode_omni_qwen_14b.py" \
      --checkpoint_path="${ckpt_path}" \
      --num_samples="${2:-3}" \
      --max_new_tokens="${3:-128}"
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
  decode)
    decode_samples "${2:-}" "${3:-}" "${4:-}"
    ;;
  pretrain)
    pretrain_xpk
    ;;
  sft)
    sft_xpk "${2:-}"
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
    echo "Usage: $0 <stitch|test_stitch|decode|pretrain|sft|pipeline|eval|list|status|logs|delete> [checkpoint_path|workload_name]"
    ;;
esac
