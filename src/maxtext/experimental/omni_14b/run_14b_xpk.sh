#!/bin/bash
# ==============================================================================
# Omni Gemma3-4B + Qwen3-14B Pretrain & SFT Execution on XPK (v4-128)
# ==============================================================================

set -e

ACTION="${1:-help}"

# Base Storage Buckets & Checkpoints
GCS_BUCKET="${GCS_BUCKET:-gs://yuchenhou-maxtext-logs}"
STITCHED_CKPT="${STITCHED_CKPT:-${GCS_BUCKET}/omni_checkpoints/omni_stitched_gemma3-4b_qwen3-14b/0/items}"

# XPK Cluster Configuration
XPK_CLUSTER="${XPK_CLUSTER:-v4-128-bodaborg-us-central2-b}"
XPK_PROJECT="${XPK_PROJECT:-cloud-tpu-multipod-dev}"
XPK_ZONE="${XPK_ZONE:-us-central2-b}"
XPK_DEVICE_TYPE="${XPK_DEVICE_TYPE:-v4-128}"
XPK_NUM_SLICES="${XPK_NUM_SLICES:-1}"
XPK_BASE_DOCKER_IMAGE="${XPK_BASE_DOCKER_IMAGE:-gcr.io/tpu-prod-env-multipod/maxtext_base_image:latest}"

USER_PREFIX="${USER_PREFIX:-yuchenhou}"
TIMESTAMP=$(date +%m%d-%H%M)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAXTEXT_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || (cd "${SCRIPT_DIR}/../../../../.." && pwd))"

# ------------------------------------------------------------------------------
# 1. ChartNet CSV Pretraining on XPK
# ------------------------------------------------------------------------------
pretrain_xpk() {
  local workload_name="${USER_PREFIX}-14b-pretrain-csv-${TIMESTAMP}"

  echo "=================================================================="
  echo ">>> [XPK PRETRAIN - ChartNet CSV] Submitting Workload: ${workload_name}"
  echo ">>> Model:            Gemma 3 4B + Qwen 3 14B"
  echo ">>> Cluster:          ${XPK_CLUSTER} (${XPK_DEVICE_TYPE}, ${XPK_ZONE})"
  echo ">>> Input Checkpoint: ${STITCHED_CKPT}"
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
      --command "export TMPDIR=/dev/shm && export PYTHONPATH=src:\${PYTHONPATH:-} && export HF_HOME=/dev/shm/huggingface && python3 -m maxtext.experimental.omni_poc.train_sft_omni src/maxtext/experimental/omni_14b/pretrain-omni-gemma3-qwen3-14b-chartnet-xpk-128-csv.yml load_parameters_path=${STITCHED_CKPT}"
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
# 2. ChartQA SFT on XPK
# ------------------------------------------------------------------------------
sft_xpk() {
  local input_ckpt="${1:-${GCS_BUCKET}/omni-gemma3-qwen3-14b/multimodal/pretrain_chartnet_xpk/omni_14b_pretrain_chartnet_xpk_128_csv/checkpoints/2100/items}"
  local workload_name="${USER_PREFIX}-14b-sft-chartqa-${TIMESTAMP}"

  echo "=================================================================="
  echo ">>> [XPK SFT - ChartQA] Submitting Workload: ${workload_name}"
  echo ">>> Model:            Gemma 3 4B + Qwen 3 14B"
  echo ">>> Cluster:          ${XPK_CLUSTER} (${XPK_DEVICE_TYPE}, ${XPK_ZONE})"
  echo ">>> Input Checkpoint: ${input_ckpt}"
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
      --command "export TMPDIR=/dev/shm && export PYTHONPATH=src:\${PYTHONPATH:-} && export HF_HOME=/dev/shm/huggingface && python3 -m maxtext.experimental.omni_poc.train_sft_omni src/maxtext/experimental/omni_14b/sft-omni-gemma3-qwen3-14b-xpk-128.yml load_parameters_path=${input_ckpt}"
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
# 3. Pipeline on XPK (Pretrain -> SFT in ONE Job)
# ------------------------------------------------------------------------------
pipeline_xpk() {
  local workload_name="${USER_PREFIX}-14b-pipeline-${TIMESTAMP}"

  echo "=================================================================="
  echo ">>> [XPK FULL PIPELINE 14B] Submitting Workload: ${workload_name}"
  echo ">>> Stage 1: ChartNet CSV Pretraining (2,172 steps)"
  echo ">>> Stage 2: ChartQA SFT (1,105 steps)"
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
      --command "export TMPDIR=/dev/shm && export PYTHONPATH=src:\${PYTHONPATH:-} && export HF_HOME=/dev/shm/huggingface && echo '=== Stage 1: Pretrain ===' && python3 -m maxtext.experimental.omni_poc.train_sft_omni src/maxtext/experimental/omni_14b/pretrain-omni-gemma3-qwen3-14b-chartnet-xpk-128-csv.yml load_parameters_path=${STITCHED_CKPT} && echo '=== Stage 2: SFT ===' && python3 -m maxtext.experimental.omni_poc.train_sft_omni src/maxtext/experimental/omni_14b/sft-omni-gemma3-qwen3-14b-xpk-128.yml load_parameters_path=${GCS_BUCKET}/omni-gemma3-qwen3-14b/multimodal/pretrain_chartnet_xpk/omni_14b_pretrain_chartnet_xpk_128_csv/checkpoints/2100/items"
  )

  echo ""
  echo ">>> Pipeline Workload submitted: ${workload_name}"
  echo "=================================================================="
}

# ------------------------------------------------------------------------------
# Dispatcher
# ------------------------------------------------------------------------------
case "$ACTION" in
  pretrain)
    pretrain_xpk
    ;;
  sft)
    sft_xpk "${2:-}"
    ;;
  pipeline)
    pipeline_xpk
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
    echo "Usage: $0 <pretrain|sft|pipeline|list|status|logs|delete> [checkpoint_path|workload_name]"
    ;;
esac
