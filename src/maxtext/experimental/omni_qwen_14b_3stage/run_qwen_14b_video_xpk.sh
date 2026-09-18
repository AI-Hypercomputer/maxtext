#!/bin/bash
# ==============================================================================
# Omni Qwen3-VL-4B Vision + Qwen3-14B LLM Video SFT Training on XPK (v4-128)
# ==============================================================================

set -e

ACTION="${1:-video_sft}"

# Global Experiment / Model Name
EXP_NAME="${EXP_NAME:-omni_qwen3_vl_14b_video_sft}"

: "${HF_TOKEN:?Error: HF_TOKEN is not set. Please run: export HF_TOKEN=hf_...}"
: "${GCS_BUCKET:?Error: GCS_BUCKET is not set. Please run: export GCS_BUCKET=gs://your-bucket}"
export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

# Base Storage Buckets & Checkpoints
STITCHED_CKPT="${STITCHED_CKPT:-${GCS_BUCKET}/omni_checkpoints/omni_qwen3_vl_14b_3stage/0/items}"
VIDEO_OUTPUT_DIR="${GCS_BUCKET}/experimental/${EXP_NAME}"
VIDEO_RUN_NAME="${EXP_NAME}_run"

# Video Dataset Configuration
HF_VIDEO_FILES="${HF_VIDEO_FILES:-/mounted/LLaVA-Video-178K/0_30_s_academic_v0_1/*.parquet}"

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
# Video SFT Workload Submission on XPK
# ------------------------------------------------------------------------------
video_sft_xpk() {
  local input_ckpt="${1:-${STITCHED_CKPT}}"
  local workload_name="${USER_PREFIX}-qwen14b-video-${TIMESTAMP}"

  echo "=================================================================="
  echo ">>> [XPK VIDEO SFT] Submitting Workload: ${workload_name}"
  echo ">>> Input Checkpoint: ${input_ckpt}"
  echo ">>> Output Dir:       ${VIDEO_OUTPUT_DIR}/${VIDEO_RUN_NAME}"
  echo ">>> Video Dataset:    ${HF_VIDEO_FILES}"
  echo ">>> Cluster:          ${XPK_CLUSTER} (${XPK_DEVICE_TYPE})"
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
      --command "export TMPDIR=/dev/shm && export PYTHONPATH=src:\${PYTHONPATH:-} && export HF_HOME=/dev/shm/huggingface && export HF_TOKEN=${HF_TOKEN} && pip install --no-cache-dir decord && python3 -m maxtext.experimental.omni_pipeline.train_sft_omni src/maxtext/experimental/omni_qwen_14b_3stage/sft-omni-qwen3-vl-14b-video-xpk-128.yml load_parameters_path=${input_ckpt} base_output_directory=${VIDEO_OUTPUT_DIR} run_name=${VIDEO_RUN_NAME} scan_layers=false ici_fsdp_parallelism=-1 ici_tensor_parallelism=4 grain_worker_count=0"
  )

  echo ""
  echo ">>> Workload submitted: ${workload_name}"
  echo ">>> View logs: $0 logs ${workload_name}"
  echo "=================================================================="
}

# ------------------------------------------------------------------------------
# Dispatcher
# ------------------------------------------------------------------------------
case "$ACTION" in
  video_sft|train)
    video_sft_xpk "${2:-}"
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
    echo "Usage: $0 <video_sft|list|status|logs|delete> [checkpoint_path|workload_name]"
    ;;
esac
