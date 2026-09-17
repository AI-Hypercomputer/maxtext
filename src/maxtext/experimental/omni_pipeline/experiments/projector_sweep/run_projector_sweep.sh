#!/bin/bash
# ==============================================================================
# End-to-End Multimodal Projector Architecture Sweep on XPK (TPU Cluster)
# ==============================================================================
# Cluster: v4-128-bodaborg-us-central2-b (cloud-tpu-multipod-dev)
#
# This script orchestrates:
#   1. Model Stitching (Local / VM CPU)
#   2. Pretraining on ChartNet via XPK (Stage 1: Alignment)
#   3. SFT Fine-tuning on ChartQA via XPK (Stage 2: Task Training)
#   4. Full End-to-End Pipeline on XPK (Pretrain -> SFT -> Eval in one job!)
#   5. Workload status inspection, live log streaming, and workload deletion
#
# Checkpoints are saved under distinct GCS subfolders keyed by architecture tag:
#   <ARCH_TAG> = l{NUM_LAYERS}_h{HIDDEN_SIZE}_{ACTIVATION}
#
# Examples:
#   # 1. Stitch all variant checkpoints on CPU:
#   ./run_projector_sweep.sh stitch all
#
#   # 2. Run FULL end-to-end pipeline (Pretrain -> SFT -> Eval) on XPK:
#   ./run_projector_sweep.sh pipeline l2_h4096_silu
#   ./run_projector_sweep.sh pipeline all
#
#   # 3. Or run stages individually:
#   ./run_projector_sweep.sh pretrain all
#   ./run_projector_sweep.sh sft all
#
#   # 4. Check cluster workload status & stream logs:
#   ./run_projector_sweep.sh list
#   ./run_projector_sweep.sh status <workload_name>
#   ./run_projector_sweep.sh logs <workload_name>
#   ./run_projector_sweep.sh delete <workload_name>
# ==============================================================================

set -e

ACTION="${1:-help}"
TARGET_VARIANT="${2:-all}"

# Base Storage Buckets
: "${HF_TOKEN:?Error: HF_TOKEN is not set. Please run: export HF_TOKEN=hf_...}"
: "${GCS_BUCKET:?Error: GCS_BUCKET is not set. Please run: export GCS_BUCKET=gs://your-bucket}"
export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

VISION_SOURCE_CKPT="${VISION_SOURCE_CKPT:-${GCS_BUCKET}/omni_checkpoints/gemma3-4b_converted/0/items}"
LLM_SOURCE_CKPT="${LLM_SOURCE_CKPT:-${GCS_BUCKET}/omni_checkpoints/qwen3-4b_converted/0/items}"

# XPK Cluster Configuration (Matched to your cluster setup)
XPK_CLUSTER="${XPK_CLUSTER:-v4-128-bodaborg-us-central2-b}"
XPK_PROJECT="${XPK_PROJECT:-cloud-tpu-multipod-dev}"
XPK_ZONE="${XPK_ZONE:-us-central2-b}"
XPK_DEVICE_TYPE="${XPK_DEVICE_TYPE:-v4-128}"
XPK_NUM_SLICES="${XPK_NUM_SLICES:-1}"
XPK_BASE_DOCKER_IMAGE="${XPK_BASE_DOCKER_IMAGE:-gcr.io/tpu-prod-env-multipod/maxtext_base_image:latest}"

# User Prefix for Workload Isolation
USER_PREFIX="${USER_PREFIX:-user}"

# Hugging Face Auth Tokens
export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

# Project Root Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OMNI_PIPELINE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MAXTEXT_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || (cd "${SCRIPT_DIR}/../../../../../.." && pwd))"
export PYTHONPATH="${MAXTEXT_ROOT}/src:${PYTHONPATH}"

# Define Architecture Sweep Matrix:
# Format: "ARCH_TAG:NUM_LAYERS:HIDDEN_SIZE:ACTIVATION"
VARIANTS=(
  "l2_h4096_gelu:2:4096:gelu"
  "l2_h4096_silu:2:4096:silu"
  "l3_h2560_gelu:3:2560:gelu"
  "l3_h2560_silu:3:2560:silu"
  "l3_h4096_gelu:3:4096:gelu"
  "l3_h4096_silu:3:4096:silu"
  "l3_h8192_gelu:3:8192:gelu"
  "l3_h8192_silu:3:8192:silu"
  "l4_h4096_gelu:4:4096:gelu"
  "l4_h4096_silu:4:4096:silu"
  "l4_h8192_gelu:4:8192:gelu"
)

# Helper: parse variant specs
get_variant_spec() {
  local target="$1"
  for entry in "${VARIANTS[@]}"; do
    local tag=$(echo "$entry" | cut -d: -f1)
    if [ "$tag" == "$target" ]; then
      echo "$entry"
      return 0
    fi
  done
  return 1
}

# ------------------------------------------------------------------------------
# 1. Checkpoint Stitching (CPU-bound)
# ------------------------------------------------------------------------------
stitch_variant() {
  local entry="$1"
  local tag=$(echo "$entry" | cut -d: -f1)
  local layers=$(echo "$entry" | cut -d: -f2)
  local hidden=$(echo "$entry" | cut -d: -f3)
  local act=$(echo "$entry" | cut -d: -f4)

  local model_yaml="${SCRIPT_DIR}/model_${tag}.yml"
  local output_ckpt="${GCS_BUCKET}/omni_checkpoints/omni_stitched_gemma3-4b_qwen3-4b_${tag}/0/items"

  echo "=================================================================="
  echo ">>> [STITCH] Variant: ${tag} (layers=${layers}, hidden=${hidden}, act=${act})"
  echo ">>> Output Path: ${output_ckpt}"
  echo "=================================================================="

  JAX_PLATFORMS=cpu python3 -m maxtext.experimental.omni_pipeline.utils.stitch_checkpoint \
    "${model_yaml}" \
    "vision_load_path=${VISION_SOURCE_CKPT}" \
    "llm_load_path=${LLM_SOURCE_CKPT}" \
    "stitched_output_path=${output_ckpt}"
}

# ------------------------------------------------------------------------------
# 2. ChartNet Pretraining on XPK (Alignment Stage)
# ------------------------------------------------------------------------------
pretrain_xpk_variant() {
  local entry="$1"
  local extra_args="${2:-}"
  local tag=$(echo "$entry" | cut -d: -f1)
  local layers=$(echo "$entry" | cut -d: -f2)
  local hidden=$(echo "$entry" | cut -d: -f3)
  local act=$(echo "$entry" | cut -d: -f4)

  local input_ckpt="${GCS_BUCKET}/omni_checkpoints/omni_stitched_gemma3-4b_qwen3-4b_${tag}/0/items"
  local base_out_dir="${GCS_BUCKET}/omni-gemma3-qwen3/multimodal/pretrain_chartnet_xpk/${tag}"
  local run_name="omni_pretrain_chartnet_${tag}_xpk128"
  local timestamp=$(date +%m%d-%H%M)
  local workload_name="${USER_PREFIX}-pretrain-${tag//_/-}-${timestamp}"

  echo "=================================================================="
  echo ">>> [XPK PRETRAIN - ChartNet] Submitting Workload: ${workload_name}"
  echo ">>> Variant:          ${tag} (layers=${layers}, hidden=${hidden}, act=${act})"
  echo ">>> Cluster:          ${XPK_CLUSTER} (${XPK_DEVICE_TYPE}, ${XPK_ZONE})"
  echo ">>> Input Checkpoint: ${input_ckpt}"
  echo ">>> Base Output Dir:  ${base_out_dir}"
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
      --command "bash src/maxtext/experimental/omni_pipeline/experiments/projector_sweep/train_variant.sh pretrain ${tag} ${layers} ${hidden} ${act}"
  )

  echo ""
  echo ">>> Workload submitted successfully: ${workload_name}"
  echo ">>> To check pod status:"
  echo "    kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${workload_name} -w"
  echo ">>> To stream logs:"
  echo "    kubectl logs \$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${workload_name} -o jsonpath='{.items[0].metadata.name}') -c jax-tpu -f"
  echo "=================================================================="
}

# ------------------------------------------------------------------------------
# 3. ChartQA SFT on XPK (Task Stage)
# ------------------------------------------------------------------------------
sft_xpk_variant() {
  local entry="$1"
  local extra_args="${2:-}"
  local tag=$(echo "$entry" | cut -d: -f1)
  local layers=$(echo "$entry" | cut -d: -f2)
  local hidden=$(echo "$entry" | cut -d: -f3)
  local act=$(echo "$entry" | cut -d: -f4)

  local base_out_dir="${GCS_BUCKET}/omni-gemma3-qwen3/multimodal/sft_chartqa_xpk/${tag}"
  local timestamp=$(date +%m%d-%H%M)
  local workload_name="${USER_PREFIX}-sft-${tag//_/-}-${timestamp}"

  echo "=================================================================="
  echo ">>> [XPK SFT - ChartQA] Submitting Workload: ${workload_name}"
  echo ">>> Variant:          ${tag} (layers=${layers}, hidden=${hidden}, act=${act})"
  echo ">>> Cluster:          ${XPK_CLUSTER} (${XPK_DEVICE_TYPE}, ${XPK_ZONE})"
  echo ">>> Base Output Dir:  ${base_out_dir}"
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
      --command "bash src/maxtext/experimental/omni_pipeline/experiments/projector_sweep/train_variant.sh sft ${tag} ${layers} ${hidden} ${act}"
  )

  echo ""
  echo ">>> Workload submitted successfully: ${workload_name}"
  echo ">>> To check pod status:"
  echo "    kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${workload_name} -w"
  echo ">>> To stream logs:"
  echo "    kubectl logs \$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${workload_name} -o jsonpath='{.items[0].metadata.name}') -c jax-tpu -f"
  echo "=================================================================="
}

# ------------------------------------------------------------------------------
# 4. Full End-to-End Pipeline on XPK (Pretrain -> SFT in ONE Job)
# ------------------------------------------------------------------------------
pipeline_xpk_variant() {
  local entry="$1"
  local extra_args="${2:-}"
  local tag=$(echo "$entry" | cut -d: -f1)
  local layers=$(echo "$entry" | cut -d: -f2)
  local hidden=$(echo "$entry" | cut -d: -f3)
  local act=$(echo "$entry" | cut -d: -f4)

  local timestamp=$(date +%m%d-%H%M)
  local workload_name="${USER_PREFIX}-pipe-${tag//_/-}-${timestamp}"

  echo "=================================================================="
  echo ">>> [XPK FULL PIPELINE] Submitting Workload: ${workload_name}"
  echo ">>> Variant:          ${tag} (layers=${layers}, hidden=${hidden}, act=${act})"
  echo ">>> Stage 1:          ChartNet Pretraining (2,172 steps)"
  echo ">>> Stage 2:          ChartQA SFT (1,105 steps)"
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
      --command "bash src/maxtext/experimental/omni_pipeline/experiments/projector_sweep/train_variant.sh pipeline ${tag} ${layers} ${hidden} ${act}"
  )

  echo ""
  echo ">>> Pipeline Workload submitted: ${workload_name}"
  echo ">>> To check pod status:"
  echo "    kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${workload_name} -w"
  echo ">>> To stream logs:"
  echo "    kubectl logs \$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${workload_name} -o jsonpath='{.items[0].metadata.name}') -c jax-tpu -f"
  echo "=================================================================="
}

# ------------------------------------------------------------------------------
# 5. Local TPU VM Smoke Test (Quick 10-step verification before XPK)
# ------------------------------------------------------------------------------
test_vm_variant() {
  local entry="$1"
  local tag=$(echo "$entry" | cut -d: -f1)
  local layers=$(echo "$entry" | cut -d: -f2)
  local hidden=$(echo "$entry" | cut -d: -f3)
  local act=$(echo "$entry" | cut -d: -f4)

  echo "=================================================================="
  echo ">>> [TPU VM SMOKE TEST] Running 10-step Pretrain & SFT verification for ${tag}"
  echo "=================================================================="

  local input_ckpt="${GCS_BUCKET}/omni_checkpoints/omni_stitched_gemma3-4b_qwen3-4b_${tag}/0/items"
  local test_out_dir="${GCS_BUCKET}/omni_test_runs/${tag}"

  python3 -m maxtext.experimental.omni_pipeline.train_sft_omni \
    "${OMNI_PIPELINE_DIR}/pretrain-omni-gemma3-qwen3-chartnet.yml" \
    "vision_connector_num_layers=${layers}" \
    "vision_connector_hidden_size=${hidden}" \
    "vision_connector_activation=${act}" \
    "load_parameters_path=${input_ckpt}" \
    "base_output_directory=${test_out_dir}" \
    "run_name=smoke_test_pretrain_${tag}" \
    "steps=10" \
    "checkpoint_period=10" \
    "eval_interval=10" \
    "eval_steps=2"
}

# ------------------------------------------------------------------------------
# Cluster Utility Functions: List, Status, Logs, Delete
# ------------------------------------------------------------------------------
xpk_list() {
  xpk workload list \
    --cluster "${XPK_CLUSTER}" \
    --project "${XPK_PROJECT}" \
    --zone "${XPK_ZONE}"
}

xpk_status() {
  local w_name="$1"
  if [ -z "$w_name" ]; then
    echo "Usage: $0 status <workload_name>"
    exit 1
  fi
  kubectl get pods -l jobset.sigs.k8s.io/jobset-name="${w_name}" -w
}

xpk_logs() {
  local w_name="$1"
  if [ -z "$w_name" ]; then
    echo "Usage: $0 logs <workload_name>"
    exit 1
  fi
  local pod_name=$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name="${w_name}" -o jsonpath='{.items[0].metadata.name}')
  kubectl logs "${pod_name}" -c jax-tpu -f
}

xpk_delete() {
  local w_name="$1"
  if [ -z "$w_name" ]; then
    echo "Usage: $0 delete <workload_name>"
    exit 1
  fi
  xpk workload delete \
    --workload "${w_name}" \
    --cluster "${XPK_CLUSTER}" \
    --project "${XPK_PROJECT}" \
    --zone "${XPK_ZONE}"
}

xpk_clean_sweep() {
  echo "Finding all workloads belonging to '${USER_PREFIX}'..."
  local sweep_jobs=$(xpk workload list --cluster "${XPK_CLUSTER}" --project "${XPK_PROJECT}" --zone "${XPK_ZONE}" 2>/dev/null | awk '{print $1}' | grep -E "^(${USER_PREFIX}-pretrain-|${USER_PREFIX}-sft-|${USER_PREFIX}-pipe-|pretrain-|sft-|pipe-)" || true)

  if [ -z "$sweep_jobs" ]; then
    echo "No matching sweep workloads found on cluster."
    return 0
  fi

  echo "The following sweep workloads will be deleted:"
  echo "$sweep_jobs"
  echo ""
  for job in $sweep_jobs; do
    echo "Deleting workload: $job ..."
    xpk workload delete \
      --workload "$job" \
      --cluster "${XPK_CLUSTER}" \
      --project "${XPK_PROJECT}" \
      --zone "${XPK_ZONE}"
  done
  echo "Sweep cleanup complete!"
}

# ------------------------------------------------------------------------------
# Main Dispatcher
# ------------------------------------------------------------------------------
run_action_for_targets() {
  local fn="$1"
  shift
  local extra="$@"

  if [ "${TARGET_VARIANT}" == "all" ]; then
    for entry in "${VARIANTS[@]}"; do
      $fn "$entry" $extra
    done
  else
    local spec=$(get_variant_spec "$TARGET_VARIANT")
    if [ -z "$spec" ]; then
      echo "Error: Unknown variant tag '$TARGET_VARIANT'."
      echo "Available variants:"
      for entry in "${VARIANTS[@]}"; do
        echo "  - $(echo "$entry" | cut -d: -f1)"
      done
      exit 1
    fi
    $fn "$spec" $extra
  fi
}

case "$ACTION" in
  stitch)
    run_action_for_targets stitch_variant
    ;;
  pretrain)
    run_action_for_targets pretrain_xpk_variant
    ;;
  sft)
    run_action_for_targets sft_xpk_variant
    ;;
  pipeline)
    run_action_for_targets pipeline_xpk_variant
    ;;
  test_vm)
    run_action_for_targets test_vm_variant
    ;;
  list)
    xpk_list
    ;;
  status)
    xpk_status "$TARGET_VARIANT"
    ;;
  logs)
    xpk_logs "$TARGET_VARIANT"
    ;;
  delete)
    xpk_delete "$TARGET_VARIANT"
    ;;
  clean_sweep)
    xpk_clean_sweep
    ;;
  help|*)
    echo "Usage: $0 <stitch|pretrain|sft|pipeline|test_vm|list|status|logs|delete> [variant_name|workload_name|all]"
    echo ""
    echo "Actions:"
    echo "  stitch   : Stitch checkpoints locally on CPU (e.g. ./run_projector_sweep.sh stitch all)"
    echo "  pipeline : Run FULL end-to-end (Pretrain -> SFT -> Eval) in one XPK job per variant"
    echo "  pretrain : Submit ChartNet pretraining workload to XPK cluster"
    echo "  sft      : Submit ChartQA SFT workload to XPK cluster"
    echo "  test_vm  : Run a quick 10-step smoke test on local TPU VM"
    echo "  list     : List all active workloads on cluster"
    echo "  status   : Watch pod status for a workload (e.g. ./run_projector_sweep.sh status <workload>)"
    echo "  logs     : Stream live logs for a workload (e.g. ./run_projector_sweep.sh logs <workload>)"
    echo "  delete   : Delete a workload on cluster (e.g. ./run_projector_sweep.sh delete <workload>)"
    echo ""
    echo "Available variant names:"
    for entry in "${VARIANTS[@]}"; do
      echo "  - $(echo "$entry" | cut -d: -f1)"
    done
    ;;
esac
