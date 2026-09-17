#!/bin/bash
# ==============================================================================
# Omni Gemma3-4B Vision + Qwen3-4B LLM 2-Stage Pipeline on XPK
#
# Stage 1: ChartNet Core CSV Table Grounding (1 Epoch = 13,267 steps, Eval Disabled)
# Stage 2: ChartQA Visual QA SFT            (5 Epochs = 1,105 steps)
#
# Schedule Computations on TPU v4-128 (128 accelerator devices, batch_size=1):
# - Stage 1 (ChartNet core subset: 1,698,111 examples):
#     1,698,111 examples * 1 epoch / (128 devices * 1 per_device_batch_size = 128)
#     = 13,266.49 -> ceil = 13,267 steps (final checkpoint at step 13266).
#     eval_interval: -1, eval_steps: 0 (intermediate evaluation disabled).
# - Stage 2 (ChartQA shuffled train split: 28,288 examples):
#     28,288 examples * 5 epochs / (128 devices * 1 per_device_batch_size = 128)
#     = 1,105 steps (final checkpoint at step 1104).
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

# Global Experiment & Working Directory
EXP_NAME="${EXP_NAME:-omni_gemma3_qwen3_2stage_core_csv_1ep}"
WORKING_DIR="${WORKING_DIR:-${GCS_BUCKET}/omni-gemma3-qwen3/multimodal/2stage_core_csv_pipeline/${EXP_NAME}}"

# Base Storage Buckets & Initial Checkpoints
VISION_SOURCE_CKPT="${VISION_SOURCE_CKPT:-${GCS_BUCKET}/omni_checkpoints/gemma3-4b_converted/0/items}"
LLM_SOURCE_CKPT="${LLM_SOURCE_CKPT:-${GCS_BUCKET}/omni_checkpoints/qwen3-4b_converted/0/items}"
STITCHED_CKPT="${STITCHED_CKPT:-${GCS_BUCKET}/omni_checkpoints/omni_stitched_gemma3-4b_qwen3-4b/0/items}"

# Pre-sharded Dataset Storage Paths
CHARTNET_CORE_DATASET_DIR="${CHARTNET_CORE_DATASET_DIR:-${GCS_BUCKET}/datasets/chartnet_core_sharded}"
CHARTQA_DATASET_DIR="${CHARTQA_DATASET_DIR:-${GCS_BUCKET}/datasets/chartqa_shuffled}"

# Stage 1: ChartNet Core CSV Table Grounding (1 Epoch = 13,267 steps on 128 devices, No Eval)
STAGE1_CONFIG="${STAGE1_CONFIG:-src/maxtext/experimental/omni_pipeline/pretrain-omni-gemma3-qwen3-chartnet-core-csv-xpk-128-1ep.yml}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-1}"
STAGE1_STEPS="${STAGE1_STEPS:-13267}"
STAGE1_CKPT_PERIOD="${STAGE1_CKPT_PERIOD:-2000}"
STAGE1_OUTPUT_DIR="${WORKING_DIR}/stage1_chartnet_core_csv_1ep"
STAGE1_RUN_NAME="${EXP_NAME}_stage1_core_csv_1ep"
STAGE1_FINAL_CKPT="${STAGE1_OUTPUT_DIR}/${STAGE1_RUN_NAME}/checkpoints/$((STAGE1_STEPS - 1))/items"

# Stage 2: ChartQA Visual QA SFT (5 Epochs = 1,105 steps on 128 devices)
STAGE2_CONFIG="${STAGE2_CONFIG:-src/maxtext/experimental/omni_pipeline/sft-omni-gemma3-qwen3-xpk-128.yml}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-5}"
STAGE2_STEPS="${STAGE2_STEPS:-1105}"
STAGE2_OUTPUT_DIR="${WORKING_DIR}/stage2_chartqa_sft"
STAGE2_RUN_NAME="${EXP_NAME}_stage2_chartqa_sft"
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
TIMESTAMP=$(date +%m%d-%H%M)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAXTEXT_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || (cd "${SCRIPT_DIR}/../../../../.." && pwd))"

# ------------------------------------------------------------------------------
# 0. Check Dataset Status (GCS and Local)
# ------------------------------------------------------------------------------
check_data_status() {
  echo "=================================================================="
  echo ">>> [DATA CHECK] Checking ChartNet Core and ChartQA Dataset Paths in GCS"
  echo "=================================================================="
  
  echo "1. Checking ChartNet Core Sharded Path in GCS: ${CHARTNET_CORE_DATASET_DIR}"
  if gcloud storage ls "${CHARTNET_CORE_DATASET_DIR}/*.parquet" >/dev/null 2>&1; then
    count=$(gcloud storage ls "${CHARTNET_CORE_DATASET_DIR}/*.parquet" | wc -l)
    echo "   [FOUND] ChartNet Core dataset exists with ${count} parquet shards."
  else
    echo "   [NOT FOUND] ChartNet Core dataset not found in GCS. Run: $0 prepare_chartnet"
  fi
  echo ""

  echo "2. Checking ChartQA Shuffled Path in GCS: ${CHARTQA_DATASET_DIR}"
  if gcloud storage ls "${CHARTQA_DATASET_DIR}/*.parquet" >/dev/null 2>&1; then
    count=$(gcloud storage ls "${CHARTQA_DATASET_DIR}/*.parquet" | wc -l)
    echo "   [FOUND] ChartQA dataset exists with ${count} parquet shards."
  else
    echo "   [NOT FOUND] ChartQA dataset not found in GCS. Run: $0 prepare_chartqa"
  fi
  echo "=================================================================="
}

# ------------------------------------------------------------------------------
# 1. Stitch Checkpoint (CPU-bound: Gemma3-4B ViT + Qwen3-4B LLM Backbone)
# ------------------------------------------------------------------------------
stitch_ckpt() {
  echo "=================================================================="
  echo ">>> [STITCH] Stitching Gemma3-4B Vision + Qwen3-4B LLM"
  echo ">>> Vision Source: ${VISION_SOURCE_CKPT}"
  echo ">>> LLM Source:    ${LLM_SOURCE_CKPT}"
  echo ">>> Output Path:   ${STITCHED_CKPT}"
  echo "=================================================================="

  (
    cd "${MAXTEXT_ROOT}"
    HF_TOKEN="${HF_TOKEN}" HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}" JAX_PLATFORMS=cpu python3 -m maxtext.experimental.omni_pipeline.utils.stitch_checkpoint \
      "${SCRIPT_DIR}/maxtext-omni-gemma3-qwen3.yml" \
      "hf_access_token=${HF_TOKEN}" \
      "vision_load_path=${VISION_SOURCE_CKPT}" \
      "llm_load_path=${LLM_SOURCE_CKPT}" \
      "stitched_output_path=${STITCHED_CKPT}"
  )
}

# ------------------------------------------------------------------------------
# 2. Stage 1: ChartNet Core CSV Table Grounding on XPK (1 Epoch = 13,267 steps)
# ------------------------------------------------------------------------------
stage1_xpk() {
  local input_ckpt="${1:-${STITCHED_CKPT}}"
  local workload_name="${USER_PREFIX}-omni-s1-csv1ep-${TIMESTAMP}"
  workload_name="${workload_name:0:39}"
  workload_name="${workload_name%-}"

  echo "=================================================================="
  echo ">>> [XPK STAGE 1 - ChartNet Core CSV (1 Epoch)] Submitting Workload: ${workload_name}"
  echo ">>> Config:           ${STAGE1_CONFIG}"
  echo ">>> Input Checkpoint: ${input_ckpt}"
  echo ">>> Working Dir:      ${STAGE1_OUTPUT_DIR}/${STAGE1_RUN_NAME}"
  echo ">>> Schedule:         ${STAGE1_EPOCHS} Epoch = ${STAGE1_STEPS} Steps (Eval Disabled)"
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
      --command "export MEGASCALE_NUM_SLICES=${XPK_NUM_SLICES} && export TMPDIR=/dev/shm && export PYTHONPATH=src:\${PYTHONPATH:-} && export HF_HOME=/dev/shm/huggingface && export HF_TOKEN=${HF_TOKEN} && export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} && python3 -m pip install --no-cache-dir -U 'orbax-checkpoint>=0.12.4' && gcloud storage cp -r ${CHARTNET_CORE_DATASET_DIR} /dev/shm/ && python3 -m maxtext.experimental.omni_pipeline.train_sft_omni ${STAGE1_CONFIG} load_parameters_path=${input_ckpt} base_output_directory=${STAGE1_OUTPUT_DIR} run_name=${STAGE1_RUN_NAME} num_epoch=${STAGE1_EPOCHS} steps=${STAGE1_STEPS} eval_interval=-1 eval_steps=0 checkpoint_period=${STAGE1_CKPT_PERIOD} hf_access_token=${HF_TOKEN} scan_layers=true grain_worker_count=0"
  )
}

# ------------------------------------------------------------------------------
# 3. Stage 2: ChartQA Visual QA SFT on XPK (5 Epochs = 1,105 steps)
# ------------------------------------------------------------------------------
stage2_xpk() {
  local input_ckpt="${1:-${STAGE1_FINAL_CKPT}}"
  local workload_name="${USER_PREFIX}-omni-s2-sft-${TIMESTAMP}"
  workload_name="${workload_name:0:39}"
  workload_name="${workload_name%-}"

  echo "=================================================================="
  echo ">>> [XPK STAGE 2 - ChartQA SFT (Projector Only)] Submitting Workload: ${workload_name}"
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
      --command "export MEGASCALE_NUM_SLICES=${XPK_NUM_SLICES} && export TMPDIR=/dev/shm && export PYTHONPATH=src:\${PYTHONPATH:-} && export HF_HOME=/dev/shm/huggingface && export HF_TOKEN=${HF_TOKEN} && export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} && python3 -m pip install --no-cache-dir -U 'orbax-checkpoint>=0.12.4' && gcloud storage cp -r ${CHARTQA_DATASET_DIR} /dev/shm/ && python3 -m maxtext.experimental.omni_pipeline.train_sft_omni ${STAGE2_CONFIG} load_parameters_path=${input_ckpt} base_output_directory=${STAGE2_OUTPUT_DIR} run_name=${STAGE2_RUN_NAME} hf_access_token=${HF_TOKEN} scan_layers=true grain_worker_count=0"
  )
}

# ------------------------------------------------------------------------------
# 4. Full 2-Stage Pipeline on XPK (Stage 1 Core CSV 1ep -> Stage 2 ChartQA SFT)
# ------------------------------------------------------------------------------
pipeline_xpk() {
  local workload_name="${USER_PREFIX}-omni-core-csv1ep-${TIMESTAMP}"
  workload_name="${workload_name:0:39}"
  workload_name="${workload_name%-}"

  echo "=================================================================="
  echo ">>> [XPK FULL 2-STAGE PIPELINE: ChartNet Core CSV (1 Epoch) -> ChartQA SFT]"
  echo ">>> Submitting Workload: ${workload_name}"
  echo ">>> Base Working Dir:    ${WORKING_DIR}"
  echo ">>> Stage 1: ChartNet Core CSV Grounding (1 Epoch = ${STAGE1_STEPS} steps -> ${STAGE1_FINAL_CKPT})"
  echo ">>> Stage 2: ChartQA Visual QA SFT       (5 Epochs = ${STAGE2_STEPS} steps  -> ${STAGE2_FINAL_CKPT})"
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
      --command "export MEGASCALE_NUM_SLICES=${XPK_NUM_SLICES} && export TMPDIR=/dev/shm && export PYTHONPATH=src:\${PYTHONPATH:-} && export HF_HOME=/dev/shm/huggingface && export HF_TOKEN=${HF_TOKEN} && export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} && python3 -m pip install --no-cache-dir -U 'orbax-checkpoint>=0.12.4' && gcloud storage cp -r ${CHARTNET_CORE_DATASET_DIR} /dev/shm/ && gcloud storage cp -r ${CHARTQA_DATASET_DIR} /dev/shm/ && echo '=== Stage 1: ChartNet Core CSV Table Grounding (1 Epoch = ${STAGE1_STEPS} steps, No Eval) ===' && python3 -m maxtext.experimental.omni_pipeline.train_sft_omni ${STAGE1_CONFIG} load_parameters_path=${STITCHED_CKPT} base_output_directory=${STAGE1_OUTPUT_DIR} run_name=${STAGE1_RUN_NAME} num_epoch=${STAGE1_EPOCHS} steps=${STAGE1_STEPS} eval_interval=-1 eval_steps=0 checkpoint_period=${STAGE1_CKPT_PERIOD} hf_access_token=${HF_TOKEN} scan_layers=true grain_worker_count=0 && echo '=== Stage 2: ChartQA Visual QA SFT (${STAGE2_STEPS} steps) ===' && python3 -m maxtext.experimental.omni_pipeline.train_sft_omni ${STAGE2_CONFIG} load_parameters_path=${STAGE1_FINAL_CKPT} base_output_directory=${STAGE2_OUTPUT_DIR} run_name=${STAGE2_RUN_NAME} hf_access_token=${HF_TOKEN} scan_layers=true grain_worker_count=0"
  )

  echo ""
  echo ">>> 2-Stage Pipeline Workload submitted: ${workload_name}"
  echo "=================================================================="
}

# ------------------------------------------------------------------------------
# 5. Dataset Preparation (Pre-download, Global Shuffle, and Shard to GCS)
# ------------------------------------------------------------------------------
prepare_chartnet_core() {
  echo "=================================================================="
  echo ">>> [DATA] Preparing Sharded ChartNet 'core' Dataset (Cached in /dev/shm)"
  echo ">>> Output Dir: ${CHARTNET_CORE_DATASET_DIR}"
  echo "=================================================================="
  TMPDIR=/dev/shm HF_HOME=/dev/shm/huggingface HF_DATASETS_CACHE=/dev/shm/huggingface/datasets HF_TOKEN="${HF_TOKEN}" python3 "${SCRIPT_DIR}/prepare_chartnet_core_sharded.py" \
    --output_dir "${CHARTNET_CORE_DATASET_DIR}" \
    --hf_token "${HF_TOKEN}"
}

prepare_chartqa() {
  echo "=================================================================="
  echo ">>> [DATA] Preparing Shuffled & Sharded ChartQA Dataset (Cached in /dev/shm)"
  echo ">>> Output Dir: ${CHARTQA_DATASET_DIR}"
  echo "=================================================================="
  TMPDIR=/dev/shm HF_HOME=/dev/shm/huggingface HF_DATASETS_CACHE=/dev/shm/huggingface/datasets HF_TOKEN="${HF_TOKEN}" python3 "${SCRIPT_DIR}/prepare_chartqa_shuffled.py" \
    --output_dir "${CHARTQA_DATASET_DIR}" \
    --num_shards 32 \
    --seed 42 \
    --hf_token "${HF_TOKEN}"
}

prepare_all_data() {
  echo "=================================================================="
  echo ">>> [DATA] Preparing all datasets for ChartNet Core 2-Stage Pipeline..."
  echo "=================================================================="
  prepare_chartnet_core
  echo ""
  prepare_chartqa
}

# ------------------------------------------------------------------------------
# 6. Evaluation (ChartQA Test Benchmark)
# ------------------------------------------------------------------------------
eval_sft() {
  local ckpt_path="${1:-${STAGE2_FINAL_CKPT}}"
  ckpt_path="${ckpt_path%/_CHECKPOINT_METADATA}"
  ckpt_path="${ckpt_path/#https:\/\/storage.googleapis.com\//gs:\/\/}"
  if [[ "${ckpt_path}" != */items ]]; then
    ckpt_path="${ckpt_path%/}/items"
  fi
  local eval_dir="${2:-${EVAL_OUTPUT_DIR}}"
  echo "=================================================================="
  echo ">>> [EVAL] Evaluating Stage 2 SFT Checkpoint: ${ckpt_path}"
  echo ">>> Destination Dir: ${eval_dir}"
  echo "=================================================================="

  MEGASCALE_NUM_SLICES=1 HF_TOKEN="${HF_TOKEN}" HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}" python3 -m maxtext.experimental.omni_pipeline.eval_sft_omni \
    "${SCRIPT_DIR}/sft-omni-gemma3-qwen3-xpk-128.yml" \
    "load_parameters_path=${ckpt_path}" \
    "base_output_directory=${eval_dir}" \
    "hf_access_token=${HF_TOKEN}" \
    "hf_path=HuggingFaceM4/ChartQA" \
    --ckpt_type=sft \
    --num_examples=-1
}

# ------------------------------------------------------------------------------
# Dispatcher
# ------------------------------------------------------------------------------
case "$ACTION" in
  data|prepare_data|predownload)
    prepare_all_data
    ;;
  check_data|data_check|data_status)
    check_data_status
    ;;
  chartnet_data|prepare_chartnet|prepare_chartnet_core)
    prepare_chartnet_core
    ;;
  chartqa_data|prepare_chartqa)
    prepare_chartqa
    ;;
  stitch)
    stitch_ckpt
    ;;
  stage1|pretrain|csv)
    stage1_xpk "${2:-}"
    ;;
  stage2|sft)
    stage2_xpk "${2:-}"
    ;;
  pipeline|all)
    pipeline_xpk
    ;;
  eval)
    eval_sft "${2:-}" "${3:-}"
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
    echo "Usage: $0 <check_data|prepare_data|prepare_chartnet|prepare_chartqa|stitch|stage1|pretrain|stage2|pipeline|eval|list|status|logs|delete> [checkpoint_path|workload_name] [eval_output_dir]"
    echo ""
    echo "Commands:"
    echo "  check_data        Verify that ChartNet Core and ChartQA parquet datasets exist in GCS"
    echo "  prepare_data      Pre-download, shard, and upload ChartNet Core & ChartQA datasets to GCS"
    echo "  stitch            Stitch Gemma 3 Vision + Qwen 3 LLM checkpoints on CPU"
    echo "  stage1 | pretrain Launch Stage 1: ChartNet Core CSV pretraining (1 epoch = ${STAGE1_STEPS} steps, no eval)"
    echo "  stage2 | sft      Launch Stage 2: ChartQA Visual QA SFT (${STAGE2_STEPS} steps) from Stage 1 checkpoint"
    echo "  pipeline          Launch full 2-stage pipeline sequentially in a single XPK job"
    echo "  eval              Run ChartQA evaluation on Stage 2 checkpoint using eval_sft_omni"
    echo "  list              List active XPK workloads"
    echo "  status <name>     Watch status of pods for a workload"
    echo "  logs <name>       Tail TPU container logs for a workload"
    echo "  delete <name>     Delete an XPK workload"
    ;;
esac
