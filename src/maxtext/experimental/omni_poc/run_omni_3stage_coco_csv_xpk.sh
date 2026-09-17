#!/bin/bash
# ==============================================================================
# Omni Gemma3-4B Vision + Qwen3-4B LLM 3-Stage Training Pipeline on XPK
#
# Stage 1: COCO-Narratives Dense Image Captioning Alignment (1,844 steps)
# Stage 2: ChartNet CSV Table Grounding                     (2,172 steps)
# Stage 3: ChartQA Visual QA SFT                           (1,105 steps)
# ==============================================================================

set -e

ACTION="${1:-help}"

: "${HF_TOKEN:?Error: HF_TOKEN is not set. Please run: export HF_TOKEN=hf_...}"
: "${GCS_BUCKET:?Error: GCS_BUCKET is not set. Please run: export GCS_BUCKET=gs://your-bucket}"
export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"


# Global Experiment & Working Directory
EXP_NAME="${EXP_NAME:-omni_gemma3_qwen3_3stage_coco_csv}"
WORKING_DIR="${WORKING_DIR:-${GCS_BUCKET}/omni-gemma3-qwen3/multimodal/3stage_coco_csv_pipeline/${EXP_NAME}}"

# Base Storage Buckets & Initial Checkpoints
VISION_SOURCE_CKPT="${VISION_SOURCE_CKPT:-${GCS_BUCKET}/omni_checkpoints/gemma3-4b_converted/0/items}"
LLM_SOURCE_CKPT="${LLM_SOURCE_CKPT:-${GCS_BUCKET}/omni_checkpoints/qwen3-4b_converted/0/items}"
STITCHED_CKPT="${STITCHED_CKPT:-${GCS_BUCKET}/omni_checkpoints/omni_stitched_gemma3-4b_qwen3-4b/0/items}"

# Pre-sharded Dataset Storage Paths
CHARTNET_DATASET_DIR="${CHARTNET_DATASET_DIR:-${GCS_BUCKET}/datasets/chartnet_sharded}"
CHARTQA_DATASET_DIR="${CHARTQA_DATASET_DIR:-${GCS_BUCKET}/datasets/chartqa_shuffled}"

# Stage 1: COCO-Narratives Alignment (1,844 steps on 128 devices)
STAGE1_OUTPUT_DIR="${WORKING_DIR}/stage1_coco_pretrain"
STAGE1_RUN_NAME="${EXP_NAME}_stage1_coco"
STAGE1_FINAL_CKPT="${STAGE1_OUTPUT_DIR}/${STAGE1_RUN_NAME}/checkpoints/1843/items"

# Stage 2: ChartNet CSV Table Grounding (2,172 steps on 128 devices)
STAGE2_OUTPUT_DIR="${WORKING_DIR}/stage2_chartnet_csv_pretrain"
STAGE2_RUN_NAME="${EXP_NAME}_stage2_chartnet_csv"
STAGE2_FINAL_CKPT="${STAGE2_OUTPUT_DIR}/${STAGE2_RUN_NAME}/checkpoints/2171/items"

# Stage 3: ChartQA Visual QA SFT (1,105 steps on 128 devices)
STAGE3_OUTPUT_DIR="${WORKING_DIR}/stage3_chartqa_sft"
STAGE3_RUN_NAME="${EXP_NAME}_stage3_chartqa_sft"
STAGE3_FINAL_CKPT="${STAGE3_OUTPUT_DIR}/${STAGE3_RUN_NAME}/checkpoints/1104/items"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${STAGE3_OUTPUT_DIR}/eval}"

# XPK Cluster Configuration
XPK_CLUSTER="${XPK_CLUSTER:-v4-128-bodaborg-us-central2-b}"
XPK_PROJECT="${XPK_PROJECT:-cloud-tpu-multipod-dev}"
XPK_ZONE="${XPK_ZONE:-us-central2-b}"
XPK_DEVICE_TYPE="${XPK_DEVICE_TYPE:-v4-128}"
XPK_NUM_SLICES="${XPK_NUM_SLICES:-1}"
XPK_BASE_DOCKER_IMAGE="${XPK_BASE_DOCKER_IMAGE:-gcr.io/tpu-prod-env-multipod/maxtext_base_image:latest}"

USER_PREFIX="${USER_PREFIX:-user}"
TIMESTAMP=$(date +%m%d-%H%M)

# Hugging Face Token & Auth
export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAXTEXT_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || (cd "${SCRIPT_DIR}/../../../../.." && pwd))"

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
    HF_TOKEN="${HF_TOKEN}" HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}" JAX_PLATFORMS=cpu python3 -m maxtext.experimental.omni_poc.utils.stitch_checkpoint \
      "${SCRIPT_DIR}/maxtext-omni-gemma3-qwen3.yml" \
      "hf_access_token=${HF_TOKEN}" \
      "vision_load_path=${VISION_SOURCE_CKPT}" \
      "llm_load_path=${LLM_SOURCE_CKPT}" \
      "stitched_output_path=${STITCHED_CKPT}"
  )
}

# ------------------------------------------------------------------------------
# 2. Stage 1: COCO-Narratives Pretraining on XPK
# ------------------------------------------------------------------------------
stage1_xpk() {
  local input_ckpt="${1:-${STITCHED_CKPT}}"
  local workload_name="${USER_PREFIX}-omni-s1-coco-${TIMESTAMP}"

  echo "=================================================================="
  echo ">>> [XPK STAGE 1 - COCO Narratives] Submitting Workload: ${workload_name}"
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
      --command "export MEGASCALE_NUM_SLICES=${XPK_NUM_SLICES} && export TMPDIR=/dev/shm && export PYTHONPATH=src:\${PYTHONPATH:-} && export HF_HOME=/dev/shm/huggingface && export HF_TOKEN=${HF_TOKEN} && export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} && pip install --no-cache-dir -U 'orbax-checkpoint>=0.12.4' && python3 -m maxtext.experimental.omni_poc.train_sft_omni src/maxtext/experimental/omni_poc/pretrain-omni-gemma3-qwen3-coco-xpk-128.yml load_parameters_path=${input_ckpt} base_output_directory=${STAGE1_OUTPUT_DIR} run_name=${STAGE1_RUN_NAME} hf_access_token=${HF_TOKEN} scan_layers=true grain_worker_count=0"
  )
}

# ------------------------------------------------------------------------------
# 3. Stage 2: ChartNet CSV Pretraining on XPK
# ------------------------------------------------------------------------------
stage2_xpk() {
  local input_ckpt="${1:-${STAGE1_FINAL_CKPT}}"
  local workload_name="${USER_PREFIX}-omni-s2-csv-${TIMESTAMP}"

  echo "=================================================================="
  echo ">>> [XPK STAGE 2 - ChartNet CSV] Submitting Workload: ${workload_name}"
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
      --command "export MEGASCALE_NUM_SLICES=${XPK_NUM_SLICES} && export TMPDIR=/dev/shm && export PYTHONPATH=src:\${PYTHONPATH:-} && export HF_HOME=/dev/shm/huggingface && export HF_TOKEN=${HF_TOKEN} && export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} && pip install --no-cache-dir -U 'orbax-checkpoint>=0.12.4' && gcloud storage cp -r ${CHARTNET_DATASET_DIR} /dev/shm/ && python3 -m maxtext.experimental.omni_poc.train_sft_omni src/maxtext/experimental/omni_poc/pretrain-omni-gemma3-qwen3-chartnet-xpk-128-csv.yml load_parameters_path=${input_ckpt} base_output_directory=${STAGE2_OUTPUT_DIR} run_name=${STAGE2_RUN_NAME} hf_access_token=${HF_TOKEN} scan_layers=true grain_worker_count=0"
  )
}

# ------------------------------------------------------------------------------
# 4. Stage 3: ChartQA SFT on XPK
# ------------------------------------------------------------------------------
stage3_xpk() {
  local input_ckpt="${1:-${STAGE2_FINAL_CKPT}}"
  local workload_name="${USER_PREFIX}-omni-s3-sft-${TIMESTAMP}"

  echo "=================================================================="
  echo ">>> [XPK STAGE 3 - ChartQA SFT (Projector Only)] Submitting Workload: ${workload_name}"
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
      --command "export MEGASCALE_NUM_SLICES=${XPK_NUM_SLICES} && export TMPDIR=/dev/shm && export PYTHONPATH=src:\${PYTHONPATH:-} && export HF_HOME=/dev/shm/huggingface && export HF_TOKEN=${HF_TOKEN} && export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} && pip install --no-cache-dir -U 'orbax-checkpoint>=0.12.4' && gcloud storage cp -r ${CHARTQA_DATASET_DIR} /dev/shm/ && python3 -m maxtext.experimental.omni_poc.train_sft_omni src/maxtext/experimental/omni_poc/sft-omni-gemma3-qwen3-xpk-128.yml load_parameters_path=${input_ckpt} base_output_directory=${STAGE3_OUTPUT_DIR} run_name=${STAGE3_RUN_NAME} hf_access_token=${HF_TOKEN} scan_layers=true grain_worker_count=0"
  )
}

# ------------------------------------------------------------------------------
# 5. Full 3-Stage Pipeline on XPK (Stage 1 -> Stage 2 -> Stage 3 in ONE Job)
# ------------------------------------------------------------------------------
pipeline_xpk() {
  local workload_name="${USER_PREFIX}-omni-3s-coco-csv-${TIMESTAMP}"

  echo "=================================================================="
  echo ">>> [XPK FULL 3-STAGE PIPELINE (COCO -> ChartNet CSV -> ChartQA)]"
  echo ">>> Submitting Workload: ${workload_name}"
  echo ">>> Base Working Dir:    ${WORKING_DIR}"
  echo ">>> Stage 1: COCO-Narratives Alignment    (1,844 steps -> ${STAGE1_FINAL_CKPT})"
  echo ">>> Stage 2: ChartNet CSV Table Grounding (2,172 steps -> ${STAGE2_FINAL_CKPT})"
  echo ">>> Stage 3: ChartQA Visual QA SFT        (1,105 steps -> ${STAGE3_FINAL_CKPT})"
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
      --command "export MEGASCALE_NUM_SLICES=${XPK_NUM_SLICES} && export TMPDIR=/dev/shm && export PYTHONPATH=src:\${PYTHONPATH:-} && export HF_HOME=/dev/shm/huggingface && export HF_TOKEN=${HF_TOKEN} && export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} && pip install --no-cache-dir -U 'orbax-checkpoint>=0.12.4' && gcloud storage cp -r ${CHARTNET_DATASET_DIR} /dev/shm/ && gcloud storage cp -r ${CHARTQA_DATASET_DIR} /dev/shm/ && echo '=== Stage 1: COCO Narratives Pretraining ===' && python3 -m maxtext.experimental.omni_poc.train_sft_omni src/maxtext/experimental/omni_poc/pretrain-omni-gemma3-qwen3-coco-xpk-128.yml load_parameters_path=${STITCHED_CKPT} base_output_directory=${STAGE1_OUTPUT_DIR} run_name=${STAGE1_RUN_NAME} hf_access_token=${HF_TOKEN} scan_layers=true grain_worker_count=0 && echo '=== Stage 2: ChartNet CSV Table Grounding ===' && python3 -m maxtext.experimental.omni_poc.train_sft_omni src/maxtext/experimental/omni_poc/pretrain-omni-gemma3-qwen3-chartnet-xpk-128-csv.yml load_parameters_path=${STAGE1_FINAL_CKPT} base_output_directory=${STAGE2_OUTPUT_DIR} run_name=${STAGE2_RUN_NAME} hf_access_token=${HF_TOKEN} scan_layers=true grain_worker_count=0 && echo '=== Stage 3: ChartQA Visual QA SFT ===' && python3 -m maxtext.experimental.omni_poc.train_sft_omni src/maxtext/experimental/omni_poc/sft-omni-gemma3-qwen3-xpk-128.yml load_parameters_path=${STAGE2_FINAL_CKPT} base_output_directory=${STAGE3_OUTPUT_DIR} run_name=${STAGE3_RUN_NAME} hf_access_token=${HF_TOKEN} scan_layers=true grain_worker_count=0"
  )

  echo ""
  echo ">>> 3-Stage Pipeline Workload submitted: ${workload_name}"
  echo "=================================================================="
}

# ------------------------------------------------------------------------------
# 6. Dataset Preparation (Pre-download, Global Shuffle, and Shard to GCS)
# ------------------------------------------------------------------------------
prepare_chartnet() {
  echo "=================================================================="
  echo ">>> [DATA] Preparing Sharded ChartNet Dataset (Cached in /dev/shm)"
  echo ">>> Output Dir: ${CHARTNET_DATASET_DIR}"
  echo "=================================================================="
  TMPDIR=/dev/shm HF_HOME=/dev/shm/huggingface HF_DATASETS_CACHE=/dev/shm/huggingface/datasets HF_TOKEN="${HF_TOKEN}" python3 "${SCRIPT_DIR}/prepare_chartnet_sharded.py" \
    --output_dir "${CHARTNET_DATASET_DIR}" \
    --num_shards 32 \
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
  echo ">>> [DATA] Preparing all datasets for 3-Stage Pipeline..."
  echo "=================================================================="
  prepare_chartnet
  echo ""
  prepare_chartqa
}

# ------------------------------------------------------------------------------
# 7. Evaluation
# ------------------------------------------------------------------------------
eval_sft() {
  local ckpt_path="${1:-${STAGE3_FINAL_CKPT}}"
  ckpt_path="${ckpt_path%/_CHECKPOINT_METADATA}"
  ckpt_path="${ckpt_path/#https:\/\/storage.googleapis.com\//gs:\/\/}"
  if [[ "${ckpt_path}" != */items ]]; then
    ckpt_path="${ckpt_path%/}/items"
  fi
  local eval_dir="${2:-${EVAL_OUTPUT_DIR}}"
  echo "=================================================================="
  echo ">>> [EVAL] Evaluating Stage 3 SFT Checkpoint: ${ckpt_path}"
  echo ">>> Destination Dir: ${eval_dir}"
  echo "=================================================================="

  MEGASCALE_NUM_SLICES=1 HF_TOKEN="${HF_TOKEN}" HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}" python3 -m maxtext.experimental.omni_poc.eval_sft_omni \
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
  chartnet_data|prepare_chartnet)
    prepare_chartnet
    ;;
  chartqa_data|prepare_chartqa)
    prepare_chartqa
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
    echo "Usage: $0 <prepare_data|prepare_chartnet|prepare_chartqa|stitch|stage1|stage2|stage3|pipeline|eval|list|status|logs|delete> [checkpoint_path|workload_name] [eval_output_dir]"
    ;;
esac
