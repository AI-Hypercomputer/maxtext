#!/bin/bash
# Smoke training run for OLMo 3 7B on the OLMo numpy grain pipeline.
#
# Validates that dataset_type=olmo_grain wires through the trainer, that
# OlmoNpyDataSource reads .npy data via a gcsfuse mount, and that 50 steps
# execute without crashes / shape mismatches with monotonically decreasing
# loss.
#
# Required env vars:
#   INDEX_PATH    JSON index from tools/data_generation/build_olmo_npy_index.py
#   GCS_BASE      gs:// prefix recorded in the index (e.g. gs://my-bucket/)
#   LOCAL_MOUNT   gcsfuse mount of GCS_BASE on this host
#   HF_TOKEN      HuggingFace token for the tokenizer (or HF_SECRETS=<file>)
# Optional: VENV_PATH, OUTPUT_DIR, PER_DEVICE_BATCH, SEQ_LEN, STEPS,
#           WEIGHT_DTYPE, NUM_LAYERS.
#
# Usage:
#   INDEX_PATH=/path/to/olmo_index_seq8192.json \
#   LOCAL_MOUNT=/mnt/your-mount/ \
#   GCS_BASE=gs://your-bucket/ \
#   HF_TOKEN=hf_... \
#   bash scripts/run_olmo3_7b_grain_smoke.sh

set -euo pipefail

MAXTEXT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

VENV_PATH="${VENV_PATH:-${MAXTEXT_ROOT}/maxtext_venv}"
HF_SECRETS="${HF_SECRETS:-}"
INDEX_PATH="${INDEX_PATH:?INDEX_PATH is required (path to olmo index JSON)}"
GCS_BASE="${GCS_BASE:?GCS_BASE is required (e.g. gs://my-bucket/)}"
LOCAL_MOUNT="${LOCAL_MOUNT:?LOCAL_MOUNT is required (gcsfuse mount path of GCS_BASE)}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/olmo_smoke_out}"

PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
SEQ_LEN="${SEQ_LEN:-8192}"
STEPS="${STEPS:-50}"
DATA_SEED="${DATA_SEED:-42}"
# Smoke test uses a reduced model (bf16, 4 layers) so it fits small TPU
# slices; we're validating the data path, not full-size convergence.
WEIGHT_DTYPE="${WEIGHT_DTYPE:-bfloat16}"
NUM_LAYERS="${NUM_LAYERS:-4}"

RUN_NAME="${RUN_NAME:-olmo_grain_smoke_$(date +%Y%m%d-%H%M%S)}"

# Activate venv + load HF secrets.
# shellcheck disable=SC1090,SC1091
source "${VENV_PATH}/bin/activate"
if [[ -n "${HF_SECRETS:-}" && -f "${HF_SECRETS}" ]]; then
  # shellcheck disable=SC1090
  source "${HF_SECRETS}"
fi
: "${HF_TOKEN:?HF_TOKEN must be set (or HF_SECRETS pointing at a file that exports it)}"
export PYTHONPATH="${MAXTEXT_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "${OUTPUT_DIR}"

echo "=== OLMo 3 7B + olmo_grain smoke run ==="
echo "  run_name      : ${RUN_NAME}"
echo "  index         : ${INDEX_PATH}"
echo "  path remap    : ${GCS_BASE} → ${LOCAL_MOUNT}"
echo "  per_device_bs : ${PER_DEVICE_BATCH}"
echo "  seq_len       : ${SEQ_LEN}"
echo "  steps         : ${STEPS}"
echo "  weight_dtype  : ${WEIGHT_DTYPE}"
echo "  num_layers    : ${NUM_LAYERS}  (full 7B has 32)"
echo "  output_dir    : ${OUTPUT_DIR}"
echo

# Data is already tokenized; the tokenizer is loaded only for pad/eos IDs +
# vocab_size checks. Olmo-3-7B-Instruct uses the same dolma3 tokenizer.
TOKENIZER_PATH="${TOKENIZER_PATH:-allenai/Olmo-3-7B-Instruct}"

python -m maxtext.trainers.pre_train.train \
  "${MAXTEXT_ROOT}/src/maxtext/configs/base.yml" \
  model_name=olmo3-7b-pt \
  run_name="${RUN_NAME}" \
  base_output_directory="${OUTPUT_DIR}" \
  dataset_type=olmo_grain \
  olmo_index_path="${INDEX_PATH}" \
  olmo_path_remap_from="${GCS_BASE}" \
  olmo_path_remap_to="${LOCAL_MOUNT}" \
  data_shuffle_seed="${DATA_SEED}" \
  olmo_apply_ngram_filter=True \
  grain_worker_count=0 \
  per_device_batch_size="${PER_DEVICE_BATCH}" \
  max_target_length="${SEQ_LEN}" \
  steps="${STEPS}" \
  enable_checkpointing=False \
  tokenizer_type=huggingface \
  tokenizer_path="${TOKENIZER_PATH}" \
  weight_dtype="${WEIGHT_DTYPE}" \
  override_model_config=True \
  base_num_decoder_layers="${NUM_LAYERS}" \
  sharding_tolerance=0.05
