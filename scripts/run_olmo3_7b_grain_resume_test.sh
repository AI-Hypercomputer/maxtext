#!/bin/bash
# End-to-end resume test for the OLMo grain pipeline (stateless sampler +
# step-derived initial_step). See scripts/run_olmo3_7b_grain_smoke.sh for
# the env-var contract; this script accepts the same vars.
#
# Plan:
#   Run A: train 50 steps from scratch, save checkpoint at step 50, exit.
#   Run B: relaunch with the SAME run_name (so the checkpoint dir is reused).
#          The trainer restores model state at step 50; our iterator factory
#          detects the latest checkpoint step and sets ``initial_step`` so
#          the data stream picks up at absolute position 50 * per_host_batch.
#          Train 25 more steps (to step 75).
#
# What success looks like:
#   * Run B's first step (step 51) reports a loss similar to Run A's step 50
#     loss. A spike or jump → model state didn't restore.
#   * No repeats: Run B's batches are NOT the same as Run A's batches at the
#     same absolute step. (Hard to assert without batch-content hashing in
#     the trainer; for the smoke we rely on the unit tests + loss continuity.)
#   * No regression: Run B's loss continues to decrease.
#
# Outputs:
#   ${LOG_A}  — first 50 steps
#   ${LOG_B}  — resumed 25 steps
#   $OUTPUT_DIR/<run_name>/checkpoints/  — Orbax checkpoint(s)

set -euo pipefail

MAXTEXT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PATH="${VENV_PATH:-${MAXTEXT_ROOT}/maxtext_venv}"
HF_SECRETS="${HF_SECRETS:-}"
INDEX_PATH="${INDEX_PATH:?INDEX_PATH is required (path to olmo index JSON)}"
GCS_BASE="${GCS_BASE:?GCS_BASE is required (e.g. gs://my-bucket/)}"
LOCAL_MOUNT="${LOCAL_MOUNT:?LOCAL_MOUNT is required (gcsfuse mount path of GCS_BASE)}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/olmo_resume_test_out}"
RUN_NAME="${RUN_NAME:-olmo_resume_$(date +%Y%m%d-%H%M%S)}"

# Where each run's stdout is teed. Keep them under OUTPUT_DIR so the
# script doesn't depend on a hard-coded absolute path.
LOG_A="${LOG_A:-${OUTPUT_DIR}/${RUN_NAME}.runA.log}"
LOG_B="${LOG_B:-${OUTPUT_DIR}/${RUN_NAME}.runB.log}"

PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
SEQ_LEN="${SEQ_LEN:-8192}"
WEIGHT_DTYPE="${WEIGHT_DTYPE:-bfloat16}"
NUM_LAYERS="${NUM_LAYERS:-4}"
DATA_SEED="${DATA_SEED:-42}"

# Run A trains 50 steps + saves a checkpoint at step 50; Run B continues to 75.
STEPS_A="${STEPS_A:-50}"
STEPS_B="${STEPS_B:-75}"
CHECKPOINT_PERIOD="${CHECKPOINT_PERIOD:-50}"

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

TOKENIZER_PATH="${TOKENIZER_PATH:-allenai/Olmo-3-7B-Instruct}"

run_train() {
  local steps="$1"
  local logfile="$2"
  echo "----- launching: steps=${steps}  log=${logfile} -----"
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
    steps="${steps}" \
    enable_checkpointing=True \
    async_checkpointing=False \
    checkpoint_period="${CHECKPOINT_PERIOD}" \
    save_checkpoint_on_completion=True \
    tokenizer_type=huggingface \
    tokenizer_path="${TOKENIZER_PATH}" \
    weight_dtype="${WEIGHT_DTYPE}" \
    override_model_config=True \
    base_num_decoder_layers="${NUM_LAYERS}" \
    sharding_tolerance=0.05 \
    2>&1 | tee "${logfile}"
}

echo "=== OLMo 3 grain resume test ==="
echo "  run_name      : ${RUN_NAME}"
echo "  output_dir    : ${OUTPUT_DIR}/${RUN_NAME}"
echo "  per_device_bs : ${PER_DEVICE_BATCH}"
echo "  seq_len       : ${SEQ_LEN}"
echo "  num_layers    : ${NUM_LAYERS}"
echo "  Run A steps   : ${STEPS_A}  (will checkpoint at step ${CHECKPOINT_PERIOD})"
echo "  Run B steps   : ${STEPS_B}  (resumed via initial_step)"
echo

# Run A
run_train "${STEPS_A}" "${LOG_A}"

echo
echo "=== Run A done. Last 3 step events: ==="
grep -E "completed step:" "${LOG_A}" | tail -3
echo

# Run B (resume)
run_train "${STEPS_B}" "${LOG_B}"

echo
echo "=== Run B done ==="
echo "First 3 step events from Run B (expect step >= ${STEPS_A}):"
grep -E "completed step:" "${LOG_B}" | head -3
echo
echo "Last 3 step events from Run B:"
grep -E "completed step:" "${LOG_B}" | tail -3
echo

echo "=== Pass criteria (manual check): ==="
echo "  1. Run B's first step number >= ${STEPS_A} (model state restored)"
echo "  2. Run B's first step loss within ~5% of Run A's last step loss"
echo "     (model continued, no re-init)"
echo "  3. Loss continues to decrease across Run B"
echo "  4. iterator log line shows 'resumed_step=${STEPS_A} initial_step=...' on Run B"
