#!/usr/bin/env bash
# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Staged TPU end-to-end verification script for multimodal DeepSeek-V4.1-Flash.
#
# Environment variables:
#   HF_CKPT_PATH            Path to the release SafeTensors checkpoint directory (required for all stages).
#   OUTPUT_DIR              Directory for summaries, metrics, and logs (required for all stages).
#   BASE_OUTPUT_DIRECTORY   Optional MaxText base_output_directory override (defaults to OUTPUT_DIR).
#   GOLDEN_TEXT_PKL         Path to 4-prompt text golden pickle (required for scanned_logits, unscanned_logits).
#   GOLDEN_MM_DIR           Optional local directory containing L1..L4, M1..M8, mm_logit.pkl (staged from MM_GOLDEN_GCS_PREFIX if unset).
#   MM_GOLDEN_GCS_PREFIX    GCS prefix for multimodal goldens when GOLDEN_MM_DIR is unset (default: gs://maxtext-deepseek/deepseek-v4.1-flash/golden/mm_v3).
#   GOLDEN_PREFILL_TEXT_PKL Path to 7-record text masked-prefill golden pickle (required when RUN_TEXT_PREFILL=1 in masked_prefill).
#   ORBAX_EXPORT_PREFIX     Optional prefix for exporting verified scanned/unscanned Orbax checkpoints in logit stages.
#   EXPORT_ORBAX            Set to "1" when train_e2e resume must also verify scanned/unscanned Orbax exports exist (default: "0").
#   GCS_LOG_PREFIX          Optional GCS prefix for archiving stage JSON/log/metrics artifacts on worker 0.
#   RESUME_PASSED_STAGES    Set to "1" to skip stages whose summary JSON already passed in GCS_RESUME_SUMMARY_PREFIX (default: "0").
#   GCS_RESUME_SUMMARY_PREFIX Directory or gs:// prefix checked when RESUME_PASSED_STAGES=1 (defaults to GCS_LOG_PREFIX).
#   PYTHON_BIN              Python interpreter executable (default: "python3").
#   TOKENIZER_PATH          Optional HuggingFace tokenizer directory path for offline TPU containers.
#   STAGES                  Comma-separated stages to execute (default: "scanned_logits,unscanned_logits,masked_prefill,train_e2e").
#   ENABLE_LONG_TEXT_GATE   Set to "1" to enable the long-text forward logit gate in logit stages (default: "0").
#   RUN_TEXT_PREFILL        Set to "1" to include GOLDEN_PREFILL_TEXT_PKL in masked_prefill (default: "0").
#   TRAIN_STEPS             Number of training steps in train_e2e (default: "20").

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
BASE_CONFIG="${MAXTEXT_CONFIGS_DIR:-${REPO_ROOT}/src/maxtext/configs}/base.yml"
THRESHOLDS_JSON="${THRESHOLDS_JSON:-${SCRIPT_DIR}/thresholds.json}"
MM_GOLDEN_GCS_PREFIX="${MM_GOLDEN_GCS_PREFIX:-gs://maxtext-deepseek/deepseek-v4.1-flash/golden/mm_v3}"
GOLDEN_MM_DIR="${GOLDEN_MM_DIR:-}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
ALLOW_REFERENCE_TIE_BRANCHES="${ALLOW_REFERENCE_TIE_BRANCHES:-1}"
TIE_BRANCH_DIR="${TIE_BRANCH_DIR:-gs://maxtext-deepseek/deepseek-v4.1-flash/prefill_goldens_v2_alt_branches/}"
TIE_BRANCH_MAX_ULP="${TIE_BRANCH_MAX_ULP:-8.0}"
STAGES="${STAGES:-scanned_logits,unscanned_logits,masked_prefill,train_e2e}"
ENABLE_LONG_TEXT_GATE="${ENABLE_LONG_TEXT_GATE:-0}"
RUN_TEXT_PREFILL="${RUN_TEXT_PREFILL:-0}"
TRAIN_STEPS="${TRAIN_STEPS:-20}"
ORBAX_EXPORT_PREFIX="${ORBAX_EXPORT_PREFIX:-}"
EXPORT_ORBAX="${EXPORT_ORBAX:-0}"
GCS_LOG_PREFIX="${GCS_LOG_PREFIX:-}"
RESUME_PASSED_STAGES="${RESUME_PASSED_STAGES:-0}"
GCS_RESUME_SUMMARY_PREFIX="${GCS_RESUME_SUMMARY_PREFIX:-${GCS_LOG_PREFIX}}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"

# Multi-worker GCS startup barrier settings
BARRIER_START_EPOCH="${BARRIER_START_EPOCH:-$(date -u +%s)}"
WORKER_ID="${WORKER_ID:-${JOB_COMPLETION_INDEX:-${TPU_WORKER_ID:-0}}}"
NUM_WORKERS="${NUM_WORKERS:-1}"
RUN_ATTEMPT_ID="${RUN_ATTEMPT_ID:-${JOB_ATTEMPT_NUMBER:-${POD_RESTART_COUNT:-}}}"
MAX_BARRIER_SKEW_SEC="${MAX_BARRIER_SKEW_SEC:-180}"
BARRIER_TIMEOUT_SEC="${BARRIER_TIMEOUT_SEC:-600}"
BARRIER_POLL_SEC="${BARRIER_POLL_SEC:-5}"
CLEAN_STALE_READY_ON_START="${CLEAN_STALE_READY_ON_START:-0}"
ENABLE_GCS_STARTUP_BARRIER="${ENABLE_GCS_STARTUP_BARRIER:-0}"
GCS_READY_DIR="${GCS_READY_DIR:-${READY_DIR:-}}"
if [[ -z "${GCS_READY_DIR}" && -n "${GCS_LOG_PREFIX}" ]]; then
  if [[ -n "${RUN_ATTEMPT_ID}" ]]; then
    GCS_READY_DIR="${GCS_LOG_PREFIX%/}/ready/attempt_${RUN_ATTEMPT_ID}"
  else
    GCS_READY_DIR="${GCS_LOG_PREFIX%/}/ready"
  fi
fi

LONG_TEXT_GATE_FLAG="--enable_long_text_gate=false"
case "${ENABLE_LONG_TEXT_GATE}" in
  0|false|FALSE)
    LONG_TEXT_GATE_FLAG="--enable_long_text_gate=false"
    ;;
  1|true|TRUE)
    LONG_TEXT_GATE_FLAG="--enable_long_text_gate=true"
    ;;
  *)
    echo "ERROR: Invalid ENABLE_LONG_TEXT_GATE=${ENABLE_LONG_TEXT_GATE}; expected 0/1 or false/true." >&2
    exit 1
    ;;
esac

IFS=',' read -r -a REQUESTED_STAGES <<< "${STAGES}"
if [[ "${#REQUESTED_STAGES[@]}" -eq 0 || -z "${STAGES}" ]]; then
  echo "ERROR: STAGES must contain one or more comma-separated stages from {scanned_logits,unscanned_logits,masked_prefill,train_e2e}." >&2
  exit 1
fi
for s_tok in "${REQUESTED_STAGES[@]}"; do
  case "${s_tok}" in
    scanned_logits|unscanned_logits|masked_prefill|train_e2e) ;;
    *)
      echo "ERROR: Invalid stage token '${s_tok}' in STAGES='${STAGES}'; expected comma-separated {scanned_logits,unscanned_logits,masked_prefill,train_e2e} without whitespace." >&2
      exit 1
      ;;
  esac
done

stage_enabled() {
  local target="$1"
  [[ ",${STAGES}," == *",${target},"* ]]
}

require_var() {
  local var_name="$1"
  local stage_label="$2"
  if [[ -z "${!var_name:-}" ]]; then
    echo "ERROR: Required environment variable ${var_name} is unset or empty for stage ${stage_label}." >&2
    exit 1
  fi
}

validate_environment() {
  for stage_id in scanned_logits unscanned_logits masked_prefill train_e2e; do
    if stage_enabled "${stage_id}"; then
      require_var "HF_CKPT_PATH" "${stage_id}"
      require_var "OUTPUT_DIR" "${stage_id}"
    fi
  done

  if stage_enabled "scanned_logits"; then
    require_var "GOLDEN_TEXT_PKL" "scanned_logits"
    if [[ -z "${GOLDEN_MM_DIR}" ]]; then
      require_var "MM_GOLDEN_GCS_PREFIX" "scanned_logits"
    fi
  fi

  if stage_enabled "unscanned_logits"; then
    require_var "GOLDEN_TEXT_PKL" "unscanned_logits"
    if [[ -z "${GOLDEN_MM_DIR}" ]]; then
      require_var "MM_GOLDEN_GCS_PREFIX" "unscanned_logits"
    fi
  fi

  if stage_enabled "masked_prefill"; then
    if [[ -z "${GOLDEN_MM_DIR}" ]]; then
      require_var "MM_GOLDEN_GCS_PREFIX" "masked_prefill"
    fi
    if [[ "${RUN_TEXT_PREFILL}" == "1" ]]; then
      require_var "GOLDEN_PREFILL_TEXT_PKL" "masked_prefill"
    fi
  fi

  if [[ -n "${OUTPUT_DIR:-}" ]]; then
    mkdir -p "${OUTPUT_DIR}"
    BASE_OUTPUT_DIRECTORY="${BASE_OUTPUT_DIRECTORY:-${OUTPUT_DIR}}"
  fi
}

TOKENIZER_ARGS=()
if [[ -n "${TOKENIZER_PATH}" ]]; then
  TOKENIZER_ARGS=(tokenizer_type=huggingface "tokenizer_path=${TOKENIZER_PATH}")
fi

sync_outputs_to_gcs() {
  local worker_id="${WORKER_ID:-${JOB_COMPLETION_INDEX:-${TPU_WORKER_ID:-0}}}"
  if [[ -z "${GCS_LOG_PREFIX}" ]]; then
    return 0
  fi
  if ! command -v gcloud >/dev/null 2>&1; then
    echo "ERROR: GCS_LOG_PREFIX=${GCS_LOG_PREFIX} is set, but gcloud was not found on PATH." >&2
    return 1
  fi
  shopt -s nullglob
  local summary_artifacts=(
      "${OUTPUT_DIR}"/*.json
      "${OUTPUT_DIR}"/masked_prefill_sample_reports.jsonl
  )
  local log_artifacts=("${OUTPUT_DIR}"/*.log "${OUTPUT_DIR}"/train_e2e_metrics.jsonl)
  shopt -u nullglob
  for artifact in "${summary_artifacts[@]}"; do
    if [[ -f "${artifact}" ]]; then
      local fname
      fname="$(basename "${artifact}")"
      if [[ "${worker_id}" == "0" ]]; then
        gcloud storage cp "${artifact}" "${GCS_LOG_PREFIX%/}/${fname}" || return 1
      fi
      gcloud storage cp "${artifact}" "${GCS_LOG_PREFIX%/}/worker_${worker_id}_${fname}" || return 1
    fi
  done

  for log_file in "${log_artifacts[@]}"; do
    if [[ -f "${log_file}" ]]; then
      local lfname
      lfname="$(basename "${log_file}")"
      if [[ "${worker_id}" == "0" ]]; then
        gcloud storage cp "${log_file}" "${GCS_LOG_PREFIX%/}/${lfname}" || return 1
      else
        gcloud storage cp "${log_file}" "${GCS_LOG_PREFIX%/}/worker_${worker_id}_${lfname}" || return 1
      fi
    fi
  done
  return 0
}

stage_already_passed() {
  local stage="$1"
  if [[ "${RESUME_PASSED_STAGES}" != "1" || -z "${GCS_RESUME_SUMMARY_PREFIX}" ]]; then
    return 1
  fi
  local summary_file=""
  case "${stage}" in
    scanned_logits) summary_file="scanned_logits_summary.json" ;;
    unscanned_logits) summary_file="unscanned_logits_summary.json" ;;
    masked_prefill) summary_file="masked_prefill_summary.json" ;;
    train_e2e) summary_file="train_e2e_summary.json" ;;
    *)
      echo "ERROR: Unknown stage ${stage} in stage_already_passed." >&2
      return 1
      ;;
  esac
  if "${PYTHON_BIN}" - "${stage}" "${GCS_RESUME_SUMMARY_PREFIX%/}/${summary_file}" "${OUTPUT_DIR}/${summary_file}" "${EXPORT_ORBAX}" "${ORBAX_EXPORT_PREFIX%/}" <<'PY'
import json
import sys
from etils import epath

stage, remote_uri, local_dst, export_orbax, orbax_prefix = sys.argv[1:6]
remote_path = epath.Path(remote_uri)
if not remote_path.exists():
  sys.exit(1)
raw_text = remote_path.read_text(encoding="utf-8")
try:
  data = json.loads(raw_text)
except json.JSONDecodeError:
  sys.exit(1)
if not isinstance(data, dict) or not ("passed" in data and data["passed"] is True):
  sys.exit(1)
if stage == "train_e2e" and export_orbax == "1" and orbax_prefix:
  if not (
      epath.Path(f"{orbax_prefix}/scanned/0/items").exists()
      and epath.Path(f"{orbax_prefix}/unscanned/0/items").exists()
  ):
    sys.exit(1)
dst_path = epath.Path(local_dst)
dst_path.parent.mkdir(parents=True, exist_ok=True)
if remote_path.as_posix() != dst_path.as_posix():
  dst_path.write_text(raw_text, encoding="utf-8")
sys.exit(0)
PY
  then
    echo "[2_test_deepseek41_mm] Stage ${stage} already passed in ${GCS_RESUME_SUMMARY_PREFIX}; skipping (RESUME_PASSED_STAGES=1)."
    return 0
  fi
  return 1
}

run_scanned_logits() {
  local orbax_args=()
  if [[ -n "${ORBAX_EXPORT_PREFIX}" ]]; then
    orbax_args=(--export_orbax_dir="${ORBAX_EXPORT_PREFIX%/}/scanned")
  fi
  local log_file="${OUTPUT_DIR}/scanned_logits.log"
  local checker_rc=0
  "${PYTHON_BIN}" -m tests.utils.deepseek_v41.logit_runner \
    "${BASE_CONFIG}" \
    base_output_directory="${BASE_OUTPUT_DIRECTORY}" \
    run_name=ds41_mm_scanned_logits \
    model_name=deepseek4.1-flash \
    override_model_config=true \
    load_parameters_path="${HF_CKPT_PATH}" \
    source_checkpoint_layout=safetensors_dynamic \
    scan_layers=true \
    max_target_length="${SCANNED_LOGITS_MAX_TARGET_LENGTH:-1024}" \
    ici_fsdp_parallelism=-1 \
    ici_expert_parallelism=16 \
    sparse_matmul=false \
    capacity_factor=-1 \
    use_multimodal=true \
    vision_encoder_block=deepseek41 \
    ds41_vision_grid='[39,39]' \
    per_device_batch_size=0.125 \
    dtype=float32 \
    weight_dtype=bfloat16 \
    dtype_mm=float32 \
    matmul_precision=highest \
    remat_policy=none \
    ds41_activation_quant=false \
    "${TOKENIZER_ARGS[@]}" \
    --run_ds41_mm_audit=true \
    --hf_checkpoint_dir_for_audit="${HF_CKPT_PATH}" \
    --golden_logits_path="${GOLDEN_TEXT_PKL}" \
    --mm_golden_logits_path="${GOLDEN_MM_DIR%/}/mm_logit.pkl" \
    --require_frozen_topk_gate=true \
    --thresholds_json="${THRESHOLDS_JSON}" \
    "${LONG_TEXT_GATE_FLAG}" \
    --summary_json="${OUTPUT_DIR}/scanned_logits_summary.json" \
    "${orbax_args[@]}" 2>&1 | tee -a "${log_file}" || checker_rc=$?

  sync_outputs_to_gcs || true
  if [[ "${checker_rc}" -ne 0 ]]; then
    echo "ERROR: Stage scanned_logits checker failed with exit code ${checker_rc}." >&2
    return "${checker_rc}"
  fi
  return 0
}

run_unscanned_logits() {
  local orbax_args=()
  if [[ -n "${ORBAX_EXPORT_PREFIX}" ]]; then
    orbax_args=(--export_orbax_dir="${ORBAX_EXPORT_PREFIX%/}/unscanned")
  fi
  local log_file="${OUTPUT_DIR}/unscanned_logits.log"
  local checker_rc=0
  "${PYTHON_BIN}" -m tests.utils.deepseek_v41.logit_runner \
    "${BASE_CONFIG}" \
    base_output_directory="${BASE_OUTPUT_DIRECTORY}" \
    run_name=ds41_mm_unscanned_logits \
    model_name=deepseek4.1-flash \
    override_model_config=true \
    load_parameters_path="${HF_CKPT_PATH}" \
    source_checkpoint_layout=safetensors_dynamic \
    scan_layers=false \
    max_target_length="${UNSCANNED_LOGITS_MAX_TARGET_LENGTH:-1024}" \
    ici_fsdp_parallelism=-1 \
    ici_expert_parallelism=16 \
    sparse_matmul=false \
    capacity_factor=-1 \
    use_multimodal=true \
    vision_encoder_block=deepseek41 \
    ds41_vision_grid='[39,39]' \
    per_device_batch_size=0.125 \
    dtype=float32 \
    weight_dtype=bfloat16 \
    dtype_mm=float32 \
    matmul_precision=highest \
    remat_policy=none \
    ds41_activation_quant=false \
    "${TOKENIZER_ARGS[@]}" \
    --run_ds41_mm_audit=true \
    --hf_checkpoint_dir_for_audit="${HF_CKPT_PATH}" \
    --golden_logits_path="${GOLDEN_TEXT_PKL}" \
    --mm_golden_logits_path="${GOLDEN_MM_DIR%/}/mm_logit.pkl" \
    --require_frozen_topk_gate=true \
    --thresholds_json="${THRESHOLDS_JSON}" \
    "${LONG_TEXT_GATE_FLAG}" \
    --summary_json="${OUTPUT_DIR}/unscanned_logits_summary.json" \
    "${orbax_args[@]}" 2>&1 | tee -a "${log_file}" || checker_rc=$?

  sync_outputs_to_gcs || true
  if [[ "${checker_rc}" -ne 0 ]]; then
    echo "ERROR: Stage unscanned_logits checker failed with exit code ${checker_rc}." >&2
    return "${checker_rc}"
  fi
  return 0
}

run_masked_prefill() {
  local log_file="${OUTPUT_DIR}/masked_prefill.log"
  local prefill_rc=0
  local prefill_text_args=()
  if [[ "${RUN_TEXT_PREFILL}" == "1" ]]; then
    prefill_text_args=(--golden_logits_path="${GOLDEN_PREFILL_TEXT_PKL}")
  fi
  local prefill_mm_dir="${OUTPUT_DIR}/masked_prefill_mm_m_only"
  rm -rf "${prefill_mm_dir}"
  mkdir -p "${prefill_mm_dir}"
  shopt -s nullglob
  local m_npz_files=("${GOLDEN_MM_DIR%/}"/M*.npz)
  shopt -u nullglob
  if [[ "${#m_npz_files[@]}" -eq 0 ]]; then
    echo "ERROR: Stage masked_prefill found zero M*.npz golden files in GOLDEN_MM_DIR=${GOLDEN_MM_DIR}." >&2
    return 1
  fi
  for m_npz in "${m_npz_files[@]}"; do
    ln -s "$(cd "$(dirname "${m_npz}")" && pwd)/$(basename "${m_npz}")" "${prefill_mm_dir}/$(basename "${m_npz}")"
  done
  "${PYTHON_BIN}" -m tests.utils.deepseek_v41.masked_prefill_runner \
    "${BASE_CONFIG}" \
    base_output_directory="${BASE_OUTPUT_DIRECTORY}" \
    run_name=ds41_mm_masked_prefill \
    model_name=deepseek4.1-flash \
    override_model_config=true \
    load_parameters_path="${HF_CKPT_PATH}" \
    source_checkpoint_layout=safetensors_dynamic \
    scan_layers=true \
    max_target_length=2048 \
    ici_fsdp_parallelism=-1 \
    ici_expert_parallelism=16 \
    sparse_matmul=false \
    capacity_factor=-1 \
    use_multimodal=true \
    vision_encoder_block=deepseek41 \
    ds41_vision_grid='[39,39]' \
    per_device_batch_size=0.125 \
    dtype=float32 \
    weight_dtype=bfloat16 \
    dtype_mm=float32 \
    matmul_precision=highest \
    remat_policy=none \
    ds41_activation_quant=false \
    "${TOKENIZER_ARGS[@]}" \
    --mm_golden_dir="${prefill_mm_dir}" \
    --require_frozen_topk_gate=true \
    --max_kl_div=0.08 \
    --min_cos_sim=0.999 \
    --enforce_length_bucket_gates=true \
    --output_logits_path="${OUTPUT_DIR}/masked_prefill_sample_reports.jsonl" \
    --output_json="${OUTPUT_DIR}/masked_prefill_summary.json" \
    --allow_reference_tie_branches="${ALLOW_REFERENCE_TIE_BRANCHES}" \
    --tie_branch_dir="${TIE_BRANCH_DIR}" \
    --tie_branch_max_ulp="${TIE_BRANCH_MAX_ULP}" \
    "${prefill_text_args[@]}" 2>&1 | tee -a "${log_file}" || prefill_rc=$?

  sync_outputs_to_gcs || true
  if [[ "${prefill_rc}" -ne 0 ]]; then
    echo "ERROR: Stage masked_prefill checker failed with exit code ${prefill_rc}." >&2
    return "${prefill_rc}"
  fi
  return 0
}

run_train_e2e() {
  local train_metrics_file="${TRAIN_METRICS_FILE:-${OUTPUT_DIR}/train_e2e_metrics.jsonl}"
  local train_log_file="${TRAIN_LOG_FILE:-${OUTPUT_DIR}/train_e2e.log}"
  local train_rc=0
  "${PYTHON_BIN}" "${REPO_ROOT}/src/maxtext/trainers/pre_train/train.py" \
    "${BASE_CONFIG}" \
    base_output_directory="${BASE_OUTPUT_DIRECTORY}" \
    run_name=ds41_mm_train_e2e \
    model_name=deepseek4.1-flash \
    override_model_config=true \
    load_parameters_path="${HF_CKPT_PATH}" \
    source_checkpoint_layout=safetensors_dynamic \
    scan_layers=true \
    dtype=bfloat16 \
    weight_dtype=bfloat16 \
    dtype_mm=bfloat16 \
    matmul_precision=default \
    remat_policy=full \
    ds41_activation_quant=true \
    ici_fsdp_parallelism=-1 \
    ici_expert_parallelism=16 \
    sparse_matmul=false \
    capacity_factor=-1 \
    opt_type=adamw \
    enable_checkpointing=false \
    save_checkpoint_on_completion=false \
    steps="${TRAIN_STEPS}" \
    learning_rate_schedule_steps="${TRAIN_STEPS}" \
    learning_rate=1e-5 \
    warmup_steps_fraction=0.0 \
    adam_weight_decay=0.1 \
    gradient_clipping_threshold=1.0 \
    load_balance_loss_weight=0.0001 \
    routed_bias_update_rate=0.001 \
    use_multimodal=true \
    vision_encoder_block=deepseek41 \
    freeze_vision_encoder_params=true \
    dataset_type=synthetic \
    max_num_images_per_example=1 \
    ds41_vision_grid='[39,39]' \
    max_target_length=512 \
    per_device_batch_size=0.125 \
    indexer_sparse_training=true \
    indexer_loss_scaling_factor=0.0 \
    log_moe_bias_norms=true \
    log_moe_dropped_frac=true \
    log_period=1 \
    "${TOKENIZER_ARGS[@]}" \
    metrics_file="${train_metrics_file}" 2>&1 | tee "${train_log_file}" || train_rc=$?

  if [[ "${train_rc}" -eq 0 && "${WORKER_ID}" == "0" ]]; then
    if ! "${PYTHON_BIN}" - "${train_metrics_file}" "${OUTPUT_DIR}/train_e2e_summary.json" "${TRAIN_STEPS}" <<'PY'
import json
import math
import sys
from pathlib import Path

metrics_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
expected_steps = int(sys.argv[3])
losses = {}
if metrics_path.is_file():
  for line in metrics_path.read_text(encoding="utf-8").splitlines():
    stripped = line.strip()
    if not stripped:
      continue
    rec = json.loads(stripped)
    if "step" in rec and "learning/loss" in rec:
      losses[int(rec["step"])] = float(rec["learning/loss"])
all_finite = bool(losses) and all(math.isfinite(v) and v > 0.0 for v in losses.values())
passed = len(losses) >= expected_steps and all_finite
summary = {
    "passed": passed,
    "all_losses_finite": all_finite,
    "expected_steps": expected_steps,
    "recorded_steps": len(losses),
}
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
if not passed:
  sys.exit(1)
PY
    then
      echo "ERROR: Stage train_e2e summary gate failed (missing steps or non-finite loss)." >&2
      train_rc=1
    fi
  fi

  sync_outputs_to_gcs || true
  if [[ "${train_rc}" -ne 0 ]]; then
    echo "ERROR: Stage train_e2e failed with exit code ${train_rc}." >&2
    return "${train_rc}"
  fi
  return 0
}

write_worker_ready_marker() {
  local ready_dir="${1:-${GCS_READY_DIR:-${READY_DIR:-}}}"
  local worker_id="${2:-${WORKER_ID:-0}}"
  local attempt_id="${3:-${RUN_ATTEMPT_ID:-}}"
  local now_epoch
  now_epoch="$(date -u +%s)"

  if [[ -z "${ready_dir}" ]]; then
    return 0
  fi

  local payload
  payload="$(printf '{"worker_id": %s, "pod": "%s", "attempt_id": "%s", "epoch": %s}\n' \
    "${worker_id}" "${HOSTNAME:-unknown}" "${attempt_id}" "${now_epoch}")"

  local target_file="${ready_dir%/}/worker_${worker_id}.ready"
  if [[ "${ready_dir}" =~ ^gs:// ]]; then
    local tmp_file
    tmp_file="$(mktemp)"
    printf '%s\n' "${payload}" > "${tmp_file}"
    gcloud storage cp "${tmp_file}" "${target_file}" >/dev/null 2>&1
    rm -f "${tmp_file}"
  else
    mkdir -p "${ready_dir}"
    printf '%s\n' "${payload}" > "${target_file}"
  fi
}

wait_for_workers_ready() {
  local ready_dir="${1:-${GCS_READY_DIR:-${READY_DIR:-}}}"
  local num_workers="${2:-${NUM_WORKERS:-1}}"
  local worker_id="${3:-${WORKER_ID:-0}}"
  local attempt_id="${4:-${RUN_ATTEMPT_ID:-}}"
  local barrier_start_epoch="${5:-${BARRIER_START_EPOCH:-$(date -u +%s)}}"
  local max_skew="${6:-${MAX_BARRIER_SKEW_SEC:-180}}"
  local timeout_sec="${7:-${BARRIER_TIMEOUT_SEC:-600}}"
  local poll_sec="${8:-${BARRIER_POLL_SEC:-5}}"

  if [[ -z "${ready_dir}" || "${num_workers}" -le 1 ]]; then
    return 0
  fi

  write_worker_ready_marker "${ready_dir}" "${worker_id}" "${attempt_id}"

  local local_check_dir
  if [[ "${ready_dir}" =~ ^gs:// ]]; then
    local_check_dir="$(mktemp -d)"
  else
    local_check_dir="${ready_dir}"
  fi

  local start_time
  start_time="$(date -u +%s)"
  local elapsed=0

  echo "[wait_for_workers_ready] Waiting for ${num_workers} workers in ${ready_dir} (attempt='${attempt_id}', barrier_start=${barrier_start_epoch}, max_skew=${max_skew}s, timeout=${timeout_sec}s)..."

  while true; do
    if [[ "${ready_dir}" =~ ^gs:// ]]; then
      rm -f "${local_check_dir}"/worker_*.ready
      gcloud storage cp "${ready_dir%/}/worker_*.ready" "${local_check_dir}/" >/dev/null 2>&1 || true
    fi

    # Self-heal if worker 0 removed early-arriving peer markers during CLEAN_STALE_READY_ON_START=1.
    if [[ ! -f "${local_check_dir}/worker_${worker_id}.ready" ]]; then
      write_worker_ready_marker "${ready_dir}" "${worker_id}" "${attempt_id}"
      if [[ "${ready_dir}" =~ ^gs:// ]]; then
        gcloud storage cp "${ready_dir%/}/worker_${worker_id}.ready" "${local_check_dir}/" >/dev/null 2>&1 || true
      fi
    fi

    if "${PYTHON_BIN}" - "${local_check_dir}" "${num_workers}" "${attempt_id}" "${barrier_start_epoch}" "${max_skew}" <<'PY'
import json
import sys
from pathlib import Path

ready_dir = Path(sys.argv[1])
num_workers = int(sys.argv[2])
expected_attempt_id = sys.argv[3]
barrier_start_epoch = int(sys.argv[4])
max_skew_sec = int(sys.argv[5])
min_allowed_epoch = barrier_start_epoch - max_skew_sec

all_ready = True
for wid in range(num_workers):
  p = ready_dir / f"worker_{wid}.ready"
  if not p.is_file():
    all_ready = False
    continue
  try:
    data = json.loads(p.read_text(encoding="utf-8"))
  except Exception:
    all_ready = False
    continue

  if not isinstance(data, dict):
    all_ready = False
    continue

  epoch = int(data.get("epoch", 0))
  attempt_id = str(data.get("attempt_id", ""))
  if expected_attempt_id and attempt_id != expected_attempt_id:
    all_ready = False
    continue
  if epoch < min_allowed_epoch:
    all_ready = False
    continue

if all_ready:
  sys.exit(0)
sys.exit(1)
PY
    then
      echo "[wait_for_workers_ready] All ${num_workers} workers ready in ${ready_dir}."
      if [[ "${ready_dir}" =~ ^gs:// ]]; then
        rm -rf "${local_check_dir}"
      fi
      return 0
    fi

    local now
    now="$(date -u +%s)"
    elapsed=$(( now - start_time ))
    if [[ "${elapsed}" -ge "${timeout_sec}" ]]; then
      echo "ERROR: Timed out waiting for ${num_workers} workers in ${ready_dir} after ${elapsed}s." >&2
      if [[ "${ready_dir}" =~ ^gs:// ]]; then
        rm -rf "${local_check_dir}"
      fi
      return 1
    fi

    sleep "${poll_sec}"
  done
}

stage_mm_goldens_from_gcs() {
  local dst_dir="${OUTPUT_DIR}/golden_mm"
  mkdir -p "${dst_dir}"
  if ! command -v gcloud >/dev/null 2>&1; then
    echo "ERROR: GOLDEN_MM_DIR is unset and gcloud was not found on PATH to stage MM_GOLDEN_GCS_PREFIX=${MM_GOLDEN_GCS_PREFIX}." >&2
    return 1
  fi
  echo "[2_test_deepseek41_mm] Staging multimodal goldens from ${MM_GOLDEN_GCS_PREFIX%/}/* to ${dst_dir}/ ..."
  if ! gcloud storage cp "${MM_GOLDEN_GCS_PREFIX%/}/*" "${dst_dir}/"; then
    echo "ERROR: Failed to stage multimodal goldens from ${MM_GOLDEN_GCS_PREFIX%/}/*." >&2
    return 1
  fi
  GOLDEN_MM_DIR="${dst_dir}"
}

main() {
  validate_environment

  if [[ "${ENABLE_GCS_STARTUP_BARRIER}" == "1" || ( "${NUM_WORKERS}" -gt 1 && -n "${GCS_READY_DIR}" ) ]]; then
    if [[ "${CLEAN_STALE_READY_ON_START}" == "1" && "${WORKER_ID}" -eq 0 && -n "${GCS_READY_DIR}" ]]; then
      if [[ "${GCS_READY_DIR}" =~ ^gs:// ]]; then
        if command -v gcloud >/dev/null 2>&1; then
          gcloud storage rm "${GCS_READY_DIR%/}/worker_*.ready" >/dev/null 2>&1 || true
        fi
      else
        rm -f "${GCS_READY_DIR%/}"/worker_*.ready >/dev/null 2>&1 || true
      fi
    fi
    wait_for_workers_ready "${GCS_READY_DIR}" "${NUM_WORKERS}" "${WORKER_ID}" "${RUN_ATTEMPT_ID}" "${BARRIER_START_EPOCH}" "${MAX_BARRIER_SKEW_SEC}" "${BARRIER_TIMEOUT_SEC}" "${BARRIER_POLL_SEC}" || exit 1
  fi

  overall_rc=0
  for stage_id in scanned_logits unscanned_logits masked_prefill train_e2e; do
    if ! stage_enabled "${stage_id}"; then
      continue
    fi
    if stage_already_passed "${stage_id}"; then
      continue
    fi
    if [[ ( "${stage_id}" == "scanned_logits" || "${stage_id}" == "unscanned_logits" || "${stage_id}" == "masked_prefill" ) && -z "${GOLDEN_MM_DIR}" ]]; then
      if ! stage_mm_goldens_from_gcs; then
        overall_rc=1
        continue
      fi
    fi
    echo "=== STAGE ${stage_id} START $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    set +e
    ( set -euo pipefail; "run_${stage_id}" )
    rc=$?
    gcs_rc=0
    if [[ "${rc}" -eq 0 ]]; then
      sync_outputs_to_gcs
      gcs_rc=$?
    fi
    set -e
    echo "=== STAGE ${stage_id} END $(date -u +%Y-%m-%dT%H:%M:%SZ) rc=${rc} ==="
    if [[ "${rc}" -ne 0 || "${gcs_rc}" -ne 0 ]]; then
      if [[ "${rc}" -eq 42 ]]; then
        overall_rc=42
        sync_outputs_to_gcs || true
        break
      elif [[ "${overall_rc}" -ne 42 ]]; then
        overall_rc=1
      fi
    fi
  done

  set +e
  sync_outputs_to_gcs
  final_gcs_rc=$?
  set -e
  if [[ "${final_gcs_rc}" -ne 0 && "${overall_rc}" -eq 0 ]]; then
    overall_rc=1
  fi

  exit "${overall_rc}"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
