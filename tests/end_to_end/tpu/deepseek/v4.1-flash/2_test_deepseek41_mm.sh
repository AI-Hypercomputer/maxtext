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

# Staged TPU end-to-end verification script for multimodal DeepSeek-V4.1-Flash (R1..R4).
#
# Environment variables:
#   HF_CKPT_PATH            Path to the release SafeTensors checkpoint directory (required for R1..R4).
#   OUTPUT_DIR              Directory for summaries, metrics, and logs (required for R1..R4).
#   BASE_OUTPUT_DIRECTORY   Optional MaxText base_output_directory override (defaults to OUTPUT_DIR).
#   GOLDEN_TEXT_PKL         Path to 4-prompt text golden pickle (required for R1, R2).
#   GOLDEN_MM_DIR           Optional local directory containing L1..L4, M1..M8, mm_logit.pkl, manifest.json (staged from MM_GOLDEN_GCS_PREFIX if unset).
#   MM_GOLDEN_GCS_PREFIX    GCS prefix for multimodal goldens when GOLDEN_MM_DIR is unset (default: gs://yaoyuchen-us-central1-fast/deepseek-v4.1-flash/golden/mm_v3).
#   GOLDEN_PREFILL_TEXT_PKL Path to 7-record text masked-prefill golden pickle (required when RUN_TEXT_PREFILL=1 in R3).
#   ORBAX_EXPORT_PREFIX     Optional prefix for exporting verified scanned/unscanned Orbax checkpoints in R1/R2.
#   EXPORT_ORBAX            Set to "1" when R4 resume must also verify scanned/unscanned Orbax exports exist (default: "0").
#   GCS_LOG_PREFIX          Optional GCS prefix for archiving stage JSON/log/metrics artifacts on worker 0.
#   RESUME_PASSED_STAGES    Set to "1" to skip stages whose summary JSON already passed in GCS_RESUME_SUMMARY_PREFIX (default: "0").
#   GCS_RESUME_SUMMARY_PREFIX Directory or gs:// prefix checked when RESUME_PASSED_STAGES=1 (defaults to GCS_LOG_PREFIX).
#   PYTHON_BIN              Python interpreter executable (default: "python3").
#   TOKENIZER_PATH          Optional HuggingFace tokenizer directory path for offline TPU containers.
#   STAGES                  Comma-separated stages to execute (default: "R1,R2,R3,R4").
#   ENABLE_TT2_GATE         Set to "1" to enable deferred TT2 long-text forward logit gate in R1/R2 (default: "0").
#   RUN_TEXT_PREFILL        Set to "1" to include GOLDEN_PREFILL_TEXT_PKL in R3 (default: "0").
#   R4_STEPS                Number of R4 training steps (default: "20").
#   A6_AOT_JSON             Path to A6 (or W1_ALL) AOT summary JSON for TT6 peak HBM check (required for R4 unless A6_AOT_HBM_GIB is set).
#   A6_AOT_HBM_GIB          Explicit A6 AOT peak HBM estimate in GiB for TT6 check (mutually exclusive with A6_AOT_JSON).
#   OVERLAY_COMMIT_SHA      Expected git commit SHA (7..40 hex) for A6_AOT_JSON provenance check (defaults to EXPECTED_OVERLAY_COMMIT_SHA or git rev-parse HEAD).
#   AOT_JSON_SHA256         Optional expected SHA-256 digest of A6_AOT_JSON verified via sha256sum -c before R4.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
BASE_CONFIG="${MAXTEXT_CONFIGS_DIR:-${REPO_ROOT}/src/maxtext/configs}/base.yml"
THRESHOLDS_JSON="${THRESHOLDS_JSON:-${SCRIPT_DIR}/thresholds.json}"
PIN_JSON="${PIN_JSON:-${SCRIPT_DIR}/golden_manifest_pin.json}"
MM_GOLDEN_GCS_PREFIX="${MM_GOLDEN_GCS_PREFIX:-gs://yaoyuchen-us-central1-fast/deepseek-v4.1-flash/golden/mm_v3}"
GOLDEN_MM_DIR="${GOLDEN_MM_DIR:-}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
EXIT_DETERMINISTIC_GATE_FAIL=42
ALLOW_REFERENCE_TIE_BRANCHES="${ALLOW_REFERENCE_TIE_BRANCHES:-1}"
TIE_BRANCH_DIR="${TIE_BRANCH_DIR:-gs://yaoyuchen-us-central1-fast/deepseek-v4.1-flash/prefill_goldens_v2_alt_branches/}"
TIE_BRANCH_MAX_ULP="${TIE_BRANCH_MAX_ULP:-8.0}"
STAGES="${STAGES:-R1,R2,R3,R4}"
ENABLE_TT2_GATE="${ENABLE_TT2_GATE:-0}"
RUN_TEXT_PREFILL="${RUN_TEXT_PREFILL:-0}"
R4_STEPS="${R4_STEPS:-20}"
ORBAX_EXPORT_PREFIX="${ORBAX_EXPORT_PREFIX:-}"
EXPORT_ORBAX="${EXPORT_ORBAX:-0}"
GCS_LOG_PREFIX="${GCS_LOG_PREFIX:-}"
GCS_RESULTS_PATH="${GCS_RESULTS_PATH:-${GCS_LOG_PREFIX}}"
RESUME_PASSED_STAGES="${RESUME_PASSED_STAGES:-0}"
GCS_RESUME_SUMMARY_PREFIX="${GCS_RESUME_SUMMARY_PREFIX:-${GCS_RESULTS_PATH}}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
A6_AOT_JSON="${A6_AOT_JSON:-}"
A6_AOT_HBM_GIB="${A6_AOT_HBM_GIB:-}"
AOT_JSON_SHA256="${AOT_JSON_SHA256:-}"
OVERLAY_COMMIT_SHA="${OVERLAY_COMMIT_SHA:-${EXPECTED_OVERLAY_COMMIT_SHA:-}}"

# Multi-worker GCS startup barrier settings
BARRIER_START_EPOCH="${BARRIER_START_EPOCH:-$(date -u +%s)}"
WORKER_ID="${WORKER_ID:-${JOB_COMPLETION_INDEX:-0}}"
NUM_WORKERS="${NUM_WORKERS:-1}"
RUN_ATTEMPT_ID="${RUN_ATTEMPT_ID:-${JOB_ATTEMPT_NUMBER:-${POD_RESTART_COUNT:-}}}"
MAX_BARRIER_SKEW_SEC="${MAX_BARRIER_SKEW_SEC:-180}"
BARRIER_TIMEOUT_SEC="${BARRIER_TIMEOUT_SEC:-600}"
BARRIER_POLL_SEC="${BARRIER_POLL_SEC:-5}"
CLEAN_STALE_READY_ON_START="${CLEAN_STALE_READY_ON_START:-0}"
ENABLE_GCS_STARTUP_BARRIER="${ENABLE_GCS_STARTUP_BARRIER:-0}"
GCS_READY_DIR="${GCS_READY_DIR:-${READY_DIR:-}}"
if [[ -z "${GCS_READY_DIR}" && -n "${GCS_RESULTS_PATH}" ]]; then
  if [[ -n "${RUN_ATTEMPT_ID}" ]]; then
    GCS_READY_DIR="${GCS_RESULTS_PATH%/}/ready/attempt_${RUN_ATTEMPT_ID}"
  else
    GCS_READY_DIR="${GCS_RESULTS_PATH%/}/ready"
  fi
fi

TT2_GATE_FLAG="--enable_tt2_gate=false"
case "${ENABLE_TT2_GATE}" in
  0|false|FALSE)
    TT2_GATE_FLAG="--enable_tt2_gate=false"
    ;;
  1|true|TRUE)
    TT2_GATE_FLAG="--enable_tt2_gate=true"
    ;;
  *)
    echo "ERROR: Invalid ENABLE_TT2_GATE=${ENABLE_TT2_GATE}; expected 0/1 or false/true." >&2
    exit 1
    ;;
esac

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

resolve_overlay_commit_sha() {
  local cand="${OVERLAY_COMMIT_SHA:-${EXPECTED_OVERLAY_COMMIT_SHA:-}}"
  if [[ -z "${cand}" ]] && command -v git >/dev/null 2>&1; then
    cand="$(git -c safe.directory=* -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || true)"
  fi
  cand="${cand,,}"
  if [[ ! "${cand}" =~ ^[0-9a-f]{7,40}$ ]]; then
    echo "ERROR: Stage R4 with A6_AOT_JSON requires a valid 7..40 hex OVERLAY_COMMIT_SHA (or git HEAD in ${REPO_ROOT}), got '${cand}'." >&2
    return 1
  fi
  printf '%s\n' "${cand}"
}

verify_r4_aot_json_sha256() {
  if [[ -n "${AOT_JSON_SHA256}" && -z "${A6_AOT_JSON}" ]]; then
    echo "ERROR: AOT_JSON_SHA256 is set, but A6_AOT_JSON is unset." >&2
    return 1
  fi
  if [[ -n "${A6_AOT_JSON}" && "${A6_AOT_JSON}" != gs://* && ! -f "${A6_AOT_JSON}" ]]; then
    echo "ERROR: A6_AOT_JSON=${A6_AOT_JSON} does not exist." >&2
    return 1
  fi
  if [[ -n "${AOT_JSON_SHA256}" ]]; then
    if ! printf '%s  %s\n' "${AOT_JSON_SHA256}" "${A6_AOT_JSON}" | sha256sum -c - >/dev/null 2>&1; then
      echo "ERROR: A6_AOT_JSON=${A6_AOT_JSON} failed sha256sum verification against AOT_JSON_SHA256=${AOT_JSON_SHA256}." >&2
      return 1
    fi
  fi
}

validate_environment() {
  for stage_id in R1 R2 R3 R4; do
    if stage_enabled "${stage_id}"; then
      require_var "HF_CKPT_PATH" "${stage_id}"
      require_var "OUTPUT_DIR" "${stage_id}"
    fi
  done

  if stage_enabled "R1"; then
    require_var "GOLDEN_TEXT_PKL" "R1"
    if [[ -z "${GOLDEN_MM_DIR}" ]]; then
      require_var "MM_GOLDEN_GCS_PREFIX" "R1"
    fi
  fi

  if stage_enabled "R2"; then
    require_var "GOLDEN_TEXT_PKL" "R2"
    if [[ -z "${GOLDEN_MM_DIR}" ]]; then
      require_var "MM_GOLDEN_GCS_PREFIX" "R2"
    fi
  fi

  if stage_enabled "R3"; then
    if [[ -z "${GOLDEN_MM_DIR}" ]]; then
      require_var "MM_GOLDEN_GCS_PREFIX" "R3"
    fi
    if [[ "${RUN_TEXT_PREFILL}" == "1" ]]; then
      require_var "GOLDEN_PREFILL_TEXT_PKL" "R3"
    fi
  fi

  if stage_enabled "R4"; then
    if [[ -z "${A6_AOT_JSON}" && -z "${A6_AOT_HBM_GIB}" ]]; then
      echo "ERROR: Stage R4 requires either A6_AOT_JSON or A6_AOT_HBM_GIB to be set for TT6 HBM verification." >&2
      exit 1
    fi
    if [[ -n "${A6_AOT_JSON}" && -n "${A6_AOT_HBM_GIB}" ]]; then
      echo "ERROR: Stage R4 requires exactly one of A6_AOT_JSON or A6_AOT_HBM_GIB, not both." >&2
      exit 1
    fi
    verify_r4_aot_json_sha256 || exit 1
    if [[ -n "${A6_AOT_JSON}" ]]; then
      OVERLAY_COMMIT_SHA="$(resolve_overlay_commit_sha)" || exit 1
      export OVERLAY_COMMIT_SHA
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
  local worker_id="${JOB_COMPLETION_INDEX:-${TPU_WORKER_ID:-0}}"
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
      "${OUTPUT_DIR}"/r3_sample_reports.jsonl
      "${OUTPUT_DIR}"/r4_bias_vl_steps.npz
  )
  local sample_artifacts=("${OUTPUT_DIR}"/samples/*.json)
  shopt -u nullglob
  for artifact in "${summary_artifacts[@]}"; do
    if [[ -f "${artifact}" ]]; then
      local fname
      fname="$(basename "${artifact}")"
      gcloud storage cp "${artifact}" "${GCS_LOG_PREFIX%/}/${fname}" || return 1
      gcloud storage cp "${artifact}" "${GCS_LOG_PREFIX%/}/worker_${worker_id}_${fname}" || return 1
    fi
  done
  for s_artifact in "${sample_artifacts[@]}"; do
    if [[ -f "${s_artifact}" ]]; then
      local sfname
      sfname="$(basename "${s_artifact}")"
      gcloud storage cp "${s_artifact}" "${GCS_LOG_PREFIX%/}/samples/${sfname}" || return 1
    fi
  done

  shopt -s nullglob
  local log_artifacts=("${OUTPUT_DIR}"/*.log "${OUTPUT_DIR}"/r4_metrics.jsonl)
  shopt -u nullglob
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

verify_stage_summary_gate() {
  local summary_file="$1"
  local log_file="${2:-}"
  local worker_id="${JOB_COMPLETION_INDEX:-${TPU_WORKER_ID:-0}}"
  if [[ "${worker_id}" != "0" ]]; then
    return 0
  fi
  if [[ ! -f "${summary_file}" && -n "${GCS_LOG_PREFIX}" && -x "$(command -v gcloud)" ]]; then
    local fname
    fname="$(basename "${summary_file}")"
    gcloud storage cp "${GCS_LOG_PREFIX%/}/${fname}" "${summary_file}" 2>/dev/null || \
    gcloud storage cp "${GCS_LOG_PREFIX%/}/worker_*_${fname}" "${summary_file}" 2>/dev/null || true
  fi
  local gcs_arg=()
  if [[ -n "${GCS_LOG_PREFIX}" ]]; then
    gcs_arg=(--gcs_prefix="${GCS_LOG_PREFIX}")
  fi
  local log_arg=()
  if [[ -n "${log_file}" ]]; then
    log_arg=(--log_file="${log_file}")
  fi
  "${PYTHON_BIN}" "${REPO_ROOT}/tests/utils/deepseek_v41_mm_audit.py" \
    --mode=verify_stage_summary_gate \
    --summary_json="${summary_file}" \
    "${log_arg[@]}" \
    "${gcs_arg[@]}"
}

mirror_stage_summary() {
  local src_json="$1"
  local dst_json="$2"
  if [[ -f "${src_json}" ]]; then
    printf '%s\n' "$(<"${src_json}")" > "${dst_json}"
  fi
}

stage_already_passed() {
  local stage="$1"
  if [[ "${RESUME_PASSED_STAGES}" != "1" || -z "${GCS_RESUME_SUMMARY_PREFIX}" ]]; then
    return 1
  fi
  local summary_file=""
  case "${stage}" in
    R1) summary_file="r1_logit_summary.json" ;;
    R2) summary_file="r2_logit_summary.json" ;;
    R3) summary_file="r3_prefill_summary.json" ;;
    R4) summary_file="r4_gates_summary.json" ;;
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
if stage == "R4" and export_orbax == "1" and orbax_prefix:
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
    if [[ "${stage}" == "R1" ]]; then
      mirror_stage_summary "${OUTPUT_DIR}/r1_logit_summary.json" "${OUTPUT_DIR}/r1_summary.json"
    elif [[ "${stage}" == "R2" ]]; then
      mirror_stage_summary "${OUTPUT_DIR}/r2_logit_summary.json" "${OUTPUT_DIR}/r2_summary.json"
    fi
    echo "[2_test_deepseek41_mm] Stage ${stage} already passed in ${GCS_RESUME_SUMMARY_PREFIX}; skipping (RESUME_PASSED_STAGES=1)."
    return 0
  fi
  return 1
}

run_R1() {
  local orbax_args=()
  if [[ -n "${ORBAX_EXPORT_PREFIX}" ]]; then
    orbax_args=(--export_orbax_dir="${ORBAX_EXPORT_PREFIX%/}/scanned")
  fi
  local r1_log_file="${OUTPUT_DIR}/r1.log"
  local checker_rc=0
  "${PYTHON_BIN}" "${REPO_ROOT}/tests/utils/deepseek_v41_mm_audit.py" \
    --mode=verify_pin \
    --pin_json_path="${PIN_JSON}" \
    --manifest_json_path="${GOLDEN_MM_DIR%/}/manifest.json" \
    --summary_json="${OUTPUT_DIR}/r1_pin_summary.json" 2>&1 | tee -a "${r1_log_file}" || return "${EXIT_DETERMINISTIC_GATE_FAIL}"
  verify_stage_summary_gate "${OUTPUT_DIR}/r1_pin_summary.json" "${r1_log_file}" || return "${EXIT_DETERMINISTIC_GATE_FAIL}"

  "${PYTHON_BIN}" "${REPO_ROOT}/tests/utils/forward_pass_logit_checker.py" \
    "${BASE_CONFIG}" \
    base_output_directory="${BASE_OUTPUT_DIRECTORY}" \
    run_name=ds41_mm_r1_logits \
    model_name=deepseek4.1-flash \
    override_model_config=true \
    load_parameters_path="${HF_CKPT_PATH}" \
    source_checkpoint_layout=safetensors_dynamic \
    scan_layers=true \
    max_target_length="${R1_MAX_TARGET_LENGTH:-1024}" \
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
    "${TT2_GATE_FLAG}" \
    --summary_json="${OUTPUT_DIR}/r1_logit_summary.json" \
    "${orbax_args[@]}" 2>&1 | tee -a "${r1_log_file}" || checker_rc=$?

  sync_outputs_to_gcs || true
  verify_stage_summary_gate "${OUTPUT_DIR}/r1_logit_summary.json" "${r1_log_file}" || return 1
  mirror_stage_summary "${OUTPUT_DIR}/r1_logit_summary.json" "${OUTPUT_DIR}/r1_summary.json"
  verify_stage_summary_gate "${OUTPUT_DIR}/r1_summary.json" "${r1_log_file}" || return 1
  return "${checker_rc}"
}

run_R2() {
  local orbax_args=()
  if [[ -n "${ORBAX_EXPORT_PREFIX}" ]]; then
    orbax_args=(--export_orbax_dir="${ORBAX_EXPORT_PREFIX%/}/unscanned")
  fi
  local r2_log_file="${OUTPUT_DIR}/r2.log"
  local checker_rc=0
  "${PYTHON_BIN}" "${REPO_ROOT}/tests/utils/deepseek_v41_mm_audit.py" \
    --mode=verify_pin \
    --pin_json_path="${PIN_JSON}" \
    --manifest_json_path="${GOLDEN_MM_DIR%/}/manifest.json" \
    --summary_json="${OUTPUT_DIR}/r2_pin_summary.json" 2>&1 | tee -a "${r2_log_file}" || return "${EXIT_DETERMINISTIC_GATE_FAIL}"
  verify_stage_summary_gate "${OUTPUT_DIR}/r2_pin_summary.json" "${r2_log_file}" || return "${EXIT_DETERMINISTIC_GATE_FAIL}"

  "${PYTHON_BIN}" "${REPO_ROOT}/tests/utils/forward_pass_logit_checker.py" \
    "${BASE_CONFIG}" \
    base_output_directory="${BASE_OUTPUT_DIRECTORY}" \
    run_name=ds41_mm_r2_logits \
    model_name=deepseek4.1-flash \
    override_model_config=true \
    load_parameters_path="${HF_CKPT_PATH}" \
    source_checkpoint_layout=safetensors_dynamic \
    scan_layers=false \
    max_target_length="${R2_MAX_TARGET_LENGTH:-1024}" \
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
    "${TT2_GATE_FLAG}" \
    --summary_json="${OUTPUT_DIR}/r2_logit_summary.json" \
    --audit_summary_json="${OUTPUT_DIR}/r2_audit_summary.json" \
    --samples_dir="${OUTPUT_DIR}/samples" \
    --stage_name=r2 \
    --gcs_results_path="${GCS_RESULTS_PATH}" \
    "${orbax_args[@]}" 2>&1 | tee -a "${r2_log_file}" || checker_rc=$?

  sync_outputs_to_gcs || true
  verify_stage_summary_gate "${OUTPUT_DIR}/r2_logit_summary.json" "${r2_log_file}" || return 1
  mirror_stage_summary "${OUTPUT_DIR}/r2_logit_summary.json" "${OUTPUT_DIR}/r2_summary.json"
  verify_stage_summary_gate "${OUTPUT_DIR}/r2_summary.json" "${r2_log_file}" || return 1
  return "${checker_rc}"
}

run_R3() {
  local r3_log_file="${OUTPUT_DIR}/r3.log"
  local r3_rc=0
  "${PYTHON_BIN}" "${REPO_ROOT}/tests/utils/deepseek_v41_mm_audit.py" \
    --mode=verify_pin \
    --pin_json_path="${PIN_JSON}" \
    --manifest_json_path="${GOLDEN_MM_DIR%/}/manifest.json" \
    --summary_json="${OUTPUT_DIR}/r3_pin_summary.json" 2>&1 | tee -a "${r3_log_file}" || return "${EXIT_DETERMINISTIC_GATE_FAIL}"
  verify_stage_summary_gate "${OUTPUT_DIR}/r3_pin_summary.json" "${r3_log_file}" || return "${EXIT_DETERMINISTIC_GATE_FAIL}"

  local prefill_text_args=()
  if [[ "${RUN_TEXT_PREFILL}" == "1" ]]; then
    prefill_text_args=(--golden_logits_path="${GOLDEN_PREFILL_TEXT_PKL}")
  fi
  local r3_mm_dir="${OUTPUT_DIR}/r3_mm_m_only"
  rm -rf "${r3_mm_dir}"
  mkdir -p "${r3_mm_dir}"
  shopt -s nullglob
  local m_npz_files=("${GOLDEN_MM_DIR%/}"/M*.npz)
  shopt -u nullglob
  if [[ "${#m_npz_files[@]}" -eq 0 ]]; then
    echo "ERROR: Stage R3 found zero M*.npz masked-prefill golden files in GOLDEN_MM_DIR=${GOLDEN_MM_DIR}." >&2
    return 1
  fi
  for m_npz in "${m_npz_files[@]}"; do
    ln -s "$(cd "$(dirname "${m_npz}")" && pwd)/$(basename "${m_npz}")" "${r3_mm_dir}/$(basename "${m_npz}")"
  done
  "${PYTHON_BIN}" "${REPO_ROOT}/tests/utils/masked_prefill_logits_checker.py" \
    "${BASE_CONFIG}" \
    base_output_directory="${BASE_OUTPUT_DIRECTORY}" \
    run_name=ds41_mm_r3_prefill \
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
    --mm_golden_dir="${r3_mm_dir}" \
    --require_frozen_topk_gate=true \
    --max_kl_div=0.08 \
    --min_cos_sim=0.999 \
    --enforce_length_bucket_gates=true \
    --output_logits_path="${OUTPUT_DIR}/r3_sample_reports.jsonl" \
    --output_json="${OUTPUT_DIR}/r3_prefill_summary.json" \
    --samples_dir="${OUTPUT_DIR}/samples" \
    --stage_name=r3 \
    --gcs_results_path="${GCS_RESULTS_PATH}" \
    --allow_reference_tie_branches="${ALLOW_REFERENCE_TIE_BRANCHES}" \
    --tie_branch_dir="${TIE_BRANCH_DIR}" \
    --tie_branch_max_ulp="${TIE_BRANCH_MAX_ULP}" \
    "${prefill_text_args[@]}" 2>&1 | tee -a "${r3_log_file}" || r3_rc=$?

  sync_outputs_to_gcs || true
  if [[ "${r3_rc}" -ne 0 ]]; then
    echo "ERROR: Stage R3 checker failed with exit code ${r3_rc}." >&2
    if [[ "${r3_rc}" -eq 42 ]]; then
      return 42
    fi
    return "${r3_rc}"
  fi
  if ! verify_stage_summary_gate "${OUTPUT_DIR}/r3_prefill_summary.json" "${r3_log_file}"; then
    echo "ERROR: Stage R3 summary gate verification failed." >&2
    return "${EXIT_DETERMINISTIC_GATE_FAIL}"
  fi
  return 0
}

run_R4() {
  local r4_aot_args=()
  if [[ -n "${A6_AOT_JSON}" && -n "${A6_AOT_HBM_GIB}" ]]; then
    echo "ERROR: Stage R4 requires exactly one of A6_AOT_JSON or A6_AOT_HBM_GIB, not both." >&2
    return 1
  elif [[ -n "${A6_AOT_JSON}" ]]; then
    verify_r4_aot_json_sha256 || return 1
    local resolved_overlay_sha=""
    resolved_overlay_sha="$(resolve_overlay_commit_sha)" || return 1
    r4_aot_args=(
      --aot_summary_json="${A6_AOT_JSON}"
      --expected_aot_commit_sha="${resolved_overlay_sha}"
    )
  elif [[ -n "${A6_AOT_HBM_GIB}" ]]; then
    verify_r4_aot_json_sha256 || return 1
    r4_aot_args=(--aot_hbm_gb="${A6_AOT_HBM_GIB}")
  else
    echo "ERROR: Stage R4 requires either A6_AOT_JSON or A6_AOT_HBM_GIB to be set for TT6 HBM verification." >&2
    return 1
  fi
  local r4_metrics_file="${R4_METRICS_FILE:-${OUTPUT_DIR}/r4_metrics.jsonl}"
  local r4_log_file="${R4_LOG_FILE:-${OUTPUT_DIR}/r4_train.log}"
  local r4_bias_vl_npz="${R4_BIAS_VL_NPZ:-${OUTPUT_DIR}/r4_bias_vl_steps.npz}"
  DS41_DUMP_BIAS_VL_NPZ="${r4_bias_vl_npz}" \
    "${PYTHON_BIN}" "${REPO_ROOT}/tests/utils/deepseek_v41_mm_audit.py" \
    --mode=run_r4_train \
    --bias_vl_npz="${r4_bias_vl_npz}" \
    "${r4_aot_args[@]}" \
    -- \
    "${REPO_ROOT}/src/maxtext/trainers/pre_train/train.py" \
    "${BASE_CONFIG}" \
    base_output_directory="${BASE_OUTPUT_DIRECTORY}" \
    run_name=ds41_mm_r4_train \
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
    steps="${R4_STEPS}" \
    learning_rate_schedule_steps="${R4_STEPS}" \
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
    "${TOKENIZER_ARGS[@]}" \
    metrics_file="${r4_metrics_file}" 2>&1 | tee "${r4_log_file}" && \
  "${PYTHON_BIN}" "${REPO_ROOT}/tests/utils/deepseek_v41_mm_audit.py" \
    --mode=verify_r4_gates \
    --log_file="${r4_log_file}" \
    --metrics_file="${r4_metrics_file}" \
    --thresholds_json="${THRESHOLDS_JSON}" \
    --hf_checkpoint_dir="${HF_CKPT_PATH}" \
    --bias_vl_npz="${r4_bias_vl_npz}" \
    --expected_steps="${R4_STEPS}" \
    --summary_json="${OUTPUT_DIR}/r4_gates_summary.json" \
    "${r4_aot_args[@]}"
  sync_outputs_to_gcs || true
  verify_stage_summary_gate "${OUTPUT_DIR}/r4_gates_summary.json" "${r4_log_file}" || return 1
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
      gcloud storage cp "${ready_dir%/}/worker_*.ready" "${local_check_dir}/" >/dev/null 2>&1 || true
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
missing = []
stale = []
invalid = []

for wid in range(num_workers):
  p = ready_dir / f"worker_{wid}.ready"
  if not p.is_file():
    all_ready = False
    missing.append(wid)
    continue
  try:
    data = json.loads(p.read_text(encoding="utf-8"))
  except Exception:
    all_ready = False
    invalid.append(wid)
    continue

  if not isinstance(data, dict):
    all_ready = False
    invalid.append(wid)
    continue

  epoch = int(data.get("epoch", 0))
  attempt_id = str(data.get("attempt_id", ""))

  if expected_attempt_id and attempt_id != expected_attempt_id:
    all_ready = False
    stale.append(f"worker_{wid}(attempt='{attempt_id}')")
    continue

  if epoch < min_allowed_epoch:
    all_ready = False
    stale.append(f"worker_{wid}(epoch={epoch} < {min_allowed_epoch})")
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
  for stage_id in R1 R2 R3 R4; do
    if ! stage_enabled "${stage_id}"; then
      continue
    fi
    if stage_already_passed "${stage_id}"; then
      continue
    fi
    if [[ ( "${stage_id}" == "R1" || "${stage_id}" == "R2" || "${stage_id}" == "R3" ) && -z "${GOLDEN_MM_DIR}" ]]; then
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
      elif [[ "${overall_rc}" -ne 42 ]]; then
        overall_rc=1
      fi
    fi
  done

  set +e
  sync_outputs_to_gcs
  final_gcs_rc=$?
  set -e
  if [[ "${final_gcs_rc}" -ne 0 ]]; then
    overall_rc=1
  fi

  exit "${overall_rc}"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
