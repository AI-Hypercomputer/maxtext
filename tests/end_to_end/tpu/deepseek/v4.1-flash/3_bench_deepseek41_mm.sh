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

# Staged TPU7x MFU & HBM benchmark runner for DeepSeek-V4.1-Flash (W2 64c & W3 128c, MM + text twin).
#
# Environment variables:
#   TOPOLOGY_MODE             Target slice mode: "64c" (W2 tpu7x-128, FSDP=8, EP=16, offload=true) or
#                             "128c" (W3 tpu7x-256, FSDP=16, EP=16, offload=true) [required].
#   OUTPUT_DIR                Local directory for bench_{64c,128c}_{mm,text}_summary.json, metrics, and logs [required].
#   BASE_OUTPUT_DIRECTORY     Optional MaxText base_output_directory override (defaults to OUTPUT_DIR).
#   STAGES                    Comma-separated stages to run: "MM,TEXT" (default), "MM", or "TEXT".
#   LOAD_PARAMETERS_PATH      Scanned Orbax items URI or SafeTensors path for parameter loading.
#   HF_CKPT_PATH              HuggingFace SafeTensors checkpoint path (used with safetensors_dynamic if LOAD_PARAMETERS_PATH is unset).
#   SOURCE_CHECKPOINT_LAYOUT  Checkpoint layout ("orbax" or "safetensors_dynamic"; auto-inferred when omitted).
#   ALLOW_UNINITIALIZED_BENCH Set to "1" to run synthetic benchmark from random init without checkpoint load (default: "0").
#   REMAT_POLICY              Optional explicit "minimal" or "full" override; defaults from AOT <= 85.0 GiB selection.
#   AOT_BATCH_JSON            Optional multi-preset AOT JSON (W2_BENCH_ALL or W3_BENCH_ALL) supplying primary+fallback AOT reports.
#   MM_AOT_JSON / MM_AOT_HBM_GIB                       Primary minimal-remat MM AOT report / GiB (A8 or A9).
#   MM_FALLBACK_AOT_JSON / MM_FALLBACK_AOT_HBM_GIB     Fallback full-remat MM AOT report / GiB (A8b or A9b).
#   TEXT_AOT_JSON / TEXT_AOT_HBM_GIB                   Primary minimal-remat text AOT report / GiB (A8t or A9t).
#   TEXT_FALLBACK_AOT_JSON / TEXT_FALLBACK_AOT_HBM_GIB Fallback full-remat text AOT report / GiB (A8tb or A9tb).
#   REMAT_FALLBACK_HBM_GIB    Threshold GiB above which AOT auto-switches minimal -> full (default: "85.0").
#   DS41_LOG_MM_NORMS         Set to "1" (default) to emit host-side live TPU HBM (`DS41_MM_NORMS: {...}`) on BOTH MM and TEXT steps.
#   REQUIRE_LIVE_HBM          Set to "true" (default) to fail closed if `DS41_MM_NORMS` live TPU `peak_hbm_gb` is missing or > 94.74 GiB.
#   BENCH_STEPS               Unprofiled benchmark steps (default: "30").
#   XPROF_STEPS               Trailing XPlane profiled steps after BENCH_STEPS (default: "3"; "0" disables profiler).
#   STEADY_START_STEP         Inclusive steady-state window start step (default: "10").
#   STEADY_END_STEP           Inclusive steady-state window end step (default: "29").
#   XPROF_GCS_PREFIX          Prefix for XPlane trace upload (defaults to GCS_LOG_PREFIX or BASE_OUTPUT_DIRECTORY).
#   GCS_LOG_PREFIX            Optional GCS prefix for archiving per-stage JSON/JSONL/log artifacts on worker 0.
#   RESUME_PASSED_STAGES      Set to "1" to skip stages whose summary JSON already passed in GCS_RESUME_SUMMARY_PREFIX (default: "0").
#   GCS_RESUME_SUMMARY_PREFIX Directory or gs:// prefix checked when RESUME_PASSED_STAGES=1 (defaults to GCS_LOG_PREFIX).
#   PYTHON_BIN                Python interpreter executable (default: "python3").
#   TOKENIZER_PATH            Optional HuggingFace tokenizer path for offline TPU containers.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
BASE_CONFIG="${REPO_ROOT}/src/maxtext/configs/base.yml"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TOPOLOGY_MODE="${TOPOLOGY_MODE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
STAGES="${STAGES:-MM,TEXT}"
LOAD_PARAMETERS_PATH="${LOAD_PARAMETERS_PATH:-}"
HF_CKPT_PATH="${HF_CKPT_PATH:-}"
SOURCE_CHECKPOINT_LAYOUT="${SOURCE_CHECKPOINT_LAYOUT:-}"
ALLOW_UNINITIALIZED_BENCH="${ALLOW_UNINITIALIZED_BENCH:-0}"
REMAT_POLICY="${REMAT_POLICY:-}"
AOT_BATCH_JSON="${AOT_BATCH_JSON:-}"
MM_AOT_JSON="${MM_AOT_JSON:-}"
MM_AOT_HBM_GIB="${MM_AOT_HBM_GIB:-}"
MM_FALLBACK_AOT_JSON="${MM_FALLBACK_AOT_JSON:-}"
MM_FALLBACK_AOT_HBM_GIB="${MM_FALLBACK_AOT_HBM_GIB:-}"
TEXT_AOT_JSON="${TEXT_AOT_JSON:-}"
TEXT_AOT_HBM_GIB="${TEXT_AOT_HBM_GIB:-}"
TEXT_FALLBACK_AOT_JSON="${TEXT_FALLBACK_AOT_JSON:-}"
TEXT_FALLBACK_AOT_HBM_GIB="${TEXT_FALLBACK_AOT_HBM_GIB:-}"
REMAT_FALLBACK_HBM_GIB="${REMAT_FALLBACK_HBM_GIB:-85.0}"
DS41_LOG_MM_NORMS="${DS41_LOG_MM_NORMS:-1}"
DS41_LOG_STEP0_MOE_DROPPED="${DS41_LOG_STEP0_MOE_DROPPED:-1}"
REQUIRE_LIVE_HBM="${REQUIRE_LIVE_HBM:-true}"

BENCH_STEPS="${BENCH_STEPS:-30}"
XPROF_STEPS="${XPROF_STEPS:-3}"
STEADY_START_STEP="${STEADY_START_STEP:-10}"
STEADY_END_STEP="${STEADY_END_STEP:-29}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-4096}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
MAX_NUM_IMAGES_PER_EXAMPLE="${MAX_NUM_IMAGES_PER_EXAMPLE:-4}"
DS41_VISION_GRID="${DS41_VISION_GRID:-[39,39]}"
DTYPE="${DTYPE:-bfloat16}"
WEIGHT_DTYPE="${WEIGHT_DTYPE:-bfloat16}"
DTYPE_MM="${DTYPE_MM:-bfloat16}"
GRAD_DTYPE="${GRAD_DTYPE:-bfloat16}"
MATMUL_PRECISION="${MATMUL_PRECISION:-default}"
NUM_VOCAB_TILING="${NUM_VOCAB_TILING:-8}"
QUANTIZATION_LOCAL_SHARD_COUNT="${QUANTIZATION_LOCAL_SHARD_COUNT:-8}"
SPARSE_MATMUL="${SPARSE_MATMUL:-true}"
MEGABLOX="${MEGABLOX:-true}"
USE_TOKAMAX_GMM="${USE_TOKAMAX_GMM:-true}"
USE_GMM_V2="${USE_GMM_V2:-true}"
ALLOW_SPLIT_PHYSICAL_AXES="${ALLOW_SPLIT_PHYSICAL_AXES:-true}"
RAGGED_BUFFER_FACTOR="${RAGGED_BUFFER_FACTOR:-2.0}"
LOG_PERIOD="${LOG_PERIOD:-1}"
DS41_ACTIVATION_QUANT="${DS41_ACTIVATION_QUANT:-false}"
DS41_PER_LAYER_REMAT="${DS41_PER_LAYER_REMAT:-true}"
DS41_ATTENTION_CHUNK="${DS41_ATTENTION_CHUNK:-512}"
DS41_ENGRAM_LOOKUP="${DS41_ENGRAM_LOOKUP:-all_to_all}"
DS41_ENGRAM_A2A_CAPACITY_FACTOR="${DS41_ENGRAM_A2A_CAPACITY_FACTOR:-2.0}"

GCS_LOG_PREFIX="${GCS_LOG_PREFIX:-}"
RESUME_PASSED_STAGES="${RESUME_PASSED_STAGES:-0}"
GCS_RESUME_SUMMARY_PREFIX="${GCS_RESUME_SUMMARY_PREFIX:-${GCS_LOG_PREFIX}}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"

if [[ -z "${TOPOLOGY_MODE}" ]]; then
  echo "ERROR: Required environment variable TOPOLOGY_MODE ('64c' or '128c') is not set." >&2
  exit 1
fi
if [[ -z "${OUTPUT_DIR}" ]]; then
  echo "ERROR: Required environment variable OUTPUT_DIR is not set." >&2
  exit 1
fi

case "${TOPOLOGY_MODE}" in
  64c)
    DEFAULT_ICI_FSDP="8"
    DEFAULT_ICI_EP="16"
    DEFAULT_OPT_OFFLOAD="true"
    WINDOW_TAG="W2_64c"
    ;;
  128c)
    DEFAULT_ICI_FSDP="16"
    DEFAULT_ICI_EP="16"
    DEFAULT_OPT_OFFLOAD="true"
    WINDOW_TAG="W3_128c"
    ;;
  *)
    echo "ERROR: Unsupported TOPOLOGY_MODE='${TOPOLOGY_MODE}'; expected '64c' or '128c'." >&2
    exit 1
    ;;
esac

ICI_FSDP_PARALLELISM="${ICI_FSDP_PARALLELISM:-${DEFAULT_ICI_FSDP}}"
ICI_EXPERT_PARALLELISM="${ICI_EXPERT_PARALLELISM:-${DEFAULT_ICI_EP}}"
OPTIMIZER_MEMORY_HOST_OFFLOAD="${OPTIMIZER_MEMORY_HOST_OFFLOAD:-${DEFAULT_OPT_OFFLOAD}}"

IFS=',' read -r -a REQUESTED_STAGES <<< "${STAGES}"
if [[ "${#REQUESTED_STAGES[@]}" -eq 0 ]]; then
  echo "ERROR: STAGES must contain at least one of MM or TEXT." >&2
  exit 1
fi
for s in "${REQUESTED_STAGES[@]}"; do
  s_trimmed="$(echo "${s}" | tr -d '[:space:]')"
  if [[ "${s_trimmed}" != "MM" && "${s_trimmed}" != "TEXT" ]]; then
    echo "ERROR: Unknown stage '${s_trimmed}' in STAGES='${STAGES}' (expected MM and/or TEXT)." >&2
    exit 1
  fi
done

stage_enabled() {
  local target="$1"
  local s
  for s in "${REQUESTED_STAGES[@]}"; do
    if [[ "$(echo "${s}" | tr -d '[:space:]')" == "${target}" ]]; then
      return 0
    fi
  done
  return 1
}

CKPT_LOAD_ARGS=()
if [[ -n "${LOAD_PARAMETERS_PATH}" ]]; then
  resolved_layout="${SOURCE_CHECKPOINT_LAYOUT}"
  if [[ -z "${resolved_layout}" ]]; then
    if [[ -n "${HF_CKPT_PATH}" && "${LOAD_PARAMETERS_PATH}" == "${HF_CKPT_PATH}" ]]; then
      resolved_layout="safetensors_dynamic"
    else
      resolved_layout="orbax"
    fi
  fi
  CKPT_LOAD_ARGS=("load_parameters_path=${LOAD_PARAMETERS_PATH}" "source_checkpoint_layout=${resolved_layout}")
elif [[ -n "${HF_CKPT_PATH}" ]]; then
  resolved_layout="${SOURCE_CHECKPOINT_LAYOUT:-safetensors_dynamic}"
  CKPT_LOAD_ARGS=("load_parameters_path=${HF_CKPT_PATH}" "source_checkpoint_layout=${resolved_layout}")
elif [[ "${ALLOW_UNINITIALIZED_BENCH}" == "1" ]]; then
  CKPT_LOAD_ARGS=()
else
  echo "ERROR: Provide LOAD_PARAMETERS_PATH (scanned Orbax or SafeTensors), HF_CKPT_PATH, or ALLOW_UNINITIALIZED_BENCH=1." >&2
  exit 1
fi

DEFAULT_RANDOM_ROUTING="false"
if [[ "${ALLOW_UNINITIALIZED_BENCH}" == "1" ]]; then
  DEFAULT_RANDOM_ROUTING="true"
fi
USE_RANDOM_ROUTING="${USE_RANDOM_ROUTING:-${DEFAULT_RANDOM_ROUTING}}"

if [[ -n "${AOT_BATCH_JSON}" ]]; then
  if [[ -z "${MM_AOT_JSON}" && -z "${MM_AOT_HBM_GIB}" ]]; then
    MM_AOT_JSON="${AOT_BATCH_JSON}"
  fi
  if [[ -z "${MM_FALLBACK_AOT_JSON}" && -z "${MM_FALLBACK_AOT_HBM_GIB}" ]]; then
    MM_FALLBACK_AOT_JSON="${AOT_BATCH_JSON}"
  fi
  if [[ -z "${TEXT_AOT_JSON}" && -z "${TEXT_AOT_HBM_GIB}" ]]; then
    TEXT_AOT_JSON="${AOT_BATCH_JSON}"
  fi
  if [[ -z "${TEXT_FALLBACK_AOT_JSON}" && -z "${TEXT_FALLBACK_AOT_HBM_GIB}" ]]; then
    TEXT_FALLBACK_AOT_JSON="${AOT_BATCH_JSON}"
  fi
fi

mkdir -p "${OUTPUT_DIR}"
BASE_OUTPUT_DIRECTORY="${BASE_OUTPUT_DIRECTORY:-${OUTPUT_DIR}}"
XPROF_GCS_PREFIX="${XPROF_GCS_PREFIX:-${GCS_LOG_PREFIX:-${BASE_OUTPUT_DIRECTORY}}}"
AOT_SELECTION_JSON="${OUTPUT_DIR}/bench_${TOPOLOGY_MODE}_aot_selection.json"

TOKENIZER_ARGS=()
if [[ -n "${TOKENIZER_PATH}" ]]; then
  TOKENIZER_ARGS=(tokenizer_type=huggingface "tokenizer_path=${TOKENIZER_PATH}")
fi

SELECT_ARGS=(
  --mode=select_remat
  "--topology_mode=${TOPOLOGY_MODE}"
  "--enabled_stages=${STAGES}"
  "--explicit_remat_policy=${REMAT_POLICY}"
  "--remat_fallback_hbm_gib=${REMAT_FALLBACK_HBM_GIB}"
  "--output_json=${AOT_SELECTION_JSON}"
)
[[ -n "${MM_AOT_JSON}" ]] && SELECT_ARGS+=("--mm_aot_json=${MM_AOT_JSON}")
[[ -n "${MM_AOT_HBM_GIB}" ]] && SELECT_ARGS+=("--mm_aot_hbm_gib=${MM_AOT_HBM_GIB}")
[[ -n "${MM_FALLBACK_AOT_JSON}" ]] && SELECT_ARGS+=("--mm_fallback_aot_json=${MM_FALLBACK_AOT_JSON}")
[[ -n "${MM_FALLBACK_AOT_HBM_GIB}" ]] && SELECT_ARGS+=("--mm_fallback_aot_hbm_gib=${MM_FALLBACK_AOT_HBM_GIB}")
[[ -n "${TEXT_AOT_JSON}" ]] && SELECT_ARGS+=("--text_aot_json=${TEXT_AOT_JSON}")
[[ -n "${TEXT_AOT_HBM_GIB}" ]] && SELECT_ARGS+=("--text_aot_hbm_gib=${TEXT_AOT_HBM_GIB}")
[[ -n "${TEXT_FALLBACK_AOT_JSON}" ]] && SELECT_ARGS+=("--text_fallback_aot_json=${TEXT_FALLBACK_AOT_JSON}")
[[ -n "${TEXT_FALLBACK_AOT_HBM_GIB}" ]] && SELECT_ARGS+=("--text_fallback_aot_hbm_gib=${TEXT_FALLBACK_AOT_HBM_GIB}")

"${PYTHON_BIN}" "${SCRIPT_DIR}/bench_metrics.py" "${SELECT_ARGS[@]}" >/dev/null

SELECTED_REMAT_POLICY="$(
  "${PYTHON_BIN}" - "${AOT_SELECTION_JSON}" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as f:
  data = json.load(f)
print(data["remat_policy"])
PY
)"

AOT_SELECTION_SUMMARY="$(
  "${PYTHON_BIN}" - "${AOT_SELECTION_JSON}" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as f:
  data = json.load(f)
parts = []
for key in ("mm_aot", "text_aot"):
  rec = data[key]
  if rec is None:
    parts.append(f"{key}=none")
  else:
    parts.append(
        f"{key}(compile_succeeded={rec['aot_compile_succeeded']},"
        f"temp={rec['aot_temp_hbm_gib']:.3f},"
        f"io={rec['aot_non_aliased_io_hbm_gib']:.3f},"
        f"naive={rec['aot_naive_sum_hbm_gib']:.3f},"
        f"clamped_cap={rec['aot_clamped_heap_cap_gib']:.3f})"
    )
print(" ".join(parts))
PY
)"

echo "[3_bench_deepseek41_mm] topology=${TOPOLOGY_MODE} remat_policy=${SELECTED_REMAT_POLICY} ${AOT_SELECTION_SUMMARY}"

sync_outputs_to_gcs() {
  local worker_id="${JOB_COMPLETION_INDEX:-${TPU_WORKER_ID:-0}}"
  if [[ -n "${GCS_LOG_PREFIX}" && "${worker_id}" == "0" ]]; then
    if ! command -v gcloud >/dev/null 2>&1; then
      echo "ERROR: GCS_LOG_PREFIX=${GCS_LOG_PREFIX} is set on worker 0, but gcloud was not found on PATH." >&2
      return 1
    fi
    shopt -s nullglob
    local files=("${OUTPUT_DIR}"/*.json "${OUTPUT_DIR}"/*.jsonl "${OUTPUT_DIR}"/*.log)
    shopt -u nullglob
    if [[ "${#files[@]}" -eq 0 ]]; then
      echo "ERROR: GCS_LOG_PREFIX=${GCS_LOG_PREFIX} is set on worker 0, but no output files exist in ${OUTPUT_DIR}." >&2
      return 1
    fi
    if ! gcloud storage cp "${files[@]}" "${GCS_LOG_PREFIX%/}/"; then
      echo "ERROR: gcloud storage cp to ${GCS_LOG_PREFIX%/}/ failed." >&2
      return 1
    fi
  fi
}

stage_already_passed() {
  local stage="$1"
  if [[ "${RESUME_PASSED_STAGES}" != "1" || -z "${GCS_RESUME_SUMMARY_PREFIX}" ]]; then
    return 1
  fi
  local mod_lower
  mod_lower="$(echo "${stage}" | tr '[:upper:]' '[:lower:]')"
  local summary_file="bench_${TOPOLOGY_MODE}_${mod_lower}_summary.json"
  if "${PYTHON_BIN}" - "${GCS_RESUME_SUMMARY_PREFIX%/}/${summary_file}" "${OUTPUT_DIR}/${summary_file}" <<'PY'
import json
import sys
from etils import epath

remote_uri, local_dst = sys.argv[1:3]
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
dst_path = epath.Path(local_dst)
dst_path.parent.mkdir(parents=True, exist_ok=True)
if remote_path.as_posix() != dst_path.as_posix():
  dst_path.write_text(raw_text, encoding="utf-8")
sys.exit(0)
PY
  then
    echo "[3_bench_deepseek41_mm] Stage ${stage} (${TOPOLOGY_MODE}) already passed in ${GCS_RESUME_SUMMARY_PREFIX}; skipping (RESUME_PASSED_STAGES=1)."
    return 0
  fi
  return 1
}

run_bench_stage() {
  local stage_id="$1"
  local start_utc="$2"
  local mod_lower
  mod_lower="$(echo "${stage_id}" | tr '[:upper:]' '[:lower:]')"
  local use_mm="false"
  local freeze_vit="false"
  local mm_extra_args=()
  if [[ "${stage_id}" == "MM" ]]; then
    use_mm="true"
    freeze_vit="true"
    mm_extra_args=(
      "max_num_images_per_example=${MAX_NUM_IMAGES_PER_EXAMPLE}"
      "ds41_vision_grid=${DS41_VISION_GRID}"
    )
  else
    use_mm="false"
    freeze_vit="false"
  fi

  local total_steps=$((BENCH_STEPS + XPROF_STEPS))
  local ckpt_period=$(( total_steps + 1000 > 100000 ? total_steps + 1000 : 100000 ))
  local run_name="ds41_bench_${TOPOLOGY_MODE}_${mod_lower}"
  local metrics_file="${OUTPUT_DIR}/bench_${TOPOLOGY_MODE}_${mod_lower}_metrics.jsonl"
  local log_file="${OUTPUT_DIR}/bench_${TOPOLOGY_MODE}_${mod_lower}.log"
  local summary_json="${OUTPUT_DIR}/bench_${TOPOLOGY_MODE}_${mod_lower}_summary.json"
  local xprof_uri=""
  local profiler_args=()
  if [[ "${XPROF_STEPS}" -gt 0 ]]; then
    xprof_uri="${XPROF_GCS_PREFIX%/}/xprof_${TOPOLOGY_MODE}_${mod_lower}"
    profiler_args=(
      profiler=xplane
      enable_tpu_profiling_options=true
      "skip_first_n_steps_for_profiler=${BENCH_STEPS}"
      "profiler_steps=${XPROF_STEPS}"
      "tensorboard_dir=${xprof_uri}"
    )
  else
    profiler_args=("profiler=")
  fi

  : > "${log_file}"
  echo "STAGE_START_UTC=${WINDOW_TAG}_${stage_id} ${start_utc}" | tee -a "${log_file}"

  DS41_LOG_MM_NORMS="${DS41_LOG_MM_NORMS}" DS41_LOG_STEP0_MOE_DROPPED="${DS41_LOG_STEP0_MOE_DROPPED}" "${PYTHON_BIN}" "${REPO_ROOT}/src/maxtext/trainers/pre_train/train.py" \
    "${BASE_CONFIG}" \
    "base_output_directory=${BASE_OUTPUT_DIRECTORY}" \
    "run_name=${run_name}" \
    model_name=deepseek4.1-flash \
    override_model_config=true \
    "${CKPT_LOAD_ARGS[@]}" \
    scan_layers=true \
    "dtype=${DTYPE}" \
    "weight_dtype=${WEIGHT_DTYPE}" \
    "dtype_mm=${DTYPE_MM}" \
    "grad_dtype=${GRAD_DTYPE}" \
    "matmul_precision=${MATMUL_PRECISION}" \
    "num_vocab_tiling=${NUM_VOCAB_TILING}" \
    "quantization_local_shard_count=${QUANTIZATION_LOCAL_SHARD_COUNT}" \
    "remat_policy=${SELECTED_REMAT_POLICY}" \
    "ds41_activation_quant=${DS41_ACTIVATION_QUANT}" \
    "ici_fsdp_parallelism=${ICI_FSDP_PARALLELISM}" \
    "ici_expert_parallelism=${ICI_EXPERT_PARALLELISM}" \
    "allow_split_physical_axes=${ALLOW_SPLIT_PHYSICAL_AXES}" \
    "optimizer_memory_host_offload=${OPTIMIZER_MEMORY_HOST_OFFLOAD}" \
    "sparse_matmul=${SPARSE_MATMUL}" \
    "megablox=${MEGABLOX}" \
    "use_tokamax_gmm=${USE_TOKAMAX_GMM}" \
    "use_gmm_v2=${USE_GMM_V2}" \
    "ragged_buffer_factor=${RAGGED_BUFFER_FACTOR}" \
    "use_random_routing=${USE_RANDOM_ROUTING}" \
    capacity_factor=-1 \
    opt_type=adamw \
    enable_checkpointing=true \
    async_checkpointing=false \
    save_checkpoint_on_start=false \
    save_checkpoint_on_completion=false \
    "checkpoint_period=${ckpt_period}" \
    "log_period=${LOG_PERIOD}" \
    "steps=${total_steps}" \
    "learning_rate_schedule_steps=${total_steps}" \
    learning_rate=1e-5 \
    warmup_steps_fraction=0.0 \
    adam_weight_decay=0.1 \
    gradient_clipping_threshold=1.0 \
    load_balance_loss_weight=0.0001 \
    routed_bias_update_rate=0.001 \
    "use_multimodal=${use_mm}" \
    vision_encoder_block=deepseek41 \
    "freeze_vision_encoder_params=${freeze_vit}" \
    dataset_type=synthetic \
    "max_target_length=${MAX_TARGET_LENGTH}" \
    "per_device_batch_size=${PER_DEVICE_BATCH_SIZE}" \
    indexer_sparse_training=true \
    indexer_loss_scaling_factor=0.0 \
    log_moe_bias_norms=false \
    "${mm_extra_args[@]}" \
    "${profiler_args[@]}" \
    "${TOKENIZER_ARGS[@]}" \
    "metrics_file=${metrics_file}" 2>&1 | tee -a "${log_file}"

  local end_utc
  end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "STAGE_END_UTC=${WINDOW_TAG}_${stage_id} ${end_utc}" | tee -a "${log_file}"

  DS41_LOG_STEP0_MOE_DROPPED="${DS41_LOG_STEP0_MOE_DROPPED}" "${PYTHON_BIN}" "${SCRIPT_DIR}/bench_metrics.py" \
    --mode=summarize_bench \
    "--topology_mode=${TOPOLOGY_MODE}" \
    "--modality=${mod_lower}" \
    "--remat_policy=${SELECTED_REMAT_POLICY}" \
    "--aot_selection_json=${AOT_SELECTION_JSON}" \
    "--require_live_hbm=${REQUIRE_LIVE_HBM}" \
    "--metrics_file=${metrics_file}" \
    "--log_file=${log_file}" \
    "--summary_json=${summary_json}" \
    "--bench_steps=${BENCH_STEPS}" \
    "--xprof_steps=${XPROF_STEPS}" \
    "--steady_start_step=${STEADY_START_STEP}" \
    "--steady_end_step=${STEADY_END_STEP}" \
    "--per_device_batch_size=${PER_DEVICE_BATCH_SIZE}" \
    "--max_target_length=${MAX_TARGET_LENGTH}" \
    "--max_num_images_per_example=${MAX_NUM_IMAGES_PER_EXAMPLE}" \
    "--ds41_vision_grid=${DS41_VISION_GRID}" \
    "--freeze_vision_encoder_params=${freeze_vit}" \
    "--xprof_trace_gcs_uri=${xprof_uri}" \
    "--stage_start_utc=${start_utc}" \
    "--stage_end_utc=${end_utc}"
}

overall_rc=0
for stage_id in MM TEXT; do
  if ! stage_enabled "${stage_id}"; then
    continue
  fi
  if stage_already_passed "${stage_id}"; then
    continue
  fi
  stage_start="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "=== STAGE ${WINDOW_TAG}_${stage_id} START ${stage_start} ==="
  set +e
  ( set -euo pipefail; run_bench_stage "${stage_id}" "${stage_start}" )
  rc=$?
  gcs_rc=0
  sync_outputs_to_gcs || gcs_rc=$?
  set -e
  stage_end="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "=== STAGE ${WINDOW_TAG}_${stage_id} END ${stage_end} rc=${rc} ==="
  if [[ "${rc}" -ne 0 || "${gcs_rc}" -ne 0 ]]; then
    overall_rc=1
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
