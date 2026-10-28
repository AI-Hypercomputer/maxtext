#!/usr/bin/env bash
# ==============================================================================
# Run Single-Slice (num_slices=1) Checkpoint Load-Time Benchmark
#
# Enables single-slice execution:
#   - num_slices=1 (4 devices per slice)
#   - skip_jax_distributed_system=True
#   - Compares `to_maxtext` (Steps 1, 2, 3) vs `safetensor` (`safetensors_dynamic` Steps 1, 2)
#
# Usage Examples:
#   ./checkpoint_load_benchmark/scripts/run_single_slice.sh --models qwen3-0.6b gemma2-2b
#   ./checkpoint_load_benchmark/scripts/run_single_slice.sh --models llama3.1-70b --scan-modes scanned unscanned
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

MODELS=("${TIER0_MODELS[@]}")
SCAN_MODES=("${DEFAULT_SCAN_MODES[@]}")
HARDWARE="${DEFAULT_HARDWARE}"
WEIGHT_DTYPE="${DEFAULT_WEIGHT_DTYPE}"
DEVICES_PER_SLICE="${DEFAULT_DEVICES_PER_SLICE}"
GCS_BASE="${DEFAULT_GCS_BASE}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
CSV_REPORT="${BENCHMARK_DIR}/reports/single_slice_${RUN_ID}.csv"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)
      shift
      MODELS=()
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        MODELS+=("$1")
        shift
      done
      ;;
    --scan-modes)
      shift
      SCAN_MODES=()
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        SCAN_MODES+=("$1")
        shift
      done
      ;;
    --hardware)
      HARDWARE="$2"
      shift 2
      ;;
    --weight-dtype)
      WEIGHT_DTYPE="$2"
      shift 2
      ;;
    --devices-per-slice)
      DEVICES_PER_SLICE="$2"
      shift 2
      ;;
    --gcs-base)
      GCS_BASE="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --csv-report)
      CSV_REPORT="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

print_banner "Single-Slice (num_slices=1) Checkpoint Load-Time Benchmark"
echo " Models            : ${MODELS[*]}"
echo " Scan Modes        : ${SCAN_MODES[*]}"
echo " Num Slices        : 1 (${DEVICES_PER_SLICE} devices/slice)"
echo " Hardware          : ${HARDWARE}"
echo " Weight Dtype      : ${WEIGHT_DTYPE}"
echo " GCS Base          : ${GCS_BASE}"
echo " Run ID            : ${RUN_ID}"
echo " CSV Report        : ${CSV_REPORT}"
echo "================================================================================"

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" "${BENCHMARK_SCRIPT}" \
  --models "${MODELS[@]}" \
  --scan-modes "${SCAN_MODES[@]}" \
  --num-slices-list 1 \
  --devices-per-slice "${DEVICES_PER_SLICE}" \
  --hardware "${HARDWARE}" \
  --weight-dtype "${WEIGHT_DTYPE}" \
  --gcs-base "${GCS_BASE}" \
  --run-id "${RUN_ID}" \
  --csv-report "${CSV_REPORT}" \
  --reuse-model-hf-cache \
  --allow-step-failure \
  "${EXTRA_ARGS[@]}"
