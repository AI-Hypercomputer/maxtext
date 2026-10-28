#!/usr/bin/env bash
# ==============================================================================
# Run Multi-Slice (num_slices>=2) Checkpoint Load-Time Benchmark (Local or Hybrid)
#
# Enables multi-slice execution:
#   - Initializes JAX distributed runtime (`jax.distributed.initialize`) across
#     N slice processes (`slice_id = 0 .. N-1`)
#   - Sets MaxText flags:
#       skip_jax_distributed_system=False
#       num_slices=<N>
#       dcn_data_parallelism=<N> (<=47B models) OR dcn_fsdp_parallelism=<N> (>=70B models)
#       ici_fsdp_parallelism=-1
#   - Each slice has `DEVICES_PER_SLICE` devices (default 4 -> 8 global devices on 2 slices)
#
# Usage Examples:
#   # Run 2-slice benchmark for qwen3-0.6b and gemma2-2b:
#   ./checkpoint_load_benchmark/scripts/run_multi_slice_local.sh --models qwen3-0.6b gemma2-2b
#
#   # Run both 1-slice and 2-slice side-by-side with explicit FSDP across slices:
#   ./checkpoint_load_benchmark/scripts/run_multi_slice_local.sh \
#     --models qwen3-30b-a3b --num-slices-list 1 2 --dcn-parallelism-axis fsdp
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

MODELS=("${TIER0_MODELS[@]}")
SCAN_MODES=("${DEFAULT_SCAN_MODES[@]}")
NUM_SLICES_LIST=(2)
DCN_AXIS="${DEFAULT_DCN_AXIS}"
HARDWARE="${DEFAULT_HARDWARE}"
WEIGHT_DTYPE="${DEFAULT_WEIGHT_DTYPE}"
DEVICES_PER_SLICE="${DEFAULT_DEVICES_PER_SLICE}"
GCS_BASE="${DEFAULT_GCS_BASE}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
CSV_REPORT="${BENCHMARK_DIR}/reports/multi_slice_${RUN_ID}.csv"
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
    --num-slices-list)
      shift
      NUM_SLICES_LIST=()
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        NUM_SLICES_LIST+=("$1")
        shift
      done
      ;;
    --dcn-parallelism-axis)
      DCN_AXIS="$2"
      shift 2
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

print_banner "Multi-Slice (num_slices=${NUM_SLICES_LIST[*]}) Checkpoint Load-Time Benchmark"
echo " Models            : ${MODELS[*]}"
echo " Scan Modes        : ${SCAN_MODES[*]}"
echo " Num Slices List   : ${NUM_SLICES_LIST[*]} (${DEVICES_PER_SLICE} devices/slice)"
echo " DCN Axis          : ${DCN_AXIS}"
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
  --num-slices-list "${NUM_SLICES_LIST[@]}" \
  --dcn-parallelism-axis "${DCN_AXIS}" \
  --devices-per-slice "${DEVICES_PER_SLICE}" \
  --hardware "${HARDWARE}" \
  --weight-dtype "${WEIGHT_DTYPE}" \
  --gcs-base "${GCS_BASE}" \
  --run-id "${RUN_ID}" \
  --csv-report "${CSV_REPORT}" \
  --reuse-model-hf-cache \
  --allow-step-failure \
  "${EXTRA_ARGS[@]}"
