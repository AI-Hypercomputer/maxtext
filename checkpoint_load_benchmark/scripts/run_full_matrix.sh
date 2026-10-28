#!/usr/bin/env bash
# ==============================================================================
# Reproduce the Full 48-Configuration Checkpoint Load-Time Benchmark Suite
# (12 Models x 2 Scan Modes [scanned, unscanned] x 2 Slice Counts [1, 2])
#
# Tiers:
#   - Tier 0 (<=32B)        : qwen3-0.6b, gemma2-2b, deepseek2-16b, qwen3-30b-a3b, qwen3-32b
#   - Tier 1 (47B-70B)      : mixtral-8x7b, llama3.1-70b
#   - Tier 2 (120B OOM)     : gpt-oss-120b
#   - Tier 3 (284B-1600B)   : deepseek4-284b, deepseek3-671b, deepseek3.2-671b, deepseek4-1600b
#
# Usage Examples:
#   # Run Tier 0 (<=32B) locally across 1 & 2 slices:
#   ./checkpoint_load_benchmark/scripts/run_full_matrix.sh --tier 0
#
#   # Run Tier 1 & Tier 2 (>32B) on the GKE TPU cluster across 1 & 2 slices:
#   ./checkpoint_load_benchmark/scripts/run_full_matrix.sh --tier 1 --use-cluster
#   ./checkpoint_load_benchmark/scripts/run_full_matrix.sh --tier 2 --use-cluster
#
#   # Run Tier 3 (DeepSeek 284B-1600B OOM tests) on the GKE TPU cluster:
#   ./checkpoint_load_benchmark/scripts/run_full_matrix.sh --tier 3 --use-cluster
#
#   # Regenerate the 48-row CSV summary from an existing run ID:
#   ./checkpoint_load_benchmark/scripts/run_full_matrix.sh --tier all --run-id 20261023_032224 --skip-completed
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

TIER="all"
USE_CLUSTER="false"
RUN_ID="${RUN_ID:-20261023_032224}"
CSV_REPORT="${BENCHMARK_DIR}/reports/benchmark_ckpt_load_time_summary.csv"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier)
      TIER="$2"
      shift 2
      ;;
    --use-cluster)
      USE_CLUSTER="true"
      shift
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

case "${TIER}" in
  0)
    SELECTED_MODELS=("${TIER0_MODELS[@]}")
    ;;
  1)
    SELECTED_MODELS=("${TIER1_MODELS[@]}")
    ;;
  2)
    SELECTED_MODELS=("${TIER2_MODELS[@]}")
    ;;
  3)
    SELECTED_MODELS=("${TIER3_OOM_MODELS[@]}")
    export STAGE_SPARSE_DEEPSEEK_OOM="1"
    ;;
  all)
    SELECTED_MODELS=("${ALL_MODELS[@]}")
    ;;
  *)
    echo "ERROR: Invalid --tier '${TIER}'. Expected one of: 0, 1, 2, 3, all." >&2
    exit 1
    ;;
esac

if [[ "${USE_CLUSTER}" == "true" ]]; then
  CLUSTER_FLAGS=()
  if [[ "${TIER}" == "3" ]]; then
    CLUSTER_FLAGS+=("--stage-sparse-oom")
  fi
  exec "${SCRIPT_DIR}/run_multi_slice_cluster.sh" \
    --models "${SELECTED_MODELS[@]}" \
    --scan-modes scanned unscanned \
    --num-slices-list 1 2 \
    --run-id "${RUN_ID}" \
    "${CLUSTER_FLAGS[@]}" \
    "${EXTRA_ARGS[@]}"
else
  print_banner "Full Matrix Benchmark Runner (Tier=${TIER}, 1 & 2 Slices)"
  echo " Models            : ${SELECTED_MODELS[*]}"
  echo " Scan Modes        : scanned unscanned"
  echo " Num Slices List   : 1 2"
  echo " Run ID            : ${RUN_ID}"
  echo " CSV Report        : ${CSV_REPORT}"
  echo "================================================================================"

  cd "${REPO_ROOT}"
  exec "${PYTHON_BIN}" "${BENCHMARK_SCRIPT}" \
    --models "${SELECTED_MODELS[@]}" \
    --scan-modes scanned unscanned \
    --num-slices-list 1 2 \
    --hardware cpu \
    --weight-dtype bfloat16 \
    --gcs-base "${DEFAULT_GCS_BASE}" \
    --run-id "${RUN_ID}" \
    --csv-report "${CSV_REPORT}" \
    --reuse-model-hf-cache \
    --allow-step-failure \
    "${EXTRA_ARGS[@]}"
fi
