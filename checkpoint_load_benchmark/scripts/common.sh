#!/usr/bin/env bash
# ==============================================================================
# Shared Setup & Helper Functions for MaxText Checkpoint Load-Time Benchmark
# ==============================================================================
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(cd "${SCRIPTS_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${BENCHMARK_DIR}/.." && pwd)"

# Load model tiers and default settings
# shellcheck source=../configs/model_tiers.env
source "${BENCHMARK_DIR}/configs/model_tiers.env"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-65536}"

# Resolve Python interpreter
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/usr/local/google/home/mesakh/Documents/mesakh/py312/bin/python" ]]; then
    PYTHON_BIN="/usr/local/google/home/mesakh/Documents/mesakh/py312/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi
export PYTHON_BIN

# Resolve HuggingFace token from environment or token file (never echoed)
if [[ -z "${HF_TOKEN:-}" ]]; then
  HF_TOKEN_FILE="${HF_TOKEN_FILE:-/usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/hf_token.txt}"
  if [[ -f "${HF_TOKEN_FILE}" ]]; then
    HF_TOKEN="$(tr -d '[:space:]' < "${HF_TOKEN_FILE}")"
    export HF_TOKEN
  fi
fi

BENCHMARK_SCRIPT="${BENCHMARK_DIR}/benchmark_ckpt_load_time.py"

print_banner() {
  local title="$1"
  echo "================================================================================"
  echo " ${title}"
  echo "================================================================================"
  echo " Repo Root     : ${REPO_ROOT}"
  echo " Benchmark Dir : ${BENCHMARK_DIR}"
  echo " Python Bin    : ${PYTHON_BIN}"
  echo "================================================================================"
}
