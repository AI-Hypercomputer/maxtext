#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Generate (and optionally apply with --apply) a smoke-test JobSet manifest for v7x, v6e, or v5e.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/usr/local/google/home/mesakh/Documents/mesakh/py312/bin/python" ]]; then
    PYTHON_BIN="/usr/local/google/home/mesakh/Documents/mesakh/py312/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

PLATFORM=""
TOPOLOGY=""
RUN_ID=""
MODEL="qwen3-0.6b"
SCAN_MODES=("scanned" "unscanned")
PIPELINES="both"
OUT_FILE=""
APPLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --platform) PLATFORM="$2"; shift 2 ;;
    --topology) TOPOLOGY="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --pipelines) PIPELINES="$2"; shift 2 ;;
    --out) OUT_FILE="$2"; shift 2 ;;
    --apply) APPLY=1; shift ;;
    --scan-modes)
      SCAN_MODES=()
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        SCAN_MODES+=("$1")
        shift
      done
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 --platform <v7x|v6e|v5e> --topology <topo> --run-id <id> [--model qwen3-0.6b] [--scan-modes scanned unscanned] [--pipelines both|to_maxtext|safetensors_dynamic] [--out <file>] [--apply]" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${PLATFORM}" || -z "${TOPOLOGY}" || -z "${RUN_ID}" ]]; then
  echo "Error: --platform, --topology, and --run-id are required." >&2
  exit 1
fi

TOPO_SLUG="${TOPOLOGY//\//-}"
if [[ -z "${OUT_FILE}" ]]; then
  OUT_FILE="/tmp/mesa-smoke-${PLATFORM}-${TOPO_SLUG}-${RUN_ID}.yaml"
fi

"${PYTHON_BIN}" "${BENCHMARK_DIR}/scripts/make_smoke_jobset.py" \
  --platform "${PLATFORM}" \
  --topology "${TOPOLOGY}" \
  --model "${MODEL}" \
  --run-id "${RUN_ID}" \
  --scan-modes "${SCAN_MODES[@]}" \
  --pipelines "${PIPELINES}" \
  --out "${OUT_FILE}"

cat "${OUT_FILE}"

read -r KUBE_CONTEXT JOBSET_NAME < <("${PYTHON_BIN}" - "${BENCHMARK_DIR}" "${PLATFORM}" "${MODEL}" "${RUN_ID}" <<'PYEOF'
import os, sys
sys.path.insert(0, sys.argv[1])
from bench import platforms
from scripts import make_smoke_jobset
p = platforms.get_profile(sys.argv[2])
print(p.kube_context, make_smoke_jobset.jobset_name(sys.argv[2], sys.argv[3], sys.argv[4]))
PYEOF
)

if [[ "${APPLY}" -eq 1 ]]; then
  echo ""
  echo "Applying ${OUT_FILE} to context ${KUBE_CONTEXT}..."
  kubectl --context="${KUBE_CONTEXT}" apply -f "${OUT_FILE}"
fi

echo ""
echo "Watch pods:"
echo "  kubectl --context=${KUBE_CONTEXT} get jobsets,pods -l app=mesa-bench-v2,jobset.sigs.k8s.io/jobset-name=${JOBSET_NAME} -o wide"
echo "Read runner log:"
echo "  kubectl --context=${KUBE_CONTEXT} logs -f ${JOBSET_NAME}-slice0-0-0"
