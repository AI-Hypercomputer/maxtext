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
# Unified multi-platform v2 launcher across v7x, v6e, and v5e clusters.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=checkpoint_load_benchmark/scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

PLATFORM=""
MODE="status"
TOPOLOGY=""
MODELS=()
RUN_ID=""
SLICES=1
MIN_SLICES=1
WORKER_ID=""
QUEUE_URI=""
SOURCE_GCS_DIR=""
TIERS=("S" "A" "C" "B")
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --platform) PLATFORM="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --topology) TOPOLOGY="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --slices) SLICES="$2"; shift 2 ;;
    --min-slices) MIN_SLICES="$2"; shift 2 ;;
    --worker) WORKER_ID="$2"; shift 2 ;;
    --queue) QUEUE_URI="$2"; shift 2 ;;
    --source-gcs-dir) SOURCE_GCS_DIR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --models)
      MODELS=()
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        MODELS+=("$1")
        shift
      done
      ;;
    --tiers)
      TIERS=()
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        TIERS+=("$1")
        shift
      done
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${PLATFORM}" ]]; then
  echo "Error: --platform <v7x|v6e|v5e> is required." >&2
  exit 1
fi

read -r KUBE_CONTEXT GCS_CODE_DIR GCS_REGIONAL_ROOT < <("${PYTHON_BIN}" - "${BENCHMARK_DIR}" "${PLATFORM}" <<'PYEOF'
import sys
sys.path.insert(0, sys.argv[1])
from bench import platforms
p = platforms.get_profile(sys.argv[2])
print(p.kube_context, p.gcs_code_dir, p.regional_gcs_data_root or p.gcs_data_root)
PYEOF
)

if [[ "${MODE}" == "status" ]]; then
  kubectl --context="${KUBE_CONTEXT}" get jobsets,jobs,pods,pvc -l app=mesa-bench-v2 -o wide
  exit 0
fi

if [[ "${MODE}" == "cleanup" ]]; then
  kubectl --context="${KUBE_CONTEXT}" delete jobsets,jobs -l app=mesa-bench-v2 --ignore-not-found
  kubectl --context="${KUBE_CONTEXT}" delete pvc -l app=mesa-bench-v2 --ignore-not-found
  exit 0
fi

if [[ "${MODE}" == "upload" ]]; then
  TMP_BUNDLE="$(mktemp -d)"
  trap 'rm -rf "${TMP_BUNDLE}"' EXIT
  "${PYTHON_BIN}" "${BENCHMARK_DIR}/scripts/build_code_tarball.py" --out "${TMP_BUNDLE}" --upload "${GCS_CODE_DIR}"
  exit 0
fi

OUT_YAML="/tmp/mesa_v2_${PLATFORM}_${MODE}_${RUN_ID:-default}.yaml"
GEN="${BENCHMARK_DIR}/scripts/generate_v2_manifests.py"

case "${MODE}" in
  step1)
    "${PYTHON_BIN}" "${GEN}" step1-cpu --platform "${PLATFORM}" --models "${MODELS[@]}" --run-id "${RUN_ID}" --out "${OUT_YAML}"
    ;;
  step1b)
    "${PYTHON_BIN}" "${GEN}" step1b-cpu --model "${MODELS[0]}" --run-id "${RUN_ID}" --out "${OUT_YAML}"
    ;;
  copy-region)
    "${PYTHON_BIN}" "${GEN}" copy-region --dest-platform "${PLATFORM}" --model "${MODELS[0]}" --source-gcs-dir "${SOURCE_GCS_DIR}" --run-id "${RUN_ID}" --out "${OUT_YAML}"
    ;;
  queue-tasks)
    QUEUE_URI="${QUEUE_URI:-${GCS_REGIONAL_ROOT}/queue/${RUN_ID:-v2}}"
    "${PYTHON_BIN}" "${GEN}" queue-tasks --platform "${PLATFORM}" --run-prefix "${RUN_ID:-v2}" --tiers "${TIERS[@]}" --upload "${QUEUE_URI}/tasks.json"
    exit 0
    ;;
  queue-worker)
    QUEUE_URI="${QUEUE_URI:-${GCS_REGIONAL_ROOT}/queue/${RUN_ID:-v2}}"
    "${PYTHON_BIN}" "${GEN}" queue-worker --platform "${PLATFORM}" --topology "${TOPOLOGY}" --slices "${SLICES}" --min-slices "${MIN_SLICES}" --queue "${QUEUE_URI}" --worker "${WORKER_ID}" --run-id "${RUN_ID}" --out "${OUT_YAML}"
    ;;
  *)
    echo "Unknown --mode: ${MODE}" >&2
    exit 1
    ;;
esac

if [[ "${DRY_RUN}" -eq 1 ]]; then
  cat "${OUT_YAML}"
else
  kubectl --context="${KUBE_CONTEXT}" apply -f "${OUT_YAML}"
fi
