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

# Turnkey local CPU reproduction suite (~3-4 min, no TPU or GCS access needed).
# Runs:
#   1. Unit test suites for PR #6398 (DeepSeek FP8/MXFP4 dequant), PR #6353 (composite HBM peak),
#      and PR #6354 (to_maxtext Orbax save concurrent GB)
#   2. Benchmark v2 4-platform harness unit tests
#   3. End-to-end local CPU model parity & buffer-hygiene checks (deepseek4-284b, gemma4-26b, glm5.2-744b)
#   4. Full PARAM_MAPPING cross-model scale & legacy-rename safety audit

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/checkpoint_load_benchmark${PYTHONPATH:+:${PYTHONPATH}}"
if [[ "${XLA_FLAGS:-}" != *"--xla_force_host_platform_device_count"* ]]; then
  export XLA_FLAGS="${XLA_FLAGS:-} --xla_force_host_platform_device_count=4"
  export XLA_FLAGS="${XLA_FLAGS# }"
fi

cd "${REPO_ROOT}"

echo "========================================================================"
echo " MaxText Checkpoint Load Benchmark & PRs (#6398, #6353, #6354) CPU Repro"
echo " Repo root : ${REPO_ROOT}"
echo " Git HEAD  : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo " Python    : $(command -v "${PYTHON_BIN}")"
echo "========================================================================"

echo ""
echo "[Step 1/4] Running PR #6398, PR #6353 & PR #6354 unit tests..."
"${PYTHON_BIN}" -m pytest -q -p no:cacheprovider \
  tests/unit/deepseek_dequantize_dynamic_test.py \
  tests/unit/hf_streaming_load_test.py \
  tests/unit/safetensors_reader_test.py \
  tests/unit/to_maxtext_test.py \
  tests/unit/checkpointing_test.py -k "not pathways"

echo ""
echo "[Step 2/4] Running Benchmark v2 harness unit tests..."
"${PYTHON_BIN}" -m pytest -q -p no:cacheprovider checkpoint_load_benchmark/tests

echo ""
echo "[Step 3/4] Running reduced-dim E2E model parity & buffer-hygiene verification..."
"${PYTHON_BIN}" checkpoint_load_benchmark/scripts/repro/verify_e2e_cpu_models.py --repo-root "${REPO_ROOT}"

echo ""
echo "[Step 4/4] Running PARAM_MAPPING cross-model safety audit..."
"${PYTHON_BIN}" checkpoint_load_benchmark/scripts/repro/audit_all_param_mappings.py --quiet

echo ""
echo "========================================================================"
echo " [PASS] All 4 local CPU reproduction suites completed with 0 failures!"
echo "========================================================================"
