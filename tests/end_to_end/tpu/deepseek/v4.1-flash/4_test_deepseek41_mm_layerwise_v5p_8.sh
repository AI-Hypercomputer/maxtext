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

# Real-weight layerwise verification of DeepSeek-V4.1-Flash multimodal on a v5p-8 TPU VM (gates L1-L4).
#
# Every variable is required; nothing falls back to a default.
#   PYTHON_BIN        Python interpreter of the VM's MaxText environment.
#   HF_CKPT_DIR       Local directory of the release HF safetensors checkpoint.
#   TOKENIZER_PATH    Local HF tokenizer directory.
#   BUNDLE_GCS        gs:// prefix holding the reference bundles, bundle_index.json and SHA256SUMS.txt.
#   BUNDLE_DIR        Local directory the bundles are downloaded to.
#   AOT_DIR           Directory of the v5p-8 AOT reports, one per sample: lw_aot_v5p_8_<sample>.json written by
#                     aot_check_layerwise.py; each must cover every program its sample's runs execute.
#   OUTPUT_DIR        Directory for per-run JSON reports and logs.
#   LW_RUNS           Space-separated runs, each "<stage>:<tier>:<sample>:<target>" (targets: see
#                     layerwise_harness.py --target). Must include every run of layerwise_required_runs.json; the
#                     planner (lw_runs.py plan) refuses it otherwise.
#
# Order: the pinned HBM probe (tier-b Engram layer, full table) runs first, then the other runs, then every run whose
# v5p-8 AOT peak exceeds the pinned threshold. Each run is its own process, so device buffers and host memory of one
# run never stay alive into the next. Every run is executed and its exit code recorded; the summary (lw_runs.py
# summarize) prints the status table with the special-status counts and exits nonzero unless every run is PASS.
# Tier a sets the matmul precision to HIGHEST in process (lw_config.configure_runtime) instead of through
# JAX_DEFAULT_MATMUL_PRECISION, which this script unsets for every run.

set -uo pipefail

for var in PYTHON_BIN HF_CKPT_DIR TOKENIZER_PATH BUNDLE_GCS BUNDLE_DIR AOT_DIR OUTPUT_DIR LW_RUNS; do
  if [[ -z "${!var:-}" ]]; then
    echo "ERROR: ${var} must be set" >&2
    exit 2
  fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src"
DATA_PINS_JSON="${SCRIPT_DIR}/layerwise_data_pins.json"
PINNED_NOISE_FLOORS_SHA256="861ff65d5ba31b9879b83f79075e69f8279dd62dd53fff0b8230be6cdc5ca510"

mkdir -p "${BUNDLE_DIR}" "${OUTPUT_DIR}" || exit 2

# 1. Host-RAM Preflight (>= 100 GiB MemAvailable required).
MEM_AVAIL_KB="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
MEM_MIN_KB=$((100 * 1024 * 1024))
MEM_AVAIL_GIB="$(awk -v kb="${MEM_AVAIL_KB:-0}" 'BEGIN {printf "%.2f", kb / (1024.0 * 1024.0)}')"
if [[ -z "${MEM_AVAIL_KB:-}" || "${MEM_AVAIL_KB}" -lt "${MEM_MIN_KB}" ]]; then
  echo "ERROR: Host-RAM preflight failed: MemAvailable=${MEM_AVAIL_GIB} GiB (< 100.00 GiB required)" >&2
  exit 1
fi
echo "[PREFLIGHT] Host RAM OK: MemAvailable=${MEM_AVAIL_GIB} GiB (>= 100.00 GiB)"

# 1a. Optional JAX Compilation Cache on persistent local SSD.
if [[ -d "/mnt/disks/data" ]]; then
  export JAX_COMPILATION_CACHE_DIR="/mnt/disks/data/jax_cache"
  mkdir -p "${JAX_COMPILATION_CACHE_DIR}"
fi

# 1b. Verify /mnt/disks/persist starts clean (0 bytes in use if mount exists).
if [[ -d "/mnt/disks/persist" ]]; then
  PERSIST_USED_KB="$(du -sk "/mnt/disks/persist" 2>/dev/null | awk '{print $1}')"
  if [[ "${PERSIST_USED_KB:-0}" -gt 0 ]]; then
    echo "WARNING: /mnt/disks/persist is not empty before run (${PERSIST_USED_KB} KB used)" >&2
  fi
fi

verify_local_bundles() {
  if [[ ! -f "${BUNDLE_DIR}/SHA256SUMS.txt" || ! -f "${BUNDLE_DIR}/bundle_index.json" || ! -f "${BUNDLE_DIR}/measured_noise_floors.json" ]]; then
    return 1
  fi
  "${PYTHON_BIN}" - "${BUNDLE_DIR}" "${DATA_PINS_JSON}" "${PINNED_NOISE_FLOORS_SHA256}" <<'PYEOF' || return 1
import json, pathlib, sys
from tests.utils.deepseek_v41_layerwise import lw_bundle, lw_checks

root = pathlib.Path(sys.argv[1])
pins = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
expected_idx_sha = pins["bundle_index_sha256"]
expected_floors_sha = pins.get("measured_noise_floors_sha256", sys.argv[3])
actual_idx_sha = lw_checks.sha256_file(root / "bundle_index.json")
actual_floors_sha = lw_checks.sha256_file(root / "measured_noise_floors.json")
if actual_idx_sha != expected_idx_sha:
  print(
      f"bundle_index.json sha256 mismatch: got {actual_idx_sha}, expected {expected_idx_sha}",
      file=sys.stderr,
  )
  sys.exit(1)
if actual_floors_sha != expected_floors_sha:
  print(
      f"measured_noise_floors.json sha256 mismatch: got {actual_floors_sha}, expected {expected_floors_sha}",
      file=sys.stderr,
  )
  sys.exit(1)
lw_bundle.load_index(root, pins_path=pathlib.Path(sys.argv[2]))
lw_bundle.load_noise_floors(root)
PYEOF
  if (cd "${BUNDLE_DIR}" && sha256sum -c --quiet SHA256SUMS.txt); then
    return 0
  fi
  return 1
}

LOCAL_BUNDLES_VERIFIED=0
if verify_local_bundles; then
  LOCAL_BUNDLES_VERIFIED=1
fi

# 2. Disk-Space Preflight on filesystems backing BUNDLE_DIR, OUTPUT_DIR, and HF_CKPT_DIR.
REF_BUNDLES_TARGET_BYTES=$((130 * 1024 * 1024 * 1024))
HEADROOM_BYTES=$((50 * 1024 * 1024 * 1024))
if [[ "${LOCAL_BUNDLES_VERIFIED}" -eq 1 ]]; then
  NEEDED_BYTES=0
else
  PRESENT_REF_KB=0
  if [[ -d "${BUNDLE_DIR}/ref_bundles" ]]; then
    PRESENT_REF_KB="$(du -sk "${BUNDLE_DIR}/ref_bundles" | awk '{print $1}')"
  fi
  PRESENT_REF_BYTES=$((PRESENT_REF_KB * 1024))
  if [[ "${PRESENT_REF_BYTES}" -ge "${REF_BUNDLES_TARGET_BYTES}" ]]; then
    NEEDED_BYTES=0
  else
    NEEDED_BYTES=$((REF_BUNDLES_TARGET_BYTES - PRESENT_REF_BYTES))
  fi
fi

for check_dir in "${BUNDLE_DIR}" "${OUTPUT_DIR}" "${HF_CKPT_DIR}"; do
  if [[ ! -d "${check_dir}" ]]; then
    echo "ERROR: Required directory ${check_dir} does not exist" >&2
    exit 1
  fi
  AVAIL_KB="$(df -Pk "${check_dir}" | awk 'NR==2 {print $4}')"
  FREE_BYTES=$((AVAIL_KB * 1024))
  if [[ "${check_dir}" == "${BUNDLE_DIR}" ]]; then
    REQ_BYTES=$((NEEDED_BYTES + HEADROOM_BYTES))
  else
    REQ_BYTES="${HEADROOM_BYTES}"
  fi
  FREE_GB="$(awk -v b="${FREE_BYTES}" 'BEGIN {printf "%.2f", b / (1024.0 * 1024.0 * 1024.0)}')"
  REQ_GB="$(awk -v b="${REQ_BYTES}" 'BEGIN {printf "%.2f", b / (1024.0 * 1024.0 * 1024.0)}')"
  if [[ "${FREE_BYTES}" -lt "${REQ_BYTES}" ]]; then
    echo "ERROR: Disk-space preflight failed for ${check_dir}: free_gb=${FREE_GB} GiB, required_gb=${REQ_GB} GiB" >&2
    exit 1
  fi
  echo "[PREFLIGHT] Disk space OK for ${check_dir}: free_gb=${FREE_GB} GiB, required_gb=${REQ_GB} GiB"
done

# 3. Skip GCS sync if local SHA256SUMS.txt + layerwise_data_pins.json verification passed; otherwise sync selectively.
if [[ "${LOCAL_BUNDLES_VERIFIED}" -eq 1 ]]; then
  echo "Local bundle directory ${BUNDLE_DIR} already verified against SHA256SUMS.txt and layerwise_data_pins.json; skipping GCS sync."
else
  gcloud storage cp "${BUNDLE_GCS}/bundle_index.json" "${BUNDLE_DIR}/bundle_index.json" || exit 2
  gcloud storage cp "${BUNDLE_GCS}/SHA256SUMS.txt" "${BUNDLE_DIR}/SHA256SUMS.txt" || exit 2
  gcloud storage cp "${BUNDLE_GCS}/measured_noise_floors.json" "${BUNDLE_DIR}/measured_noise_floors.json" || exit 2
  mkdir -p "${BUNDLE_DIR}/ref_bundles" || exit 2
  gcloud storage rsync --recursive "${BUNDLE_GCS}/ref_bundles" "${BUNDLE_DIR}/ref_bundles" || exit 2
  "${PYTHON_BIN}" - "${BUNDLE_DIR}/bundle_index.json" "${BUNDLE_DIR}/missing_bundle_paths.txt" <<'PYEOF' || exit 2
import json, pathlib, sys
idx_path = pathlib.Path(sys.argv[1])
out_path = pathlib.Path(sys.argv[2])
root = idx_path.parent
bundles = json.loads(idx_path.read_text(encoding="utf-8"))["bundles"]
present_shas = {meta["sha256"] for rel, meta in bundles.items() if (root / rel).is_file()}
missing = []
for rel, meta in sorted(bundles.items()):
  if rel.startswith("ref_bundles/") and not (root / rel).is_file() and meta["sha256"] not in present_shas:
    missing.append(rel)
    present_shas.add(meta["sha256"])
out_path.write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")
PYEOF
  while IFS= read -r rel_path; do
    if [[ -n "${rel_path}" && ! -f "${BUNDLE_DIR}/${rel_path}" ]]; then
      mkdir -p "$(dirname "${BUNDLE_DIR}/${rel_path}")" || exit 2
      gcloud storage cp "${BUNDLE_GCS}/${rel_path}" "${BUNDLE_DIR}/${rel_path}" || exit 2
    fi
  done < "${BUNDLE_DIR}/missing_bundle_paths.txt"
  rm -f "${BUNDLE_DIR}/missing_bundle_paths.txt"
  verify_local_bundles || exit 2
fi

PLAN="${OUTPUT_DIR}/run_plan.tsv"
EXIT_CODES="${OUTPUT_DIR}/exit_codes.tsv"
"${PYTHON_BIN}" -m tests.utils.deepseek_v41_layerwise.lw_runs plan \
  --lw-runs="${LW_RUNS}" --aot-dir="${AOT_DIR}" --num-devices=4 --output="${PLAN}" || exit 2
: > "${EXIT_CODES}"

"${PYTHON_BIN}" -m tests.utils.deepseek_v41_layerwise.verify_layerwise \
  --lw-runs="${PLAN}" --aot-dir="${AOT_DIR}" --preflight-only

while IFS=$'\t' read -r tag stage tier sample target peak <&3; do
  env_args=()
  if [[ "${tier}" == "code_parity" || "${tier}" == "production" ]]; then
    env_args+=("LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=65536")
  fi
  echo "[LW_STAGE_TIMING] start ${tag} (AOT peak ${peak} GiB/device) $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  env -u JAX_DEFAULT_MATMUL_PRECISION "${env_args[@]}" "${PYTHON_BIN}" -m tests.utils.deepseek_v41_layerwise.lw_runs run \
    --stage="${stage}" \
    --precision-tier="${tier}" \
    --sample="${sample}" \
    --target="${target}" \
    --bundle-dir="${BUNDLE_DIR}" \
    --hf-ckpt-dir="${HF_CKPT_DIR}" \
    --tokenizer-path="${TOKENIZER_PATH}" \
    --aot-json="${AOT_DIR}/lw_aot_v5p_8_${sample}.json" \
    --output-json="${OUTPUT_DIR}/${tag}.json" 2>&1 | tee "${OUTPUT_DIR}/${tag}.log"
  printf '%s\t%s\n' "${tag}" "${PIPESTATUS[0]}" >> "${EXIT_CODES}"
  echo "[LW_STAGE_TIMING] end ${tag} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
done 3< "${PLAN}"

# 4. Post-run persist check.
if [[ -d "/mnt/disks/persist" ]]; then
  POST_PERSIST_USED_KB="$(du -sk "/mnt/disks/persist" 2>/dev/null | awk '{print $1}')"
  echo "[POSTFLIGHT] /mnt/disks/persist footprint: ${POST_PERSIST_USED_KB:-0} KB"
fi


# 5. Step 5: Chained 40-layer forward, hidden norm consistency, and Step-1 R4 pretraining parity
echo "[STEP 5] Running chained 40-layer forward, hidden norm consistency, and step-1 R4 parity..."
"${PYTHON_BIN}" -m tests.utils.deepseek_v41_layerwise.lw_chain \
  --hf-ckpt-dir="${HF_CKPT_DIR}" \
  --bundle-dir="${BUNDLE_DIR}" \
  --output-dir="${OUTPUT_DIR}/step5_chain" \
  --weight-cache-dir="${LW_STEP1_WEIGHT_CACHE_DIR:-/mnt/disks/data/ds41_lw_step1_weight_cache}" || exit 2

"${PYTHON_BIN}" -m tests.utils.deepseek_v41_layerwise.lw_runs summarize \
  --plan="${PLAN}" --exit-codes="${EXIT_CODES}" --output-dir="${OUTPUT_DIR}"
