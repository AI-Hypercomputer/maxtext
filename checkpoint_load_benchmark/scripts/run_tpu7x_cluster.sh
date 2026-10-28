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
# ==============================================================================
# Run Single-Slice (num_slices=1, 8 TPU cores) and Multi-Slice (num_slices=2,
# 16 TPU cores) Checkpoint Load-Time Benchmarks on the GKE TPU v7 (tpu7x)
# Cluster (bodaborg-tpu7x-nap).
#
# Supports:
#   --priority p1    : Priority 1 (>200B models: qwen3-235b-a22b, deepseek4-284b,
#                      qwen3.5-397b-a17b, qwen3-480b-a35b, deepseek3-671b)
#                      - Step 1: safetensors_dynamic WITH Lance's 3 PRs (#5706+#5828+#5705)
#                      - Step 2: safetensors_dynamic BEFORE Lance's 3 PRs (origin/main)
#                      - Step 3: to_maxtext vs. safetensors_dynamic comparison
#   --priority p2    : Priority 2 (16-model benchmark suite across 1 and 2 slices)
#   --priority all   : Both Priority 1 and Priority 2 (default)
#   --priority p1-completion
#                    : Upload configs/p1_completion_tasks.json as the GCS task queue
#                      (gs://mesa-maxtext/mesakh_tpu7x_bench/queue/p1/tasks.json) and deploy its
#                      workers: pair g (2 slices) and mesakh-ctrl-f1 (single-slice tasks on the
#                      surviving daemon pod mesakh-tpu7x-f1). Re-running only re-uploads the task
#                      list; finished tasks (queue/p1/done/*.json) are never re-run.
#   --pair <a..f>    : Deploy only a specific 2-slice pod pair (a, b, c, d, e, f)
#   --status         : Show current pod and job status on bodaborg-tpu7x-nap
#   --cleanup        : Delete all benchmark pods and services on bodaborg-tpu7x-nap
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=checkpoint_load_benchmark/scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

CLUSTER_CONTEXT="${CLUSTER_CONTEXT:-gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap}"
GCS_CODE_DIR="${GCS_CODE_DIR:-gs://mesa-maxtext/mesakh_tpu7x_bench/code}"
QUEUE_P1="${QUEUE_P1:-gs://mesa-maxtext/mesakh_tpu7x_bench/queue/p1}"
PRIORITY="all"
PAIR_FILTER=""
ACTION="deploy"
MANIFEST_DIR="${MANIFEST_DIR:-/tmp/tpu7x_manifests}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --priority)
      PRIORITY="$2"
      shift 2
      ;;
    --pair)
      PAIR_FILTER="$2"
      shift 2
      ;;
    --status)
      ACTION="status"
      shift
      ;;
    --cleanup)
      ACTION="cleanup"
      shift
      ;;
    --upload-only)
      ACTION="upload"
      shift
      ;;
    --manifest-dir)
      MANIFEST_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 [--priority p1|p2|all|p1-completion] [--pair a|b|c|d|e|f] [--status] [--cleanup] [--upload-only]" >&2
      exit 1
      ;;
  esac
done

if [[ "${ACTION}" == "status" ]]; then
  kubectl --context="${CLUSTER_CONTEXT}" get pods,svc -l app=mesakh-tpu7x-bench -o wide
  exit 0
fi

if [[ "${ACTION}" == "cleanup" ]]; then
  echo "[run_tpu7x_cluster] Cleaning up all mesakh-tpu7x-bench pods and services on ${CLUSTER_CONTEXT}..."
  kubectl --context="${CLUSTER_CONTEXT}" delete pod,svc -l app=mesakh-tpu7x-bench --ignore-not-found
  kubectl --context="${CLUSTER_CONTEXT}" delete svc mesakh-w1-a mesakh-w1-b mesakh-w1-c mesakh-w1-d mesakh-w1-e mesakh-w1-f --ignore-not-found
  exit 0
fi

echo "================================================================================"
echo " MaxText TPU v7 (tpu7x) Cluster Benchmark Launcher (bodaborg-tpu7x-nap)"
echo " Cluster Context : ${CLUSTER_CONTEXT}"
echo " Machine Type    : tpu7x-standard-4t (2x2x1 = 4 chips / 8 cores / 813.8 GB HBM / 1,014.5 GB RAM per slice)"
echo " Priority Mode   : ${PRIORITY}"
echo " Pair Filter     : ${PAIR_FILTER:-all}"
echo "================================================================================"

# 1. Sync benchmark harness, bootstrap script, task-queue worker and the HBM-fix utils to GCS code bucket
echo "[1/3] Uploading harness, bootstrap, task queue and fixed utils to ${GCS_CODE_DIR}..."
"${PYTHON_BIN}" -c "
from etils import epath
files = {
    '${BENCHMARK_DIR}/benchmark_ckpt_load_time.py': 'benchmark_ckpt_load_time.py',
    '${BENCHMARK_DIR}/bootstrap_tpu7x.py': 'bootstrap_tpu7x.py',
    '${BENCHMARK_DIR}/scripts/tpu7x_task_queue.py': 'tpu7x_task_queue.py',
    '${BENCHMARK_DIR}/patches/hbmfix/src/maxtext/checkpoint_conversion/utils/load_dynamic.py': 'load_dynamic.py',
    '${BENCHMARK_DIR}/patches/hbmfix/src/maxtext/checkpoint_conversion/utils/tensor_handling.py': 'tensor_handling.py',
}
for local_p, name in files.items():
    gcs_p = '${GCS_CODE_DIR}/' + name
    epath.Path(gcs_p).write_bytes(open(local_p, 'rb').read())
    print(f'  Uploaded {local_p} -> {gcs_p}')
"

if [[ "${ACTION}" == "upload" ]]; then
  exit 0
fi

# 2. Generate Kubernetes manifests
echo "[2/3] Generating Kubernetes manifests in ${MANIFEST_DIR}..."
MANIFEST_OUT_DIR="${MANIFEST_DIR}" "${PYTHON_BIN}" "${BENCHMARK_DIR}/generate_tpu7x_manifests.py"

# 3. Apply manifests to bodaborg-tpu7x-nap
echo "[3/3] Deploying pod pairs to ${CLUSTER_CONTEXT}..."
if [[ -n "${PAIR_FILTER}" ]]; then
  kubectl --context="${CLUSTER_CONTEXT}" delete pod "mesakh-tpu7x-${PAIR_FILTER}0" "mesakh-tpu7x-${PAIR_FILTER}1" --ignore-not-found
  "${PYTHON_BIN}" -c "
import yaml, subprocess
docs = list(yaml.safe_load_all(open('${MANIFEST_DIR}/tpu7x_benchmark_pods.yaml')))
sel = [d for d in docs if d['metadata']['name'] in ('mesakh-w1-${PAIR_FILTER}', 'mesakh-tpu7x-${PAIR_FILTER}1', 'mesakh-tpu7x-${PAIR_FILTER}0')]
payload = yaml.safe_dump_all(sel, sort_keys=False)
subprocess.run(['kubectl', '--context=${CLUSTER_CONTEXT}', 'apply', '-f', '-'], input=payload, text=True, check=True)
"
elif [[ "${PRIORITY}" == "p1" ]]; then
  kubectl --context="${CLUSTER_CONTEXT}" apply -f "${MANIFEST_DIR}/tpu7x_p1_pods.yaml"
elif [[ "${PRIORITY}" == "p2" ]]; then
  kubectl --context="${CLUSTER_CONTEXT}" apply -f "${MANIFEST_DIR}/tpu7x_p2_pods.yaml"
elif [[ "${PRIORITY}" == "p1-completion" ]]; then
  "${PYTHON_BIN}" -c "
from etils import epath
epath.Path('${QUEUE_P1}/tasks.json').write_bytes(open('${BENCHMARK_DIR}/configs/p1_completion_tasks.json', 'rb').read())
print('  Uploaded task list -> ${QUEUE_P1}/tasks.json')
"
  kubectl --context="${CLUSTER_CONTEXT}" apply -f "${MANIFEST_DIR}/tpu7x_p1_completion_pods.yaml"
  if kubectl --context="${CLUSTER_CONTEXT}" get pod mesakh-tpu7x-f1 >/dev/null 2>&1; then
    kubectl --context="${CLUSTER_CONTEXT}" apply -f "${MANIFEST_DIR}/tpu7x_ctrl_f1_pod.yaml"
  fi
else
  kubectl --context="${CLUSTER_CONTEXT}" apply -f "${MANIFEST_DIR}/tpu7x_benchmark_pods.yaml"
fi

echo ""
echo "Deployed! Check live status and logs with:"
echo "  ./checkpoint_load_benchmark/scripts/run_tpu7x_cluster.sh --status"
echo "  kubectl --context=${CLUSTER_CONTEXT} logs -f mesakh-tpu7x-a0"
