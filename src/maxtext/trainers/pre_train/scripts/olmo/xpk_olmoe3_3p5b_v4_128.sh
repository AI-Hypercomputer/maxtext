#!/bin/bash
#
# Copyright 2023-2026 Google LLC
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
#
# Cloud TPU GKE / XPK launcher for OLMo 3.5 Small (olmoe3-3p5b) on
# cluster `v4-128-bodaborg-us-central2-b` (Project: `cloud-tpu-multipod-dev`,
# Zone: `us-central2-b`, Node Pools: `v4-128-bodaborg-us-central2-b-np-0..3`,
# Machine Type: `ct4p-hightpu-4t`, 16 nodes / 64 TPU v4 chips = `4x4x4` per pool).
#
# Supports both `xpk workload create` and direct `kubectl apply` (GKE JobSet)
# when `xpk` CLI is not installed in PATH.
#
# Usage:
#   # 1. Authenticate GKE cluster credentials (once):
#   bash src/maxtext/trainers/pre_train/scripts/olmo/xpk_olmoe3_3p5b_v4_128.sh auth
#
#   # 2. Submit single-slice 1x v4-128 (64 chips, 4x4x4, FSDP=4, EP=16) at 8k:
#   SEQ_LEN=8192 bash src/maxtext/trainers/pre_train/scripts/olmo/xpk_olmoe3_3p5b_v4_128.sh submit
#
#   # 3. Submit single-slice 1x v4-128 (64 chips, 4x4x4, FSDP=4, EP=16) at 4k:
#   SEQ_LEN=4096 bash src/maxtext/trainers/pre_train/scripts/olmo/xpk_olmoe3_3p5b_v4_128.sh submit
#
#   # 4. Submit 4-slice 4x v4-128 (256 chips across np-0..np-3, DCN_DATA=4, FSDP=4, EP=16):
#   XPK_NUM_SLICES=4 DCN_DATA_PARALLELISM=4 SEQ_LEN=8192 \
#     bash src/maxtext/trainers/pre_train/scripts/olmo/xpk_olmoe3_3p5b_v4_128.sh submit
#
#   # 5. Submit 4 parallel 1x v4-128 jobs pinned one per node pool (np-0, np-1, np-2, np-3):
#   bash src/maxtext/trainers/pre_train/scripts/olmo/xpk_olmoe3_3p5b_v4_128.sh submit_all_pools
#
#   # 6. Monitor or delete workload:
#   bash src/maxtext/trainers/pre_train/scripts/olmo/xpk_olmoe3_3p5b_v4_128.sh monitor [workload]
#   bash src/maxtext/trainers/pre_train/scripts/olmo/xpk_olmoe3_3p5b_v4_128.sh delete [workload]

set -euo pipefail

if ! command -v xpk >/dev/null 2>&1 && [[ -x "${HOME}/venvp3/bin/xpk" ]]; then
  export PATH="${HOME}/venvp3/bin:${PATH}"
fi

MODE="${1:-submit}"

# -------------------------- Cluster Defaults --------------------------
: "${XPK_PROJECT:=cloud-tpu-multipod-dev}"
: "${XPK_CLUSTER:=v4-128-bodaborg-us-central2-b}"
: "${XPK_ZONE:=us-central2-b}"
: "${XPK_DEVICE_TYPE:=v4-128}"
: "${XPK_NUM_SLICES:=1}"
: "${XPK_PRIORITY:=medium}"
: "${XPK_DOCKER_IMAGE:=gcr.io/${XPK_PROJECT}/maxtext-olmo35:latest}"
: "${XPK_BASE_OUTPUT_DIR:=gs://${XPK_PROJECT}-maxtext-output/olmoe3_3p5b}"

# -------------------------- Model & Parallelism Defaults --------------------------
: "${SEQ_LEN:=8192}"
: "${PER_DEVICE_BATCH:=1.0}"
: "${GRAD_ACCUM_STEPS:=1}"
: "${STEPS:=30}"
: "${LOG_PERIOD:=10}"
: "${DATASET_TYPE:=synthetic}"
: "${DCN_DATA_PARALLELISM:=${XPK_NUM_SLICES}}"
: "${ICI_FSDP_PARALLELISM:=4}"
: "${ICI_EXPERT_PARALLELISM:=16}"
: "${PROFILER:=}"
: "${NODE_POOL:=}"  # Optional: e.g. v4-128-bodaborg-us-central2-b-np-0

: "${XPK_WORKLOAD:=olmoe3-v4-l${SEQ_LEN}-s${XPK_NUM_SLICES}-$(printf '%04d' $((RANDOM % 10000)))}"
: "${XPK_RUN_NAME:=${XPK_WORKLOAD}}"

LAST_WORKLOAD_FILE="${XPK_LAST_WORKLOAD_FILE:-${HOME}/.xpk_last_workload_olmoe3}"

auth_cluster() {
  echo "Fetching credentials for GKE cluster ${XPK_CLUSTER} (${XPK_PROJECT}, ${XPK_ZONE})..."
  gcloud container clusters get-credentials "${XPK_CLUSTER}" \
    --project="${XPK_PROJECT}" \
    --zone="${XPK_ZONE}"
  kubectl get nodes -L cloud.google.com/gke-nodepool,cloud.google.com/gke-tpu-topology
}

build_inner_command() {
  cat <<EOF
set -euo pipefail
export PYTHONPATH=/deps/src:\${PYTHONPATH:-}
export RUN_NAME='${XPK_RUN_NAME}'
export OUTPUT_DIR='${XPK_BASE_OUTPUT_DIR}'
export DATASET_TYPE='${DATASET_TYPE}'
export SEQ_LEN='${SEQ_LEN}'
export PER_DEVICE_BATCH='${PER_DEVICE_BATCH}'
export GRAD_ACCUM_STEPS='${GRAD_ACCUM_STEPS}'
export STEPS='${STEPS}'
export LOG_PERIOD='${LOG_PERIOD}'
export DCN_DATA_PARALLELISM='${DCN_DATA_PARALLELISM}'
export ICI_FSDP_PARALLELISM='${ICI_FSDP_PARALLELISM}'
export ICI_EXPERT_PARALLELISM='${ICI_EXPERT_PARALLELISM}'
export PROFILER='${PROFILER}'
export VENV_PATH=/__skip_venv__
bash /deps/src/maxtext/trainers/pre_train/scripts/olmo/run_olmoe3_3p5b_v4.sh
EOF
}

submit_via_xpk() {
  local cmd
  cmd="$(build_inner_command | tr '\n' '; ')"
  xpk workload create \
    --cluster "${XPK_CLUSTER}" \
    --workload "${XPK_WORKLOAD}" \
    --priority="${XPK_PRIORITY}" \
    --tpu-type="${XPK_DEVICE_TYPE}" \
    --num-slices="${XPK_NUM_SLICES}" \
    --project="${XPK_PROJECT}" \
    --zone="${XPK_ZONE}" \
    --docker-image="${XPK_DOCKER_IMAGE}" \
    --command "${cmd}"
}

submit_via_kubectl_jobset() {
  local inner_cmd
  inner_cmd="$(build_inner_command)"
  local node_selector_extra=""
  if [[ -n "${NODE_POOL}" ]]; then
    node_selector_extra="cloud.google.com/gke-nodepool: \"${NODE_POOL}\""
  fi

  kubectl apply -f - <<EOF
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  name: ${XPK_WORKLOAD}
  labels:
    app: olmoe3-3p5b-v4
spec:
  failurePolicy:
    maxRestarts: 0
  replicatedJobs:
  - name: slice
    replicas: ${XPK_NUM_SLICES}
    template:
      spec:
        parallelism: 16
        completions: 16
        backoffLimit: 0
        template:
          spec:
            restartPolicy: Never
            nodeSelector:
              cloud.google.com/gke-tpu-accelerator: tpu-v4-podslice
              cloud.google.com/gke-tpu-topology: 4x4x4
              ${node_selector_extra}
            containers:
            - name: maxtext-runner
              image: ${XPK_DOCKER_IMAGE}
              command: ["bash", "-lc"]
              args:
              - |
$(echo "${inner_cmd}" | sed 's/^/                /')
              resources:
                limits:
                  google.com/tpu: 4
                requests:
                  google.com/tpu: 4
EOF
}

submit_workload() {
  echo "=== Submitting OLMo 3.5 Small (olmoe3-3p5b) to Cloud TPU GKE ==="
  echo "  Workload  : ${XPK_WORKLOAD}"
  echo "  Cluster   : ${XPK_CLUSTER} (${XPK_PROJECT}, ${XPK_ZONE})"
  echo "  Topology  : ${XPK_DEVICE_TYPE} (4x4x4, 16 nodes / 64 chips) x ${XPK_NUM_SLICES} slice(s)"
  echo "  Node Pool : ${NODE_POOL:-<auto-scheduled across np-0..np-3>}"
  echo "  Seq Len   : ${SEQ_LEN}"
  echo "  Sharding  : DCN_DATA=${DCN_DATA_PARALLELISM}, ICI_FSDP=${ICI_FSDP_PARALLELISM}, ICI_EP=${ICI_EXPERT_PARALLELISM}"
  echo "  Image     : ${XPK_DOCKER_IMAGE}"
  echo "  Output    : ${XPK_BASE_OUTPUT_DIR}/${XPK_RUN_NAME}"
  echo

  if command -v xpk >/dev/null 2>&1 && [[ -z "${NODE_POOL}" ]]; then
    submit_via_xpk
  else
    submit_via_kubectl_jobset
  fi
  echo "${XPK_WORKLOAD}" > "${LAST_WORKLOAD_FILE}"
}

submit_all_pools() {
  # Launches 4 concurrent single-slice v4-128 benchmark runs across np-0..np-3:
  #   np-0: SEQ_LEN=4096, opt_lean_v3 (context=device, moe_dispatch=device)
  #   np-1: SEQ_LEN=8192, hc5 (context=remat, moe_dispatch=remat, moe_x_sorted=device)
  #   np-2: SEQ_LEN=8192, hc5 + profiler=xplane
  #   np-3: SEQ_LEN=8192, hc5 + grad_accum=2
  NODE_POOL="v4-128-bodaborg-us-central2-b-np-0" \
    SEQ_LEN=4096 XPK_WORKLOAD="olmoe3-np0-l4096-hc5" \
    submit_workload

  NODE_POOL="v4-128-bodaborg-us-central2-b-np-1" \
    SEQ_LEN=8192 XPK_WORKLOAD="olmoe3-np1-l8192-hc5" \
    submit_workload

  NODE_POOL="v4-128-bodaborg-us-central2-b-np-2" \
    SEQ_LEN=8192 PROFILER=xplane XPK_WORKLOAD="olmoe3-np2-l8192-xprof" \
    submit_workload

  NODE_POOL="v4-128-bodaborg-us-central2-b-np-3" \
    SEQ_LEN=8192 GRAD_ACCUM_STEPS=2 XPK_WORKLOAD="olmoe3-np3-l8192-ga2" \
    submit_workload
}

monitor_workload() {
  local workload="${2:-$(cat "${LAST_WORKLOAD_FILE}" 2>/dev/null || true)}"
  if [[ -z "${workload}" ]]; then
    echo "ERROR: no workload specified and none found in ${LAST_WORKLOAD_FILE}." >&2
    exit 1
  fi
  echo "Monitoring pods for JobSet ${workload}..."
  kubectl get pods -l "jobset.sigs.k8s.io/jobset-name=${workload}" -o wide
  local pod0
  pod0=$(kubectl get pods -l "jobset.sigs.k8s.io/jobset-name=${workload}" \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
  if [[ -n "${pod0}" ]]; then
    echo "--- Streaming logs from ${pod0} ---"
    kubectl logs -f "${pod0}"
  fi
}

delete_workload() {
  local workload="${2:-$(cat "${LAST_WORKLOAD_FILE}" 2>/dev/null || true)}"
  if [[ -z "${workload}" ]]; then
    echo "ERROR: no workload specified." >&2
    exit 1
  fi
  if command -v xpk >/dev/null 2>&1; then
    xpk workload delete --cluster "${XPK_CLUSTER}" --workload "${workload}" \
      --project="${XPK_PROJECT}" --zone="${XPK_ZONE}"
  else
    kubectl delete jobset "${workload}"
  fi
}

case "${MODE}" in
  auth)             auth_cluster ;;
  submit)           submit_workload ;;
  submit_all_pools) submit_all_pools ;;
  monitor)          monitor_workload "$@" ;;
  delete)           delete_workload "$@" ;;
  *)
    echo "ERROR: unknown mode '${MODE}' (expected: auth | submit | submit_all_pools | monitor | delete)" >&2
    exit 1
    ;;
esac
