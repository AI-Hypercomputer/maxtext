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
# XPK wrapper for run_olmo3_7b_stage1.sh. Submits a multi-host TPU workload
# that runs the launcher inside the MaxText runner image. Stays thin — the
# inner script is the source of truth for hyperparameters; this script only
# handles cluster/storage/image plumbing and forwards env vars.
#
# Pattern mirrors src/maxtext/trainers/post_train/distillation/scripts/run_distill_xpk.sh.
#
# Usage:
#   bash src/maxtext/trainers/pre_train/scripts/olmo/xpk_olmo3_7b_stage1.sh submit
#   bash src/maxtext/trainers/pre_train/scripts/olmo/xpk_olmo3_7b_stage1.sh monitor
#   bash src/maxtext/trainers/pre_train/scripts/olmo/xpk_olmo3_7b_stage1.sh resume_until_done
#
# Required env:
#   XPK_CLUSTER, XPK_PROJECT, XPK_ZONE, XPK_DEVICE_TYPE, XPK_BASE_OUTPUT_DIR
#   OLMO_INDEX_PATH      JSON index from build_olmo_npy_index.py (in-container path)
#   OLMO_GCS_BASE        gs:// prefix recorded in the index
#   HF_TOKEN             tokenizer download (or HF_SECRETS=<file>)
#
# Optional (one of):
#   LOAD_PARAMETERS_PATH cold-start init weights (Orbax dir; e.g. converted
#                        AI2 stage1-stepN snapshot). If unset, MaxText
#                        random-inits from olmo3-7b-pt.
#
# Optional (with defaults):
#   XPK_DOCKER_IMAGE    gcr.io/${XPK_PROJECT}/maxtext-olmo3:latest
#                       Image must be PUSHED to GCR/GAR. Build+push:
#                         sudo bash src/dependencies/scripts/docker_build_dependency_image.sh \
#                           MODE=stable WORKFLOW=pre-training
#                         sudo bash src/dependencies/scripts/docker_upload_runner.sh \
#                           CLOUD_IMAGE_NAME=maxtext-olmo3 PROJECT=$XPK_PROJECT
#                       (after `sudo gcloud auth configure-docker gcr.io`)
#   XPK_WORKLOAD        olmo3-s1-NNNN  (DNS-label safe; keep stable across resumes)
#   XPK_PRIORITY        medium
#   XPK_NUM_SLICES      1
#   XPK_RUN_NAME        olmo3_7b_stage1  (subdir under XPK_BASE_OUTPUT_DIR)
#   XPK_TOTAL_DEVICES   128  (tpu7x-4x4x4); update if XPK_DEVICE_TYPE changes
#   OLMO_LOCAL_MOUNT    /tmp/olmo-data  (gcsfuse mount path inside container)
#   TARGET_GLOBAL_BATCH 512  (instances; ×8192 seq = 4.19M tokens)
#   STEPS               1420000  (≈6T tokens / 4M per step)
#   WARMUP_STEPS        2000  (absolute, in schedule space)
#   LR_SCHEDULE_STEPS   STEPS — for partial overlay runs, set to the reference
#                       horizon (e.g. 1193317 for OLMo-3 pretrain-1's 5T-token
#                       schedule) so warmup stays absolute and cosine matches.
#   LEARNING_RATE       3.0e-4
#   COSINE_FINAL_FRAC   0.1
#   CHECKPOINT_PERIOD   1000
#   DATA_SEED           42
#   STEPS_OVERRIDE      empty — only used by resume_until_done as termination target
#   MAX_RETRIES         10  — only used by resume_until_done
#   RETRY_BACKOFF_SECONDS 300 — sleep between resubmits in resume_until_done
#   HF_SECRETS          empty — file that exports HF_TOKEN (kept off shell history)
#
# Restart policy:
#   L1 (pod):       GKE restarts crashed pods; trainer auto-resumes via Orbax.
#   L2 (workload):  resume_until_done deletes + resubmits with the same
#                   XPK_WORKLOAD/XPK_BASE_OUTPUT_DIR (Orbax restores state).
#   L3 (human):     resume_until_done exits non-zero after MAX_RETRIES.

set -euo pipefail

MODE="${1:-submit}"

require_env() {
  local missing=()
  for v in "$@"; do
    [ -z "${!v:-}" ] && missing+=("$v")
  done
  if [ "${#missing[@]}" -gt 0 ]; then
    echo "ERROR: required env vars not set: ${missing[*]}" >&2
    exit 1
  fi
}

# -------------------------- defaults --------------------------
: "${XPK_DOCKER_IMAGE:=gcr.io/${XPK_PROJECT:-PROJECT_UNSET}/maxtext-olmo3:latest}"
# Workload names are DNS labels: lowercase, [a-z0-9-] only. Fixed-width 4-digit
# suffix disambiguates concurrent submits; override XPK_WORKLOAD to keep the
# same name across resumes (Orbax picks up the latest checkpoint).
: "${XPK_WORKLOAD:=olmo3-s1-$(printf '%04d' $((RANDOM % 10000)))}"
: "${XPK_PRIORITY:=medium}"
: "${XPK_NUM_SLICES:=1}"
: "${XPK_RUN_NAME:=olmo3_7b_stage1}"
: "${OLMO_LOCAL_MOUNT:=/tmp/olmo-data}"
: "${XPK_TOTAL_DEVICES:=128}"
: "${TARGET_GLOBAL_BATCH:=512}"
: "${STEPS:=1414078}"
: "${WARMUP_STEPS:=2000}"
: "${LR_SCHEDULE_STEPS:=${STEPS}}"
: "${LEARNING_RATE:=3.0e-4}"
: "${COSINE_FINAL_FRAC:=0.1}"
: "${CHECKPOINT_PERIOD:=1000}"
: "${DATA_SEED:=42}"
: "${MAX_RETRIES:=10}"
# Sleep between preemption-recovery resubmits. Bumped from the original 60s so
# Kueue has time to free capacity after a preemption wave; override to taste.
: "${RETRY_BACKOFF_SECONDS:=300}"
: "${HF_SECRETS:=}"

# Container-side index path. Absolute paths pass through (typical: a gcsfuse-
# mounted bucket like /tmp/olmo-data/...); relative paths are resolved against
# /deps (the runner image's WORKDIR, where the repo source is copied).
if [[ "${OLMO_INDEX_PATH:-}" = /* ]]; then
  OLMO_INDEX_PATH_IN_CONTAINER="${OLMO_INDEX_PATH}"
else
  OLMO_INDEX_PATH_IN_CONTAINER="/deps/${OLMO_INDEX_PATH:-}"
fi

# Read HF_TOKEN at submit time so it can be forwarded; the container can't see
# the host's home dir.
if [ -z "${HF_TOKEN:-}" ] && [ -n "$HF_SECRETS" ] && [ -f "$HF_SECRETS" ]; then
  # shellcheck disable=SC1090
  source "$HF_SECRETS"
fi

LAST_WORKLOAD_FILE="${XPK_LAST_WORKLOAD_FILE:-${HOME}/.xpk_last_workload_olmo}"

# -------------------------- submit --------------------------
submit_workload() {
  echo "Workload:    $XPK_WORKLOAD"
  echo "Cluster:     $XPK_CLUSTER ($XPK_PROJECT, $XPK_ZONE)"
  echo "Device:      $XPK_DEVICE_TYPE x ${XPK_NUM_SLICES} slice(s)"
  echo "Image:       $XPK_DOCKER_IMAGE"
  echo "Output dir:  $XPK_BASE_OUTPUT_DIR/$XPK_RUN_NAME"
  echo "Index:       $OLMO_INDEX_PATH_IN_CONTAINER"
  echo "Data bucket: $OLMO_GCS_BASE → $OLMO_LOCAL_MOUNT (gcsfuse)"
  echo

  # MOUNT_GCSFUSE=1 tells the inner launcher to invoke setup_gcsfuse.sh — each
  # pod mounts its own copy at startup, no cluster-side configuration required.
  # VENV_PATH=/__skip_venv__ skips the venv lookup so the launcher falls
  # through to the runner image's system Python.
  xpk workload create \
    --cluster "$XPK_CLUSTER" \
    --workload "$XPK_WORKLOAD" \
    --priority="$XPK_PRIORITY" \
    --tpu-type="$XPK_DEVICE_TYPE" \
    --num-slices="$XPK_NUM_SLICES" \
    --project="$XPK_PROJECT" \
    --zone="$XPK_ZONE" \
    --docker-image="$XPK_DOCKER_IMAGE" \
    --command "set -euo pipefail; \
export PYTHONPATH=/deps/src; \
export HF_TOKEN='${HF_TOKEN}'; \
export INDEX_PATH='${OLMO_INDEX_PATH_IN_CONTAINER}'; \
export GCS_BASE='${OLMO_GCS_BASE}'; \
export LOCAL_MOUNT='${OLMO_LOCAL_MOUNT}'; \
export OUTPUT_DIR='${XPK_BASE_OUTPUT_DIR}'; \
export RUN_NAME='${XPK_RUN_NAME}'; \
export LOAD_PARAMETERS_PATH='${LOAD_PARAMETERS_PATH}'; \
export TARGET_GLOBAL_BATCH='${TARGET_GLOBAL_BATCH}'; \
export TOTAL_DEVICES='${XPK_TOTAL_DEVICES}'; \
export STEPS='${STEPS}'; \
export WARMUP_STEPS='${WARMUP_STEPS}'; \
export LR_SCHEDULE_STEPS='${LR_SCHEDULE_STEPS}'; \
export LEARNING_RATE='${LEARNING_RATE}'; \
export COSINE_FINAL_FRAC='${COSINE_FINAL_FRAC}'; \
export CHECKPOINT_PERIOD='${CHECKPOINT_PERIOD}'; \
export DATA_SEED='${DATA_SEED}'; \
export MOUNT_GCSFUSE=1; \
export VENV_PATH=/__skip_venv__; \
bash /deps/src/maxtext/trainers/pre_train/scripts/olmo/run_olmo3_7b_stage1.sh"

  echo "$XPK_WORKLOAD" > "$LAST_WORKLOAD_FILE"
}

# -------------------------- monitor --------------------------
# Stream pod logs for the most recently submitted workload (or one passed as
# the second positional arg).
monitor_workload() {
  local workload="${2:-$(cat "$LAST_WORKLOAD_FILE" 2>/dev/null || true)}"
  if [ -z "$workload" ]; then
    echo "ERROR: no workload to monitor (none recorded in $LAST_WORKLOAD_FILE)." >&2
    exit 1
  fi
  echo "Monitoring $workload"

  while true; do
    local phases
    phases=$(kubectl get pods -l "jobset.sigs.k8s.io/jobset-name=${workload}" \
              -o jsonpath='{.items[*].status.phase}' 2>/dev/null || echo "")
    if echo "$phases" | grep -q Running; then break; fi
    if [ -n "$phases" ] && ! echo "$phases" | grep -qE "Pending|ContainerCreating"; then
      echo "No Running pod; phases: $phases"
      local one
      one=$(kubectl get pods -l "jobset.sigs.k8s.io/jobset-name=${workload}" \
            -o name 2>/dev/null | head -1)
      [ -n "$one" ] && kubectl logs "$one" --all-containers 2>&1 | tail -40
      exit 1
    fi
    echo "waiting for Running (phases: ${phases:-none})..."; sleep 15
  done

  local n max out_log
  n=$(kubectl get pods -l "jobset.sigs.k8s.io/jobset-name=${workload}" -o name | wc -l)
  max=$((n * 4))
  out_log="${HOME}/training-${workload}.log"
  echo "Streaming logs from $n pods (--max-log-requests=${max}) → ${out_log}"
  kubectl logs -f -l "jobset.sigs.k8s.io/jobset-name=${workload}" \
    --all-containers --max-log-requests="$max" --prefix \
    2>&1 | tee "$out_log"
}

# -------------------------- resume_until_done --------------------------
# Auto-resubmit loop. Each iteration submits with the SAME XPK_WORKLOAD and
# OUTPUT_DIR, so the trainer auto-restores from the latest checkpoint. Exits
# when checkpoint step ≥ STEPS_OVERRIDE or after MAX_RETRIES.
resume_until_done() {
  if [ -z "${STEPS_OVERRIDE:-}" ]; then
    echo "ERROR: STEPS_OVERRIDE must be set so the loop knows when to stop." >&2
    exit 1
  fi
  local target="$STEPS_OVERRIDE"
  local retry=0

  while [ "$retry" -lt "$MAX_RETRIES" ]; do
    echo "=== resume attempt $((retry + 1)) / $MAX_RETRIES (target steps: $target) ==="
    submit_workload

    # Wait for the workload to leave the running state. Kueue can evict the
    # whole JobSet on preemption, in which case `terminalState` is never set
    # and the resource is eventually deleted — without the NotFound check the
    # loop would hang forever waiting on a dead workload.
    local zero_active_count=0
    while true; do
      sleep 60
      # Probe existence first; NotFound means Kueue deleted it.
      if ! kubectl get jobset "$XPK_WORKLOAD" >/dev/null 2>&1; then
        echo "Workload $XPK_WORKLOAD deleted (likely Kueue preemption)."
        break
      fi
      local terminal
      terminal=$(kubectl get jobset "$XPK_WORKLOAD" \
        -o jsonpath='{.status.terminalState}' 2>/dev/null || echo "")
      if [ -n "$terminal" ]; then
        echo "Workload $XPK_WORKLOAD reached terminal state: $terminal"
        break
      fi
      # Suspended-by-Kueue (preempted but not yet deleted): active=0 across
      # all replicatedJobs with no terminalState. After 2 consecutive minutes
      # of zero-active, treat as preempted so we can resubmit instead of
      # waiting forever.
      local actives
      actives=$(kubectl get jobset "$XPK_WORKLOAD" \
        -o jsonpath='{.status.replicatedJobsStatus[*].active}' 2>/dev/null \
        | tr ' ' '\n' | grep -E '^[1-9]' | head -1)
      if [ -z "$actives" ]; then
        zero_active_count=$((zero_active_count + 1))
        if [ "$zero_active_count" -ge 2 ]; then
          echo "Workload $XPK_WORKLOAD has zero active pods for 2+ min; treating as preempted."
          break
        fi
      else
        zero_active_count=0
      fi
    done

    local last_step
    last_step=$(gcloud storage ls "${XPK_BASE_OUTPUT_DIR}/${XPK_RUN_NAME}/checkpoints/" 2>/dev/null \
                 | grep -oE '/[0-9]+/$' | tr -d '/' | sort -n | tail -1)
    last_step=${last_step:-0}
    echo "Latest checkpoint step on disk: ${last_step}"

    if [ "$last_step" -ge "$target" ]; then
      echo "Reached target step ${target}. Done."
      return 0
    fi

    xpk workload delete --workload="$XPK_WORKLOAD" \
      --cluster="$XPK_CLUSTER" --project="$XPK_PROJECT" --zone="$XPK_ZONE" --force \
      >/dev/null 2>&1 || true

    retry=$((retry + 1))
    echo "Resubmitting from step ${last_step} (attempt $((retry + 1))/${MAX_RETRIES}) in ${RETRY_BACKOFF_SECONDS}s..."
    sleep "$RETRY_BACKOFF_SECONDS"
  done

  echo "ERROR: max_retries=${MAX_RETRIES} reached without completing ${target} steps." >&2
  return 1
}

# -------------------------- dispatch --------------------------
case "$MODE" in
  submit|monitor|resume_until_done)
    require_env XPK_CLUSTER XPK_PROJECT XPK_ZONE XPK_DEVICE_TYPE \
                XPK_BASE_OUTPUT_DIR OLMO_INDEX_PATH OLMO_GCS_BASE \
                HF_TOKEN
    : "${LOAD_PARAMETERS_PATH:=}"  # optional; empty → MaxText random init
    case "$MODE" in
      submit)            submit_workload ;;
      monitor)           monitor_workload "$@" ;;
      resume_until_done) resume_until_done ;;
    esac
    ;;
  *)
    echo "Unknown mode: $MODE (use submit|monitor|resume_until_done)" >&2
    exit 1
    ;;
esac
