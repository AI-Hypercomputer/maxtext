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
# Reference launcher for MaxText distillation on a GKE TPU cluster via XPK.
# Treat this as a starting template — copy it, adapt the env vars and
# `--command` body for your cluster + run, and submit from CI / a tmux / screen.
#
# The script expects a base image at $XPK_BASE_IMAGE. Build the MaxText base
# with:
#   sudo bash src/dependencies/scripts/docker_build_dependency_image.sh \
#     MODE=stable WORKFLOW=post-training
#
# Then layer tunix on top via `prep_image` (below). We install tunix WITH deps
# (it needs google-metrax + kagglehub at runtime) and then force-reinstall
# jax/jaxlib/libtpu back to versions compatible with the base image's libtpu.
#
# Usage:
#   bash src/maxtext/trainers/post_train/distillation/scripts/run_distill_xpk.sh prep_image          # one-time image layering
#   bash src/maxtext/trainers/post_train/distillation/scripts/run_distill_xpk.sh upload_runner       # bake workspace + push to GCR
#   bash src/maxtext/trainers/post_train/distillation/scripts/run_distill_xpk.sh submit             # fire-and-forget; returns in ~60s
#   bash src/maxtext/trainers/post_train/distillation/scripts/run_distill_xpk.sh monitor            # stream logs for the last submit
#   bash src/maxtext/trainers/post_train/distillation/scripts/run_distill_xpk.sh resume_until_done  # auto-retry loop for long jobs
#
# Reading logs on-demand:
#   - GCP Logs Explorer URL that XPK prints at submit time (best for searching/filtering).
#   - `kubectl logs <pod> -c jax-tpu-1 --tail=100`  for a quick ad-hoc look.
#   - `kubectl logs <pod> --previous`               for a crashed container's last lines.
#
# Watch pod state changes (restarts, failures) in the background:
#   nohup kubectl get pods -l jobset.sigs.k8s.io/jobset-name=$XPK_WORKLOAD --watch \
#       > ~/distill-events.log 2>&1 &
#   disown
#   # Then: grep -iE 'error|crashloop|restart' ~/distill-events.log
#
# Restart policy (three layers):
#   L1 pod-level     GKE restarts a crashed pod automatically; the trainer resumes
#                    from the latest checkpoint via `maybe_restore` — no action needed.
#   L2 workload-level when the whole JobSet terminates, `resume_until_done` deletes + re-submits
#                    with the same XPK_WORKLOAD/XPK_BASE_OUTPUT_DIR (checkpoint resume, again).
#   L3 human          `resume_until_done` exits non-zero after MAX_RETRIES; page yourself from there.
#
# REQUIRED env vars (no defaults; the script exits with an error if unset):
#   XPK_CLUSTER          GKE cluster name
#   XPK_PROJECT          GCP project hosting the cluster
#   XPK_ZONE             cluster zone (e.g. us-central1-a)
#   XPK_DEVICE_TYPE      e.g. tpu7x-4x4x4, v5p-128
#   XPK_BASE_OUTPUT_DIR  GCS prefix for run outputs (each run writes to
#                        ${XPK_BASE_OUTPUT_DIR}/${XPK_WORKLOAD}/)
#
# OPTIONAL env vars (with defaults):
#   XPK_BASE_IMAGE       default: maxtext_base_image. A slash in the name
#                        (e.g. gcr.io/...) switches xpk from --base-docker-image
#                        (buildx re-push on each submit) to --docker-image
#                        (pull from registry). Prefer the registry path after
#                        the first submit.
#   XPK_WORKLOAD         default: d-${USER:0:8}-${RANDOM} (~14 chars max).
#                        Keep ≲16 chars: some clusters cap derived
#                        resource names at 49 chars
#                        (default-jobset-<workload>-<5>-...-<5>).
#   XPK_PRIORITY         default: medium
#   XPK_NUM_SLICES       default: 1
#   XPK_DISTILL_CONFIG   default: src/maxtext/configs/post_train/distillation.yml
#   XPK_RUN_NAME         default: distill_run — passed as MaxText run_name; becomes
#                        the subdir under base_output_directory where checkpoints
#                        and TB logs land (...${OUTPUT_DIR}/${XPK_RUN_NAME}/...).
#                        resume_until_done lists this subdir to find the latest step.
#   XPK_USE_GCSFUSE      default: 1 — mount XPK_DATASET_BUCKET via gcsfuse and
#                        point grain at the local mount path. ~10x faster than
#                        direct gs:// reads for ArrayRecord shards. Set to 0
#                        to bypass gcsfuse and read directly from gs://.
#   XPK_DATASET_BUCKET   default: maxtext-dataset
#   XPK_DATASET_SUBPATH  default: array-record/climbmix/*.arrayrecord
#                        The script always sets grain_train_files from these
#                        two, overriding the YAML in both modes.
#   STEPS_OVERRIDE       default: empty — yml `steps` is used unless set
#   CHECKPOINT_PERIOD_OVERRIDE  default: empty — yml `checkpoint_period` is used
#   MAX_RETRIES          default: 10 — only used by resume_until_done
#
# Feature-mapping / distillation loss hyperparameters (always passed to the
# trainer; override yml values). Defaults enable feature mapping on the first
# 8 layers. Set DISTILL_BETA=0.0 to disable feature mapping.
#   DISTILL_ALPHA          default: 0.5
#   DISTILL_TEMPERATURE    default: 1.0
#   DISTILL_BETA           default: 1.0   (>0 enables feature-map loss;
#                          requires scan_layers=True and enable_nnx=True)
#   DISTILL_LAYER_INDICES  default: [0,1,2,3,4,5,6,7]  (no spaces inside brackets)
#
# Image pinning (used by prep_image):
#   TUNIX_SOURCE  pip-installable spec for tunix.
#                 default: git+https://github.com/google/tunix@110932a8395086511228483312131841521695c1
#                 Use "google-tunix==<ver>" once a pypi release ships with the
#                 multi-host shard_input fix.
#   JAX_PIN       default: 0.10.0  — version to pin back after tunix deps resolve.
#                 Must be ≥ 0.10.0 (tunix's flax dep imports jax.extend.core.Effect).
#   JAXLIB_PIN    default: 0.10.0
#   LIBTPU_PIN    default: 0.0.39
#
# upload_runner env vars:
#   XPK_RUNNER_IMAGE_NAME  default: maxtext_base_image — GCR short name.
#   XPK_RUNNER_IMAGE_TAG   default: ${USER}-distill — per-user tag avoids
#                          clobbering shared :latest. Override (or set USER) if
#                          your shell $USER produces an awkward tag, e.g.
#                          XPK_RUNNER_IMAGE_TAG=agagik-distill. Pushes to
#                          gcr.io/$XPK_PROJECT/$XPK_RUNNER_IMAGE_NAME:$XPK_RUNNER_IMAGE_TAG.
#
# Resume on failure:
#   `resume_until_done` reuses the same XPK_BASE_OUTPUT_DIR/XPK_WORKLOAD across
#   restarts, so the trainer auto-restores from the latest checkpoint each time.

set -euo pipefail
MODE="${1:-submit}"

# -------------------------- required env --------------------------
# Collects all missing vars so the user fixes them in one pass, not N runs.
# Not called for `prep_image` — that mode only needs XPK_BASE_IMAGE + TUNIX_SOURCE
# + JAX_PIN family, all of which have defaults.
require_env() {
  local missing=()
  for v in "$@"; do
    [ -z "${!v:-}" ] && missing+=("$v")
  done
  if [ "${#missing[@]}" -gt 0 ]; then
    echo "ERROR: required env vars not set: ${missing[*]}" >&2
    echo "See this script's header for descriptions and example values." >&2
    exit 1
  fi
}

# -------------------------- defaults --------------------------
: "${XPK_BASE_IMAGE:=maxtext_base_image}"
: "${XPK_WORKLOAD:=d-${USER:0:8}-${RANDOM}}"
: "${XPK_PRIORITY:=medium}"
: "${XPK_NUM_SLICES:=1}"
: "${XPK_DISTILL_CONFIG:=src/maxtext/configs/post_train/distillation.yml}"
: "${XPK_RUN_NAME:=distill_run}"
: "${XPK_USE_GCSFUSE:=1}"
: "${XPK_DATASET_BUCKET:=maxtext-dataset}"
: "${XPK_DATASET_SUBPATH:=array-record/climbmix/*.arrayrecord}"
: "${MAX_RETRIES:=10}"

# Feature-mapping / distillation loss hyperparameters.
: "${DISTILL_ALPHA:=0.5}"
: "${DISTILL_TEMPERATURE:=1.0}"
: "${DISTILL_BETA:=1.0}"
: "${DISTILL_LAYER_INDICES:=[0,1,2,3,4,5,6,7]}"

# Image pinning (used by prep_image).
: "${TUNIX_SOURCE:=git+https://github.com/google/tunix@110932a8395086511228483312131841521695c1}"
: "${JAX_PIN:=0.10.0}"
: "${JAXLIB_PIN:=0.10.0}"
: "${LIBTPU_PIN:=0.0.39}"

# Computed at top-level so both submit_workload and resume_until_done can read it.
# `${:-}` keeps `set -u` happy for `prep_image`, which doesn't need XPK_BASE_OUTPUT_DIR.
OUTPUT_DIR="${XPK_BASE_OUTPUT_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR%/}/${XPK_WORKLOAD}"
# Default to $HOME, not /tmp: when `submit` is invoked under `sudo -E` (needed
# so xpk can reach the docker daemon), some environments refuse bash redirects
# from root to pre-existing user-owned files in /tmp. $HOME avoids that class
# of failure. Override XPK_LAST_WORKLOAD_FILE if you want a different path.
LAST_WORKLOAD_FILE="${XPK_LAST_WORKLOAD_FILE:-${HOME}/.xpk_last_workload}"

# CLI overrides for the trainer. The distillation hyperparameters always
# flow through so the values in this script are the source of truth; the
# step / checkpoint_period overrides only apply when the env var is set.
extra_cli="distill_alpha=${DISTILL_ALPHA} \
distill_temperature=${DISTILL_TEMPERATURE} \
distill_beta=${DISTILL_BETA} \
distill_layer_indices=${DISTILL_LAYER_INDICES} \
enable_nnx=True"
if [ -n "${STEPS_OVERRIDE:-}" ]; then
  extra_cli="$extra_cli learning_rate_schedule_steps=${STEPS_OVERRIDE} steps=${STEPS_OVERRIDE}"
fi
if [ -n "${CHECKPOINT_PERIOD_OVERRIDE:-}" ]; then
  extra_cli="$extra_cli checkpoint_period=${CHECKPOINT_PERIOD_OVERRIDE}"
fi

# Build grain_train_files (configs leave it empty); pick local mount or direct gs://.
gcsfuse_prelude=""
if [ "$XPK_USE_GCSFUSE" = "1" ]; then
  gcsfuse_prelude="bash src/dependencies/scripts/setup_gcsfuse.sh \
    DATASET_GCS_BUCKET=${XPK_DATASET_BUCKET} MOUNT_PATH=/tmp/gcsfuse;"
  grain_files_override="grain_train_files=/tmp/gcsfuse/${XPK_DATASET_SUBPATH}"
else
  grain_files_override="grain_train_files=gs://${XPK_DATASET_BUCKET}/${XPK_DATASET_SUBPATH}"
fi

# -------------------------- prep_image --------------------------
# Adds tunix and repins jax/libtpu on top of $XPK_BASE_IMAGE, then retags
# the result as $XPK_BASE_IMAGE (the original tag is overwritten).
# Safe to re-run: pins are force-reinstalled, but layers accumulate.
# To reset to a clean base, rebuild via docker_build_dependency_image.sh.
prep_image() {
  echo "== layering on ${XPK_BASE_IMAGE} =="
  echo "  tunix   : ${TUNIX_SOURCE}"
  echo "  jax     : ${JAX_PIN}"
  echo "  jaxlib  : ${JAXLIB_PIN}"
  echo "  libtpu  : ${LIBTPU_PIN}"
  if ! sudo docker image inspect "$XPK_BASE_IMAGE" >/dev/null 2>&1; then
    echo "ERROR: base image $XPK_BASE_IMAGE not found locally. Build it first:" >&2
    echo "  sudo bash src/dependencies/scripts/docker_build_dependency_image.sh MODE=stable WORKFLOW=post-training" >&2
    exit 1
  fi
  local tmp; tmp=$(mktemp -d)
  cat > "$tmp/Dockerfile" <<EOF
FROM $XPK_BASE_IMAGE
# 1. Install tunix WITH deps (google-metrax + kagglehub are runtime requirements).
RUN pip install --no-cache-dir --force-reinstall "$TUNIX_SOURCE"
# 2. Repin jax/libtpu so the image's libtpu and the installed jax stay compatible.
RUN pip install --no-cache-dir --force-reinstall --no-deps \\
      "jax==$JAX_PIN" "jaxlib==$JAXLIB_PIN" "libtpu==$LIBTPU_PIN"
EOF
  sudo docker build -t "$XPK_BASE_IMAGE" -f "$tmp/Dockerfile" "$tmp"
  rm -rf "$tmp"
  # Sanity check: verify the installed shard_input carries the upstream fix.
  sudo docker run --rm "$XPK_BASE_IMAGE" python -c "
import inspect, tunix
from tunix.sft import sharding_utils
src = inspect.getsource(sharding_utils.shard_input)
assert 'is_fully_addressable' in src, 'tunix install does not contain the shard_input fix'
print(f'tunix {tunix.__version__}: shard_input fix present.')
"
}

# -------------------------- upload_runner --------------------------
# Bakes ./src into the layered image and pushes to
# gcr.io/$XPK_PROJECT/$XPK_RUNNER_IMAGE_NAME:$XPK_RUNNER_IMAGE_TAG.
# Does the build/tag/push inline rather than calling docker_upload_runner.sh,
# because that script hardcodes :latest and would clobber the shared tag.
upload_runner() {
  : "${XPK_RUNNER_IMAGE_NAME:=maxtext_base_image}"
  : "${XPK_RUNNER_IMAGE_TAG:=${USER}-distill}"
  local target="gcr.io/${XPK_PROJECT}/${XPK_RUNNER_IMAGE_NAME}:${XPK_RUNNER_IMAGE_TAG}"
  echo "== upload_runner -> ${target} =="
  if ! sudo docker image inspect "$XPK_BASE_IMAGE" >/dev/null 2>&1; then
    echo "ERROR: base image $XPK_BASE_IMAGE not found locally. Run prep_image first." >&2
    exit 1
  fi
  local runner_local="${XPK_BASE_IMAGE}__runner"
  sudo docker build --no-cache \
    --build-arg "BASEIMAGE=${XPK_BASE_IMAGE}" \
    --build-arg "PACKAGE_DIR=src" \
    -f src/dependencies/dockerfiles/maxtext_runner.Dockerfile \
    -t "$runner_local" .
  sudo docker tag "$runner_local" "$target"
  sudo docker push "$target"
  echo "Pushed: $target"
}

# -------------------------- submit --------------------------
submit_workload() {
  echo "Workload:    $XPK_WORKLOAD"
  echo "Cluster:     $XPK_CLUSTER ($XPK_PROJECT, $XPK_ZONE)"
  echo "Device:      $XPK_DEVICE_TYPE x ${XPK_NUM_SLICES} slice(s)"
  echo "Base image:  $XPK_BASE_IMAGE"
  echo "Output dir:  $OUTPUT_DIR"
  echo "Config:      $XPK_DISTILL_CONFIG"
  [ -n "$extra_cli" ] && echo "Overrides:  $extra_cli"

  # Registry path (contains slash) → --docker-image (pull on cluster);
  # local tag → --base-docker-image (buildx re-push).
  local image_flag="--base-docker-image"
  if [[ "$XPK_BASE_IMAGE" == *"/"* ]]; then
    image_flag="--docker-image"
  fi
  echo "Image flag:  $image_flag=$XPK_BASE_IMAGE"

  # PYTHONPATH covers both image flows: /deps/src (upload_runner-baked) and /app/src (xpk crane overlay).
  xpk workload create \
    --cluster "$XPK_CLUSTER" \
    --workload "$XPK_WORKLOAD" \
    --priority="$XPK_PRIORITY" \
    --tpu-type="$XPK_DEVICE_TYPE" \
    --num-slices="$XPK_NUM_SLICES" \
    --project="$XPK_PROJECT" \
    --zone="$XPK_ZONE" \
    "$image_flag=$XPK_BASE_IMAGE" \
    --command "export PYTHONPATH=/deps/src:/app/src; \
export BASE_OUTPUT_DIRECTORY=${OUTPUT_DIR}; \
${gcsfuse_prelude} \
python3 -m maxtext.trainers.post_train.distillation.train_distill ${XPK_DISTILL_CONFIG} \
  run_name=${XPK_RUN_NAME} \
  base_output_directory=\$BASE_OUTPUT_DIRECTORY \
  ${grain_files_override} \
  ${extra_cli} \
  save_checkpoint_on_completion=True"

  echo "$XPK_WORKLOAD" > "$LAST_WORKLOAD_FILE"
}

# -------------------------- monitor --------------------------
# Streams kubectl logs for the most recently submitted workload (or pass
# the workload name as the second arg).
monitor_workload() {
  local workload="${2:-$(cat "$LAST_WORKLOAD_FILE" 2>/dev/null || true)}"
  if [ -z "$workload" ]; then
    echo "ERROR: no workload to monitor (none recorded in $LAST_WORKLOAD_FILE)." >&2
    exit 1
  fi
  echo "Monitoring $workload"

  # Wait for any pod to reach Running, but bail if all pods finish without ever running.
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

  local n
  n=$(kubectl get pods -l "jobset.sigs.k8s.io/jobset-name=${workload}" -o name | wc -l)
  local max=$((n * 4))   # raise above 2 × num_pods (each pod has 2 containers)
  local out_log="${HOME}/training-${workload}.log"
  echo "Streaming logs from $n pods (--max-log-requests=${max}) → ${out_log}"
  kubectl logs -f -l "jobset.sigs.k8s.io/jobset-name=${workload}" \
    --all-containers --max-log-requests="$max" --prefix \
    2>&1 | tee "$out_log"
}

# -------------------------- resume_until_done --------------------------
# Auto-resubmit loop. Each iteration submits with the SAME XPK_WORKLOAD +
# OUTPUT_DIR, so the trainer auto-restores from the latest checkpoint.
# Exits when the latest checkpoint reaches STEPS_OVERRIDE or MAX_RETRIES.
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

    while true; do
      sleep 60
      local terminal
      terminal=$(kubectl get jobset "$XPK_WORKLOAD" \
        -o jsonpath='{.status.terminalState}' 2>/dev/null || echo "")
      if [ -n "$terminal" ]; then
        echo "Workload $XPK_WORKLOAD reached terminal state: $terminal"
        break
      fi
    done

    # latest checkpoint step on disk (subdirs named after the step number)
    local last_step
    last_step=$(gcloud storage ls "${OUTPUT_DIR}/${XPK_RUN_NAME}/checkpoints/" 2>/dev/null \
                 | grep -oE '/[0-9]+/$' | tr -d '/' | sort -n | tail -1)
    last_step=${last_step:-0}
    echo "Latest checkpoint step on disk: ${last_step}"

    if [ "$last_step" -ge "$target" ]; then
      echo "Reached target step ${target}. Done."
      return 0
    fi

    # Free the workload name so we can resubmit with the same name.
    xpk workload delete --workload="$XPK_WORKLOAD" \
      --cluster="$XPK_CLUSTER" --project="$XPK_PROJECT" --zone="$XPK_ZONE" --force \
      >/dev/null 2>&1 || true

    retry=$((retry + 1))
    echo "Resubmitting from step ${last_step} (attempt $((retry + 1))/${MAX_RETRIES}) in 60s..."
    sleep 60
  done

  echo "ERROR: max_retries=${MAX_RETRIES} reached without completing ${target} steps." >&2
  return 1
}

# -------------------------- dispatch --------------------------
case "$MODE" in
  prep_image)
    prep_image
    ;;
  upload_runner)
    require_env XPK_PROJECT  # GCR target; default gcloud project is usually wrong here.
    upload_runner
    ;;
  submit|monitor|resume_until_done)
    require_env XPK_CLUSTER XPK_PROJECT XPK_ZONE XPK_DEVICE_TYPE XPK_BASE_OUTPUT_DIR
    case "$MODE" in
      submit)            submit_workload ;;
      monitor)           monitor_workload "$@" ;;
      resume_until_done) resume_until_done ;;
    esac
    ;;
  *)
    echo "Unknown mode: $MODE (use prep_image|upload_runner|submit|monitor|resume_until_done)" >&2
    exit 1
    ;;
esac
