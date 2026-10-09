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
# Reference launcher for MaxText distillation on a GKE TPU cluster via
# Cluster Toolkit (gcluster).
# Treat this as a starting template — copy it, adapt the env vars and
# `--command` body for your cluster + run, and submit from CI / a tmux / screen.
#
# The script expects a base image at $DOCKER_IMAGE (or $CTK_BASE_IMAGE). `prep_image`
# (below) builds the MaxText TPU post-training Docker image as described in
# https://maxtext.readthedocs.io/page/tutorials/build_maxtext.html#tpu-post-training-docker-image
# (equivalent to running `build_maxtext_docker_image WORKFLOW=post-training` from
# an activated MaxText virtual environment). Post-training dependencies such as
# Tunix come from the pins in src/dependencies/extra_deps/post_train_github_deps.txt.
# Then bake your local `./src` into a runner image pushed to GCR via `upload_runner`.
#
# Usage:
#   bash src/maxtext/trainers/post_train/distillation/scripts/run_distill_ctk.sh prep_image          # one-time image build
#   bash src/maxtext/trainers/post_train/distillation/scripts/run_distill_ctk.sh upload_runner       # bake workspace + push to GCR
#   bash src/maxtext/trainers/post_train/distillation/scripts/run_distill_ctk.sh submit             # fire-and-forget job submission
#   bash src/maxtext/trainers/post_train/distillation/scripts/run_distill_ctk.sh monitor            # stream logs for the last submit
#   bash src/maxtext/trainers/post_train/distillation/scripts/run_distill_ctk.sh cleanup            # cancel/delete the last workload
#   bash src/maxtext/trainers/post_train/distillation/scripts/run_distill_ctk.sh resume_until_done  # auto-retry loop for long jobs
#
# Reading logs on-demand:
#   - GCP Logs Explorer URL that `gcluster job submit` prints at submit time.
#   - `gcluster job logs <workload> --main-only=false`
#   - `kubectl logs <pod> -c jax-tpu --tail=100`    for a quick ad-hoc look.
#   - `kubectl logs <pod> --previous`               for a crashed container's last lines.
#
# Watch pod state changes (restarts, failures) in the background:
#   nohup kubectl get pods -l jobset.sigs.k8s.io/jobset-name=$CTK_WORKLOAD --watch \
#       > ~/distill-events.log 2>&1 &
#   disown
#   # Then: grep -iE 'error|crashloop|restart' ~/distill-events.log
#
# Restart policy (three layers):
#   L1 pod-level     GKE restarts a crashed pod automatically; the trainer resumes
#                    from the latest checkpoint via `maybe_restore` — no action needed.
#   L2 workload-level when the whole JobSet terminates, `resume_until_done` cancels + re-submits
#                    with the same CTK_WORKLOAD/BASE_OUTPUT_DIRECTORY (checkpoint resume, again).
#   L3 human          `resume_until_done` exits non-zero after MAX_RETRIES; page yourself from there.
#
# REQUIRED env vars (no defaults; the script exits with an error if unset):
#   GKE_CLUSTER (or CTK_CLUSTER)                GKE cluster name
#   PROJECT_ID (or CTK_PROJECT)                 GCP project hosting the cluster
#   LOCATION (or CTK_ZONE / ZONE)               cluster region or zone (e.g. europe-west4, us-central1-a)
#   COMPUTE_TYPE (or CTK_COMPUTE_TYPE)          e.g. tpu7x-standard-4t, ct5p-hightpu-4t, ct6e-hightpu-4t
#   TOPOLOGY (or CTK_TOPOLOGY)                  e.g. 4x4x4, 16x16
#   BASE_OUTPUT_DIRECTORY (or CTK_BASE_OUTPUT_DIR)
#                                               GCS prefix for run outputs (each run writes to
#                                               ${BASE_OUTPUT_DIRECTORY}/${CTK_WORKLOAD}/)
#
# OPTIONAL env vars (with defaults):
#   DOCKER_IMAGE (or CTK_BASE_IMAGE)
#                        default: maxtext_base_image. A slash in the name
#                        (e.g. gcr.io/...) uses `--image` (pull from registry);
#                        otherwise uses `--base-image` with `--build-context=.`.
#                        Prefer the registry path after `upload_runner`.
#   CTK_WORKLOAD         default: d-${USER:0:8}-${RANDOM} (~14 chars max).
#                        Must be <= 28 chars (or <= 22 chars with Pathways) and a valid DNS label.
#   CTK_PRIORITY         default: medium
#   NUM_SLICES (or CTK_NUM_SLICES)
#                        default: 1
#   CTK_DISTILL_CONFIG   default: src/maxtext/configs/post_train/distillation.yml
#   RUN_NAME (or CTK_RUN_NAME)
#                        default: distill_run — passed as MaxText run_name; becomes
#                        the subdir under base_output_directory where checkpoints
#                        and TB logs land (...${OUTPUT_DIR}/${RUN_NAME}/...).
#                        resume_until_done lists this subdir to find the latest step.
#   CTK_USE_GCSFUSE      default: 1 — mount CTK_DATASET_BUCKET via GKE GCS FUSE CSI
#                        (`--mount`) and point grain at the local mount path.
#                        Set to 0 to bypass gcsfuse and read directly from gs://.
#   CTK_DATASET_BUCKET   default: maxtext-dataset
#   CTK_DATASET_SUBPATH  default: array-record/climbmix/*.arrayrecord
#                        The script always sets grain_train_files from these
#                        two, overriding the YAML in both modes.
#   CTK_HF_CACHE_DIR     default: /dev/shm/hf — HF `datasets` Arrow cache dir.
#   STEPS_OVERRIDE       default: empty — yml `steps` is used unless set
#   CHECKPOINT_PERIOD_OVERRIDE  default: empty — yml `checkpoint_period` is used
#   MAX_RETRIES          default: 10 — only used by resume_until_done
#   STUDENT_CKPT_PATH    default: empty — overrides student_overrides.load_parameters_path if set
#   TEACHER_CKPT_PATH    default: empty — overrides teacher_overrides.load_parameters_path if set
#   TOKENIZER_PATH       default: empty — overrides tokenizer_path if set
#   HF_TOKEN             default: empty — overrides hf_access_token if set
#
# Feature-mapping / distillation loss hyperparameters (always passed to the
# trainer; override yml values). Defaults enable feature mapping on the first
# 8 layers. Set DISTILL_BETA=0.0 to disable feature mapping.
#   DISTILL_ALPHA          default: 0.5
#   DISTILL_TEMPERATURE    default: 1.0
#   DISTILL_BETA           default: 1.0   (>0 enables feature-map loss;
#                          requires scan_layers=True)
#   DISTILL_LAYER_INDICES  default: [0,1,2,3,4,5,6,7]  (no spaces inside brackets)
#
# upload_runner env vars:
#   CTK_RUNNER_IMAGE_NAME  default: maxtext_base_image — GCR short name.
#   CTK_RUNNER_IMAGE_TAG   default: ${USER}-distill — per-user tag avoids
#                          clobbering shared :latest. Pushes to
#                          gcr.io/$PROJECT_ID/$CTK_RUNNER_IMAGE_NAME:$CTK_RUNNER_IMAGE_TAG.

set -euo pipefail
MODE="${1:-submit}"

# -------------------------- required env --------------------------
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

# -------------------------- normalize aliases --------------------------
export GKE_CLUSTER="${GKE_CLUSTER:-${CTK_CLUSTER:-}}"
export PROJECT_ID="${PROJECT_ID:-${CTK_PROJECT:-}}"
export LOCATION="${LOCATION:-${CTK_LOCATION:-${CTK_ZONE:-${ZONE:-}}}}"
export COMPUTE_TYPE="${COMPUTE_TYPE:-${CTK_COMPUTE_TYPE:-}}"
export TOPOLOGY="${TOPOLOGY:-${CTK_TOPOLOGY:-}}"
export BASE_OUTPUT_DIRECTORY="${BASE_OUTPUT_DIRECTORY:-${CTK_BASE_OUTPUT_DIR:-}}"

# -------------------------- defaults --------------------------
: "${CTK_BASE_IMAGE:=${DOCKER_IMAGE:-maxtext_base_image}}"
: "${CTK_WORKLOAD:=d-${USER:0:8}-${RANDOM}}"
: "${CTK_PRIORITY:=medium}"
: "${CTK_NUM_SLICES:=${NUM_SLICES:-1}}"
: "${CTK_DISTILL_CONFIG:=src/maxtext/configs/post_train/distillation.yml}"
: "${CTK_RUN_NAME:=${RUN_NAME:-distill_run}}"
: "${CTK_USE_GCSFUSE:=1}"
: "${CTK_DATASET_BUCKET:=maxtext-dataset}"
: "${CTK_DATASET_SUBPATH:=array-record/climbmix/*.arrayrecord}"
: "${CTK_HF_CACHE_DIR:=/dev/shm/hf}"
: "${MAX_RETRIES:=10}"

# Feature-mapping / distillation loss hyperparameters.
: "${DISTILL_ALPHA:=0.5}"
: "${DISTILL_TEMPERATURE:=1.0}"
: "${DISTILL_BETA:=1.0}"
: "${DISTILL_LAYER_INDICES:=[0,1,2,3,4,5,6,7]}"

OUTPUT_DIR="${BASE_OUTPUT_DIRECTORY:-}"
OUTPUT_DIR="${OUTPUT_DIR%/}/${CTK_WORKLOAD}"
LAST_WORKLOAD_FILE="${CTK_LAST_WORKLOAD_FILE:-${HOME}/.ctk_last_workload}"

# CLI overrides for the trainer.
extra_cli="distill_alpha=${DISTILL_ALPHA} \
distill_temperature=${DISTILL_TEMPERATURE} \
distill_beta=${DISTILL_BETA} \
distill_layer_indices=${DISTILL_LAYER_INDICES}"
if [ -n "${STEPS_OVERRIDE:-}" ]; then
  extra_cli="$extra_cli learning_rate_schedule_steps=${STEPS_OVERRIDE} steps=${STEPS_OVERRIDE}"
fi
if [ -n "${CHECKPOINT_PERIOD_OVERRIDE:-}" ]; then
  extra_cli="$extra_cli checkpoint_period=${CHECKPOINT_PERIOD_OVERRIDE}"
fi
if [ -n "${STUDENT_CKPT_PATH:-}" ]; then
  extra_cli="$extra_cli student_overrides.load_parameters_path=${STUDENT_CKPT_PATH}"
fi
if [ -n "${TEACHER_CKPT_PATH:-}" ]; then
  extra_cli="$extra_cli teacher_overrides.load_parameters_path=${TEACHER_CKPT_PATH}"
fi
if [ -n "${TOKENIZER_PATH:-}" ]; then
  extra_cli="$extra_cli tokenizer_path=${TOKENIZER_PATH} tokenizer_type=huggingface"
fi
if [ -n "${HF_TOKEN:-}" ]; then
  extra_cli="$extra_cli hf_access_token=${HF_TOKEN}"
fi

# Build grain_train_files (configs leave it empty); pick GCS FUSE mount or direct gs://.
mount_args=()
if [ "$CTK_USE_GCSFUSE" = "1" ]; then
  mount_args=(--mount="gs://${CTK_DATASET_BUCKET};/tmp/gcsfuse;ro")
  grain_files_override="grain_train_files=/tmp/gcsfuse/${CTK_DATASET_SUBPATH}"
else
  grain_files_override="grain_train_files=gs://${CTK_DATASET_BUCKET}/${CTK_DATASET_SUBPATH}"
fi

# Optional: stage the YAML from GCS instead of baking via upload_runner.
yaml_prelude=""
if [ -n "${CTK_YAML_GCS:-}" ]; then
  yaml_prelude="gcloud storage cp \"${CTK_YAML_GCS}\" \"${CTK_DISTILL_CONFIG}\";"
fi

# Optional: stage HF tokenizer files from GCS for models whose tokenizer isn't
# baked into the image (e.g. gpt-oss).
tokenizer_prelude=""
if [ -n "${CTK_TOKENIZER_GCS:-}" ] && [ -n "${CTK_TOKENIZER_LOCAL:-}" ]; then
  tokenizer_prelude="mkdir -p \"${CTK_TOKENIZER_LOCAL}\" && gcloud storage rsync \"${CTK_TOKENIZER_GCS}\" \"${CTK_TOKENIZER_LOCAL}\";"
fi

# Default v7x XLA flags. The default vmem limit (32 MB) is too small for
# tokamax splash backward; we need ≥60 MB.
default_libtpu_args="--xla_tpu_scoped_vmem_limit_kib=61440 \
--xla_tpu_enable_all_experimental_scheduler_features=true \
--xla_tpu_enable_scheduler_memory_pressure_tracking=true \
--xla_tpu_host_transfer_overlap_limit=24 \
--xla_tpu_aggressive_opt_barrier_removal=ENABLED \
--xla_lhs_prioritize_async_depth_over_stall=ENABLED \
--xla_tpu_enable_ag_backward_pipelining=true \
--xla_should_allow_loop_variant_parameter_in_chain=ENABLED \
--xla_should_add_loop_invariant_op_in_chain=ENABLED \
--xla_max_concurrent_host_send_recv=100 \
--xla_tpu_scheduler_percent_shared_memory_limit=100 \
--xla_latency_hiding_scheduler_rerun=2"
libtpu_init_args=$(printf '%s' "${CTK_LIBTPU_INIT_ARGS:-$default_libtpu_args}" | tr -s '[:space:]' ' ')

# -------------------------- configure_cluster --------------------------
configure_cluster() {
  if ! command -v gcluster >/dev/null 2>&1; then
    echo "ERROR: gcluster CLI not found in PATH. Install Cluster Toolkit first." >&2
    exit 1
  fi
  gcloud config set project "$PROJECT_ID" >/dev/null
  gcloud container clusters get-credentials "$GKE_CLUSTER" \
    --location "$LOCATION" \
    --project "$PROJECT_ID" >/dev/null
  gcluster job config set project "$PROJECT_ID"
  gcluster job config set cluster "$GKE_CLUSTER"
  gcluster job config set location "$LOCATION"
}

# -------------------------- prep_image --------------------------
# Builds the MaxText TPU post-training Docker image following
# https://maxtext.readthedocs.io/page/tutorials/build_maxtext.html#tpu-post-training-docker-image.
# `build_maxtext_docker_image` always produces a local image named
# `maxtext_base_image`; it is retagged as $CTK_BASE_IMAGE if that differs.
# Must be run from the MaxText repo root with the MaxText virtual environment
# activated (it provides the `build_maxtext_docker_image` console script).
prep_image() {
  if ! command -v build_maxtext_docker_image >/dev/null 2>&1; then
    echo "ERROR: build_maxtext_docker_image not found in PATH. Activate the MaxText virtual environment first;" >&2
    echo "  see https://maxtext.readthedocs.io/page/tutorials/build_maxtext.html" >&2
    exit 1
  fi
  echo "== building TPU post-training image -> ${CTK_BASE_IMAGE} =="
  # Run under sudo like the other docker calls in this script; keep PATH so the
  # venv console script (and its python) is still found.
  sudo env "PATH=$PATH" build_maxtext_docker_image WORKFLOW=post-training
  if [ "$CTK_BASE_IMAGE" != "maxtext_base_image" ]; then
    sudo docker tag maxtext_base_image "$CTK_BASE_IMAGE"
  fi
  # Sanity check: verify the installed shard_input carries the upstream fix.
  sudo docker run --rm "$CTK_BASE_IMAGE" python -c "
import inspect, tunix
from tunix.sft import sharding_utils
src = inspect.getsource(sharding_utils.shard_input)
assert 'is_fully_addressable' in src, 'tunix install does not contain the shard_input fix'
print(f'tunix {tunix.__version__}: shard_input fix present.')
"
}

# -------------------------- upload_runner --------------------------
upload_runner() {
  : "${CTK_RUNNER_IMAGE_NAME:=maxtext_base_image}"
  : "${CTK_RUNNER_IMAGE_TAG:=${USER}-distill}"
  local target="gcr.io/${PROJECT_ID}/${CTK_RUNNER_IMAGE_NAME}:${CTK_RUNNER_IMAGE_TAG}"
  echo "== upload_runner -> ${target} =="
  if ! sudo docker image inspect "$CTK_BASE_IMAGE" >/dev/null 2>&1; then
    echo "ERROR: base image $CTK_BASE_IMAGE not found locally. Run prep_image first." >&2
    exit 1
  fi
  local runner_local="${CTK_BASE_IMAGE}__runner"
  sudo docker build --no-cache \
    --build-arg "BASEIMAGE=${CTK_BASE_IMAGE}" \
    --build-arg "PACKAGE_DIR=src" \
    -f src/dependencies/dockerfiles/maxtext_runner.Dockerfile \
    -t "$runner_local" .
  sudo docker tag "$runner_local" "$target"
  sudo docker push "$target"
  echo "Pushed: $target"
}

# -------------------------- submit --------------------------
submit_workload() {
  configure_cluster

  echo "Workload:      $CTK_WORKLOAD"
  echo "Cluster:       $GKE_CLUSTER ($PROJECT_ID, $LOCATION)"
  echo "Compute/Topo:  $COMPUTE_TYPE / $TOPOLOGY x ${CTK_NUM_SLICES} slice(s)"
  echo "Image:         $CTK_BASE_IMAGE"
  echo "Output dir:    $OUTPUT_DIR"
  echo "Config:        $CTK_DISTILL_CONFIG"
  [ -n "$extra_cli" ] && echo "Overrides:     $extra_cli"

  local image_args=()
  if [[ "$CTK_BASE_IMAGE" == *"/"* ]]; then
    image_args=(--image="$CTK_BASE_IMAGE")
  else
    image_args=(--base-image="$CTK_BASE_IMAGE" --build-context=.)
  fi
  echo "Image args:    ${image_args[*]}"

  gcluster job submit \
    "${image_args[@]}" \
    --name="$CTK_WORKLOAD" \
    --priority="$CTK_PRIORITY" \
    --compute-type="$COMPUTE_TYPE" \
    --topology="$TOPOLOGY" \
    --num-slices="$CTK_NUM_SLICES" \
    "${mount_args[@]}" \
    --command="export PYTHONPATH=/deps/src:/app/src; \
export BASE_OUTPUT_DIRECTORY=${OUTPUT_DIR}; \
export LIBTPU_INIT_ARGS='${libtpu_init_args}'; \
export TMPDIR=/dev/shm; export JAX_COMPILATION_CACHE_DIR=/dev/shm/jax_cache; \
export HF_HOME=${CTK_HF_CACHE_DIR}; export HF_DATASETS_CACHE=${CTK_HF_CACHE_DIR}/datasets; mkdir -p ${CTK_HF_CACHE_DIR}/datasets; \
${yaml_prelude} \
${tokenizer_prelude} \
python3 -m maxtext.trainers.post_train.distillation.train_distill ${CTK_DISTILL_CONFIG} \
  run_name=${CTK_RUN_NAME} \
  base_output_directory=\$BASE_OUTPUT_DIRECTORY \
  ${grain_files_override} \
  ${extra_cli} \
  save_checkpoint_on_completion=True"

  echo "$CTK_WORKLOAD" > "$LAST_WORKLOAD_FILE"
}

# -------------------------- monitor --------------------------
monitor_workload() {
  local workload="${2:-$(cat "$LAST_WORKLOAD_FILE" 2>/dev/null || true)}"
  if [ -z "$workload" ]; then
    echo "ERROR: no workload to monitor (none recorded in $LAST_WORKLOAD_FILE)." >&2
    exit 1
  fi
  echo "Monitoring $workload"
  gcluster job list || true
  kubectl get jobset -l "gcluster.google.com/workload=${workload}" || true

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
  local max=$((n * 4))   # raise above 2 x num_pods
  local out_log="${HOME}/training-${workload}.log"
  echo "Streaming logs from $n pods (--max-log-requests=${max}) -> ${out_log}"
  kubectl logs -f -l "jobset.sigs.k8s.io/jobset-name=${workload}" \
    --all-containers --max-log-requests="$max" --prefix \
    2>&1 | tee "$out_log"
}

# -------------------------- cleanup --------------------------
cleanup_workload() {
  local workload="${2:-$(cat "$LAST_WORKLOAD_FILE" 2>/dev/null || true)}"
  if [ -z "$workload" ]; then
    echo "ERROR: no workload to clean up (none recorded in $LAST_WORKLOAD_FILE)." >&2
    exit 1
  fi
  echo "Canceling workload $workload"
  gcluster job cancel "$workload" || kubectl delete jobset "$workload" --ignore-not-found
}

# -------------------------- resume_until_done --------------------------
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
      terminal=$(kubectl get jobset "$CTK_WORKLOAD" \
        -o jsonpath='{.status.terminalState}' 2>/dev/null || echo "")
      if [ -n "$terminal" ]; then
        echo "Workload $CTK_WORKLOAD reached terminal state: $terminal"
        break
      fi
    done

    # latest checkpoint step on disk (subdirs named after the step number)
    local last_step
    last_step=$(gcloud storage ls "${OUTPUT_DIR}/${CTK_RUN_NAME}/checkpoints/" 2>/dev/null \
                 | grep -oE '/[0-9]+/$' | tr -d '/' | sort -n | tail -1)
    last_step=${last_step:-0}
    echo "Latest checkpoint step on disk: ${last_step}"

    if [ "$last_step" -ge "$target" ]; then
      echo "Reached target step ${target}. Done."
      return 0
    fi

    # Free the workload name so we can resubmit with the same name.
    gcluster job cancel "$CTK_WORKLOAD" >/dev/null 2>&1 \
      || kubectl delete jobset "$CTK_WORKLOAD" --ignore-not-found >/dev/null 2>&1 \
      || true

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
    require_env PROJECT_ID  # GCR target; default gcloud project is usually wrong here.
    upload_runner
    ;;
  submit|resume_until_done)
    require_env GKE_CLUSTER PROJECT_ID LOCATION COMPUTE_TYPE TOPOLOGY BASE_OUTPUT_DIRECTORY
    case "$MODE" in
      submit)            submit_workload ;;
      resume_until_done) resume_until_done ;;
    esac
    ;;
  monitor)
    monitor_workload "$@"
    ;;
  cleanup|delete)
    cleanup_workload "$@"
    ;;
  *)
    echo "Unknown mode: $MODE (use prep_image|upload_runner|submit|monitor|cleanup|resume_until_done)" >&2
    exit 1
    ;;
esac
