#!/bin/bash
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

# ==============================================================================
# Threaded (non-SPMD) streaming DiLoCo on Pathways, one DiLoCo replica per slice.
#
# Builds an image with the local tree and submits it with `xpk workload create-pathways`.
# Threaded DiLoCo needs a single controller, so it runs on Pathways only. xpk's default Pathways head-pod
# resources must fit your cluster's CPU node pool.
#
# Example (2 x v6e-8, Qwen3-8B, one fragment synced per step):
#   CLUSTER="my-cluster" PROJECT="my-project" ZONE="my-zone" \
#   DEVICE_TYPE="v6e-8" NUM_SLICES="2" XPK_WORKLOAD="tdlco-01" \
#   BASE_OUTPUT_DIRECTORY="gs://my-bucket/maxtext-logs" \
#   bash src/maxtext/trainers/diloco/scripts/run_threaded_streaming_diloco.sh
#
# The defaults are a throughput smoke test (synthetic data, 100 steps, bfloat16 weights, outer LR 0.1); see
# docs/tutorials/diloco_pretraining.md for the recommended training settings.
#
# Profiling every slice: set PROFILE=true and PROFILER_MAX_NUM_HOSTS to the total number of TPU hosts across all
# slices (e.g. 4 for 2 x v6e-8 with 2 hosts per slice), and PROFILER_CHIPS_PER_HOST to the TPU chips per host
# (default 4). Each learner's steps appear as `train_learner_<i>` in the trace.
# ==============================================================================
set -euo pipefail

# Build from the repository root regardless of the caller's working directory.
cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.."

# cluster
CLUSTER="${CLUSTER:?Set CLUSTER}"
PROJECT="${PROJECT:?Set PROJECT}"
ZONE="${ZONE:?Set ZONE}"
RESERVATION="${RESERVATION:-}"

# resources
NUM_SLICES="${NUM_SLICES:-2}"
DEVICE_TYPE="${DEVICE_TYPE:-v6e-8}"

# workload
XPK_WORKLOAD="${XPK_WORKLOAD:-tdlco-$(date +%m%d-%H%M)}"  # Keep under 20 characters.
RUNNAME="${RUNNAME:-${XPK_WORKLOAD}}"
BASE_OUTPUT_DIRECTORY="${BASE_OUTPUT_DIRECTORY:?Set BASE_OUTPUT_DIRECTORY, e.g. gs://my-bucket/maxtext-logs}"
DOCKER_IMAGE_BASE="${DOCKER_IMAGE_BASE:-us-docker.pkg.dev/tpu-prod-env-multipod/maxtext-images/maxtext_jax_stable:latest}"
MY_IMAGE="${MY_IMAGE:-gcr.io/${PROJECT}/$(whoami | tr '[:upper:]' '[:lower:]')-runner:${XPK_WORKLOAD}}"

MODEL_NAME="${MODEL_NAME:-qwen3-8b}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-2048}"
STEPS="${STEPS:-100}"
# Threaded DiLoCo writes metrics every LOG_PERIOD steps and at the last step.
LOG_PERIOD="${LOG_PERIOD:-1}"
# bfloat16 is a throughput setting: the parameters, and so the fragments the outer step averages, are stored in
# bfloat16. Set WEIGHT_DTYPE=float32 to match the SPMD script (base.yml default).
WEIGHT_DTYPE="${WEIGHT_DTYPE:-bfloat16}"
# Input pipeline arguments; threaded DiLoCo supports dataset_type synthetic, grain, tfds and hf. For grain, tfds
# and hf, also pass the data location (e.g. dataset_path, grain_train_files or hf_path) and tokenizer_path here.
DATASET_ARGS="${DATASET_ARGS:-dataset_type=synthetic}"

# DiLoCo. qwen3-8b has 36 decoder layers: 36 layer fragments + 1 non-layer fragment.
DILOCO_NUM_FRAGMENTS="${DILOCO_NUM_FRAGMENTS:-37}"
DILOCO_SYNC_PERIOD="${DILOCO_SYNC_PERIOD:-${DILOCO_NUM_FRAGMENTS}}"  # One fragment synced per step.
# Steps a learner trains past a sync step before applying its synced fragment, while the transfer and outer step run.
DILOCO_NUM_COMM_OVERLAP_STEPS="${DILOCO_NUM_COMM_OVERLAP_STEPS:-5}"
DILOCO_COMM_OVERLAP_ALPHA="${DILOCO_COMM_OVERLAP_ALPHA:-0.0}"
DILOCO_OUTER_LR="${DILOCO_OUTER_LR:-0.1}"  # The tutorial recommends 0.3-0.9 (e.g. 0.7).
DILOCO_OUTER_MOMENTUM="${DILOCO_OUTER_MOMENTUM:-0.9}"
# Spreads the embedding / output-head rows over the layer fragments instead of syncing them in one step.
# Requires DILOCO_NUM_FRAGMENTS >= 3 and does not support shard_optimizer_over_data.
DILOCO_BUCKETIZE_NON_SCANNED="${DILOCO_BUCKETIZE_NON_SCANNED:-true}"

# profiling
PROFILE="${PROFILE:-false}"
PROFILER_SKIP_STEPS="${PROFILER_SKIP_STEPS:-40}"
PROFILER_MAX_NUM_HOSTS="${PROFILER_MAX_NUM_HOSTS:-1}"
PROFILER_CHIPS_PER_HOST="${PROFILER_CHIPS_PER_HOST:-4}"

PROFILER_ARGS=""
case "${PROFILE,,}" in
  1|true|yes)
    if (( PROFILER_SKIP_STEPS >= STEPS )); then
      echo "PROFILE needs STEPS (${STEPS}) > PROFILER_SKIP_STEPS (${PROFILER_SKIP_STEPS})." >&2
      exit 1
    fi
    PROFILER_ARGS="profiler=xplane skip_first_n_steps_for_profiler=${PROFILER_SKIP_STEPS} \
      profiler_steps=${PROFILER_STEPS:-5} profiler_max_num_hosts=${PROFILER_MAX_NUM_HOSTS} \
      upload_all_profiler_results=true enable_tpu_profiling_options=true \
      tpu_num_chips_to_profile_per_task=${PROFILER_CHIPS_PER_HOST}"
    ;;
  0|false|no|"") ;;
  *)
    echo "PROFILE must be one of 1/true/yes or 0/false/no, got '${PROFILE}'." >&2
    exit 1
    ;;
esac

CMD="export PYTHONPATH=/app/src:\$PYTHONPATH && cd /app/src/ && python3 maxtext/trainers/pre_train/train.py \
  maxtext/configs/base.yml \
  run_name=${RUNNAME} \
  base_output_directory=${BASE_OUTPUT_DIRECTORY} \
  ${DATASET_ARGS} \
  model_name=${MODEL_NAME} \
  weight_dtype=${WEIGHT_DTYPE} \
  dtype=bfloat16 \
  per_device_batch_size=${PER_DEVICE_BATCH_SIZE} \
  max_target_length=${MAX_TARGET_LENGTH} \
  steps=${STEPS} \
  log_period=${LOG_PERIOD} \
  enable_checkpointing=false \
  enable_diloco=true \
  enable_streaming_diloco=true \
  enable_threaded_diloco=true \
  dcn_diloco_parallelism=${NUM_SLICES} \
  num_diloco_fragments=${DILOCO_NUM_FRAGMENTS} \
  diloco_sync_period=${DILOCO_SYNC_PERIOD} \
  num_communication_overlapping_steps=${DILOCO_NUM_COMM_OVERLAP_STEPS} \
  communication_overlapping_alpha=${DILOCO_COMM_OVERLAP_ALPHA} \
  diloco_outer_lr=${DILOCO_OUTER_LR} \
  diloco_outer_momentum=${DILOCO_OUTER_MOMENTUM} \
  diloco_bucketize_non_scanned=${DILOCO_BUCKETIZE_NON_SCANNED} \
  ${PROFILER_ARGS}"

echo "Building ${MY_IMAGE} from $(pwd)..."
docker build -t "${MY_IMAGE}" -f - . <<EOF
FROM ${DOCKER_IMAGE_BASE}
WORKDIR /app
COPY . .
RUN find /app -name "*.pyc" -delete && find /app -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
EOF
docker push "${MY_IMAGE}"

XPK_ARGS=(
  --workload "${XPK_WORKLOAD}"
  --docker-image "${MY_IMAGE}"
  --command "${CMD}"
  --num-slices "${NUM_SLICES}"
  --tpu-type "${DEVICE_TYPE}"
  --priority "${PRIORITY:-medium}"
  --cluster "${CLUSTER}"
  --project "${PROJECT}"
  --zone "${ZONE}"
)
if [ -n "${RESERVATION}" ]; then
  XPK_ARGS+=(--reservation "${RESERVATION}")
fi

echo "Creating Pathways workload ${XPK_WORKLOAD}..."
xpk workload create-pathways "${XPK_ARGS[@]}"
