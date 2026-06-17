#!/bin/bash
# Script to launch the MaxText RL weight loading latency reproduction workload using XPK.

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/t/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -eu

# Usage check
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 [apply_patch: 0|1] [disable_persistent_cache: 0|1] [optional_base_docker_image]"
  echo "Example: $0 0 0"
  exit 1
fi

APPLY_PATCH=$1
DISABLE_PERSISTENT_CACHE=${2:-0}
BASE_DOCKER_IMAGE=${3:-"gcr.io/tpu-prod-env-multipod/maxtext_post_training_nightly:latest"}

USER_PREFIX=${USER:-"repro"}
TIMESTAMP=${OVERRIDE_TIMESTAMP:-$(date +%H%M%S)}
WORKLOAD_NAME="${USER_PREFIX}-rl-${TIMESTAMP}"
GCS_SCRATCH="gs://cloud-pathways-staging/tmp/${USER_PREFIX}-rl-cold-${TIMESTAMP}"

echo "=== Launching XPK Repro Workload ==="
echo "Workload Name: ${WORKLOAD_NAME}"
echo "JAX Compilation Cache GCS Location: ${GCS_SCRATCH}"
echo "Apply Patch: ${APPLY_PATCH}"
echo "Disable Persistent Cache: ${DISABLE_PERSISTENT_CACHE}"
echo "Docker Image: ${BASE_DOCKER_IMAGE}"
echo "===================================="

# Ensure xpk is available
if [ ! -f "maxtext_venv/bin/xpk" ]; then
  echo "Error: XPK CLI not found at maxtext_venv/bin/xpk. Please set up the virtual environment."
  exit 1
fi

maxtext_venv/bin/xpk workload create-pathways \
  --workload="${WORKLOAD_NAME}" \
  --tpu-type=v5p-64 \
  --num-slices=1 \
  --project=cloud-tpu-multipod-dev \
  --zone=europe-west4 \
  --cluster=mlperf-v5p \
  --base-docker-image="${BASE_DOCKER_IMAGE}" \
  --custom-pathways-server-args="--gcs_scratch_location=${GCS_SCRATCH}" \
  --custom-pathways-proxy-server-args="--gcs_scratch_location=${GCS_SCRATCH}" \
  --custom-pathways-worker-args="--gcs_scratch_location=${GCS_SCRATCH}" \
  --command="APPLY_PATCH=${APPLY_PATCH} DISABLE_PERSISTENT_CACHE=${DISABLE_PERSISTENT_CACHE} bash run_gke_35b.sh \
    src/maxtext/configs/post_train/rl.yml \
    model_name=qwen3.5-35b-a3b \
    tokenizer_path=Qwen/Qwen3.5-35B-A3B \
    run_name=rl-qwen3.5-35b-repro-32chips \
    base_output_directory=gs://igorts_europe/ttl=30d/rl-qwen3.5-35b-repro-32chips \
    batch_size=32 \
    num_batches=2 \
    num_test_batches=0 \
    chips_per_vm=4 \
    scan_layers=True \
    hbm_utilization_vllm=0.75 \
    rollout_data_parallelism=4 \
    rollout_tensor_parallelism=4 \
    rl.num_generations=8 \
    train_micro_batch_size=8 \
    rollout_micro_batch_size=8 \
    dataset_name=openai/gsm8k \
    max_target_length=1024 \
    max_prefill_predict_length=256 \
    enable_checkpointing=false \
    convert_checkpoint_if_possible=false \
    vllm_load_format=dummy \
    vllm_additional_config='{\"block_size\":256}'"
