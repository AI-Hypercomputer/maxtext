#!/bin/bash

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Run MaxText elastic training with Qwen3 0.6B.
#
# Called inside the training container by the JobSet manifest.
# All flags are passed via environment variables so the manifest
# stays clean and the command is easy to modify.
#
# Required env vars:
#   BUCKET_NAME   - GCS bucket for output and scratch
#   RUN_NAME      - training run name (used in checkpoint path)
#   DATASET       - "synthetic" or "glaive"
#
# Optional env vars (debugging):
#   DEBUGGING     - "true" to enable all debugging features (default: "false")

set -euo pipefail

# Dataset-specific flags
if [ "${DATASET}" = "glaive" ]; then
  DATASET_FLAGS=(
    dataset_type=grain
    grain_file_type=arrayrecord
    "grain_train_files=gs://${BUCKET_NAME}/data/glaive-fc-v2/train.array_record*"
    grain_worker_count=2
    tokenize_train_data=true
    tokenizer_path=/tokenizer
    num_epoch=50
  )
else
  DATASET_FLAGS=(
    dataset_type=synthetic
    reuse_example_batch=1
  )
fi

# Debugging flags: DEBUGGING=true adds XProf profiling + goodput monitoring.
DEBUG_FLAGS=()
if [ "${DEBUGGING:-false}" = "true" ]; then
  DEBUG_FLAGS=(
    profiler=xplane
    profiler_steps=5
    skip_first_n_steps_for_profiler=1
    enable_goodput_recording=True
    monitor_goodput=True
    goodput_upload_interval_seconds=30
    enable_checkpoint_cloud_logger=True
  )
  echo "Debugging enabled: xplane profiler + goodput monitoring"
fi

# Training. checkpoint_period=100 keeps a recent checkpoint for fast recovery;
# qwen3-0.6b keeps the checkpoint (~7 GiB) under the proxy memory limit. See the
# README appendix for the rationale and how to run a larger model.
python3 -m maxtext.trainers.pre_train.train \
  src/maxtext/configs/base.yml \
  base_output_directory=gs://${BUCKET_NAME}/output \
  run_name=${RUN_NAME} \
  model_name=qwen3-0.6b \
  per_device_batch_size=1 \
  enable_checkpointing=true \
  enable_single_controller=True \
  remat_policy=full \
  steps=5000 \
  checkpoint_period=100 \
  max_target_length=2048 \
  attention=flash \
  gcs_metrics=True \
  elastic_enabled=true \
  elastic_timeout_seconds=300 \
  elastic_max_retries=10 \
  "${DATASET_FLAGS[@]}" \
  "${DEBUG_FLAGS[@]}"
