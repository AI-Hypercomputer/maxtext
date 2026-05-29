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

# Phase 2: SFT Baseline Metrics Evaluation (Pre-DPO)
# This script runs a single-step DPO training run to establish baseline alignment metrics.

set -e

if [ -f .env ]; then
  source .env
fi

if [ -z "$HF_TOKEN" ]; then
  echo "Error: HF_TOKEN is not set. Please export HF_TOKEN before running."
  exit 1
fi

export HF_TOKEN=$HF_TOKEN

RUN_NAME="dpo-baseline-qwen-v11-$(date +%s)"

# Ensure a fresh run by cleaning up any prior output directories under this run name
gsutil -m rm -rf gs://igorts_europe/ttl=30d/dpo_quals/maxtext_baseline/${RUN_NAME} || true

# Execute DPO training for exactly 1 step to evaluate SFT baseline alignment metrics
python3 -m maxtext.trainers.post_train.dpo.train_dpo \
    run_name="${RUN_NAME}" \
    base_output_directory="gs://igorts_europe/ttl=30d/dpo_quals/maxtext_baseline" \
    model_name="qwen2.5-1.5b" \
    tokenizer_path="Qwen/Qwen2.5-1.5B-Instruct" \
    load_parameters_path="gs://igorts_europe/converted_checkpoints/qwen2.5-1.5b-instruct_mcjax_v2/0/items" \
    dataset_type=hf \
    hf_path="argilla/distilabel-intel-orca-dpo-pairs" \
    train_split="train" \
    hf_eval_split="train" \
    train_data_columns="['input', 'chosen', 'rejected']" \
    eval_data_columns="['input', 'chosen', 'rejected']" \
    per_device_batch_size=2 \
    max_target_length=1024 \
    steps=1 \
    eval_interval=1 \
    eval_steps=5 \
    learning_rate=1e-6 \
    weight_dtype=bfloat16 \
    scan_layers=True \
    log_config=0 \
    async_checkpointing=False \
    enable_checkpointing=True
