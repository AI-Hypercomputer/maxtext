#!/bin/bash
# DPO Fine-Tuning Verification Script (MaxText-Tunix Integration)
# This script runs the 100-step alignment task on Qwen2.5-1.5B.

# Load HF Token from environment or .env if present (optional for public models)
if [ -f .env ]; then
  source .env
fi

if [ -n "$HF_TOKEN" ]; then
  export HF_TOKEN=$HF_TOKEN
fi

# Ensure venv is active
if [ -z "$VIRTUAL_ENV" ]; then
  source ~/maxtext_venv/bin/activate
fi

export PYTHONPATH=src

RUN_NAME="dpo-verification-qwen-v3"

# Ensure a fresh run by cleaning up the output directory
gsutil -m rm -rf gs://igorts_europe/ttl=30d/dpo_quals/maxtext_run/${RUN_NAME} || true

python3 -m maxtext.trainers.post_train.dpo.train_dpo \
    run_name="${RUN_NAME}" \
    base_output_directory="gs://igorts_europe/ttl=30d/dpo_quals/maxtext_run" \
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
    steps=100 \
    eval_interval=20 \
    eval_steps=10 \
    learning_rate=1e-6 \
    weight_dtype=bfloat16 \
    dpo_beta=0.1 \
    scan_layers=False \
    log_config=0 \
    enable_checkpointing=True \
    async_checkpointing=False
