#!/bin/bash
# Phase 4: Legacy MaxText DPO Comparison
# This script runs the original DPO implementation from the main branch.

# Navigate to maxtext2 worktree
cd /home/igorts_google_com/git/maxtext2

# Setup environment
export HF_TOKEN=$(grep HF_TOKEN /home/igorts_google_com/git/maxtext/.env | cut -d'=' -f2)

# Run legacy DPO using pre_train.train entry point
# We now use the 3-column Argilla dataset to match Phase 2
/home/igorts_google_com/git/maxtext2/maxtext_venv/bin/python3 -m maxtext.trainers.pre_train.train \
    run_name="dpo-legacy-comparison-v2" \
    base_output_directory="gs://igorts_europe/ttl=30d/dpo_quals/legacy_run" \
    model_name="qwen2.5-1.5b" \
    tokenizer_path="Qwen/Qwen2.5-1.5B-Instruct" \
    load_parameters_path="gs://igorts_europe/converted_checkpoints/qwen2.5-1.5b-instruct_mcjax/0/items" \
    use_dpo=True \
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
    ici_fsdp_parallelism=4 \
    ici_tensor_parallelism=1
