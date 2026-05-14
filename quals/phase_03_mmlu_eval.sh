#!/bin/bash
# Phase 3: MMLU Evaluation
# Ensure venv is active
if [ -z "$VIRTUAL_ENV" ]; then
  source ~/maxtext_venv/bin/activate
fi

export PYTHONPATH=src

SFT_CKPT="gs://igorts_europe/converted_checkpoints/qwen2.5-1.5b-instruct_mcjax_v2/0/items"
DPO_CKPT="gs://igorts_europe/ttl=30d/dpo_quals/maxtext_run/dpo-verification-qwen-v3/inference_ckpt/0/items"

echo "=== Running MMLU on SFT Baseline ==="
TPU_VISIBLE_CHIPS=0 python3 -m benchmarks.mmlu.mmlu_eval src/maxtext/configs/base.yml model_name="qwen2.5-1.5b" tokenizer_path="Qwen/Qwen2.5-1.5B-Instruct" load_parameters_path=$SFT_CKPT scan_layers=False log_config=0 ici_fsdp_parallelism=1 ici_tensor_parallelism=1 ici_data_parallelism=1 per_device_batch_size=1 max_prefill_predict_length=1024 max_target_length=2048 add_bos=False hardware=tpu enable_single_controller=True --max_examples=500

echo "=== Running MMLU on DPO Checkpoint ==="
TPU_VISIBLE_CHIPS=0 python3 -m benchmarks.mmlu.mmlu_eval src/maxtext/configs/base.yml model_name="qwen2.5-1.5b" tokenizer_path="Qwen/Qwen2.5-1.5B-Instruct" load_parameters_path=$DPO_CKPT scan_layers=False log_config=0 ici_fsdp_parallelism=1 ici_tensor_parallelism=1 ici_data_parallelism=1 per_device_batch_size=1 max_prefill_predict_length=1024 max_target_length=2048 add_bos=False hardware=tpu enable_single_controller=True --max_examples=500
