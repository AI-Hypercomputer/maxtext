#!/bin/bash
# Script to run RL with Qwen3-0.6b native vLLM model (dummy load format) to check loading latency

set -e

# Setup environment variables as in tests/post_training/integration/single_host_train_rl_test.py
export NEW_MODEL_DESIGN=1
export TPU_BACKEND_TYPE=jax
export SKIP_JAX_PRECOMPILE=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# GCP paths
export BASE_OUTPUT_DIRECTORY="gs://igorts_europe/ttl=30d"

# Run command
/home/igorts_google_com/maxtext_venv/bin/python3 -m maxtext.trainers.post_train.rl.train_rl \
  src/maxtext/configs/post_train/rl.yml \
  model_name=qwen3-0.6b \
  tokenizer_path=Qwen/Qwen3-0.6B \
  run_name=rl-qwen3-0.6b-repro \
  base_output_directory="${BASE_OUTPUT_DIRECTORY}" \
  batch_size=8 \
  num_batches=2 \
  num_test_batches=0 \
  chips_per_vm=8 \
  scan_layers=True \
  hbm_utilization_vllm=0.75 \
  rollout_data_parallelism=-1 \
  rollout_tensor_parallelism=1 \
  rl.num_generations=8 \
  train_micro_batch_size=8 \
  rollout_micro_batch_size=8 \
  dataset_name=openai/gsm8k \
  max_target_length=1024 \
  max_prefill_predict_length=256 \
  enable_checkpointing=false \
  convert_checkpoint_if_possible=false \
  vllm_load_format=dummy
