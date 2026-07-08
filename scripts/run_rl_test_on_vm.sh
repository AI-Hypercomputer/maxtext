#!/bin/bash
set -ex

cd ~/workspace/maxtext
source ~/maxtext_venv/bin/activate
export PYTHONPATH=~/workspace/maxtext/src:$PYTHONPATH
export HF_TOKEN=''

export MAXTEXT_CKPT_PATH=gs://maxtext-gemma/gemma4/e2b/converted/2026-06-30-20-39/0/items
export BASE_OUTPUT_DIRECTORY=gs://mazumdera-test-bucket-europe-west4/gemma4-e2b
export MODEL=gemma4-e2b
export CHIPS_PER_VM=4
export RUN_NAME="rl-$(date +%Y%m%d-%H%M%S)"

python3 -m maxtext.trainers.post_train.rl.train_rl \
  model_name=${MODEL?} \
  load_parameters_path=${MAXTEXT_CKPT_PATH?} \
  run_name=${RUN_NAME?} \
  base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
  chips_per_vm=${CHIPS_PER_VM?} \
  num_batches=10 \
  scan_layers=False \
  rl.use_agentic_rollout=False \
  profiler=xplane \
  skip_first_n_steps_for_profiler=5 \
  profiler_steps=2 \
  vllm_hf_overrides='{"architectures": ["MaxTextForCausalLM"]}' \
  vllm_additional_config='{"maxtext_config": {"model_name": "gemma4-e2b", "model_call_mode": "inference", "enable_dp_attention": false, "allow_split_physical_axes": true, "log_config": false, "weight_dtype": "bfloat16", "scan_layers": false}}'
