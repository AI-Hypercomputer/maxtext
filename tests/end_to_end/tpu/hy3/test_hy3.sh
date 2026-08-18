#!/bin/bash

# This script tests the Hy3 architecture in MaxText using the hy3-tiny synthetic configuration.
# Since the real Tencent Hunyuan V3 model is 295B (which is too large for automated CI daily tests),
# this tests the exact architecture pathways (GQA with QK-Norm attention, DeepSeek-V3-style
# aux-loss-free sigmoid+bias MoE routing) on a miniature configuration. Hy3 does not use MLA.

# Flow:
# 1. Run pre-training with the tiny config for 10 steps.
# 2. Run inference decoding on the resulting checkpoint.

set -ex

export MODEL_NAME='hy3-tiny'
export RUN_ID="hy3-test-$(date +%Y-%m-%d-%H-%M-%S)"

if [ -z "${BASE_OUTPUT_PATH}" ]; then
  export BASE_OUTPUT_PATH=gs://runner-maxtext-logs/${RUN_ID}
  echo "BASE_OUTPUT_PATH is not set, defaulting to gs://runner-maxtext-logs/${RUN_ID}"
fi
BASE_OUTPUT_PATH=${BASE_OUTPUT_PATH%/}

# Step 1: Run pre-training (tokamax_gmm / generic MoE path)
# We test with tiny config to ensure XLA compilation succeeds for the architecture logic.
python3 -m maxtext.trainers.pre_train.train \
  ${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}/models/hy3-tiny.yml \
  base_output_directory=${BASE_OUTPUT_PATH} \
  run_name=${RUN_ID} \
  tokenizer_type=huggingface \
  tokenizer_path=tencent/Hy3 \
  dataset_type=synthetic \
  enable_checkpointing=True \
  dtype=bfloat16 \
  weight_dtype=bfloat16 \
  per_device_batch_size=1 \
  steps=10 \
  max_target_length=128 \
  ici_fsdp_parallelism=-1 \
  ici_expert_parallelism=1 \
  save_checkpoint_on_completion=True

# Step 2: Run inference decoding using the newly trained checkpoint
python3 -m maxtext.inference.decode \
  ${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}/models/hy3-tiny.yml \
  base_output_directory=${BASE_OUTPUT_PATH} \
  run_name=${RUN_ID}-decode \
  tokenizer_type=huggingface \
  tokenizer_path=tencent/Hy3 \
  load_parameters_path=${BASE_OUTPUT_PATH}/${RUN_ID}/checkpoints/10/items \
  scan_layers=False \
  dtype=bfloat16 \
  weight_dtype=bfloat16 \
  per_device_batch_size=1 \
  max_prefill_predict_length=32 \
  max_target_length=64 \
  ici_fsdp_parallelism=1 \
  prompt="Hello, this is a test prompt for Hy3."

echo "hy3-tiny CI tests completed successfully!"
