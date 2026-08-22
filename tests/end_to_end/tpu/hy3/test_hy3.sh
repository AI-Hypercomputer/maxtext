#!/bin/bash

# This script tests the Hy3 architecture in MaxText using the hy3-tiny synthetic configuration.
# Since the real Tencent Hunyuan V3 model is 295B (which is too large for automated CI daily tests),
# this tests the exact architecture pathways (GQA with QK-Norm attention, DeepSeek-V3-style
# aux-loss-free sigmoid+bias MoE routing) on a miniature configuration. Hy3 does not use MLA.

# Flow:
# 1. Run pre-training with the tiny config for 10 steps (saves a scanned full-state checkpoint).
# 2. Convert the full-state checkpoint into a params-only, unscanned checkpoint
#    (generate_param_only_checkpoint with force_unroll=true) -- decoding needs an
#    unscanned checkpoint, and the train step above only produces a scanned one.
# 3. Run inference decoding on the unscanned checkpoint from step 2.

# Example Usage: export HF_TOKEN=<huggingface_access_token>; bash test_hy3.sh

set -ex

export MODEL_NAME='hy3-tiny'
export RUN_ID="hy3-test-$(date +%Y-%m-%d-%H-%M-%S)"

if [ -z "${BASE_OUTPUT_PATH}" ]; then
  export BASE_OUTPUT_PATH=gs://runner-maxtext-logs/${RUN_ID}
  echo "BASE_OUTPUT_PATH is not set, defaulting to gs://runner-maxtext-logs/${RUN_ID}"
fi
BASE_OUTPUT_PATH=${BASE_OUTPUT_PATH%/}

# Step 1: Run pre-training (default/generic MoE path -- no sparse_matmul or
# use_tokamax_gmm override here, this just exercises base.yml's defaults).
# We test with tiny config to ensure XLA compilation succeeds for the architecture logic.
python3 -m maxtext.trainers.pre_train.train \
  ${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}/base.yml \
  model_name=${MODEL_NAME} \
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

# Step 2: Convert the full-state (scanned) checkpoint from step 1 into a
# params-only, unscanned checkpoint suitable for decoding. Checkpoint steps
# are 0-indexed, so `steps=10` above produces its last checkpoint at "9".
python3 -m maxtext.utils.generate_param_only_checkpoint \
  ${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}/base.yml \
  model_name=${MODEL_NAME} \
  base_output_directory=${BASE_OUTPUT_PATH} \
  load_full_state_path=${BASE_OUTPUT_PATH}/${RUN_ID}/checkpoints/9/items \
  run_name=${RUN_ID}-unscanned \
  force_unroll=true

# Step 3: Run inference decoding using the unscanned params-only checkpoint
python3 -m maxtext.inference.decode \
  ${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}/base.yml \
  model_name=${MODEL_NAME} \
  base_output_directory=${BASE_OUTPUT_PATH} \
  run_name=${RUN_ID}-decode \
  tokenizer_type=huggingface \
  tokenizer_path=tencent/Hy3 \
  load_parameters_path=${BASE_OUTPUT_PATH}/${RUN_ID}-unscanned/checkpoints/0/items \
  scan_layers=False \
  dtype=bfloat16 \
  weight_dtype=bfloat16 \
  per_device_batch_size=1 \
  max_prefill_predict_length=32 \
  max_target_length=64 \
  ici_fsdp_parallelism=1 \
  prompt="Hello, this is a test prompt for Hy3."

echo "hy3-tiny CI tests completed successfully!"
