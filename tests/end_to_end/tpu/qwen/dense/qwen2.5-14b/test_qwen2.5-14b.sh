#!/bin/bash

# This script runs end-to-end tests for qwen2.5-14b on MaxText.
# The flow of this file is as follows:
# 1. Convert the HuggingFace checkpoint to MaxText-compatible checkpoint (scanned and unscanned).
# 2. Run logit check against the HuggingFace model.
# 3. Run SFT.

# Example Usage: export BASE_OUTPUT_PATH=<GCS_bucket_path> bash test_qwen2.5-14b.sh

set -ex

export MODEL_NAME='qwen2.5-14b'
export HF_MODEL_ID='Qwen/Qwen2.5-14B-Instruct'
export TOKENIZER_PATH=${HF_MODEL_ID}

if [ -z "${BASE_OUTPUT_PATH}" ]; then
    export BASE_OUTPUT_PATH=gs://runner-maxtext-logs/$(date +%Y-%m-%d-%H-%M)
    echo "BASE_OUTPUT_PATH is not set, using BASE_OUTPUT_PATH = ${BASE_OUTPUT_PATH}"
fi

export BASE_OUTPUT_PATH_PREFIX="${BASE_OUTPUT_PATH}/${MODEL_NAME}/$(date +%Y-%m-%d-%H-%M)"

# Installing torch for deps in forward_pass_logit_checker.py
#python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# Step 1: Checkpoint conversion
echo "--- Starting Checkpoint Conversion ---"

# 1.1 Convert checkpoint to `scanned` format
JAX_PLATFORMS=cpu python3 -m maxtext.checkpoint_conversion.to_maxtext \
  "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"/base.yml \
  model_name=${MODEL_NAME} \
  base_output_directory=${BASE_OUTPUT_PATH_PREFIX}/scanned \
  run_name=scanned_conversion \
  tokenizer_path=${TOKENIZER_PATH} \
  async_checkpointing=false \
  scan_layers=true

export SCANNED_CKPT_PATH=${BASE_OUTPUT_PATH_PREFIX}/scanned/0/items

# 1.2 Convert checkpoint to `unscanned` format
JAX_PLATFORMS=cpu python3 -m maxtext.checkpoint_conversion.to_maxtext \
  "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"/base.yml \
  model_name=${MODEL_NAME} \
  base_output_directory=${BASE_OUTPUT_PATH_PREFIX}/unscanned \
  run_name=unscanned_conversion \
  tokenizer_path=${TOKENIZER_PATH} \
  async_checkpointing=false \
  scan_layers=false

export UNSCANNED_CKPT_PATH=${BASE_OUTPUT_PATH_PREFIX}/unscanned/0/items

# Step 2: Forward pass logit checker
echo "--- Starting Forward Pass Logit Checker ---"
# 2.1 Check unscanned checkpoint
JAX_PLATFORMS=cpu python3 -m tests.utils.forward_pass_logit_checker \
    "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"/base.yml \
    run_name=forward_pass_test_unscanned \
    model_name=${MODEL_NAME} \
    tokenizer_path=${TOKENIZER_PATH} \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    max_prefill_predict_length=4 \
    max_target_length=4 \
    dataset_type=synthetic \
    scan_layers=false \
    per_device_batch_size=1 \
    skip_jax_distributed_system=True \
    weight_dtype=bfloat16 \
    --max_kl_div=0.015 \
    --run_hf_model=True \
    --hf_model_path=${HF_MODEL_ID}

# 2.2 Check scanned checkpoint
JAX_PLATFORMS=cpu python3 -m tests.utils.forward_pass_logit_checker \
    "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"/base.yml \
    run_name=forward_pass_test_scanned \
    model_name=${MODEL_NAME} \
    tokenizer_path=${TOKENIZER_PATH} \
    load_parameters_path=${SCANNED_CKPT_PATH} \
    max_prefill_predict_length=4 \
    max_target_length=4 \
    dataset_type=synthetic \
    scan_layers=true \
    per_device_batch_size=1 \
    skip_jax_distributed_system=True \
    weight_dtype=bfloat16 \
    --max_kl_div=0.015 \
    --run_hf_model=True \
    --hf_model_path=${HF_MODEL_ID}

# Step 3: SFT
echo "--- Starting SFT ---"
python3 -m maxtext.trainers.post_train.sft.train_sft \
    "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"/post_train/sft.yml \
    base_output_directory=${BASE_OUTPUT_PATH_PREFIX}/finetuned \
    run_name=sft_test \
    model_name=${MODEL_NAME} \
    tokenizer_path=${TOKENIZER_PATH} \
    tokenizer_type=huggingface \
    load_parameters_path=${SCANNED_CKPT_PATH} \
    dataset_type=hf \
    scan_layers=true \
    per_device_batch_size=4 \
    learning_rate=1.3e-5 \
    steps=5 \
    max_target_length=1024 \
    weight_dtype=bfloat16
