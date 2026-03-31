#!/bin/bash


# This script runs checkpoint conversion and an end-to-end MMLU perplexity test on Qwen3-0.6b.


# Pre-requisites:
# 1. Set HF_TOKEN environment variable to your Hugging Face access token with read permissions
# export HF_TOKEN=<Hugging Face access token>



set -ex
idx=$(date +%Y-%m-%d-%H-%M)
MODEL_NAME='qwen3-0.6b'
export MODEL_VARIATION='0.6b'
HF_GOLDEN_MODEL='Qwen/Qwen3-0.6B'
TOKENIZER_PATH="${MAXTEXT_ASSETS_ROOT:-${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/assets/tokenizers}}"'/qwen3-tokenizer'



export MODEL_BUCKET=gs://maxtext-qwen/qwen3



echo "Step 1: Convert the checkpoint downloaded from Hugging Face to make it compatible with MaxText."
python3 -m maxtext.checkpoint_conversion.to_maxtext "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"//base.yml \
    model_name=${MODEL_NAME} \
    hf_access_token=${HF_TOKEN} \
    base_output_directory=${MODEL_BUCKET}/${MODEL_VARIATION}/unscanned/${idx} \
    scan_layers=false



export UNSCANNED_CKPT_PATH=${MODEL_BUCKET}/${MODEL_VARIATION}/unscanned/${idx}/0/items




echo "Step 2: Run MMLU eval on the unscanned checkpoint"
PYTHONPATH=src:. python3 -m benchmarks.mmlu.mmlu_eval_perplexity \
    "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"//base.yml \
    tokenizer_path=${TOKENIZER_PATH} \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    model_name=${MODEL_NAME} \
    per_device_batch_size=1 \
    max_target_length=1024 \
    steps=5



if [ $? -eq 0 ]
then
    echo "Successfully ran mmlu_eval_perplexity for $MODEL_NAME."
else
    echo "Failed evaluating mmlu_eval_perplexity for $MODEL_NAME."
    exit 1
fi