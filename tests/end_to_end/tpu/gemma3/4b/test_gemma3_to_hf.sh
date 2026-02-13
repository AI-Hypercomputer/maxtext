#!/bin/bash

# This script is both an end-to-end test that runs once a day on a v4-8 and documentation for how to get started with Gemma3-4B.

# The flow of this script is as follows:
# 1. Convert a MaxText checkpoint to a Hugging Face model checkpoint.
# 2. Run a forward pass check to compare the logits and KL divergence between the converted ckpt and original golden HF model.

# Pre-requisites:
# 1. Set HF_TOKEN environment variable to your Hugging Face access token with read permissions
# export HF_TOKEN=<Hugging Face access token>

set -ex
idx=$(date +%Y-%m-%d-%H-%M)
MODEL_NAME='gemma3-4b'
export MODEL_VARIATION='4b'
TOKENIZER_PATH="${MAXTEXT_ASSETS_ROOT:-${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/assets/tokenizers}}"'/tokenizer.gemma3'
# To convert the multimodal model, make sure the use_multimodal is set to be true
USE_MULTIMODAL=false

# Installing torch for deps in forward_pass_logit_checker.py
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# After downloading checkpoints, copy them to GCS bucket at $CHKPT_BUCKET \
# Non-Googlers please remember to use separate GCS paths for uploading model weights from kaggle ($CHKPT_BUCKET) and MaxText compatible weights ($MODEL_BUCKET).
# Non-Googlers please remember to point these variables to GCS buckets that you own, this script uses internal buckets for testing.
export MODEL_BUCKET=gs://maxtext-gemma/unified/gemma3/hf
# Here is an example of qwen3-4b maxtext checkpoint, converted from Qwen/Qwen3-4B
export CKPT_PATH=gs://maxtext-gemma/unified/gemma3/4b/unscanned/2025-08-05-18-18/0/items

# You can upload to huggingface hub or GCS by uncommenting the HF_CKPT_PATH and using it as base_output_directory
# export HF_CKPT_PATH=${MODEL_BUCKET}/${MODEL_VARIATION}/hf/${idx}
export LOCAL_PATH=./tmp/hf/${MODEL_NAME}/${idx}

python3 -m MaxText.utils.ckpt_conversion.to_huggingface "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"//base.yml \
    model_name=${MODEL_NAME} \
    hf_access_token=${HF_TOKEN} \
    load_parameters_path=${CKPT_PATH} \
    base_output_directory=${LOCAL_PATH} \
    use_multimodal=${USE_MULTIMODAL} \
    scan_layers=false

# Alternatively, if uploaded the converted ckpt, HF requires local storage of model and please uncomment below
# mkdir -p "${LOCAL_PATH}"
# gcloud storage cp -r ${HF_CKPT_PATH}/** ${LOCAL_PATH}
# echo "Copied from ${HF_CKPT_PATH} to ${LOCAL_PATH}"

# We also test whether the forward pass logits match the original HF model
# to get higher precision (eg. float32) run on CPU with `JAX_PLATFORMS=cpu`
python3 -m tests.utils.forward_pass_logit_checker "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"//base.yml \
    tokenizer_path=${TOKENIZER_PATH} \
    load_parameters_path=${CKPT_PATH} \
    model_name=${MODEL_NAME} \
    use_multimodal=${USE_MULTIMODAL} \
    scan_layers=false \
    --hf_model_path=${LOCAL_PATH} \
    --max_kl_div=0.015 \
    --run_hf_model=true
