#!/bin/bash

# End-to-end conversion + forward-pass logit check for Gemma 4 E4B (instruction-tuned).
#
# Multimodal is gated off for E2B / E4B by `MaxTextConfig.validate_model_size`
# and is not exercised here. See `tests/end_to_end/tpu/gemma4/Run_Gemma4.md`
# for an overview.

set -ex
idx=$(date +%Y-%m-%d-%H-%M)

MODEL_NAME='gemma4-e4b'
export MODEL_VARIATION='e4b-it'
TOKENIZER_PATH='google/gemma-4-E4B-it'
USE_SCAN_LAYERS=false  # Per-layer KV sharing is incompatible with nn.scan.

# Installing torch for deps in forward_pass_logit_checker.py
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# After downloading checkpoints, copy them to GCS bucket at $MODEL_BUCKET
export MODEL_BUCKET='gs://maxtext-gemma/gemma4'
export HF_MODEL='path/to/your/hf/gemma-4-E4B-it'

# HF -> MaxText conversion:
python3 -m maxtext.checkpoint_conversion.to_maxtext "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"/base.yml \
    model_name=${MODEL_NAME} \
    hf_access_token=${HF_TOKEN} \
    --hf_model_path=${HF_MODEL} \
    base_output_directory=${MODEL_BUCKET}/${MODEL_VARIATION}/converted/${idx} \
    use_multimodal=false \
    scan_layers=${USE_SCAN_LAYERS}

export MAXTEXT_CKPT_PATH=${MODEL_BUCKET}/${MODEL_VARIATION}/converted/${idx}/0/items

# Forward-pass logit check: runs HF on the fly and compares to MaxText.
python3 -m tests.utils.forward_pass_logit_checker "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"/base.yml \
    tokenizer_path=${TOKENIZER_PATH} \
    load_parameters_path=${MAXTEXT_CKPT_PATH} \
    model_name=${MODEL_NAME} \
    use_multimodal=false \
    scan_layers=${USE_SCAN_LAYERS} \
    per_device_batch_size=1 \
    dtype=float32 \
    attention=dot_product \
    --max_kl_div=0.03 \
    --run_hf_model=true \
    --hf_model_path=${HF_MODEL}
