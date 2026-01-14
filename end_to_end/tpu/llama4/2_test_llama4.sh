#!/bin/bash

# This file, combined with step 1 in the same directory, runs on daily basis and demonstrates:
# 1. Converts the Llama4-Maverick HuggingFace checkpoint to MaxText (Orbax) format using a CPU VM.
# 2. Takes the MaxText (unscanned Orbax) checkpoint to run inference on a TPU VM.

# The flow of this file is to take the MaxText (unscanned Orbax) checkpoint and run inference on a TPU VM.

# Example Usage: export BASE_OUTPUT_PATH=/path/to/GCS/bucket; export MODEL_VARIATION=llama4-17b-[16e/128e]; bash end_to_end/tpu/llama4/2_test_llama4.sh
# Use the same BASE_OUTPUT_PATH and MODEL_VARIATION for both 1_test_llama4.sh & 1_test_llama4.sh.

# In order to generate the Llama4 golden logits, please see this script: tests/assets/logits_generation/golden_llama4_17b_16e_128e_export.ipynb

set -ex
idx=$(date +%Y-%m-%d)

# By default, we'll use "llama4-17b-16e"
if [ -z "${MODEL_VARIATION}" ]; then
    export MODEL_VARIATION="llama4-17b-16e"
    echo "MODEL_VARIATION is not set, using MODEL_VARIATION = ${MODEL_VARIATION}"
    export TOKENIZER_PATH=meta-llama/Llama-4-Scout-17B-16E
fi

# Installing torch for deps in forward_pass_logit_checker.py
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

if [ -z "${BASE_OUTPUT_PATH}" ]; then
    # Non-Googlers please remember to point BASE_OUTPUT_PATH to GCS buckets that you own, this script uses internal buckets for testing.
    export BASE_OUTPUT_PATH=gs://runner-maxtext-logs/$(date +%Y-%m-%d)
    echo "BASE_OUTPUT_PATH is not set"
fi
BASE_OUTPUT_PATH=${BASE_OUTPUT_PATH%/}
echo using BASE_OUTPUT_PATH = ${BASE_OUTPUT_PATH}


export UNSCANNED_CKPT_PATH=${BASE_OUTPUT_PATH}/unscanned/0/items

# Step 2: run logit checking
python3 -m tests.forward_pass_logit_checker "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}/"configs/base.yml tokenizer_path=${TOKENIZER_PATH} load_parameters_path=${UNSCANNED_CKPT_PATH} run_name=forward_pass_test_${MODEL_VARIATION} attention=dot_product per_device_batch_size=1 model_name=${MODEL_VARIATION} max_prefill_predict_length=4 max_target_length=4 scan_layers=false --atol=0.01 --rtol=0.01 async_checkpointing=false sparse_matmul=false weight_dtype=float32 dtype=float32 activations_in_float32=true matmul_precision=float32 float32_logits=true float32_qk_product=true ici_expert_parallelism=16
