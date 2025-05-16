#!/bin/bash

# This file, combined with step 1 in the same directory, runs on daily basis and demonstrates:
# 1. Converts the Llama4-Maverick HuggingFace checkpoint to MaxText (Orbax) format using a CPU VM.
# 2. Takes the MaxText (unscanned Orbax) checkpoint to run inference on a TPU VM.

# The flow of this file is to take the MaxText (unscanned Orbax) checkpoint and run inference on a TPU VM.

# Example Usage: export BASE_OUTPUT_PATH=/path/to/GCS/bucket; export MODEL_VARIATION=llama4-17b-[16e/128e]; bash end_to_end/tpu/llama4/2_test_llama4.sh
# Use the same BASE_OUTPUT_PATH and MODEL_VARIATION for both 1_test_llama4.sh & 1_test_llama4.sh.

# # In order to generate the Llama4 golden logits, please see this script: MaxText/scratch_code/golden_llama4_17b_16e_128e_export.ipynb

set -ex
idx=$(date +%Y-%m-%d)

# Installing torch for deps in forward_pass_logit_checker.py
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

if [ -z "${BASE_OUTPUT_PATH}" ]; then
    # Non-Googlers please remember to point BASE_OUTPUT_PATH to GCS buckets that you own, this script uses internal buckets for testing.
    export BASE_OUTPUT_PATH=gs://runner-maxtext-logs/$(date +%Y-%m-%d)/
    echo "BASE_OUTPUT_PATH is not set, using BASE_OUTPUT_PATH = ${BASE_OUTPUT_PATH}"
fi

# By default, we'll use "llama4-17b-16e"
if [ -z "${MODEL_VARIATION}" ]; then
    export MODEL_VARIATION="llama4-17b-16e"
    echo "MODEL_VARIATION is not set, using MODEL_VARIATION = ${MODEL_VARIATION}"
fi

export TOKENIZER_PATH=gs://maxtext-llama/${MODEL_VARIATION}/hf-checkpoint/
export MODEL_BUCKET=gs://maxtext-llama/${MODEL_VARIATION}

export UNSCANNED_CKPT_PATH=${MODEL_BUCKET}/${idx}/unscanned/0/items

# Step 2: run logit checking
python3 -m MaxText.tests.forward_pass_logit_checker  MaxText/configs/base.yml tokenizer_path=${TOKENIZER_PATH} load_parameters_path=${UNSCANNED_CKPT_PATH} run_name=forward_pass_test_${MODEL_VARIATION} attention=dot_product per_device_batch_size=1 model_name=${MODEL_VARIATION} max_prefill_predict_length=4 max_target_length=4 scan_layers=false --atol=0.5 --rtol=0.5 async_checkpointing=false sparse_matmul=false weight_dtype=float32 dtype=float32
