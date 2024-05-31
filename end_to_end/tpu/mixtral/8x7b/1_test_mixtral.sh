#!/bin/bash

# This file, combined with step 2 in the same directory, runs on daily basis and demonstrates:
# 1. Converts the Mistral PyTorch checkpoint to MaxText(orbax) format using a CPU VM.
# 2. Takes the MaxText(orbax) checkpoint to run inference, fine-tuning, and pre-training on a TPU VM.

# The flow of this file is to convert the Mistral PyTorch checkpoint to MaxText (orbax) format using a CPU VM.

# Example Usage: export BASE_OUTPUT_PATH=/path/to/GCS/bucket; bash end_to_end/tpu/mixtral/8x7b/1_test_mixtral.sh
# Use the same BASE_OUTPUT_PATH for both 1_test_mixtral.sh & 2_test_mixtral.sh.

set -ex
MODEL_VARIATION='8x7b'

if [ -z "${BASE_OUTPUT_PATH}" ]; then
    # Non-Googlers please remember to point BASE_OUTPUT_PATH to GCS buckets that you own, this script uses internal buckets for testing.
    export BASE_OUTPUT_PATH=gs://runner-maxtext-logs/$(date +%Y-%m-%d-%H-%M)/
    echo "BASE_OUTPUT_PATH is not set, using BASE_OUTPUT_PATH = ${BASE_OUTPUT_PATH}"
fi

# Download checkpoint
pip3 install torch
gcloud storage cp -r gs://maxtext-external/mixtral-8x7B-v0.1-Instruct /tmp

# Convert it to MaxText(orbax) format - scanned ckpt (for loop implementation)
JAX_PLATFORMS=cpu python3 MaxText/llama_or_mistral_ckpt.py --base-model-path=/tmp/mixtral-8x7B-v0.1-Instruct --model-size=mixtral-8x7b --maxtext-model-path=${BASE_OUTPUT_PATH}${MODEL_VARIATION}/scanned_ckpt/
echo "Wrote MaxText compatible scanned checkpoint to ${BASE_OUTPUT_PATH}${MODEL_VARIATION}/scanned_ckpt"

# Convert it to MaxText(orbax) format - scanned ckpt (matmul implementation)
JAX_PLATFORMS=cpu python3 MaxText/llama_or_mistral_ckpt.py --base-model-path=/tmp/mixtral-8x7B-v0.1-Instruct --model-size=mixtral-8x7b --maxtext-model-path=${BASE_OUTPUT_PATH}${MODEL_VARIATION}/matmul_scanned_ckpt/ --moe-matmul=True
echo "Wrote MaxText compatible scanned checkpoint to ${BASE_OUTPUT_PATH}${MODEL_VARIATION}/matmul_scanned_ckpt"

# Generate unscanned ckpt for efficient decoding test
export SCANNED_CHECKPOINT=${BASE_OUTPUT_PATH}${MODEL_VARIATION}/scanned_ckpt/0/items
export RUN_NAME=unscanned_ckpt
JAX_PLATFORMS=cpu python MaxText/generate_param_only_checkpoint.py MaxText/configs/base.yml async_checkpointing=false base_output_directory=${BASE_OUTPUT_PATH} load_parameters_path=${SCANNED_CHECKPOINT} run_name=${RUN_NAME} model_name='mixtral-8x7b' force_unroll=true
echo "Wrote MaxText compatible unscanned checkpoint to ${BASE_OUTPUT_PATH}/${RUN_NAME}/checkpoints"
