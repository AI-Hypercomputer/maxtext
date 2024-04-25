#!/bin/bash

# This file, combined with step 2 in the same directory, runs on daily basis and demonstrates:
# 1. Converts the Mistral PyTorch checkpoint to MaxText(orbax) format using a CPU VM.
# 2. Takes the MaxText (orbax) checkpoint to run inference and fine-tuning on a TPU VM.

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

# Download checkpoint, convert it to MaxText(orbax) format
pip3 install torch
gcloud storage cp -r gs://maxtext-external/mixtral-8x7B-v0.1-Instruct /tmp
JAX_PLATFORMS=cpu python3 MaxText/llama_or_mistral_ckpt.py --base-model-path /tmp/mixtral-8x7B-v0.1-Instruct --model-size mixtral-8x7b --maxtext-model-path ${BASE_OUTPUT_PATH}${MODEL_VARIATION}/decode-ckpt-maxtext/
echo "Wrote MaxText compatible checkpoint to ${BASE_OUTPUT_PATH}${MODEL_VARIATION}/decode-ckpt-maxtext"
