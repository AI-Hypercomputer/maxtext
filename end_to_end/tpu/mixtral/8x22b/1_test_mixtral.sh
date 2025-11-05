#!/bin/bash

# This file, combined with step 2 in the same directory, runs on daily basis and demonstrates:
# 1. Converts the Mistral PyTorch checkpoint to MaxText(orbax) format using a CPU VM.
# 2. Takes the MaxText(orbax) checkpoint to run inference, fine-tuning, and pre-training on a TPU VM.

# The flow of this file is to convert the Mistral PyTorch checkpoint to MaxText (orbax) format using a CPU VM.

# Example Usage: export BASE_OUTPUT_PATH=/path/to/GCS/bucket; bash end_to_end/tpu/mixtral/8x22b/1_test_mixtral.sh
# Use the same BASE_OUTPUT_PATH for both 1_test_mixtral.sh & 2_test_mixtral.sh.

set -ex
MODEL_VARIATION='8x22b'

if [ -z "${BASE_OUTPUT_PATH}" ]; then
    # Non-Googlers please remember to point BASE_OUTPUT_PATH to GCS buckets that you own, this script uses internal buckets for testing.
    export BASE_OUTPUT_PATH=gs://runner-maxtext-logs/$(date +%Y-%m-%d)/
    echo "BASE_OUTPUT_PATH is not set, using BASE_OUTPUT_PATH = ${BASE_OUTPUT_PATH}"
fi

# Download checkpoint
python3 -m pip install torch
MODEL_NAME="Mixtral-8x22B-Instruct-v0.1"

PARAM_DIR="$HOME/tempdisk"
mkdir -p "$PARAM_DIR"
[[ ! -z $(ls "$PARAM_DIR") ]] && fusermount -u "$PARAM_DIR"
gcsfuse --implicit-dirs maxtext-external "$PARAM_DIR"
# alternatively: $ gcloud storage cp -r "gs://maxtext-external/$MODEL_NAME" $PARAM_DIR

# Convert it to MaxText(orbax) format - scanned ckpt
JAX_PLATFORMS=cpu python3 -m MaxText.utils.ckpt_scripts.llama_or_mistral_ckpt --base-model-path="$PARAM_DIR/$MODEL_NAME" --model-size=mixtral-8x22b --maxtext-model-path=${BASE_OUTPUT_PATH}/${MODEL_VARIATION}/scanned_ckpt/
echo "Wrote MaxText compatible scanned checkpoint to ${BASE_OUTPUT_PATH}/${MODEL_VARIATION}/scanned_ckpt"

# unmount the gcsfuse directory
fusermount -u "$PARAM_DIR"

# rdyro(TODO): Temporarily disable the generation of unscanned checkpoints to
# save time, as this process takes a couple of hours and is not utilized in
# subsequent tests. The unscanned test is already running for 8x7b, so this
# removal should not impact overall testing.

