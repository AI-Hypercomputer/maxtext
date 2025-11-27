#!/bin/bash

# This file, combined with step 2 in the same directory, runs on daily basis and demonstrates:
# 1. Converts the Llama4-Maverick HuggingFace checkpoint to MaxText (Orbax) format using a CPU VM.
# 2. Takes the MaxText (unscanned Orbax) checkpoint to run inference on a TPU VM.

# The flow of this file is to convert the Llama4 (Scout/Maverick) HuggingFace checkpoint to MaxText (Orbax) format using a CPU VM.

# Example Usage: export BASE_OUTPUT_PATH=/path/to/GCS/bucket; export MODEL_VARIATION=llama4-17b-[16e/128e]; bash end_to_end/tpu/llama4/1_test_llama4.sh
# Use the same BASE_OUTPUT_PATH and MODEL_VARIATION for both 1_test_llama4.sh & 1_test_llama4.sh.

# In order to generate the Llama4 golden logits, please see this script: src/MaxText/scratch_code/golden_llama4_17b_16e_128e_export.ipynb

set -ex
idx=$(date +%Y-%m-%d)


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

uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# Step 1:
# After downloading checkpoints, copy them to GCS bucket at $CHKPT_BUCKET \
# Non-Googlers please remember to use separate GCS paths for uploading model weights from HuggingFace ($CHKPT_BUCKET) and MaxText compatible weights
# ($MODEL_BUCKET). Non-Googlers please remember to point these variables to GCS buckets that you own, this script uses internal buckets for testing.
# You can use the HuggingFace checkpoint at https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E for Scout and
# https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E for Maverick
export CHKPT_BUCKET=gs://maxtext-llama/${MODEL_VARIATION}/hf-checkpoint/
export MODEL_BUCKET=gs://maxtext-llama/${MODEL_VARIATION}

# In the following command, we are copying the HF checkpoint into a local directory `tmp` -- you are free to use a different local directory than /tmp/,
gcloud storage cp -r ${CHKPT_BUCKET} /tmp

export LOCATION_OF_HF_CHKPT_ON_DISK=/tmp/hf-checkpoint

JAX_PLATFORMS=cpu python3 -m MaxText.utils.ckpt_scripts.llama4_ckpt_unscanned --base-model-path ${LOCATION_OF_HF_CHKPT_ON_DISK} --maxtext-model-path ${MODEL_BUCKET}/${idx}/unscanned --model-size ${MODEL_VARIATION} --huggingface-checkpoint True
echo "Wrote MaxText compatible unscanned checkpoint to ${MODEL_BUCKET}/${idx}/unscanned"
