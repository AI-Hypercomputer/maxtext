#!/bin/bash

# This file, combined with step 2 in the same directory, runs on daily basis and demonstrates:
# 1. Converts the Mixtral PyTorch checkpoint to MaxText(orbax) format using a CPU VM.
# 2. Takes the MaxText(orbax) checkpoint to run inference, fine-tuning, and pre-training on a TPU VM.

# The flow of this file is to convert the Mistral PyTorch checkpoint to MaxText (orbax) format using a CPU VM.

# Example Usage: export BASE_OUTPUT_PATH=/path/to/GCS/bucket; bash end_to_end/tpu/mixtral/8x7b/1_test_mixtral.sh
# Use the same BASE_OUTPUT_PATH for both 1_test_mixtral.sh & 2_test_mixtral.sh.

set -ex
MODEL_VARIATION='8x7b'

if [ -z "${BASE_OUTPUT_PATH}" ]; then
    # Non-Googlers please remember to point BASE_OUTPUT_PATH to GCS buckets that you own, this script uses internal buckets for testing.
    export BASE_OUTPUT_PATH=gs://runner-maxtext-logs/$(date +%Y-%m-%d)/
    echo "BASE_OUTPUT_PATH is not set"
fi
BASE_OUTPUT_PATH=${BASE_OUTPUT_PATH%/}
echo using BASE_OUTPUT_PATH = ${BASE_OUTPUT_PATH}

# Download checkpoint
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

MODEL_NAME="mixtral-8x7B-v0.1-Instruct"
PARAM_DIR="$HOME/tempdisk"
mkdir -p "$PARAM_DIR"
[[ ! -z $(ls "$PARAM_DIR") ]] && fusermount -u "$PARAM_DIR"
gcsfuse --implicit-dirs maxtext-external "$PARAM_DIR"
# alternatively: gcloud storage cp -r "gs://maxtext-external/$MODEL_NAME" /tmp

# Convert it to MaxText(orbax) format - scanned ckpt
JAX_PLATFORMS=cpu python3 -m MaxText.utils.ckpt_scripts.llama_or_mistral_ckpt --base-model-path="$PARAM_DIR/$MODEL_NAME" --model-size=mixtral-8x7b --maxtext-model-path=${BASE_OUTPUT_PATH}/${MODEL_VARIATION}/scanned_ckpt/
echo "Wrote MaxText compatible scanned checkpoint to ${BASE_OUTPUT_PATH}/${MODEL_VARIATION}/scanned_ckpt"

# unmount the gcsfuse directory
fusermount -u "$PARAM_DIR"

# Generate unscanned ckpt for efficient decoding test
export SCANNED_CHECKPOINT=${BASE_OUTPUT_PATH}/${MODEL_VARIATION}/scanned_ckpt/0/items
export RUN_NAME=unscanned_ckpt
JAX_PLATFORMS=cpu python3 -m MaxText.generate_param_only_checkpoint "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}/"configs/base.yml async_checkpointing=false base_output_directory=${BASE_OUTPUT_PATH} load_parameters_path=${SCANNED_CHECKPOINT} run_name=${RUN_NAME} model_name='mixtral-8x7b' force_unroll=true skip_jax_distributed_system=True
echo "Wrote MaxText compatible unscanned checkpoint to ${BASE_OUTPUT_PATH}/${RUN_NAME}/checkpoints"
