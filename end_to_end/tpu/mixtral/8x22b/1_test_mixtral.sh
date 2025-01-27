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
    export BASE_OUTPUT_PATH=gs://runner-maxtext-logs/ranran/$(date +%Y-%m-%d)/
    echo "BASE_OUTPUT_PATH is not set, using BASE_OUTPUT_PATH = ${BASE_OUTPUT_PATH}"
fi

# Download checkpoint
pip3 install torch
MODEL_NAME="mixtral-8x22B-Instruct-v0.3"
# MODEL_NAME="Mixtral-8x22B-Instruct-v0.1"

PARAM_DIR="$HOME/tempdisk"
# OUTPUT_PARAM_DIR="$HOME/tempoutput"
mkdir -p "$PARAM_DIR"
# mkdir -p "$OUTPUT_PARAM_DIR"
[[ ! -z $(ls "$PARAM_DIR") ]] && fusermount -u "$PARAM_DIR"
# [[ ! -z $(ls "$OUTPUT_PARAM_DIR") ]] && fusermount -u "$OUTPUT_PARAM_DIR"
# gcsfuse --implicit-dirs maxtext-external "$PARAM_DIR"
# gcsfuse -o ro --implicit-dirs --type-cache-max-size-mb=-1 --stat-cache-max-size-mb=-1 --kernel-list-cache-ttl-secs=-1 --metadata-cache-ttl-secs=-1 maxtext-external "$PARAM_DIR"

# gcsfuse -o ro --implicit-dirs --log-severity=debug \
#         --type-cache-max-size-mb=-1 --stat-cache-max-size-mb=-1 --kernel-list-cache-ttl-secs=-1 --metadata-cache-ttl-secs=-1 \
#         --log-file=$HOME/gcsfuse_$TIMESTAMP.json "$DATASET_GCS_BUCKET" "$MOUNT_PATH"
# gcsfuse --implicit-dirs runner-maxtext-logs "$OUTPUT_PARAM_DIR"
# alternatively: $ gcloud storage cp -r "gs://maxtext-external/$MODEL_NAME" $PARAM_DIR
gcloud storage cp -r "gs://maxtext-external/$MODEL_NAME" $PARAM_DIR

# Convert it to MaxText(orbax) format - scanned ckpt
JAX_PLATFORMS=cpu python3 MaxText/llama_or_mistral_ckpt.py --base-model-path="$PARAM_DIR/$MODEL_NAME" --model-size=mixtral-8x22b --maxtext-model-path=${OUTPUT_PARAM_DIR}/ranran/${MODEL_NAME}/scanned_ckpt/ --checkpoint-type=safetensors
echo "Wrote MaxText compatible scanned checkpoint to ${BASE_OUTPUT_PATH}/ranran/${MODEL_VARIATION}/scanned_ckpt_fix"

# unmount the gcsfuse directory
fusermount -u "$PARAM_DIR"

# rdyro(TODO): Temporarily disable the generation of unscanned checkpoints to
# save time, as this process takes a couple of hours and is not utilized in
# subsequent tests. The unscanned test is already running for 8x7b, so this
# removal should not impact overall testing.

# Generate unscanned ckpt for efficient decoding test
# export SCANNED_CHECKPOINT=${BASE_OUTPUT_PATH}/${MODEL_VARIATION}/scanned_ckpt/0/items
# export SCANNED_CHECKPOINT=${OUTPUT_PARAM_DIR}/ranran/${MODEL_NAME}/scanned_ckpt/0/items
# export RUN_NAME=unscanned_ckpt
# JAX_PLATFORMS=cpu python MaxText/generate_param_only_checkpoint.py MaxText/configs/base.yml async_checkpointing=false base_output_directory=${OUTPUT_PARAM_DIR} load_parameters_path=${SCANNED_CHECKPOINT} run_name=${RUN_NAME} model_name='mixtral-8x22b' force_unroll=true
# echo "Wrote MaxText compatible unscanned checkpoint to ${OUTPUT_PARAM_DIR}/${RUN_NAME}/checkpoints"

# fusermount -u "$OUTPUT_PARAM_DIR"

# # Generate unscanned ckpt for efficient decoding test
# export SCANNED_CHECKPOINT=gs://ranran-multipod-dev/xlml/8x22b/origin/8x22b/scanned_ckpt/0/items
# export RUN_NAME=unscanned_ckpt
# JAX_PLATFORMS=cpu python MaxText/generate_param_only_checkpoint.py MaxText/configs/base.yml async_checkpointing=false base_output_directory=${BASE_OUTPUT_PATH} load_parameters_path=${SCANNED_CHECKPOINT} run_name=${RUN_NAME} model_name='mixtral-8x22b' weight_dtype=bfloat16 force_unroll=true
# echo "Wrote MaxText compatible unscanned checkpoint to gs://ranran-multipod-dev/xlml/8x22b/origin/8x22b/${RUN_NAME}/checkpoints"