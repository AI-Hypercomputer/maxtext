#!/bin/bash

# Converts a MaxText checkpoint to a Hugging Face model checkpoint.

# Usage:
# export RUN_ID=$(date +%Y-%m-%d-%H-%M-%S)
# bash test_gemma4_to_hf.sh $RUN_ID $CHECKPOINT_PATH $USE_MULTIMODAL $SCAN_LAYERS

set -ex

run_id=$1
CKPT_PATH=$2
USE_MULTIMODAL=${3:-false}
SCAN_LAYERS=${4:-false}

MODEL_NAME='gemma4-26b'
BASE_OUTPUT_DIRECTORY="gs://runner-maxtext-logs/${MODEL_NAME}"

if [ "${SCAN_LAYERS,,}" = "true" ]; then
    scan_status="scanned"
else
    scan_status="unscanned"
fi

python3 -m maxtext.checkpoint_conversion.to_huggingface \
    model_name=${MODEL_NAME} \
    tokenizer_type="huggingface" \
    load_parameters_path=${CKPT_PATH} \
    base_output_directory=${BASE_OUTPUT_DIRECTORY}/to_huggingface/${scan_status}/${run_id} \
    use_multimodal=${USE_MULTIMODAL} \
    scan_layers=$SCAN_LAYERS