#!/bin/bash

# Converts a MaxText checkpoint to a Hugging Face model checkpoint for GPTOSS-20B.

set -ex

run_id=$1
CKPT_PATH=$2
SCAN_LAYERS=${3:-false}

export MODEL_NAME='gpt-oss-20b'

if [ -z "${BASE_OUTPUT_PATH}" ]; then
  export BASE_OUTPUT_PATH=gs://runner-maxtext-logs/${MODEL_NAME}
fi
BASE_OUTPUT_PATH=${BASE_OUTPUT_PATH%/}

if [ "${SCAN_LAYERS,,}" = "true" ]; then
    scan_status="scanned"
else
    scan_status="unscanned"
fi

python3 -m maxtext.checkpoint_conversion.to_huggingface \
    model_name=${MODEL_NAME} \
    tokenizer_type="huggingface" \
    load_parameters_path=${CKPT_PATH} \
    base_output_directory=${BASE_OUTPUT_PATH}/to_huggingface/${scan_status}/${run_id} \
    use_multimodal=false \
    scan_layers=$SCAN_LAYERS
