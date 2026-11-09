#!/bin/bash

# Converts a MaxText checkpoint to a Hugging Face model checkpoint.

# Usage:
# export RUN_ID=$(date +%Y-%m-%d-%H-%M-%S)
# bash tests/end_to_end/cpu/qwen3/30b/test_qwen3_to_hf.sh $RUN_ID $CHECKPOINT_PATH $SCAN_LAYERS

set -ex

# Avoid conflict between M_RUN_NAME env var and CLI run_name argument in pyconfig
unset M_RUN_NAME

run_id=${1:?"Error: run_id (argument 1) is required"}
CKPT_PATH=${2:?"Error: CKPT_PATH (argument 2) is required"}
SCAN_LAYERS=${3:-false}

MODEL_NAME='qwen3-30b-a3b-base'
BASE_OUTPUT_DIRECTORY="${BASE_OUTPUT_DIRECTORY:-gs://runner-maxtext-logs/${MODEL_NAME}}"

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
    scan_layers=$SCAN_LAYERS \
    checkpoint_storage_use_zarr3=False \
    checkpoint_storage_use_ocdbt=False \
    hardware=cpu \
    skip_jax_distributed_system=true
