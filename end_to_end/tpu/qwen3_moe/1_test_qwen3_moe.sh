#!/bin/bash

# Example script to convert a Qwen3 MoE checkpoint to MaxText format.
set -ex
idx=$(date +%Y-%m-%d)

if [ -z "${BASE_OUTPUT_PATH}" ]; then
    export BASE_OUTPUT_PATH=gs://runner-maxtext-logs/$(date +%Y-%m-%d)/
    echo "BASE_OUTPUT_PATH is not set, using BASE_OUTPUT_PATH = ${BASE_OUTPUT_PATH}"
fi

if [ -z "${MODEL_VARIATION}" ]; then
    export MODEL_VARIATION="qwen3-moe"
    echo "MODEL_VARIATION is not set, using MODEL_VARIATION = ${MODEL_VARIATION}"
fi

python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

export CHKPT_BUCKET=gs://maxtext-qwen3/${MODEL_VARIATION}/hf-checkpoint/
export MODEL_BUCKET=gs://maxtext-qwen3/${MODEL_VARIATION}

gcloud storage cp -r ${CHKPT_BUCKET} /tmp
export LOCATION_OF_HF_CHKPT_ON_DISK=/tmp/hf-checkpoint

JAX_PLATFORMS=cpu python3 -m MaxText.convert_qwen3_moe_ckpt --base_model_path ${LOCATION_OF_HF_CHKPT_ON_DISK} --maxtext_model_path ${MODEL_BUCKET}/${idx}/unscanned

