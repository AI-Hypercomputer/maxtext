#!/bin/bash
set -ex

# Set output directory
export BASE_OUTPUT_DIRECTORY="gs://chfu-a4-west3-ckpts/checkpoints/llama31-8b/maxtext-base"

export PYTHONPATH=$PYTHONPATH:src

/home/chfu_google_com/maxtext/maxtext_venv/bin/python3 -m MaxText.utils.ckpt_conversion.to_maxtext src/MaxText/configs/base.yml \
    model_name=llama3.1-8b \
    base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    hf_access_token=${HF_TOKEN} \
    scan_layers=false \
    --hf_model_path=/tmp/llama31-8b-hf
