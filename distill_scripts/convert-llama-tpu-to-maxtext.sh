#!/bin/bash
set -ex

export HF_MODEL='llama3.1-8b'
export MODEL='llama3.1-8b'
export TOKENIZER='meta-llama/Llama-3.1-8B'
export RUN_NAME=student_checkpoint
export BASE_OUTPUT_DIRECTORY=/tmp/maxtext_checkpoint
export PYTHONPATH=$(pwd)/src

# copy student checkpoint to local dir
# gsutil -m -o "GSUtil:check_hashes=never" cp -r gs://chfu-eu-ckpts/checkpoints/hf/Llama-3.1-8B-TPU /tmp/

export HF_MODEL_PATH=/tmp/Llama-3.1-8B-TPU/

JAX_PLATFORMS=cpu python3 -m MaxText.utils.ckpt_conversion.to_maxtext src/MaxText/configs/base.yml \
    model_name=${HF_MODEL} \
    hf_access_token=${HF_TOKEN} \
    base_output_directory=${BASE_OUTPUT_DIRECTORY}/${RUN_NAME} \
    scan_layers=True hardware=cpu skip_jax_distributed_system=true \
    base_num_query_heads=16 \
    head_dim=256 \
    base_num_kv_heads=4 \
    --hf_model_path=$HF_MODEL_PATH