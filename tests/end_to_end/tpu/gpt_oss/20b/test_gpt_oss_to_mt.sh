#!/bin/bash

# Converts GPTOSS-20B HuggingFace checkpoint to MaxText format and validates logit correctness.

set -ex

export PYTHONPATH=src

run_id=${1:-$(date +%Y-%m-%d-%H-%M-%S)}
export MODEL_NAME='gpt-oss-20b'
export TOKENIZER_PATH='openai/gpt-oss-20b'

if [ -z "${BASE_OUTPUT_PATH}" ]; then
  export BASE_OUTPUT_PATH=gs://runner-maxtext-logs/${MODEL_NAME}
fi
BASE_OUTPUT_PATH=${BASE_OUTPUT_PATH%/}
echo "Using BASE_OUTPUT_PATH = ${BASE_OUTPUT_PATH}"

if [ -z "${CKPT_DISK_LOCATION}" ]; then
  export CKPT_BUCKET=gs://maxtext-model-checkpoints/gpt-oss-20b/hf-bf16
  gcloud storage cp -r ${CKPT_BUCKET} /tmp
  export CKPT_DISK_LOCATION=/tmp/hf-bf16
fi

# 1. Convert to scanned checkpoint (for training)
JAX_PLATFORMS=cpu python3 -m maxtext.checkpoint_conversion.standalone_scripts.convert_gpt_oss_ckpt \
    --base-model-path ${CKPT_DISK_LOCATION} \
    --maxtext-model-path ${BASE_OUTPUT_PATH}/scanned/${run_id} \
    --model-size ${MODEL_NAME}

SCANNED_CKPT_PATH=${BASE_OUTPUT_PATH}/scanned/${run_id}/0/items
echo "Scanned checkpoint path: ${SCANNED_CKPT_PATH}"

# 2. Convert to unscanned checkpoint (for inference)
JAX_PLATFORMS=cpu python3 -m maxtext.checkpoint_conversion.standalone_scripts.convert_gpt_oss_unscanned_ckpt \
    --base-model-path ${CKPT_DISK_LOCATION} \
    --maxtext-model-path ${BASE_OUTPUT_PATH}/unscanned/${run_id} \
    --model-size ${MODEL_NAME}

UNSCANNED_CKPT_PATH=${BASE_OUTPUT_PATH}/unscanned/${run_id}/0/items
echo "Unscanned checkpoint path: ${UNSCANNED_CKPT_PATH}"

# 3. Logit correctness check
if [ ! -f /tmp/golden_data_gpt-oss-20b.jsonl ]; then
  gcloud storage cp gs://maxtext-test-assets/golden_data_gpt-oss-20b.jsonl /tmp/golden_data_gpt-oss-20b.jsonl
fi

SPARSE_MATMUL="True"
MEGABLOX="True"

python3 -m tests.utils.forward_pass_logit_checker \
    base_output_directory=${BASE_OUTPUT_PATH} \
    model_name=${MODEL_NAME} \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    scan_layers=false \
    attention=dot_product \
    sparse_matmul=${SPARSE_MATMUL} \
    megablox=${MEGABLOX} \
    --golden_logits_path=/tmp/golden_data_gpt-oss-20b.jsonl \
    --max_kl_div=0.01
