#!/bin/bash

# Validates the GPTOSS-20B Supervised Fine-Tuning (SFT) pipeline.

set -ex

run_id=${1:-$(date +%Y-%m-%d-%H-%M-%S)}
export MODEL_NAME='gpt-oss-20b'
export TOKENIZER_PATH='openai/gpt-oss-20b'

if [ -z "${BASE_OUTPUT_PATH}" ]; then
  export BASE_OUTPUT_PATH=gs://runner-maxtext-logs/${MODEL_NAME}
fi
BASE_OUTPUT_PATH=${BASE_OUTPUT_PATH%/}

export SCANNED_CKPT_PATH=${BASE_OUTPUT_PATH}/scanned/${run_id}/0/items

export SPARSE_MATMUL="True"
export MEGABLOX="True"
export SFT_ATTENTION="flash"

# 1. Run Supervised Fine-Tuning
python3 -m maxtext.trainers.post_train.sft.train_sft_native \
    "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs/post_train}"/sft.yml \
    base_output_directory=${BASE_OUTPUT_PATH}/sft \
    run_name=${run_id} \
    model_name=${MODEL_NAME} \
    tokenizer_type=huggingface \
    tokenizer_path=${TOKENIZER_PATH} \
    dataset_type=hf \
    enable_checkpointing=true \
    async_checkpointing=false \
    load_parameters_path=${SCANNED_CKPT_PATH} \
    scan_layers=True \
    attention=${SFT_ATTENTION} \
    sparse_matmul=${SPARSE_MATMUL} \
    megablox=${MEGABLOX} \
    dtype=bfloat16 \
    weight_dtype=bfloat16 \
    per_device_batch_size=4 \
    steps=5 \
    max_target_length=1024 \
    ici_fsdp_parallelism=1 \
    ici_expert_parallelism=4 \
    gcs_metrics=true \
    abort_on_nan_loss=false

# 2. Run Decoding on the newly produced SFT checkpoint
python3 -m maxtext.inference.decode \
    base_output_directory=${BASE_OUTPUT_PATH} \
    run_name=decode_sft \
    model_name=${MODEL_NAME} \
    tokenizer_path=${TOKENIZER_PATH} \
    load_parameters_path=${BASE_OUTPUT_PATH}/sft/${run_id}/checkpoints/4/items \
    scan_layers=True \
    attention=dot_product \
    sparse_matmul=${SPARSE_MATMUL} \
    megablox=${MEGABLOX} \
    prompt="I love to" \
    ici_tensor_parallelism=4
