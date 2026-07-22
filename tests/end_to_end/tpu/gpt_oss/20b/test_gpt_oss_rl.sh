#!/bin/bash

# Validates the GPTOSS-20B Reinforcement Learning (RL) pipeline using GRPO.

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
export ATTENTION="flash"
export VLLM_ADDITIONAL_CONFIG='{"maxtext_config": {"model_name": "gpt-oss-20b", "log_config": "false"}}'

# 1. Run GRPO Reinforcement Learning
python3 -m maxtext.trainers.post_train.rl.train_rl \
    base_output_directory=${BASE_OUTPUT_PATH}/rl \
    load_parameters_path=${SCANNED_CKPT_PATH} \
    run_name=${run_id} \
    rl.loss_algo='grpo' \
    scan_layers=true \
    num_batches=5 \
    batch_size=1 \
    num_test_batches=5 \
    model_name=${MODEL_NAME} \
    checkpoint_storage_use_zarr3=False \
    checkpoint_storage_use_ocdbt=False \
    rollout_tensor_parallelism=1 \
    attention=${ATTENTION} \
    sparse_matmul=${SPARSE_MATMUL} \
    megablox=${MEGABLOX} \
    vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}' \
    vllm_additional_config="${VLLM_ADDITIONAL_CONFIG}"

# 2. Run Verification Decoding on the newly produced actor checkpoint
python3 -m maxtext.inference.decode \
    base_output_directory=${BASE_OUTPUT_PATH} \
    run_name=decode_rl \
    model_name=${MODEL_NAME} \
    tokenizer_path=${TOKENIZER_PATH} \
    load_parameters_path=${BASE_OUTPUT_PATH}/rl/${run_id}/checkpoints/actor/4/items \
    scan_layers=True \
    attention=dot_product \
    sparse_matmul=${SPARSE_MATMUL} \
    megablox=${MEGABLOX} \
    prompt="I love to" \
    ici_tensor_parallelism=4
