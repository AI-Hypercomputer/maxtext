#!/bin/bash

# Validates the Qwen3-30B pre-training pipeline using a pre-converted MaxText checkpoint.

# The flow of this script is as follows:
# 1. Run inference on the pre-converted checkpoint.
# 2. Run pre-training starting from the pre-converted checkpoint.
# 3. Run inference on the checkpoint produced by the pre-training run.

# Usage:
# export HF_TOKEN=<your Hugging Face access token>
# export RUN_ID=$(date +%Y-%m-%d-%H-%M-%S)
# bash test_qwen3_to_mt.sh $RUN_ID
# bash test_qwen3.sh $RUN_ID


set -ex

run_id=${1:-$(date +%Y-%m-%d-%H-%M-%S)}
MODEL_NAME='qwen3-30b-a3b-base'


BASE_OUTPUT_DIRECTORY=gs://runner-maxtext-logs/${MODEL_NAME}
UNSCANNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/to_maxtext/unscanned/${run_id}/0/items
SCANNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/to_maxtext/scanned/${run_id}/0/items
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

DATASET_PATH=gs://maxtext-dataset

# Step 1: Run inference on the pre-converted checkpoint
python3 -m maxtext.inference.decode \
    model_name=${MODEL_NAME} tokenizer_type="huggingface" \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    per_device_batch_size=1 run_name=${run_id} \
    max_prefill_predict_length=8 max_target_length=16 steps=1 async_checkpointing=false \
    checkpoint_storage_use_zarr3=False checkpoint_storage_use_ocdbt=False \
    scan_layers=false prompt='I love to' attention=\'dot_product\'

# Step 2: Run pre-training starting from the pre-converted checkpoint
python3 -m maxtext.trainers.pre_train.train \
    base_output_directory=${BASE_OUTPUT_DIRECTORY}/train \
    dataset_path=${DATASET_PATH} tokenizer_type="huggingface" \
    load_parameters_path=${SCANNED_CKPT_PATH} \
    per_device_batch_size=0.125 run_name=${run_id} \
    max_target_length=64 steps=5 async_checkpointing=false \
    checkpoint_storage_use_zarr3=False checkpoint_storage_use_ocdbt=False \
    model_name=${MODEL_NAME} scan_layers=true \
    remat_policy=full \
    ici_tensor_parallelism=4 ici_fsdp_parallelism=2 weight_dtype=bfloat16 dtype=bfloat16 opt_type=sgd optimizer_memory_host_offload=true

# Step 3: Run inference on the checkpoint produced by the pre-training run
python3 -m maxtext.inference.decode \
    model_name=${MODEL_NAME} tokenizer_type="huggingface" \
    load_parameters_path=${BASE_OUTPUT_DIRECTORY}/train/${run_id}/checkpoints/4/items \
    per_device_batch_size=1 run_name=${run_id} \
    max_prefill_predict_length=8 max_target_length=16 steps=4 async_checkpointing=false \
    checkpoint_storage_use_zarr3=False checkpoint_storage_use_ocdbt=False \
    scan_layers=true prompt='I love to' attention=\'dot_product\'
