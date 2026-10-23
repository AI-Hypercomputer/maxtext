#!/bin/bash

# Validates the Llama3.1-8b 2-node multi-host GPU pre-training pipeline using a pre-converted MaxText checkpoint.

# The flow of this script is as follows:
# 1. Run 5 steps of 2-node (multi-host) GPU pre-training starting from the pre-converted checkpoint and save checkpoints to GCS.

# Usage:
# export HF_TOKEN=<your Hugging Face access token>
# export RUN_ID=$(date +%Y-%m-%d-%H-%M-%S)
# bash test_llama3.1_8b_to_mt.sh $RUN_ID
# Multi-host execution requires JAX coordinator environment variables (JAX_COORDINATOR_IP,
# JAX_COORDINATOR_PORT, NNODES=2, NODE_RANK) or launcher integration (e.g. Slurm/srun).
# srun bash test_llama3.1_8b_2node.sh $RUN_ID


set -ex

# Avoid conflict between M_RUN_NAME env var and CLI run_name argument in pyconfig
unset M_RUN_NAME

# Disable one-shot AllReduce kernel to avoid cross-host symmetric memory registration failure on non-MNNVL slices
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_unsupported_use_all_reduce_one_shot_kernel=false"

run_id=${1:?"Error: run_id (argument 1) is required"}
train_run_name=${2:-${run_id}_2node}
MODEL_NAME='llama3.1-8b'

# Non-Googlers please remember to point `BASE_OUTPUT_DIRECTORY` to the GCS paths where you have the scanned and unscanned checkpoints stored
BASE_OUTPUT_DIRECTORY="${BASE_OUTPUT_DIRECTORY:-gs://runner-maxtext-logs/${MODEL_NAME}}"
UNSCANNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/to_maxtext/unscanned/${run_id}/0/items

# Non-Googlers please remember to point `DATASET_PATH` to the GCS bucket where you have your training data
DATASET_PATH="${DATASET_PATH:-gs://maxtext-dataset}"

# Step 1: Run 5 steps of 2-node GPU pre-training on the converted checkpoint and save checkpoints to GCS
python3 -m maxtext.trainers.pre_train.train \
    base_output_directory=${BASE_OUTPUT_DIRECTORY}/train \
    dataset_type=grain \
    grain_file_type=tfrecord \
    dataset_path=${DATASET_PATH} tokenizer_type="huggingface" \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    per_device_batch_size=1 run_name=${train_run_name} \
    max_target_length=2048 steps=5 async_checkpointing=false \
    checkpoint_storage_use_zarr3=False checkpoint_storage_use_ocdbt=False \
    model_name=${MODEL_NAME} hardware=gpu attention=cudnn_flash_te \
    max_segments_per_seq=32 \
    scan_layers=false
