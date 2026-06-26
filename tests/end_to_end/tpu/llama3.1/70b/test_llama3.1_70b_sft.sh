#!/bin/bash

# Validates the Llama3.1-70b SFT pipeline using a pre-converted MaxText checkpoint.

# The flow of this script is as follows:
# 1. Run inference on the pre-converted checkpoint.
# 2. Run SFT starting from the pre-converted checkpoint.
# 3. Run inference on the checkpoint produced by the SFT run.

# Usage:
# export HF_TOKEN=<your Hugging Face access token>
# export RUN_ID=$(date +%Y-%m-%d-%H-%M-%S)
# bash test_llama3.1_70b_to_mt.sh $RUN_ID
# bash test_llama3.1_70b_sft.sh $RUN_ID


set -ex

run_id=${1:-$(date +%Y-%m-%d-%H-%M-%S)}
use_pathways=${2:-false}
MODEL_NAME='llama3.1-70b'

# Non-Googlers please remember to point `BASE_OUTPUT_DIRECTORY` to the GCS paths where you have the scanned and unscanned checkpoints stored
BASE_OUTPUT_DIRECTORY=gs://runner-maxtext-logs/${MODEL_NAME}
UNSCANNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/to_maxtext/unscanned/${run_id}/0/items
SCANNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/to_maxtext/scanned/${run_id}/0/items

# Step 1: Run inference on the original checkpoint converted from Hugging Face
python3 -m maxtext.inference.vllm_decode \
    model_name=${MODEL_NAME} \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    tokenizer_path='meta-llama/Llama-3.1-70B-Instruct' \
    vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}' \
    hbm_utilization_vllm=0.85 \
    prompt="Suggest some famous landmarks in London." \
    use_chat_template=True scan_layers=false enable_single_controller=${use_pathways} \
    ici_tensor_parallelism=8

# Step 2: Run SFT on the converted checkpoint
python3 -m maxtext.trainers.post_train.sft.train_sft \
    base_output_directory=${BASE_OUTPUT_DIRECTORY}/sft \
    load_parameters_path=${SCANNED_CKPT_PATH} \
    per_device_batch_size=1 run_name=${run_id} \
    steps=5 scan_layers=true \
    model_name=${MODEL_NAME} tokenizer_path='meta-llama/Llama-3.1-70B-Instruct' \
    enable_single_controller=${use_pathways} \
    checkpoint_storage_use_zarr3=False checkpoint_storage_use_ocdbt=False

# Step 3: Run inference on the checkpoint generated from the previous run
python3 -m maxtext.inference.vllm_decode \
    model_name=${MODEL_NAME} \
    load_parameters_path=${BASE_OUTPUT_DIRECTORY}/sft/${run_id}/checkpoints/5/model_params \
    tokenizer_path='meta-llama/Llama-3.1-70B-Instruct' \
    vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}' \
    hbm_utilization_vllm=0.85 \
    prompt="Suggest some famous landmarks in London." \
    use_chat_template=True scan_layers=true enable_single_controller=${use_pathways} \
    ici_tensor_parallelism=8


