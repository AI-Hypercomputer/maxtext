#!/bin/bash

# Validates the Gemma3-4B SFT pipeline using a pre-converted MaxText checkpoint.

# The flow of this script is as follows:
# 1. Run inference on the pre-converted checkpoint.
# 2. Run SFT starting from the pre-converted checkpoint.
# 3. Run inference on the checkpoint produced by the SFT run.
# 4. Convert the checkpoint produced by the SFT run back to HuggingFace format.

# Usage:
# export HF_TOKEN=<your Hugging Face access token>
# export RUN_ID=$(date +%Y-%m-%d-%H-%M)
# bash test_gemma3_to_mt.sh $RUN_ID
# bash test_gemma3_sft.sh $RUN_ID


set -ex

run_id=${1:-$(date +%Y-%m-%d-%H-%M)}
MODEL_NAME='gemma3-4b'

# Non-Googlers please remember to point `BASE_OUTPUT_DIRECTORY` to the GCS paths where you have the scanned and unscanned checkpoints stored
BASE_OUTPUT_DIRECTORY=gs://runner-maxtext-logs/${MODEL_NAME}
UNSCANNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/to_maxtext/unscanned/${run_id}/0/items
SCANNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/to_maxtext/scanned/${run_id}/0/items

# Step 1: Install torch
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# Step 2: Run inference on the original checkpoint converted from Hugging Face
python3 -m maxtext.inference.vllm_decode \
    model_name=${MODEL_NAME} \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}' \
    hbm_utilization_vllm=0.5 \
    prompt="Suggest some famous landmarks in London." \
    use_chat_template=True scan_layers=false

# Step 3: Run SFT on the converted checkpoint
python3 -m maxtext.trainers.post_train.sft.train_sft \
    base_output_directory=${BASE_OUTPUT_DIRECTORY}/sft \
    load_parameters_path=${SCANNED_CKPT_PATH} \
    per_device_batch_size=1 run_name=${run_id} \
    steps=5 scan_layers=true \
    model_name=${MODEL_NAME} enable_single_controller=True \
    checkpoint_storage_use_zarr3=False checkpoint_storage_use_ocdbt=False

# Step 4: Run inference on the checkpoint generated from the previous run
python3 -m maxtext.inference.vllm_decode \
    model_name=${MODEL_NAME} \
    load_parameters_path=${BASE_OUTPUT_DIRECTORY}/sft/${run_id}/checkpoints/5/model_params \
    vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}' \
    hbm_utilization_vllm=0.5 \
    prompt="Suggest some famous landmarks in London." \
    use_chat_template=True scan_layers=true

# Step 5: Convert the checkpoint from MaxText format to Hugging Face format
python3 -m maxtext.checkpoint_conversion.to_huggingface \
    model_name=${MODEL_NAME} \
    load_parameters_path=${BASE_OUTPUT_DIRECTORY}/sft/${run_id}/checkpoints/5/model_params \
    base_output_directory=${BASE_OUTPUT_DIRECTORY}/to_huggingface/unscanned/${run_id} \
    use_multimodal=false scan_layers=true
