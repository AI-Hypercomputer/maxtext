#!/bin/bash

# Validates the Qwen3 SFT pipeline using a pre-converted MaxText checkpoint.


# The flow of this script is as follows:
# 1. Run inference on the pre-converted checkpoint.
# 2. Run SFT starting from the pre-converted checkpoint.
# 3. Run inference on the checkpoint produced by the SFT run.

# Usage:
# export HF_TOKEN=<your Hugging Face access token>
# export RUN_ID=$(date +%Y-%m-%d-%H-%M-%S)
# bash test_qwen3_to_mt.sh $RUN_ID
# bash test_qwen3_sft.sh $RUN_ID


set -ex
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export NEW_MODEL_DESIGN=1
run_id=${1:-$(date +%Y-%m-%d-%H-%M-%S)}
use_pathways=${2:-false}
MODEL_NAME='qwen3-30b-a3b-base'

BASE_OUTPUT_DIRECTORY=gs://runner-maxtext-logs/${MODEL_NAME}
UNSCANNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/to_maxtext/unscanned/${run_id}/0/items
SCANNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/to_maxtext/scanned/${run_id}/0/items

python3 -m maxtext.inference.vllm_decode \
    model_name=${MODEL_NAME} \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}' \
    hbm_utilization_vllm=0.85 \
    use_chat_template=True scan_layers=false enable_single_controller=${use_pathways} \
    ici_tensor_parallelism=4

python3 -m maxtext.trainers.post_train.sft.train_sft \
    base_output_directory=${BASE_OUTPUT_DIRECTORY}/sft \
    load_parameters_path=${SCANNED_CKPT_PATH} \
    per_device_batch_size=0.125 run_name=${run_id} \
    steps=5 scan_layers=true \
    model_name=${MODEL_NAME} enable_single_controller=${use_pathways} \
    checkpoint_storage_use_zarr3=False checkpoint_storage_use_ocdbt=False \
    remat_policy=full use_tunix_gradient_accumulation=False \
    lora.enable_lora=true lora.lora_rank=8 \
    ici_tensor_parallelism=1 ici_fsdp_parallelism=8 ici_expert_parallelism=1 max_target_length=64 weight_dtype=bfloat16 dtype=bfloat16 opt_type=sgd

python3 -m maxtext.inference.vllm_decode \
    model_name=${MODEL_NAME} \
    load_parameters_path=${BASE_OUTPUT_DIRECTORY}/sft/${run_id}/checkpoints/5/model_params \
    vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}' \
    hbm_utilization_vllm=0.85 \
    use_chat_template=True scan_layers=true enable_single_controller=${use_pathways} \
    ici_tensor_parallelism=4
