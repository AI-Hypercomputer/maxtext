#!/bin/bash

# Validates the GPT-OSS-20B LoRA pipeline using a pre-converted MaxText checkpoint.

# The flow of this script is as follows:
# 1. Run LoRA starting from the pre-converted checkpoint.
# 2. Run inference on the checkpoint produced by the LoRA run.
# 3. Convert the checkpoint produced by the LoRA run back to HuggingFace format.

# Usage:
# export HF_TOKEN=<your Hugging Face access token>
# export RUN_ID=$(date +%Y-%m-%d-%H-%M)
# bash test_gpt_oss_to_mt.sh $RUN_ID
# bash test_gpt_oss_lora.sh $RUN_ID

set -ex

run_id=${1:-$(date +%Y-%m-%d-%H-%M-%S)}
export MODEL_NAME='gpt-oss-20b'
export TOKENIZER_PATH='openai/gpt-oss-20b'

export BASE_OUTPUT_PATH=gs://runner-maxtext-logs/gpt-oss-20b
SCANNED_CKPT_PATH=${BASE_OUTPUT_PATH}/scanned/${run_id}/0/items

SPARSE_MATMUL="True"
MEGABLOX="True"
LORA_ATTENTION="flash"

# Step 1: Run LoRA SFT on the converted checkpoint
python3 -m maxtext.trainers.post_train.sft.train_sft \
    base_output_directory=${BASE_OUTPUT_PATH}/lora \
    run_name=${run_id} \
    model_name=${MODEL_NAME} \
    tokenizer_path=${TOKENIZER_PATH} \
    dataset_name=lucasmccabe-lmt/math_alpaca \
    dataset_type=hf \
    enable_checkpointing=true \
    async_checkpointing=false \
    load_parameters_path=${SCANNED_CKPT_PATH} \
    scan_layers=True \
    attention=${LORA_ATTENTION} \
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
    abort_on_nan_loss=false \
    lora.enable_lora=True \
    lora.lora_rank=16 \
    lora.lora_alpha=32.0 \
    enable_nnx=True \
    pure_nnx_decoder=True \
    enable_single_controller=True \
    checkpoint_storage_use_zarr3=False checkpoint_storage_use_ocdbt=False

# Step 2: Run inference decoding on the checkpoint generated from the previous run
python3 -m maxtext.inference.decode \
    base_output_directory=${BASE_OUTPUT_PATH} \
    run_name=decode_lora \
    model_name=${MODEL_NAME} \
    tokenizer_path=${TOKENIZER_PATH} \
    load_parameters_path=${SCANNED_CKPT_PATH} \
    lora.enable_lora=True \
    lora.lora_restore_path=${BASE_OUTPUT_PATH}/lora/${run_id}/checkpoints/4/items \
    lora.lora_rank=16 \
    lora.lora_alpha=32.0 \
    scan_layers=True \
    attention=dot_product \
    sparse_matmul=${SPARSE_MATMUL} \
    megablox=${MEGABLOX} \
    prompt="I love to" \
    ici_tensor_parallelism=4

# Step 3: Convert the checkpoint from MaxText format to Hugging Face format
python3 -m maxtext.checkpoint_conversion.to_huggingface \
    model_name=${MODEL_NAME} \
    load_parameters_path=${SCANNED_CKPT_PATH} \
    lora.lora_restore_path=${BASE_OUTPUT_PATH}/lora/${run_id}/checkpoints/4/items \
    base_output_directory=${BASE_OUTPUT_PATH}/to_huggingface/unscanned/${run_id} \
    scan_layers=true \
    enable_nnx=True \
    pure_nnx_decoder=True
