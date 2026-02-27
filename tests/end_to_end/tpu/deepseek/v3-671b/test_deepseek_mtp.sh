#!/bin/bash

# This file is documentation for how to get started with DeepSeek v3 MTP on v5p-256.

# The flow of this file is as follows:
# 1. Convert the checkpoint downloaded from HuggingFace to a MaxText-compatible format, including MTP layers.
# 2. Run fine-tuning with MTP enabled.

set -ex
idx=$(date +%Y-%m-%d-%H-%M)
export MODEL_NAME='deepseek3-671b'
export TOKENIZER_PATH='deepseek-ai/DeepSeek-V3'

# Step 1: Convert the HuggingFace Checkpoint with MTP Enabled
# Note: Non-Googlers should use their own GCS buckets.
# The --enable_mtp flag is crucial for processing the MTP-specific layers.
export CHKPT_BUCKET=gs://maxtext-deepseek/deepseek3-671b/hf
export MODEL_BUCKET=gs://maxtext-deepseek/deepseek3-671b
JAX_PLATFORMS=cpu python3 -m maxtext.checkpoint_conversion.standalone_scripts.convert_deepseek_family_ckpt \
    --base_model_path ${CHKPT_BUCKET} \
    --maxtext_model_path ${MODEL_BUCKET}/${idx} \
    --model_size ${MODEL_NAME} \
    --enable_mtp

# Step 2: Set up Checkpoint Path
# We will use the 'scanned' checkpoint directly for fine-tuning.
export CONVERTED_CHECKPOINT=${MODEL_BUCKET}/${idx}/0/items

# Step 3: Run Training
# Note: Non-Googlers should point DATASET_PATH and BASE_OUTPUT_DIRECTORY to their own GCS buckets.
export DATASET_PATH=gs://maxtext-dataset
export BASE_OUTPUT_DIRECTORY=gs://runner-maxtext-logs

# Run fine-tuning with MTP enabled
# We add `mtp_num_layers=1` and `mtp_loss_scaling_factor=0.1` to activate the MTP block.
python3 -m maxtext.trainers.pre_train.train "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"//base.yml \
    base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    dataset_path=${DATASET_PATH} \
    load_parameters_path=${CONVERTED_CHECKPOINT} \
    run_name=mtp_fine_tuning_${idx} \
    per_device_batch_size=4 \
    model_name=${MODEL_NAME} \
    ici_fsdp_parallelism=128 \
    steps=10000 \
    max_target_length=2048 \
    tokenizer_type=huggingface \
    tokenizer_path=${TOKENIZER_PATH} \
    attention=flash \
    dtype=bfloat16 \
    weight_dtype=bfloat16 \
    enable_checkpointing=true \
    mtp_num_layers=1 \
    mtp_loss_scaling_factor=0.1
