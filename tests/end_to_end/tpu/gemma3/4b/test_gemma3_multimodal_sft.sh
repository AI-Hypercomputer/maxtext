#!/bin/bash

# Validates the Gemma3-4B SFT multimodal pipeline using a pre-converted MaxText checkpoint.

# The flow of this script is as follows:
# 1. Run inference on the pre-converted checkpoint.
# 2. Run SFT of Gemma3-4B on ChartQA dataset with the converted checkpoint.
# 3. Run inference on the checkpoint produced by the SFT run.

# Usage:
# export HF_TOKEN=<your Hugging Face access token>
# export RUN_ID=$(date +%Y-%m-%d-%H-%M-%S)
# bash test_gemma3_to_mt.sh $RUN_ID true
# bash test_gemma3_multimodal_sft.sh $RUN_ID

# Note: You can stop at any step if you just want to run part of the flow.

set -ex

run_id=${1:-$(date +%Y-%m-%d-%H-%M-%S)}
MODEL_NAME='gemma3-4b'

# Non-Googlers please remember to point `BASE_OUTPUT_DIRECTORY` to the GCS paths where you have the scanned and unscanned checkpoints stored
BASE_OUTPUT_DIRECTORY=gs://runner-maxtext-logs/${MODEL_NAME}
UNSCANNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/to_maxtext/unscanned/${run_id}/0/items
SCANNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/to_maxtext/scanned/${run_id}/0/items

# Step 1: Install google-jetstream
python3 -m pip install google-jetstream@https://github.com/AI-Hypercomputer/JetStream/archive/29329e8e73820993f77cfc8efe34eb2a73f5de98.zip --no-deps

# Step 2: Run inference on the original checkpoint converted from Hugging Face
python3 -m maxtext.inference.decode \
    model_name=${MODEL_NAME} \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    per_device_batch_size=1 \
    run_name=${run_id} \
    max_prefill_predict_length=272 \
    max_target_length=300 \
    steps=1 \
    async_checkpointing=false \
    scan_layers=false \
    use_multimodal=True \
    tokenizer_type=huggingface \
    prompt=\'Describe\ image\ \<start_of_image\>\' \
    image_path=\'tests/assets/test_image.jpg\' \
    attention=\'dot_product\' skip_jax_distributed_system=True

# Step 3: Run SFT on the MaxText checkpoint on ChartQA dataset
python -m maxtext.trainers.post_train.sft.train_sft_native "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"/post_train/sft-vision-chartqa.yml \
    run_name=${run_id} \
    model_name=${MODEL_NAME} \
    per_device_batch_size=1 \
    max_prefill_predict_length=1024 max_target_length=2048 \
    steps=5 \
    scan_layers=false async_checkpointing=False \
    attention=dot_product \
    dataset_type=hf hf_path=parquet \
    hf_train_files=gs://aireenmei-multipod/dataset/hf/chartqa/train-* \
    base_output_directory=${BASE_OUTPUT_DIRECTORY}/multimodal/sft \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    dtype=bfloat16 weight_dtype=bfloat16 sharding_tolerance=0.05 \
    checkpoint_storage_use_zarr3=False checkpoint_storage_use_ocdbt=False

# Step 4: Run inference on the checkpoint generated from the previous run
python3 -m maxtext.inference.decode \
    model_name=${MODEL_NAME} \
    load_parameters_path=${BASE_OUTPUT_DIRECTORY}/multimodal/sft/${run_id}/checkpoints/4/items \
    per_device_batch_size=1 \
    run_name=${run_id} \
    max_prefill_predict_length=272 \
    max_target_length=300 \
    steps=1 \
    async_checkpointing=false \
    scan_layers=false \
    use_multimodal=true \
    tokenizer_type=huggingface \
    prompt=\'Describe\ image\ \<start_of_image\>\' \
    image_path=\'tests/assets/test_image.jpg\' \
    attention=\'dot_product\'

