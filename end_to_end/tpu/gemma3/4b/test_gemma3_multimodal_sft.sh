#!/bin/bash

# This file contains an end-to-end Airflow nightly test, designed to run once a day on a v5p-8, along with documentation to guide users in getting started with Gemma3-4B.

# The flow of this file is as follows:
# 1. Convert the checkpoint downloaded from Hugging Face to make it compatible with MaxText
# 2. Run multimodal decoding of Gemma3-4B, with the converted checkpoint.
# 3. Run supervised finetuning (SFT) of Gemma3-4B on ChartQA dataset with the converted checkpoint.
# 4. Run decoding from the finetuned checkpoint from step 3, seeing the short answer from SFT.
# 5. Convert the SFT checkpoint back to HuggingFace format.

# Note: You can stop at any step if you just want to run part of the flow.

set -ex
idx=$(date +%Y-%m-%d-%H-%M)
MODEL_NAME='gemma3-4b'
export MODEL_VARIATION='4b'
HF_TOKEN='' # Important!!! Save your hf access token here
HF_GOLDEN_MODEL='google/gemma-3-4b-pt'
TOKENIZER_PATH="${MAXTEXT_ASSETS_ROOT:-${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText/assets}}"'/tokenizer.gemma3'
# To convert the multimodal model, make sure the use_multimodal is set to be true
USE_MULTIMODAL=true
SCAN_LAYERS=false
SFT_STEPS=10

# Installing torch for deps in forward_pass_logit_checker.py
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# After downloading checkpoints, copy them to GCS bucket at $MODEL_BUCKET \
# Non-Googlers please remember to point these variables to GCS buckets that you own, this script uses internal buckets for testing.
export MODEL_BUCKET=gs://maxtext-gemma/unified/gemma3

# 1. Convert the HuggingFace checkpoint to MaxText unscanned ckpt:
python3 -m MaxText.utils.ckpt_conversion.to_maxtext "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}/"configs/base.yml \
    model_name=${MODEL_NAME} \
    hf_access_token=${HF_TOKEN} \
    base_output_directory=${MODEL_BUCKET}/${MODEL_VARIATION}/unscanned/${idx} \
    use_multimodal=${USE_MULTIMODAL} \
    scan_layers=${SCAN_LAYERS}

# 2. Decode the converted checkpoint to make sure it works
export UNSCANNED_CKPT_PATH=${MODEL_BUCKET}/${MODEL_VARIATION}/unscanned/${idx}/0/items
python3 -m MaxText.decode "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}/"configs/base.yml model_name=${MODEL_NAME} tokenizer_path=${TOKENIZER_PATH} load_parameters_path=${UNSCANNED_CKPT_PATH} per_device_batch_size=1 run_name=ht_test max_prefill_predict_length=272 max_target_length=300 steps=1 async_checkpointing=false scan_layers=$SCAN_LAYERS use_multimodal=${USE_MULTIMODAL} prompt=\'Describe\ image\ \<start_of_image\>\' image_path=\'src/MaxText/test_assets/test_image.jpg\' attention=\'dot_product\'

# 3. SFT the MaxText converted checkpoint on ChartQA dataset
export BASE_OUTPUT_DIRECTORY=${MODEL_BUCKET}/${MODEL_VARIATION}/unscanned/sft
python -m MaxText.sft_trainer "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}/"configs/sft-vision-chartqa.yml \
    run_name=$idx \
    model_name=$MODEL_NAME tokenizer_path="google/gemma-3-4b-pt" \
    per_device_batch_size=1 \
    max_prefill_predict_length=1024 max_target_length=2048 \
    steps=$SFT_STEPS \
    scan_layers=$SCAN_LAYERS async_checkpointing=False \
    attention=dot_product \
    dataset_type=hf hf_path=parquet hf_access_token=$HF_TOKEN \
    hf_train_files=gs://aireenmei-multipod/dataset/hf/chartqa/train-* \
    base_output_directory=$BASE_OUTPUT_DIRECTORY \
    load_parameters_path=$UNSCANNED_CKPT_PATH \
    dtype=bfloat16 weight_dtype=bfloat16 sharding_tolerance=0.05

# 4. Decode from the finetuned checkpoint from step 3
export FINAL_CKPT_STEP=$((SFT_STEPS - 1))
export FINETUNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/${idx}/checkpoints/${FINAL_CKPT_STEP}/items
python3 -m MaxText.decode "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}/"configs/base.yml model_name=${MODEL_NAME} tokenizer_path=${TOKENIZER_PATH} load_parameters_path=${FINETUNED_CKPT_PATH} per_device_batch_size=1 run_name=ht_test max_prefill_predict_length=272 max_target_length=300 steps=1 async_checkpointing=false scan_layers=$SCAN_LAYERS use_multimodal=${USE_MULTIMODAL} prompt=\'Describe\ image\ \<start_of_image\>\' image_path=\'src/MaxText/test_assets/test_image.jpg\' attention=\'dot_product\'

# 5. Convert the SFT checkpoint back to HuggingFace format.
export LOCAL_PATH=./tmp/hf/${MODEL_NAME}/${idx}
export CKPT_PATH="gs://maxtext-gemma/unified/gemma3/4b/unscanned/sft/2025-08-08-18-28/2025-08-08-18-28/checkpoints/9/items"
python3 -m MaxText.utils.ckpt_conversion.to_huggingface "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}/"configs/base.yml \
    model_name=${MODEL_NAME} \
    hf_access_token=${HF_TOKEN} \
    load_parameters_path=${CKPT_PATH} \
    base_output_directory=${LOCAL_PATH} \
    use_multimodal=${USE_MULTIMODAL} \
    scan_layers=$SCAN_LAYERS
