#!/bin/bash

:'

# This script demonstrates a full end-to-end workflow for Supervised Fine-Tuning (SFT)
# a pre-trained model using MaxText. The script fine-tunes a pre-trained model using 
# the `sft_trainer` on the `HuggingFaceH4/ultrachat_200k` dataset.

# Commands to run fine-tuning:
  export HF_TOKEN=<Hugging Face access token>

  export PRE_TRAINED_MODEL=llama3.1-8b

  # Chat tokenizer for the model
  export PRE_TRAINED_MODEL_TOKENIZER=meta-llama/Llama-3.1-8B-Instruct

  # MaxText-compatible model checkpoint
  export PRE_TRAINED_MODEL_CKPT_PATH=<gcs path for model checkpoint>

  # Output directory to store run logs
  export BASE_OUTPUT_DIRECTORY=<output directory>

  # Number of fine-tuning steps to run
  export STEPS=100

  bash end_to_end/tpu/run_sft.sh
'

set -xe

RUN_NAME=sft-$(date +%Y-%m-%d-%H-%M-%S)
PER_DEVICE_BATCH_SIZE=1

python3 -m MaxText.sft_trainer MaxText/configs/sft.yml \
    run_name=${RUN_NAME} base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    model_name=${PRE_TRAINED_MODEL} load_parameters_path=${PRE_TRAINED_MODEL_CKPT_PATH} \
    hf_access_token=$HF_TOKEN tokenizer_path=${PRE_TRAINED_MODEL_TOKENIZER} \
    per_device_batch_size=${PER_DEVICE_BATCH_SIZE} steps=${STEPS}

# Get the latest fine-tuned model checkpoint
CHECKPOINTS_PATH=${BASE_OUTPUT_DIRECTORY}/${RUN_NAME}-hf/checkpoints
checkpoints=$(gcloud storage ls $CHECKPOINTS_PATH)
integer_dirs=()
for dir in $checkpoints; do
  dir_name=$(basename "$dir")
  if [[ "$dir_name" =~ ^[0-9]+$ ]]; then
    integer_dirs+=("$dir_name")
  fi
done
sorted_dirs=($(printf '%s\n' "${integer_dirs[@]}" | sort -n))
largest_dir="${sorted_dirs[-1]}"
FINE_TUNED_MODEL_CKPT_PATH=${CHECKPOINTS_PATH}/${largest_dir}/items

echo "Fine-tuned model checkpoint: " $FINE_TUNED_MODEL_CKPT_PATH
