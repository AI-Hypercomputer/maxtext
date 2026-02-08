#!/bin/bash

:'
# This script demonstrates a full end-to-end workflow for Supervised Fine-Tuning (SFT)
# a pre-trained model using MaxText. The script fine-tunes a pre-trained model using
# the `sft_trainer` on the `HuggingFaceH4/ultrachat_200k` dataset and produces a
# fine-tuned model in the Hugging Face format.
#
# This script supports two scenarios:
#
# 1. **Run SFT on a Hugging Face Checkpoint**
#    The script will automatically convert a Hugging Face checkpoint
#    to a MaxText checkpoint, run SFT, and then convert the fine-tuned
#    checkpoint back to the Hugging Face format.
#
# 2. **Run SFT on a MaxText Checkpoint**
#    The script will run SFT on a pre-converted MaxText checkpoint and
#    then convert the fine-tuned checkpoint back to the Hugging Face format.
#
# --- Common environment variables for both the scenarios ---
# export HF_TOKEN=<Hugging Face access token>
#
# # Set the model name and its corresponding chat tokenizer
# export PRE_TRAINED_MODEL=gemma2-2b
# export PRE_TRAINED_MODEL_TOKENIZER=google/gemma-2-2b-it
#
# # Output directory to store run logs
# export BASE_OUTPUT_DIRECTORY=<output directory>
#
# # Number of fine-tuning steps to run
# export STEPS=100
# export PER_DEVICE_BATCH_SIZE=1
#
# --- Scenario 1: Run SFT on a Hugging Face Checkpoint ---
# PRE_TRAINED_MODEL_CKPT_PATH should be unset for this scenario
# bash tests/end_to_end/tpu/run_sft.sh
#
# --- Scenario 2: Run SFT on a MaxText Checkpoint ---
# Set the GCS path to the pre-converted MaxText checkpoint
# export PRE_TRAINED_MODEL_CKPT_PATH=<gcs path for model checkpoint>
# bash tests/end_to_end/tpu/run_sft.sh
'

set -xe

RUN_NAME=$(date +%Y-%m-%d-%H-%M-%S)

# Convert the Hugging Face checkpoint to MaxText format if PRE_TRAINED_MODEL_CKPT_PATH is not set
if [ -z "${PRE_TRAINED_MODEL_CKPT_PATH}" ]; then
  echo "PRE_TRAINED_MODEL_CKPT_PATH is not set. Converting Hugging Face checkpoint to MaxText format."
  CONVERTED_CKPT_DIR=${BASE_OUTPUT_DIRECTORY}/${PRE_TRAINED_MODEL}/${RUN_NAME}/maxtext-checkpoint
  python3 -m MaxText.utils.ckpt_conversion.to_maxtext "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"//base.yml \
      model_name=${PRE_TRAINED_MODEL} \
      hf_access_token=${HF_TOKEN} \
      base_output_directory=${CONVERTED_CKPT_DIR} \
      scan_layers=True
  export PRE_TRAINED_MODEL_CKPT_PATH=${CONVERTED_CKPT_DIR}/0/items
fi
echo "Running fine-tuning on checkpoint: ${PRE_TRAINED_MODEL_CKPT_PATH}"

# Run Supervised Fine-Tuning on MaxText checkpoint using HuggingFaceH4/ultrachat_200k dataset
python3 -m MaxText.sft_trainer "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"//sft.yml \
    run_name=${RUN_NAME} base_output_directory=${BASE_OUTPUT_DIRECTORY}/${PRE_TRAINED_MODEL} \
    model_name=${PRE_TRAINED_MODEL} load_parameters_path=${PRE_TRAINED_MODEL_CKPT_PATH} \
    hf_access_token=$HF_TOKEN tokenizer_path=${PRE_TRAINED_MODEL_TOKENIZER} \
    per_device_batch_size=${PER_DEVICE_BATCH_SIZE} steps=${STEPS}

# Get the latest fine-tuned model checkpoint
CHECKPOINTS_PATH=${BASE_OUTPUT_DIRECTORY}/${PRE_TRAINED_MODEL}/${RUN_NAME}/checkpoints
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
echo "Fine-tuned model checkpoint: ${FINE_TUNED_MODEL_CKPT_PATH}"

# Convert the fine-tuned MaxText checkpoint to Hugging Face checkpoint
export LOCAL_PATH=./tmp/hf/${PRE_TRAINED_MODEL}/${RUN_NAME}
python3 -m MaxText.utils.ckpt_conversion.to_huggingface "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"//base.yml \
    model_name=${PRE_TRAINED_MODEL} \
    hf_access_token=${HF_TOKEN} \
    load_parameters_path=${FINE_TUNED_MODEL_CKPT_PATH} \
    base_output_directory=${LOCAL_PATH} \
    scan_layers=False
echo "Converted Hugging Face checkpoint saved to: ${LOCAL_PATH}"
