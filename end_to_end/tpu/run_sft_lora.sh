#!/bin/bash

:'
# This script demonstrates a full end-to-end workflow for Supervised Fine-Tuning (SFT)
# using LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning with MaxText.
# The script fine-tunes Llama 3.1 8B using the `sft_trainer` with LoRA enabled
# on the `HuggingFaceH4/ultrachat_200k` dataset.
#
# LoRA Benefits:
# - Reduces trainable parameters significantly (typically <1% of original model)
# - Faster training and lower memory usage
# - Optional QLoRA support with NF4 quantization for even more memory efficiency
#
# This script supports two scenarios:
#
# 1. **Run LoRA SFT on a Hugging Face Checkpoint**
#    The script will automatically convert a Hugging Face checkpoint
#    to a MaxText checkpoint, run LoRA SFT, and then convert the fine-tuned
#    checkpoint back to the Hugging Face format.
#
# 2. **Run LoRA SFT on a MaxText Checkpoint**
#    The script will run LoRA SFT on a pre-converted MaxText checkpoint and
#    then convert the fine-tuned checkpoint back to the Hugging Face format.
#
# --- Common environment variables for both the scenarios ---
# export HF_TOKEN=<Hugging Face access token>
#
# # Output directory to store run logs
# export BASE_OUTPUT_DIRECTORY=<output directory>
#
# # Number of fine-tuning steps to run
# export STEPS=100
# export PER_DEVICE_BATCH_SIZE=1
#
# # LoRA-specific parameters (optional, defaults provided)
# export LORA_RANK=8              # Rank of LoRA decomposition (default: 8)
# export LORA_ALPHA=16            # LoRA scaling factor (default: 16)
# export QUANTIZE_LORA=False      # Enable QLoRA with NF4 quantization (default: False)
#
# --- Scenario 1: Run LoRA SFT on a Hugging Face Checkpoint ---
# PRE_TRAINED_MODEL_CKPT_PATH should be unset for this scenario
# bash end_to_end/tpu/llama3.1/8b/run_sft_lora.sh
#
# --- Scenario 2: Run LoRA SFT on a MaxText Checkpoint ---
# Set the GCS path to the pre-converted MaxText checkpoint
# export PRE_TRAINED_MODEL_CKPT_PATH=<gcs path for model checkpoint>
# bash end_to_end/tpu/llama3.1/8b/run_sft_lora.sh
'

set -xe

RUN_NAME=$(date +%Y-%m-%d-%H-%M-%S)
PRE_TRAINED_MODEL=llama3.1-8b
PRE_TRAINED_MODEL_TOKENIZER=meta-llama/Llama-3.1-8B-Instruct

# Set default LoRA parameters if not provided
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
QUANTIZE_LORA=${QUANTIZE_LORA:-False}


# Convert the Hugging Face checkpoint to MaxText format if PRE_TRAINED_MODEL_CKPT_PATH is not set
if [ -z "${PRE_TRAINED_MODEL_CKPT_PATH}" ]; then
  echo "PRE_TRAINED_MODEL_CKPT_PATH is not set. Converting Hugging Face checkpoint to MaxText format."
  CONVERTED_CKPT_DIR=${BASE_OUTPUT_DIRECTORY}/${PRE_TRAINED_MODEL}/${RUN_NAME}/maxtext-checkpoint
  python3 -m MaxText.utils.ckpt_conversion.to_maxtext "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}"/configs/base.yml \
      model_name=${PRE_TRAINED_MODEL} \
      hf_access_token=${HF_TOKEN} \
      base_output_directory=${CONVERTED_CKPT_DIR} \
      scan_layers=True
  export PRE_TRAINED_MODEL_CKPT_PATH=${CONVERTED_CKPT_DIR}/0/items
fi
echo "Running LoRA fine-tuning on checkpoint: ${PRE_TRAINED_MODEL_CKPT_PATH}"

# Run Supervised Fine-Tuning with LoRA on MaxText checkpoint using HuggingFaceH4/ultrachat_200k dataset
python3 -m MaxText.sft.sft_trainer "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}"/configs/sft.yml \
    run_name=${RUN_NAME} base_output_directory=${BASE_OUTPUT_DIRECTORY}/${PRE_TRAINED_MODEL} \
    model_name=${PRE_TRAINED_MODEL} load_parameters_path=${PRE_TRAINED_MODEL_CKPT_PATH} \
    hf_access_token=$HF_TOKEN tokenizer_path=${PRE_TRAINED_MODEL_TOKENIZER} \
    per_device_batch_size=${PER_DEVICE_BATCH_SIZE} steps=${STEPS} \
    profiler=xplane max_target_length=1024 weight_dtype=bfloat16 \
    use_lora=True lora_rank=${LORA_RANK} lora_alpha=${LORA_ALPHA} quantize_lora=${QUANTIZE_LORA}

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
FINE_TUNED_MODEL_CKPT_PATH=${CHECKPOINTS_PATH}/${largest_dir}
echo "Fine-tuned LoRA model checkpoint: ${FINE_TUNED_MODEL_CKPT_PATH}"
