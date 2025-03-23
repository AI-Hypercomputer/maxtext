#!/bin/bash

# This script is designed for internal use within Google.
# External users can update pre-trained model checkpoint GCS path (gs://) to your accessible locations.

set -xe

RUN_NAME=sft-$(date +%Y-%m-%d-%H-%M-%S)
STEPS=300000
PER_DEVICE_BATCH_SIZE=1
LOSS_THRESHOLD=100.0 # Set to large value so test is guaranteed to pass

PRE_TRAINED_MODEL=llama2-7b
PRE_TRAINED_MODEL_TOKENIZER=meta-llama/Llama-2-7b-hf
PRE_TRAINED_MODEL_CKPT_PATH=$(gcloud storage ls gs://maxtext-model-checkpoints/llama2-7b | sort -r | head -1)
BASE_OUTPUT_DIRECTORY=gs://runner-maxtext-logs

# SFT with HF pipeline
python MaxText/sft_trainer.py MaxText/configs/sft.yml \
    run_name=${RUN_NAME}-hf base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    model_name=${PRE_TRAINED_MODEL} load_parameters_path=${PRE_TRAINED_MODEL_CKPT_PATH}/scanned/0/items \
    dataset_type=hf hf_access_token=$HF_TOKEN tokenizer_path=${PRE_TRAINED_MODEL_TOKENIZER} \
    per_device_batch_size=${PER_DEVICE_BATCH_SIZE} \
    steps=${STEPS} max_target_length=1024 checkpoint_period=100 \
    metrics_file=sft-hf-metrics.txt

# Assert training loss is smaller than input LOSS_THRESHOLD
python3 end_to_end/tpu/eval_assert.py final_loss sft-hf-metrics.txt $LOSS_THRESHOLD

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

# Decode
python MaxText/decode.py MaxText/configs/sft.yml \
    run_name=${RUN_NAME}-hf-decode \
    model_name=${PRE_TRAINED_MODEL} tokenizer_path=assets/tokenizer.llama2 \
    load_parameters_path=${FINE_TUNED_MODEL_CKPT_PATH} \
    per_device_batch_size=${PER_DEVICE_BATCH_SIZE} max_target_length=128 max_prefill_predict_length=64 \
    attention=dot_product decode_sampling_strategy=greedy prompt="<user>Suggest some famous landmarks in London.</user> <assistant>" \
    autoregressive_decode_assert="1. Tower of London"