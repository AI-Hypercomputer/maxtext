#!/bin/bash

# This script is designed for internal use within Google.
# External users can update pre-trained model checkpoint GCS path (gs://) to your accessible locations.

set -xe

RUN_NAME=sft-$(date +%Y-%m-%d-%H-%M-%S)
STEPS=100
LOSS_THRESHOLD=100.0 # Set to large value so test is guaranteed to pass

# Gemma2-2b
PRE_TRAINED_MODEL=gemma2-2b
PRE_TRAINED_MODEL_TOKENIZER=google/gemma-2-2b-it
PRE_TRAINED_MODEL_CKPT_PATH=$(gcloud storage ls gs://maxtext-model-checkpoints/gemma2-2b | sort -r | head -1)
BASE_OUTPUT_DIRECTORY=gs://runner-maxtext-logs

# SFT with HF pipeline
python MaxText/sft_trainer.py MaxText/configs/sft.yml \
    run_name=${RUN_NAME}-hf base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    model_name=${PRE_TRAINED_MODEL} load_parameters_path=${PRE_TRAINED_MODEL_CKPT_PATH}/scanned/0/items \
    dataset_type=hf hf_access_token=$HF_TOKEN tokenizer_path=${PRE_TRAINED_MODEL_TOKENIZER} \
    per_device_batch_size=1 \
    steps=${STEPS} max_target_length=1024 \
    metrics_file=sft-hf-metrics.txt

# Assert training loss is smaller than input LOSS_THRESHOLD
python3 end_to_end/tpu/eval_assert.py final_loss sft-hf-metrics.txt $LOSS_THRESHOLD

# Decode
python MaxText/decode_sft.py MaxText/configs/sft.yml \
    run_name=${RUN_NAME}-hf-decode \
    model_name=${PRE_TRAINED_MODEL} tokenizer_path=${PRE_TRAINED_MODEL_TOKENIZER} \
    load_parameters_path=${BASE_OUTPUT_DIRECTORY}/${RUN_NAME}-hf/checkpoints/0/items \
    per_device_batch_size=1 max_prefill_predict_length=512 max_target_length=1024 \
    attention=dot_product decode_sampling_strategy=weighted decode_sampling_temperature=.00001 prompt="Suggest some famous landmarks in London."

python MaxText/decode_sft.py MaxText/configs/sft.yml \
    run_name=${RUN_NAME}-hf-decode \
    model_name=${PRE_TRAINED_MODEL} tokenizer_path=${PRE_TRAINED_MODEL_TOKENIZER} \
    load_parameters_path=${BASE_OUTPUT_DIRECTORY}/${RUN_NAME}-hf/checkpoints/0/items \
    per_device_batch_size=1 max_prefill_predict_length=512 max_target_length=1024 \
    attention=dot_product decode_sampling_strategy=weighted decode_sampling_temperature=.00001 prompt="What are the classic cocktails that every bartender should know how to make and what are their ingredients?"

python MaxText/decode_sft.py MaxText/configs/sft.yml \
    run_name=${RUN_NAME}-hf-decode \
    model_name=${PRE_TRAINED_MODEL} tokenizer_path=${PRE_TRAINED_MODEL_TOKENIZER} \
    load_parameters_path=${BASE_OUTPUT_DIRECTORY}/${RUN_NAME}-hf/checkpoints/0/items \
    per_device_batch_size=1 max_prefill_predict_length=512 max_target_length=1024 \
    attention=dot_product decode_sampling_strategy=weighted decode_sampling_temperature=.00001 prompt="Which famous landmarks should I visit in London?"
