#!/bin/bash

# This script is designed for internal use within Google.
# External users can update pre-trained model checkpoint GCS path (gs://) to your accessible locations.

set -xe

RUN_NAME=sft-$(date +%Y-%m-%d-%H-%M-%S)

PRE_TRAINED_MODEL=gemma2-2b
PRE_TRAINED_MODEL_TOKENIZER=google/gemma-2-2b-it
PRE_TRAINED_MODEL_CKPT_PATH=gs://maxtext-model-checkpoints/gemma2-2b/2025-01-29-19-26/scanned
BASE_OUTPUT_DIRECTORY=gs://runner-maxtext-logs

# HF pipeline
python MaxText/sft_trainer.py MaxText/configs/sft.yml \
    run_name=${RUN_NAME}-hf base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    model_name=${PRE_TRAINED_MODEL} load_parameters_path=${PRE_TRAINED_MODEL_CKPT_PATH}/0/items \
    dataset_type=hf hf_access_token=$HF_TOKEN tokenizer_path=${PRE_TRAINED_MODEL_TOKENIZER} \
    per_device_batch_size=0.5 allow_split_physical_axes=True \
    ici_data_parallelism=2 ici_tensor_parallelism=2 ici_fsdp_parallelism=1 \
    steps=10 max_target_length=1024
