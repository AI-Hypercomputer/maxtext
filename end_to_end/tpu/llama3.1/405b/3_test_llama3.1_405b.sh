#!/bin/bash

# This file tests the quantization of the Llama3.1-405b checkpoint, and assumes an unscanned checkpoint already exists.

set -ex

# We install torch CPU because the checkpoint conversion script does not need a TPU/GPU
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# This is defined in 2_test_llama3.1_405b.sh
export MODEL_VARIATION='llama3.1-405b'
export UNSCANNED_CHECKPOINT=gs://maxtext-llama/llama3.1_405b_bf16/unscanned/0/items

# Non-Googlers please remember to point `SAVE_QUANT_PARAMS_PATH` to the GCS bucket where you want to save your quantized checkpoint
export SAVE_QUANT_PARAMS_PATH=gs://maxtext-llama/llama3.1_405b_int8

export QUANTIZE_TYPE="int8"

JAX_PLATFORMS=cpu python3 -m MaxText.load_and_quantize_checkpoint \
    MaxText/configs/base.yml \
    tokenizer_path=assets/tokenizer_llama3.tiktoken \
    tokenizer_type=tiktoken \
    load_parameters_path=${UNSCANNED_CHECKPOINT} \
    max_prefill_predict_length=1024 \
    max_target_length=2048 \
    model_name=${MODEL_VARIATION} \
    ici_fsdp_parallelism=1 \
    ici_autoregressive_parallelism=1 \
    ici_tensor_parallelism=-1 \
    scan_layers=false \
    weight_dtype=bfloat16 \
    per_device_batch_size=1 \
    attention=dot_product \
    quantization=${QUANTIZE_TYPE} \
    save_quantized_params_path=${SAVE_QUANT_PARAMS_PATH} \
    async_checkpointing=false
