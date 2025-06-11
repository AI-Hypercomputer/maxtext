#!/bin/bash

# Run inference on a converted Qwen3 MoE checkpoint.
set -ex
idx=$(date +%Y-%m-%d)

python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

if [ -z "${BASE_OUTPUT_PATH}" ]; then
    export BASE_OUTPUT_PATH=gs://runner-maxtext-logs/$(date +%Y-%m-%d)/
fi

if [ -z "${MODEL_VARIATION}" ]; then
    export MODEL_VARIATION="qwen3-moe"
fi

export TOKENIZER_PATH=gs://maxtext-qwen3/${MODEL_VARIATION}/hf-checkpoint/
export MODEL_BUCKET=gs://maxtext-qwen3/${MODEL_VARIATION}
export UNSCANNED_CKPT_PATH=${MODEL_BUCKET}/${idx}/unscanned/0/items

python3 -m MaxText.tests.forward_pass_logit_checker MaxText/configs/base.yml tokenizer_path=${TOKENIZER_PATH} load_parameters_path=${UNSCANNED_CKPT_PATH} run_name=forward_pass_test_${MODEL_VARIATION} attention=dot_product per_device_batch_size=1 model_name=${MODEL_VARIATION} max_prefill_predict_length=4 max_target_length=4 scan_layers=false --atol=0.5 --rtol=0.5 async_checkpointing=false sparse_matmul=false weight_dtype=float32 dtype=float32

