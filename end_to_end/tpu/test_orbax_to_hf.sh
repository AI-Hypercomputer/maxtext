#!/bin/bash

# This script is to test the flow of MaxText/llama_mistral_mixtral_orbax_to_hf.py.
# Steps in the script:
# 1. compare logits of MaxText orbax ckpt and HF:meta-llama/Llama-3.1-8B ckpt
# 2. Convert MaxText orbax ckpt to HF using MaxText/llama_mistral_mixtral_orbax_to_hf.py
# 3. Confirm the logits match for MaxText orbax ckpt and the new HF ckpt created in step 2.
set -ex
export MODEL_VARIATION='llama3.1-8b'

export BASE_OUTPUT_PATH=gs://runner-maxtext-logs/2024-12-18-17-35

export CONVERTED_CHECKPOINT=${BASE_OUTPUT_PATH}/${MODEL_VARIATION}/scanned_chkpt/0/items
export RUN_NAME=unscann_llama3.1
# We defined path to unscanned checkpoint created in 1_test_llama3.1_8b.sh
export UNSCANNED_CKPT_PATH=${BASE_OUTPUT_PATH}/${RUN_NAME}/checkpoints/0/items

# python3 MaxText/tests/forward_pass_logit_checker.py MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_PATH} \
#     tokenizer_path=assets/tokenizer_llama3.tiktoken load_parameters_path=${UNSCANNED_CKPT_PATH} \
#     run_name=forward_pass_test per_device_batch_size=1 model_name=${MODEL_VARIATION} max_prefill_predict_length=3 max_target_length=4 \
#     dataset_type=synthetic dtype=float32 activations_in_float32=true matmul_precision=float32 async_checkpointing=false \
#     scan_layers=false

# converting MaxText orbax ckpt to HF

# JAX_PLATFORMS=cpu python3 MaxText/llama_mistral_mixtral_orbax_to_hf.py MaxText/configs/base.yml base_output_directory=gs://runner-maxtext-logs \
#     load_parameters_path=gs://runner-maxtext-logs/2024-12-18-17-35/llama3.1-8b/scanned_chkpt/0/items run_name=convert_to_hf \
#     model_name=llama3.1-8b hf_model_path=/home/mohitkhatwani/maxtext/hf_llama3.1_new/

# comparing logits of the HF ckpt above

python3 MaxText/tests/forward_pass_logit_checker.py MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_PATH} \
    tokenizer_path=assets/tokenizer_llama3.tiktoken load_parameters_path=${UNSCANNED_CKPT_PATH} \
    run_name=forward_pass_test per_device_batch_size=1 model_name=${MODEL_VARIATION} max_prefill_predict_length=3 max_target_length=4 \
    dataset_type=synthetic dtype=float32 activations_in_float32=true matmul_precision=float32 async_checkpointing=false \
    scan_layers=false --golden_logits_path="MaxText/test_assets/golden_data_new_llama3_1_8b.jsonl" 
