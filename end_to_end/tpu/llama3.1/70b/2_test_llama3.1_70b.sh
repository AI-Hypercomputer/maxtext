#!/bin/bash

# This file is both an integration test that runs once a day on a v4-128 and documentation for how to get started with Llama3.1-70b. 
# Please make sure you have run end_to_end/tpu/llama3.1/70b/1_test_llama3.1_70b.sh before running commands from this file. 

# The flow of this file is as follows:
# 1. Run decoding, finetuning of Llama3.1-70B with the converted checkpoint obtained from end_to_end/tpu/llama3.1/70b/1_test_llama3.1_70b.sh. Also, run pretraining of Llama3.1-70B
# 2. Convert the scanned checkpoint from step 1 into unscanned checkpoint format and run more efficient decoding.
# 3. Run decoding from the finetuned checkpoint from step 1


# Example Usage: export BASE_OUTPUT_PATH=/path/to/GCS/bucket; bash end_to_end/tpu/llama3.1/70b/2_test_llama3.1_70b.sh
# Use the same BASE_OUTPUT_PATH as end_to_end/tpu/llama3.1/70b/1_test_llama3.1_70b.sh
# Please note that in these two scripts (1_test_llama3.1_70b.sh and 2_test_llama3.1_70b.sh) BASE_OUTPUT_PATH is assumed to be already a unique path across multiple runs and 
# the subfolders names aka RUN_NAMEs are static. Please remember to change BASE_OUTPUT_PATH across different runs.

set -ex

if [ -z "${RUN_ID}" ]; then
    echo "Please set the RUN_ID used to create checkpoint from 1st script in this folder"
fi

# Installing torch for deps in forward_pass_logit_chekcker.py
pip install torch --index-url https://download.pytorch.org/whl/cpu

export MODEL='llama3.1-70b'
export BASE_OUTPUT_DIRECTORY=gs://runner-maxtext-logs
export CKPT_BUCKET=gs://maxtext-model-checkpoints
export DATASET_PATH=gs://maxtext-dataset
export ASYNC_CHECKPOINTING=false
export UNSCANNED_CHECKPOINT=${CKPT_BUCKET}/${MODEL}/${RUN_ID}/unscanned/checkpoints/0/items
export SCANNED_CHECKPOINT=${CKPT_BUCKET}/${MODEL}/${RUN_ID}/scanned/0/items
export HF_CHECKPOINT=${CKPT_BUCKET}/${MODEL}/${RUN_ID}/huggingface

# We run decoding on the `UNSCANNED_CKPT_PATH` for efficient decoding on the unscanned version of the checkpoint. Note that this checkpoint only has parameters and no optimizer state. 
# So, we use it by specifying`load_parameters_path=${UNSCANNED_CHECKPOINT}`
python MaxText/decode.py MaxText/configs/base.yml tokenizer_path=assets/tokenizer_llama3.tiktoken load_parameters_path=${UNSCANNED_CHECKPOINT} per_device_batch_size=1 run_name=runner_$(date +%Y-%m-%d-%H-%M) max_prefill_predict_length=4 max_target_length=16 dataset_type=synthetic async_checkpointing=false scan_layers=false model_name=${MODEL} attention=dot_product prompt="I love to"

# We can also run decoding (albeit in a bit unoptimized way) by using the scanned converted checkpoint located at `CONVERTED_CHECKPOINT`. Note again that this checkpoint only has parameters and no optimizer state. So, we use it by specifying`load_parameters_path=${CONVERTED_CHECKPOINT}`
python MaxText/decode.py MaxText/configs/base.yml tokenizer_path=assets/tokenizer_llama3.tiktoken load_parameters_path=${SCANNED_CHECKPOINT} per_device_batch_size=1 run_name=runner_$(date +%Y-%m-%d-%H-%M) max_prefill_predict_length=4 max_target_length=16 dataset_type=synthetic async_checkpointing=false model_name=${MODEL} attention=dot_product prompt="I love to"

# Alternatively, we skip to running finetuning by using the scanned converted checkpoint located at `SCANNED_CHECKPOINT`. Again, we use it by specifying`load_parameters_path=${SCANNED_CHECKPOINT}`. Note that scanned checkpoint helps with efficient finetuning
export FINETUNE_RUN_NAME=runner_finetune
python MaxText/train.py MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_DIRECTORY} dataset_path=${DATASET_PATH} tokenizer_path=assets/tokenizer_llama3.tiktoken load_parameters_path=${SCANNED_CHECKPOINT} per_device_batch_size=1 run_name=${FINETUNE_RUN_NAME} steps=10 async_checkpointing=false model_name=${MODEL} checkpoint_period=5

# We also test whether the forward pass logits match the golden logits for LLama3.1-8B
python3 MaxText/tests/forward_pass_logit_checker.py MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_DIRECTORY} tokenizer_path=assets/tokenizer_llama3.tiktoken load_parameters_path=${UNSCANNED_CHECKPOINT} run_name=forward_pass_test per_device_batch_size=1 model_name=${MODEL} max_prefill_predict_length=4 max_target_length=4 dataset_type=synthetic dtype=float32 activations_in_float32=true matmul_precision=float32 async_checkpointing=false scan_layers=false  --max_kl_div=1e-4

# TODO(b/391634569): converting to HF checkpoint OOMs
# Copy converted MaxText converted Huggingface checkpoint
# gcloud storage cp -r ${HF_CHECKPOINT} /tmp

# # Test whether the forward pass logits match the golden logits for Huggingface checkpoint converted from MaxText orbax checkpoint
# python3 MaxText/tests/forward_pass_logit_checker.py MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_DIRECTORY} tokenizer_path=assets/tokenizer_llama3.tiktoken load_parameters_path=${UNSCANNED_CHECKPOINT} run_name=forward_pass_test_hf per_device_batch_size=1 model_name=${MODEL} max_prefill_predict_length=3 max_target_length=4 dataset_type=synthetic dtype=float32 activations_in_float32=true matmul_precision=float32 async_checkpointing=false scan_layers=false --hf_model_path=/tmp/huggingface --max_kl_div=1e-4
