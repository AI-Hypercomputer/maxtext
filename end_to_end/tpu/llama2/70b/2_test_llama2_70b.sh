#!/bin/bash

# This file is both an integration test that runs once a day on a v4-128 and documentation for how to get started with Llama2-70b. 
# Please make sure you have run end_to_end/tpu/llama2/70b/1_test_llama2_70b.sh before running commands from this file. 

# The flow of this file is as follows:
# 1. Run decoding, finetuning of Llama2-70B with the converted checkpoint obtained from end_to_end/tpu/llama2/70b/1_test_llama2_70b.sh. Also, run pretraining of Llama2-70B
# 2. Run more efficient decoding with the unscanned checkpoint obtained from end_to_end/tpu/llama2/70b/1_test_llama2_70b.sh.
# 3. Run decoding from the finetuned checkpoint from step 1


# Example Usage: export BASE_OUTPUT_PATH=/path/to/GCS/bucket; bash end_to_end/tpu/llama2/70b/2_test_llama2_70b.sh
# Use the same BASE_OUTPUT_PATH as end_to_end/tpu/llama2/70b/1_test_llama2_70b.sh
# Please note that in these two scripts (1_test_llama2_70b.sh and 2_test_llama2_70b.sh) BASE_OUTPUT_PATH is assumed to be already a unique path across multiple runs and 
# the subfolders names aka RUN_NAMEs are static. Please remember to change BASE_OUTPUT_PATH across different runs.

set -ex

if [ -z "${RUN_ID}" ]; then
    echo "Please set the RUN_ID used to create checkpoint from 1st script in this folder"
fi

# Installing torch for deps in forward_pass_logit_chekcker.py
pip install torch --index-url https://download.pytorch.org/whl/cpu

export MODEL='llama2-70b'
export BASE_OUTPUT_PATH=gs://runner-maxtext-logs
export CKPT_BUCKET=gs://maxtext-model-checkpoints
# Non-Googlers please remember to point `DATASET_PATH` to the GCS bucket where you have your training data
export DATASET_PATH=gs://maxtext-dataset
export ASYNC_CHECKPOINTING=true # True so that jax distributed system is initialized
export UNSCANNED_CHECKPOINT=${CKPT_BUCKET}/${MODEL}/${RUN_ID}/unscanned/checkpoints/0/items
export SCANNED_CHECKPOINT=${CKPT_BUCKET}/${MODEL}/${RUN_ID}/scanned/0/items
export HF_CHECKPOINT=${CKPT_BUCKET}/${MODEL}/${RUN_ID}/huggingface

# We run decoding on the `UNSCANNED_CHECKPOINT` for efficient decoding on the unscanned version of the checkpoint. Note that this checkpoint only has parameters and no optimizer state. 
# So, we use it by specifying`load_parameters_path=${CONVERTED_CHECKPOINT}`
python MaxText/decode.py MaxText/configs/base.yml base_output_directory=gs://runner-maxtext-logs tokenizer_path=assets/tokenizer.llama2 load_parameters_path=${UNSCANNED_CHECKPOINT} per_device_batch_size=1 run_name=runner_$(date +%Y-%m-%d-%H-%M) max_prefill_predict_length=4 max_target_length=16 dataset_type=synthetic async_checkpointing=${ASYNC_CHECKPOINTING} scan_layers=false model_name=${MODEL} attention=dot_product prompt="I love to"

# We can also run decoding (albeit in a bit unoptimized way) by using the scanned converted checkpoint located at `SCANNED_CHECKPOINT`. Note again that this checkpoint only has parameters and no optimizer state. So, we use it by specifying`load_parameters_path=${CONVERTED_CHECKPOINT}`
python MaxText/decode.py MaxText/configs/base.yml base_output_directory=gs://runner-maxtext-logs tokenizer_path=assets/tokenizer.llama2 load_parameters_path=${SCANNED_CHECKPOINT} per_device_batch_size=1 run_name=runner_$(date +%Y-%m-%d-%H-%M) max_prefill_predict_length=4 max_target_length=16 dataset_type=synthetic async_checkpointing=${ASYNC_CHECKPOINTING} model_name=${MODEL} attention=dot_product prompt="I love to"

# Alternatively, we skip to running finetuning by using the scanned converted checkpoint located at `CONVERTED_CHECKPOINT`. Again, we use it by specifying`load_parameters_path=${CONVERTED_CHECKPOINT}`. Note that scanned checkpoint helps with efficient finetuning
export FINETUNE_RUN_NAME=runner_finetune_${RUN_ID}
python MaxText/train.py MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_PATH} dataset_path=${DATASET_PATH} tokenizer_path=assets/tokenizer.llama2 load_parameters_path=${SCANNED_CHECKPOINT} per_device_batch_size=1 run_name=${FINETUNE_RUN_NAME} steps=10 async_checkpointing=${ASYNC_CHECKPOINTING} model_name=${MODEL} checkpoint_period=5

# Note that the finetune run checkpoint generates the `full state` which has both parameters and optimizer state. For decoding, we only need to use the parameters. 
# So, we can use the `MaxText/generate_param_only_checkpoint.py` to convert the full state checkpoint into a parameter only checkpoint for more efficient memory use. Note that the path provided to the flag `load_full_state_path` is the path to the checkpoint subdirectory inside the `BASE_OUTPUT_PATH` from our previous finetuning run.
# `force_unroll=true` is converting the output parameter only checkpoint into an unscanned format for efficient decoding
export PARAM_RUN_NAME=param_chkpt
python MaxText/generate_param_only_checkpoint.py MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_PATH} load_full_state_path=${BASE_OUTPUT_PATH}/${FINETUNE_RUN_NAME}/checkpoints/5/items run_name=${PARAM_RUN_NAME} model_name=${MODEL} force_unroll=true

# Now, run decoding on the checkpoint generated from our finetune run.
python MaxText/decode.py MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_PATH} tokenizer_path=assets/tokenizer.llama2 load_parameters_path=${BASE_OUTPUT_PATH}/${PARAM_RUN_NAME}/checkpoints/0/items per_device_batch_size=1 run_name=runner_$(date +%Y-%m-%d-%H-%M) max_prefill_predict_length=4 max_target_length=16 dataset_type=synthetic steps=10 async_checkpointing=${ASYNC_CHECKPOINTING} scan_layers=false model_name=${MODEL} attention=dot_product prompt="I love to"

# We also test whether the forward pass logits match the golden logits for Llama2-70b
python3 MaxText/tests/forward_pass_logit_checker.py --atol=0.2 --rtol=0.2 MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_PATH} load_parameters_path=${UNSCANNED_CHECKPOINT} run_name=forward_pass_test per_device_batch_size=1 model_name=llama2-70b ici_tensor_parallelism=4 max_prefill_predict_length=4 max_target_length=4 dataset_type=synthetic dtype=float32 scan_layers=false async_checkpointing=${ASYNC_CHECKPOINTING}

# Copy converted MaxText converted Huggingface checkpoint
# TODO(b/391634569)
# gcloud storage cp -r ${HF_CHECKPOINT} /tmp

# # Test whether the forward pass logits match the golden logits for Huggingface checkpoint converted from MaxText orbax checkpoint
# python3 MaxText/tests/forward_pass_logit_checker.py MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_PATH} run_name=forward_pass_test_hf ici_tensor_parallelism=4 model_name=${MODEL} max_prefill_predict_length=4 max_target_length=4 dataset_type=synthetic dtype=float32 activations_in_float32=true matmul_precision=float32 async_checkpointing=false scan_layers=false --hf_model_path=/tmp/huggingface --max_kl_div=1e-4
