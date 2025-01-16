#!/bin/bash

# This file is both an integration test that runs once a day on a v4-8 and documentation for how to get started with Gemma2-2b.

# The flow of this file is as follows:
# 1. Convert the checkpoint downloaded from Kaggle to make it compatible with MaxText
# 2. Run decoding, finetuning of Gemma2-2B with the converted checkpoint. Also, run pretraining of Gemma2-2B
# 3. Convert the scanned checkpoint from step 1 into unscanned checkpoint format and run more efficient decoding.
# 4. Run decoding from the finetuned checkpoint from step 2


set -ex

if [ -z "${RUN_ID}" ]; then
    echo "Please set the RUN_ID used to create checkpoint from 1st script in this folder"
fi

# Installing torch for deps in forward_pass_logit_chekcker.py
pip install torch --index-url https://download.pytorch.org/whl/cpu

export MODEL='gemma2-2b'
export BASE_OUTPUT_DIRECTORY=gs://runner-maxtext-logs
export CKPT_BUCKET=gs://maxtext-model-checkpoints
export DATASET_PATH=gs://maxtext-dataset
export ASYNC_CHECKPOINTING=false
export UNSCANNED_CHECKPOINT=${CKPT_BUCKET}/${MODEL}/${RUN_ID}/unscanned/checkpoints/0/items
export SCANNED_CHECKPOINT=${CKPT_BUCKET}/${MODEL}/${RUN_ID}/scanned/0/items
export HF_CHECKPOINT=${CKPT_BUCKET}/${MODEL}/${RUN_ID}/huggingface

# We run decoding on the `UNSCANNED_CKPT_PATH` for efficient decoding on the unscanned version of the checkpoint. Note that this checkpoint only has parameters and no optimizer state. 
# So, we use it by specifying`load_parameters_path=${UNSCANNED_CHECKPOINT}`
python MaxText/decode.py MaxText/configs/base.yml tokenizer_path=assets/tokenizer.gemma load_parameters_path=${UNSCANNED_CHECKPOINT} per_device_batch_size=1 run_name=runner_$(date +%Y-%m-%d-%H-%M) max_prefill_predict_length=8 max_target_length=16 dataset_type=synthetic steps=10 async_checkpointing=false scan_layers=false model_name=${MODEL} attention=dot_product prompt="I love to"

# We can also run decoding (albeit in a bit unoptimized way) by using the scanned converted checkpoint located at `SCANNED_CHECKPOINT`. Note again that this checkpoint only has parameters and no optimizer state. So, we use it by specifying`load_parameters_path=${CONVERTED_CHECKPOINT}`
python MaxText/decode.py MaxText/configs/base.yml tokenizer_path=assets/tokenizer.gemma load_parameters_path=${SCANNED_CHECKPOINT} per_device_batch_size=1 run_name=runner_$(date +%Y-%m-%d-%H-%M) max_prefill_predict_length=8 max_target_length=16 dataset_type=synthetic steps=10 async_checkpointing=false model_name=${MODEL} attention=dot_product prompt="I love to"

# Alternatively, we skip to running finetuning by using the scanned converted checkpoint located at `CONVERTED_CHECKPOINT`. Again, we use it by specifying`load_parameters_path=${CONVERTED_CHECKPOINT}`. Note that scanned checkpoint helps with efficient finetuning
export FINETUNE_RUN_NAME=runner_finetune_${idx}
python MaxText/train.py MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_DIRECTORY} dataset_path=${DATASET_PATH} tokenizer_path=assets/tokenizer.gemma load_parameters_path=${CONVERTED_CHECKPOINT} per_device_batch_size=1 run_name=${FINETUNE_RUN_NAME} max_target_length=8192 steps=10 async_checkpointing=false model_name=${MODEL} checkpoint_period=5

# We also run pre-training, this is similar to the finetuning command except we don't pass any checkpoint directory to load parameters from
python MaxText/train.py MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_DIRECTORY} dataset_path=${DATASET_PATH} tokenizer_path=assets/tokenizer.gemma per_device_batch_size=1 run_name=runner_pretrain_${idx} max_target_length=8192 steps=5 enable_checkpointing=false model_name=${MODEL}

# Note that the finetune run checkpoint generates the `full state` which has both parameters and optimizer state. For decoding, we only need to use the parameters. 
# So, we can use the `MaxText/generate_param_only_checkpoint.py` to convert the full state checkpoint into a parameter only checkpoint for more efficient memory use. Note that the path provided to the flag `load_full_state_path` is the path to the checkpoint subdirectory inside the `BASE_OUTPUT_DIRECTORY` from our previous finetuning run.
# `force_unroll=true` is converting the output parameter only checkpoint into an unscanned format for efficient decoding
export PARAM_RUN_NAME=param_chkpt_${idx}
python MaxText/generate_param_only_checkpoint.py MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_DIRECTORY} load_full_state_path=${BASE_OUTPUT_DIRECTORY}/${FINETUNE_RUN_NAME}/checkpoints/5/items run_name=${PARAM_RUN_NAME} model_name=${MODEL} force_unroll=true

# Now, run decoding on the checkpoint generated from our finetune run.
python MaxText/decode.py MaxText/configs/base.yml tokenizer_path=assets/tokenizer.gemma load_parameters_path=${BASE_OUTPUT_DIRECTORY}/${PARAM_RUN_NAME}/checkpoints/0/items per_device_batch_size=1 run_name=runner_$(date +%Y-%m-%d-%H-%M) max_prefill_predict_length=8 max_target_length=16 dataset_type=synthetic steps=10 async_checkpointing=false scan_layers=false model_name=${MODEL} attention=dot_product prompt="I love to"

# We also test whether the forward pass logits match the golden logits for Gemma2-2b
python3 MaxText/tests/forward_pass_logit_checker.py  MaxText/configs/base.yml tokenizer_path=assets/tokenizer.gemma load_parameters_path=${UNSCANNED_CKPT_PATH} run_name=forward_pass_test_gemma2b per_device_batch_size=1 model_name=${MODEL} max_prefill_predict_length=4 max_target_length=4 dataset_type=synthetic scan_layers=false attention=dot_product --max_kl_div=0.01

# Copy converted MaxText converted Huggingface checkpoint
gcloud storage cp -r ${HF_CHECKPOINT} /tmp

# Test whether the forward pass logits match the golden logits for Huggingface checkpoint converted from MaxText orbax checkpoint
python3 MaxText/tests/forward_pass_logit_checker.py MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_DIRECTORY} run_name=forward_pass_test_hf ici_tensor_parallelism=4 model_name=${MODEL} max_prefill_predict_length=4 max_target_length=4 dataset_type=synthetic dtype=float32 activations_in_float32=true matmul_precision=float32 async_checkpointing=${ASYNC_CHECKPOINTING} scan_layers=false --vocab_size=256000 --hf_model_path=/tmp/huggingface --max_kl_div=1e-2
