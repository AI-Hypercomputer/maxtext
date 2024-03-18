#!/bin/bash

# This script is designed for internal use within Google. External users can adapt it by:
#  - Updating GCS paths (gs://) to your accessible locations.
#  - Using the checkpoint generated from train.py or available one in open source (https://llama.meta.com/llama-downloads/).

set -ex
idx=$(date +%Y-%m-%d-%H-%M)

export M_ENABLE_CHECKPOINTING=true
export M_BASE_OUTPUT_DIRECTORY=gs://runner-maxtext-logs
export M_DATASET_PATH=gs://maxtext-dataset
export M_ASYNC_CHECKPOINTING=false

# converted_checkpoint is generated from test_llama2_13b_1_convert.sh Step 1 & 2
export _idx=2024-03-19-00-40
export converted_checkpoint=gs://maxtext-llama/test/${_idx}/decode-ckpt-maxtext/0/items

# Step 4: Generate unrolled parameter checkpoint from Llama2's converted checkpoint
export PARAMETER_CHECKPOINT_RUN=generate_param_only_checkpoint_original_${idx}
python3 MaxText/generate_param_only_checkpoint.py MaxText/configs/base.yml load_parameters_path=${converted_checkpoint} run_name=${PARAMETER_CHECKPOINT_RUN} model_name='llama2-13b' force_unroll=true

# Step 5: Load unrolled parameter checkpoint into decode.py and verify result
unset M_LOAD_PARAMETERS_PATH
python3 MaxText/decode.py MaxText/configs/base.yml load_parameters_path=${M_BASE_OUTPUT_DIRECTORY}/${PARAMETER_CHECKPOINT_RUN}/checkpoints/0/items run_name=runner_direct_${idx} per_device_batch_size=1 model_name='llama2-13b' ici_tensor_parallelism=4 max_prefill_predict_length=4  max_target_length=16 prompt="I love to" autoregressive_decode_assert="read. I love to write. I love to teach." attention=dot_product scan_layers=false

# Step 6: Finetuning
python3 MaxText/train.py MaxText/configs/base.yml load_parameters_path=${converted_checkpoint} run_name=runner_finetuning_${idx}  per_device_batch_size=1 model_name='llama2-13b' ici_tensor_parallelism=4 steps=10 max_target_length=1024 per_device_batch_size=1 checkpoint_period=5
export finetuned_checkpoint=${M_BASE_OUTPUT_DIRECTORY}/runner_finetuning_${idx}/checkpoints/5/items
echo "finetuned_checkpoint: ${finetuned_checkpoint}"

# Step 7: Pre-training
python3 MaxText/train.py MaxText/configs/base.yml run_name=runner_pretraining_${idx}  per_device_batch_size=1 model_name='llama2-13b' ici_tensor_parallelism=4 steps=10 max_target_length=1024 per_device_batch_size=1

# Step 8: Generate unrolled parameter checkpoint from Llama2's "fine-tuning" run
export PARAMETER_CHECKPOINT_RUN=generate_param_only_checkpoint_finetuned_${idx}
python3 MaxText/generate_param_only_checkpoint.py MaxText/configs/base.yml load_full_state_path=${finetuned_checkpoint} run_name=${PARAMETER_CHECKPOINT_RUN} model_name='llama2-13b' force_unroll=true

# Step 9: Load fine-tuned parameter checkpoint into decode.py
unset M_LOAD_PARAMETERS_PATH
python3 MaxText/decode.py MaxText/configs/base.yml load_parameters_path=${M_BASE_OUTPUT_DIRECTORY}/${PARAMETER_CHECKPOINT_RUN}/checkpoints/0/items run_name=runner_decode_finetuned_${idx} per_device_batch_size=1 model_name='llama2-13b' ici_tensor_parallelism=4 max_prefill_predict_length=4  max_target_length=16 prompt="I love to" attention=dot_product scan_layers=false
