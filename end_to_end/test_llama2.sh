#!/bin/bash
set -e
idx=$(date +%Y-%m-%d-%H-%M)

export base_ckpt_path=gs://maxtext-llama/test/2024-01-15-06-49/decode-ckpt-maxtext/0/default
export M_ENABLE_CHECKPOINTING=true
export M_BASE_OUTPUT_DIRECTORY=gs://runner-maxtext-logs
export M_DATASET_PATH=gs://maxtext-dataset
export M_ASYNC_CHECKPOINTING=false

#TODO(internal bug -- migrate to XLML)
#pip install torch
#gsutil cp -r gs://maxtext-llama/llama2-7b/meta-ckpt /tmp/
#python3 MaxText/llama_or_mistral_ckpt.py --base-model-path /tmp/meta-ckpt --model-size llama2-7b --maxtext-model-path gs://maxtext-llama/test/${idx}/decode-ckpt-maxtext/

# Load after directly from parameter checkpoint
python3 MaxText/decode.py MaxText/configs/base.yml load_parameters_path=${base_ckpt_path} run_name=runner_direct_${idx} per_device_batch_size=1 model_name='llama2-7b' ici_tensor_parallelism=4 max_prefill_predict_length=4  max_target_length=16 prompt="I love to" autoregressive_decode_assert="read. I love to write. I love to share." attention=dot_product
#TODO(Training with Llama is not complete)
python3 MaxText/train.py MaxText/configs/base.yml load_parameters_path=${base_ckpt_path} run_name=runner_${idx}  per_device_batch_size=1 model_name='llama2-7b' ici_tensor_parallelism=4 steps=10 max_target_length=1024 per_device_batch_size=1 attention=dot_product

# generate parameter checkpoint from Llama's "fine-tuning" run
unset M_LOAD_PARAMETERS_PATH
export PARAMETER_CHECKPOINT_RUN=generate_param_only_checkpoint_${idx}
python3 MaxText/generate_param_only_checkpoint.py MaxText/configs/base.yml load_full_state_path=${M_BASE_OUTPUT_DIRECTORY}/runner_${idx}/checkpoints/0/default run_name=${PARAMETER_CHECKPOINT_RUN} model_name='llama2-7b' force_unroll=true

export new_ckpt_path=${M_BASE_OUTPUT_DIRECTORY}/${PARAMETER_CHECKPOINT_RUN}/checkpoints/0/default

# Load fine-tuned parameter checkpoint into decode.py
python3 MaxText/decode.py MaxText/configs/base.yml load_parameters_path=${new_ckpt_path} run_name=runner_direct_${idx} per_device_batch_size=1 model_name='llama2-7b' ici_tensor_parallelism=4 max_prefill_predict_length=4  max_target_length=16 prompt="I love to" autoregressive_decode_assert="read. I love to write. I love to share." attention=dot_product scan_layers=false