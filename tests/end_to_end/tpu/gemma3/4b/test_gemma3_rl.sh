#!/bin/bash

# Validates the Gemma3-4B RL pipeline using a pre-converted MaxText checkpoint.

# The flow of this script is as follows:
# 1. Run inference on the pre-converted checkpoint.
# 2. Run RL starting from the pre-converted checkpoint.
# 3. Run inference on the checkpoint produced by the RL run.

# Usage:
# export HF_TOKEN=<your Hugging Face access token>
# export RUN_ID=$(date +%Y-%m-%d-%H-%M-%S)
# bash test_gemma3_to_mt.sh $RUN_ID
# bash test_gemma3_rl.sh $RUN_ID


set -ex

export VLLM_WORKER_MULTIPROC_METHOD='spawn'
export MODEL_IMPL_TYPE='flax_nnx'
export GRPC_ENABLE_FORK_SUPPORT='0'
export JAX_RANDOM_WEIGHTS='1'
export VLLM_ENABLE_V1_MULTIPROCESSING='0'
export SKIP_JAX_PRECOMPILE='1'
export NEW_MODEL_DESIGN='0'

run_id=${1:-$(date +%Y-%m-%d-%H-%M-%S)}
use_pathways=${2:-false}
MODEL_NAME='gemma3-4b'

# Non-Googlers please remember to point `BASE_OUTPUT_DIRECTORY` to the GCS paths where you have the scanned and unscanned checkpoints stored
BASE_OUTPUT_DIRECTORY=gs://runner-maxtext-logs/${MODEL_NAME}
UNSCANNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/to_maxtext/unscanned/${run_id}/0/items
SCANNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/to_maxtext/scanned/${run_id}/0/items

# Step 1: Run inference on the original checkpoint converted from Hugging Face
python3 -m maxtext.inference.vllm_decode \
    model_name=${MODEL_NAME} \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}' \
    vllm_additional_config='{"maxtext_config": {"model_name": "gemma3-4b", "log_config": "false"}}' \
    hbm_utilization_vllm=0.5 \
    prompt='Suggest some famous landmarks in London.' \
    use_chat_template=True scan_layers=false enable_single_controller=${use_pathways}

# Step 2: Run RL on the converted checkpoint
python3 -m maxtext.trainers.post_train.rl.train_rl \
    base_output_directory=${BASE_OUTPUT_DIRECTORY}/rl \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    run_name=${run_id} rl.loss_algo='grpo' scan_layers=false \
    num_batches=2 batch_size=1 num_test_batches=2 \
    model_name=${MODEL_NAME} enable_single_controller=${use_pathways} \
    checkpoint_storage_use_zarr3=False checkpoint_storage_use_ocdbt=False \
    chips_per_vm=4 \
    rollout_tensor_parallelism=4 \
    vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}' \
    vllm_additional_config='{"maxtext_config": {"model_name": "gemma3-4b", "log_config": "false"}}' \
    hbm_utilization_vllm=0.5


# Step 3: Run inference on the checkpoint generated from the previous run
python3 -m maxtext.inference.vllm_decode \
    model_name=${MODEL_NAME} \
    load_parameters_path=${BASE_OUTPUT_DIRECTORY}/rl/${run_id}/checkpoints/actor/2/model_params \
    vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}' \
    vllm_additional_config='{"maxtext_config": {"model_name": "gemma3-4b", "log_config": "false"}}' \
    hbm_utilization_vllm=0.5 \
    prompt='Suggest some famous landmarks in London.' \
    use_chat_template=True scan_layers=false enable_single_controller=${use_pathways}
