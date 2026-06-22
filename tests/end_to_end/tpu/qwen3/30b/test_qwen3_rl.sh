#!/bin/bash

# Validates the Qwen3 RL pipeline using a pre-converted MaxText checkpoint.

# The flow of this script is as follows:
# 1. Run inference on the pre-converted checkpoint.
# 2. Run RL starting from the pre-converted checkpoint.
# 3. Run inference on the checkpoint produced by the RL run.

# Usage:
# export HF_TOKEN=<your Hugging Face access token>
# export RUN_ID=$(date +%Y-%m-%d-%H-%M-%S)
# bash test_qwen3_to_mt.sh $RUN_ID
# bash test_qwen3_rl.sh $RUN_ID

set -ex
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export NEW_MODEL_DESIGN=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=1.0
run_id=${1:-$(date +%Y-%m-%d-%H-%M-%S)}
use_pathways=${2:-false}
MODEL_NAME='qwen3-30b-a3b-base'

BASE_OUTPUT_DIRECTORY=gs://runner-maxtext-logs/${MODEL_NAME}
UNSCANNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/to_maxtext/unscanned/${run_id}/0/items
SCANNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/to_maxtext/scanned/${run_id}/0/items

# python3 -m maxtext.inference.vllm_decode \
#     model_name=${MODEL_NAME} \
#     load_parameters_path=${UNSCANNED_CKPT_PATH} \
#     vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}' \
#     hbm_utilization_vllm=0.85 \
#     use_chat_template=True scan_layers=false enable_single_controller=${use_pathways} \
#     ici_tensor_parallelism=4

python3 -m maxtext.trainers.post_train.rl.train_rl \
    "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"/post_train/rl.yml \
    base_output_directory=${BASE_OUTPUT_DIRECTORY}/rl \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    run_name=${run_id}-v12 rl.loss_algo='grpo' scan_layers=false \
    num_batches=5 batch_size=1 num_test_batches=5 \
    model_name=${MODEL_NAME} enable_single_controller=${use_pathways} \
    checkpoint_storage_use_zarr3=False checkpoint_storage_use_ocdbt=False \
    rollout_tensor_parallelism=1 rollout_expert_parallelism=8 \
    vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}' \
    vllm_additional_config='{"maxtext_config": {"model_name": "'${MODEL_NAME}'", "allow_split_physical_axes": true, "scan_layers": false}}' \
    remat_policy=full hbm_utilization_vllm=0.75 use_pathways=false \
    ici_tensor_parallelism=1 ici_fsdp_parallelism=8 ici_expert_parallelism=1 max_target_length=16 weight_dtype=bfloat16 dtype=bfloat16 opt_type=sgd \
    enable_tunix_perf_metrics=True rl.use_agentic_rollout=True chips_per_vm=8

python3 -m maxtext.inference.vllm_decode \
    model_name=${MODEL_NAME} \
    load_parameters_path=${BASE_OUTPUT_DIRECTORY}/rl/${run_id}-v12/checkpoints/actor/4/items \
    vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}' \
    hbm_utilization_vllm=0.85 \
    use_chat_template=True scan_layers=false enable_single_controller=${use_pathways} \
    ici_tensor_parallelism=1 ici_expert_parallelism=8
