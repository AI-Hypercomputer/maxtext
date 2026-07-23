#!/bin/bash
# Validates the Qwen3 RL pipeline using a pre-converted MaxText checkpoint.
set -ex
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export JAX_RANDOM_WEIGHTS=1
export SKIP_JAX_PRECOMPILE=1
export NEW_MODEL_DESIGN=1
export TPU_MIN_LOG_LEVEL=0
export TF_CPP_MIN_LOG_LEVEL=0
export TPU_STDERR_LOG_LEVEL=0

export PYTHONPATH=${PYTHONPATH:-.}:$(pwd)

run_id=${1:-$(date +%Y-%m-%d-%H-%M-%S)}
use_pathways=${2:-False}
MODEL_NAME='qwen3-0.6b'

BASE_OUTPUT_DIRECTORY=gs://runner-maxtext-logs/${MODEL_NAME}
UNSCANNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/to_maxtext/unscanned/${run_id}/0/items
SCANNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/to_maxtext/scanned/${run_id}/0/items

python3 -m maxtext.inference.vllm_decode \
    model_name=${MODEL_NAME} \
    load_parameters_path=${SCANNED_CKPT_PATH} \
    vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}' \
    hbm_utilization_vllm=0.15 \
    prompt='Suggest some famous landmarks in London.' \
    max_target_length=256 max_num_batched_tokens=256 \
    ici_tensor_parallelism=8  \
    allow_split_physical_axes=True prefuse_moe_weights=True \
    use_chat_template=True scan_layers=True enable_single_controller=${use_pathways}

# test scenario 1: MaxText to MaxText (VLLM's MaxTextForCausalLM backend) 
python3 -m maxtext.trainers.post_train.rl.train_rl \
    "$PWD/src/maxtext/configs"/post_train/rl.yml \
    base_output_directory=${BASE_OUTPUT_DIRECTORY}/rl \
    load_parameters_path=${SCANNED_CKPT_PATH} \
    run_name=${run_id} rl.loss_algo='grpo' scan_layers=True \
    num_batches=5 batch_size=4 train_micro_batch_size=1 num_test_batches=5 \
    model_name=${MODEL_NAME} enable_single_controller=${use_pathways} enable_dropout=False \
    checkpoint_storage_use_zarr3=False checkpoint_storage_use_ocdbt=False \
    rollout_tensor_parallelism=8 \
    vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}' \
    vllm_additional_config='{"maxtext_config": {"model_name": "'${MODEL_NAME}'", "allow_split_physical_axes": true, "scan_layers": true, "prefuse_moe_weights": true}}' \
    remat_policy=full hbm_utilization_vllm=0.15 use_pathways=${use_pathways} \
    chips_per_vm=8 ici_tensor_parallelism=4\
    max_target_length=512 weight_dtype=bfloat16 dtype=bfloat16 opt_type=sgd \
    enable_tunix_perf_metrics=True rl.num_generations=16 \
    debug.rl=False rl.reshard_chunk_size=1 

python3 -m maxtext.inference.vllm_decode \
    model_name=${MODEL_NAME} \
    load_parameters_path=${BASE_OUTPUT_DIRECTORY}/rl/${run_id}/checkpoints/actor/5/model_params \
    vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}' \
    hbm_utilization_vllm=0.15 \
    prompt='Suggest some famous landmarks in London.' \
    max_target_length=256 max_num_batched_tokens=256 \
    ici_tensor_parallelism=8  \
    allow_split_physical_axes=True prefuse_moe_weights=True \
    use_chat_template=True scan_layers=True enable_single_controller=${use_pathways}

