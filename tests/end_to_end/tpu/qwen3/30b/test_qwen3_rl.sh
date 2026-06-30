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

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=upb
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY="PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export JAX_RANDOM_WEIGHTS=1
export SKIP_JAX_PRECOMPILE=1
export NEW_MODEL_DESIGN=1
export TPU_MIN_LOG_LEVEL=0
export TF_CPP_MIN_LOG_LEVEL=0
export TPU_STDERR_LOG_LEVEL=0

# Force Python to use "spawn" instead of "fork" for multiprocessing to prevent gRPC socket corruption
cat << 'EOF' > usercustomize.py
import multiprocessing
try:
    multiprocessing.set_start_method("spawn", force=True)
except Exception:
    pass
EOF
export PYTHONPATH=${PYTHONPATH:-.}:$(pwd)
run_id=${1:-$(date +%Y-%m-%d-%H-%M-%S)}
use_pathways=${2:-True}
MODEL_NAME='qwen3-30b-a3b-base'

BASE_OUTPUT_DIRECTORY=gs://runner-maxtext-logs/${MODEL_NAME}
UNSCANNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/to_maxtext/unscanned/${run_id}/0/items
SCANNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/to_maxtext/scanned/${run_id}/0/items

python3 -m maxtext.inference.vllm_decode \
    model_name=${MODEL_NAME} \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}' \
    hbm_utilization_vllm=0.85 \
    prompt='Suggest some famous landmarks in London.' \
    max_target_length=256 max_num_batched_tokens=256 \
    ici_tensor_parallelism=8 ici_data_parallelism=2 allow_split_physical_axes=True prefuse_moe_weights=True \
    use_chat_template=True scan_layers=False enable_single_controller=${use_pathways}

python3 -m maxtext.trainers.post_train.rl.train_rl \
    "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"/post_train/rl.yml \
    base_output_directory=${BASE_OUTPUT_DIRECTORY}/rl \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    run_name=${run_id} rl.loss_algo='grpo' scan_layers=False \
    num_batches=5 batch_size=8 train_micro_batch_size=8 num_test_batches=5 \
    model_name=${MODEL_NAME} enable_single_controller=${use_pathways} \
    checkpoint_storage_use_zarr3=False checkpoint_storage_use_ocdbt=False \
    rollout_tensor_parallelism=8 \
    vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}' \
    vllm_additional_config='{"maxtext_config": {"model_name": "'${MODEL_NAME}'", "allow_split_physical_axes": true, "scan_layers": false, "prefuse_moe_weights": true}}' \
    remat_policy=full hbm_utilization_vllm=0.55 use_pathways=${use_pathways} \
    ici_tensor_parallelism=1 ici_fsdp_parallelism=-1 ici_expert_parallelism=8 \
    max_sequence_length=1024 max_target_length=512 weight_dtype=bfloat16 dtype=bfloat16 opt_type=sgd \
    enable_tunix_perf_metrics=True rl.use_agentic_rollout=True rl.num_generations=16

python3 -m maxtext.inference.vllm_decode \
    model_name=${MODEL_NAME} \
    load_parameters_path=${BASE_OUTPUT_DIRECTORY}/rl/${run_id}/checkpoints/actor/4/items \
    vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}' \
    hbm_utilization_vllm=0.85 \
    prompt='Suggest some famous landmarks in London.' \
    max_target_length=256 max_num_batched_tokens=256 \
    ici_tensor_parallelism=8 ici_data_parallelism=2 allow_split_physical_axes=True prefuse_moe_weights=True \
    use_chat_template=True scan_layers=False enable_single_controller=${use_pathways}
