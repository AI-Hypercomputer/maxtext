#!/bin/bash

# Validates the GPT-OSS-20b RL pipeline using a pre-converted MaxText checkpoint.

# The flow of this script is as follows:
# 1. Run RL starting from the pre-converted checkpoint.

# Usage:
# export HF_TOKEN=<your Hugging Face access token>
# export RUN_ID=$(date +%Y-%m-%d-%H-%M-%S)
# bash test_gpt_oss_to_mt.sh $RUN_ID
# bash test_gpt_oss_rl.sh $RUN_ID

set -ex

export VLLM_WORKER_MULTIPROC_METHOD='spawn'
export MODEL_IMPL_TYPE='flax_nnx'
export GRPC_ENABLE_FORK_SUPPORT='0'
export JAX_RANDOM_WEIGHTS='1'
export VLLM_ENABLE_V1_MULTIPROCESSING='0'
export SKIP_JAX_PRECOMPILE='1'
export NEW_MODEL_DESIGN='0'

if [ -z "$1" ]; then
  echo "Error: run_id argument is required."
  exit 1
fi
run_id=$1
use_pathways=${2:-false}
export MODEL_NAME='gpt-oss-20b'

# Non-Googlers please remember to point BASE_OUTPUT_DIRECTORY to the GCS paths where you have the scanned and unscanned checkpoints stored
BASE_OUTPUT_DIRECTORY=gs://runner-maxtext-logs/${MODEL_NAME}
SCANNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/to_maxtext/scanned/${run_id}/0/items

# Step 1: Run RL on the converted checkpoint
python3 -m maxtext.trainers.post_train.rl.train_rl \
    base_output_directory=${BASE_OUTPUT_DIRECTORY}/rl \
    load_parameters_path=${SCANNED_CKPT_PATH} \
    run_name=${run_id} \
    num_batches=5 \
    batch_size=16 \
    train_micro_batch_size=16 \
    rollout_micro_batch_size=16 \
    profiler=xplane \
    profiler_steps=2 \
    base_emb_dim=2880 \
    vocab_size=201088 \
    enable_dp_attention=False \
    async_scheduling=True \
    chips_per_vm=4 \
    scan_layers=true \
    model_name=${MODEL_NAME} \
    tokenizer_path='unsloth/gpt-oss-20b-BF16' \
    chat_template_path=maxtext/examples/chat_templates/gpt_oss_rl.json \
    enable_tunix_perf_metrics=True \
    max_prefill_predict_length=1024 \
    max_target_length=2048 \
    enable_single_controller=${use_pathways} \
    checkpoint_storage_use_zarr3=False \
    checkpoint_storage_use_ocdbt=False \
    rollout_data_parallelism=4 \
    rollout_tensor_parallelism=8 \
    hbm_utilization_vllm=0.8