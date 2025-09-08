#!/bin/bash

'
# This script is designed for internal use within Google.
# External users can update pre-trained model checkpoint GCS path (gs://) to your accessible locations.
# Usage:
  HF_TOKEN=<huggingface access token> \
  MODEL=llama3.3-70b TOKENIZER=meta-llama/Llama-3.3-70B \
  NUM_SAMPLERS=4 DEVICES_PER_SAMPLER=8 \
  TRAINING_PER_DEVICE_BATCH_SIZE=1 \
  INFERENCE_PER_DEVICE_BATCH_SIZE=4 \
  STEPS=20 \
  bash end_to_end/tpu/test_grpo.sh
'

set -xe

BASE_OUTPUT_DIRECTORY=gs://runner-maxtext-logs
RUN_NAME=grpo-$(date +%Y-%m-%d-%H-%M-%S)

JAX_PLATFORMS=proxy 
JAX_BACKEND_TARGET=grpc://127.0.0.1:29000
ENABLE_PATHWAYS_PERSISTENCE='1'
HF_TOKEN=${HF_TOKEN}

MAX_PREFILL_LENGTH=128
MAX_TARGET_LENGTH=256
NUM_GENERATIONS=2


COMMON_ARGS="model_name=${MODEL} base_output_directory=${BASE_OUTPUT_DIRECTORY} \
max_prefill_predict_length=${MAX_PREFILL_LENGTH} max_target_length=${MAX_TARGET_LENGTH} \
enable_checkpointing=false async_checkpointing=false \
tokenizer_type=huggingface tokenizer_path=${TOKENIZER} \
dataset_type=hf hf_path='trl-lib/tldr' \
enable_single_controller=true \
dtype=bfloat16 weight_dtype=bfloat16 \
allow_split_physical_axes=true enable_goodput_recording=false monitor_goodput=false \
profiler=xplane skip_first_n_steps_for_profiler=10 profiler_steps=5"

TRAINING_ARGS="run_name=${RUN_NAME} scan_layers=true \
inference_replicas=${NUM_SAMPLERS} inference_devices_per_replica=${DEVICES_PER_SAMPLER} \
inference_rollouts=5 \
per_device_batch_size=${TRAINING_PER_DEVICE_BATCH_SIZE} num_generations=${NUM_GENERATIONS} steps=${STEPS}"

INFERENCE_ARGS="run_name=grpo scan_layers=false \
per_device_batch_size=${INFERENCE_PER_DEVICE_BATCH_SIZE} \
ici_data_parallelism=${NUM_SAMPLERS} ici_tensor_parallelism=${DEVICES_PER_SAMPLER}"

JAX_PLATFORMS=proxy JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 ENABLE_PATHWAYS_PERSISTENCE='1' \
    python3 -m MaxText.experimental.rl.grpo_trainer "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}"/experimental/rl/grpo.yml  \
    ${COMMON_ARGS} ${TRAINING_ARGS} ${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}/experimental/rl/grpo_inference.yml \
    ${COMMON_ARGS} ${INFERENCE_ARGS}
