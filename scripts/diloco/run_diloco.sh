#!/bin/bash

# This script launches a DiLoCo pre-training workload on a GKE cluster using XPK.

set -e

# --- Environment Setup ---
if ! pip show xpk &> /dev/null; then
    echo "xpk not found in the environment. Please install it by running:"
    echo "uv pip install -e .[runner] --resolution=lowest"
    exit 1
fi

# --- Environment Variables ---
export PROJECT_ID="${PROJECT_ID:-cloud-tpu-multipod-dev}"
export CLUSTER_NAME="${CLUSTER_NAME:-auto-v5p-8-bodaborg}"
export ZONE="${ZONE:-europe-west4-b}"
export RESERVATION="${RESERVATION:-}"
export BASE_OUTPUT_DIRECTORY="${BASE_OUTPUT_DIRECTORY:-gs://chriszuo-maxtext-logs}" # change to your own GCS bucket for logging and checkpointing
export DATASET_PATH="${DATASET_PATH:-gs://chriszuo-maxtext-datasets}" # change to your own GSC bucket for datasets. Make sure datasets exists
export DOCKER_IMAGE="${DOCKER_IMAGE:-gcr.io/tpu-prod-env-multipod/maxtext_jax_stable:2026-06-04}" # should update if later versions come up
export TPU_TYPE="${TPU_TYPE:-v5p-8}"  # At least v5p-32 is needed to run Qwen3-30b-a3b. For v5p-8 you may need to decrease the PER_DEVICE_BATCH_SIZE
export NUM_SLICES="${NUM_SLICES:-2}"  # you need at least two slices to let diloco take effect
export WORKLOAD_NAME="${WORKLOAD_NAME:-$(whoami)-diloco-v5p-$(date +%Y%m%d-%H%M%S)}" # this will be the name of run, for logging purposes

# --- Hyperparameters ---
export MODEL_NAME="${MODEL_NAME:-qwen3-8b}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
export MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-2048}"
export DILOCO_SYNC_PERIOD="${DILOCO_SYNC_PERIOD:-10}"
export DILOCO_OUTER_LR="${DILOCO_OUTER_LR:-0.3}"
export DILOCO_OUTER_MOMENTUM="${DILOCO_OUTER_MOMENTUM:-0.9}"
export TRAINING_STEPS="${TRAINING_STEPS:-20}"

# --- Variable Validation ---
if [ -z "$PROJECT_ID" ] || [ -z "$CLUSTER_NAME" ] || [ -z "$ZONE" ]; then
    echo "Error: PROJECT_ID, CLUSTER_NAME, or ZONE is not set."
    exit 1
fi

if [ -z "$BASE_OUTPUT_DIRECTORY" ] || [ -z "$DATASET_PATH" ]; then
    echo "Error: BASE_OUTPUT_DIRECTORY or DATASET_PATH is not set."
    exit 1
fi

if [ "$NUM_SLICES" -lt 2 ]; then
    echo "Warning: NUM_SLICES is less than 2. DiLoCo will not take effect."
fi

# MaxText command
MAXTEXT_COMMAND="cd /deps/src/ && python3 maxtext/trainers/pre_train/train.py \
maxtext/configs/base.yml \
run_name=$WORKLOAD_NAME \
save_config_to_gcs=true \
base_output_directory=$BASE_OUTPUT_DIRECTORY \
dataset_path=$DATASET_PATH \
dataset_name='c4/en:3.0.1' \
eval_dataset_name='c4/en:3.0.1' \
model_name=$MODEL_NAME \
tokenizer_type=huggingface \
tokenizer_path=maxtext/assets/tokenizers/qwen3-tokenizer \
per_device_batch_size=$PER_DEVICE_BATCH_SIZE \
max_target_length=$MAX_TARGET_LENGTH \
enable_diloco=true \
dcn_diloco_parallelism=$NUM_SLICES \
diloco_sync_period=$DILOCO_SYNC_PERIOD \
diloco_outer_lr=$DILOCO_OUTER_LR \
diloco_outer_momentum=$DILOCO_OUTER_MOMENTUM \
steps=$TRAINING_STEPS"

# Workload Creation
echo "Submitting DiLoCo job to XPK..."
xpk workload create \
  --cluster="$CLUSTER_NAME" \
  --project="$PROJECT_ID" \
  --reservation="$RESERVATION" \
  --zone="$ZONE" \
  --tpu-type="$TPU_TYPE" \
  --num-slices="$NUM_SLICES" \
  --docker-image="${DOCKER_IMAGE}" \
  --workload="${WORKLOAD_NAME}" \
  --command="${MAXTEXT_COMMAND}"
