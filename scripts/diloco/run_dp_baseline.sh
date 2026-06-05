#!/bin/bash

# This script launches a Data Parallel (DP) baseline pre-training workload on a GKE cluster using XPK.

set -e

# --- Environment Setup ---
if ! pip show xpk &> /dev/null; then
    echo "xpk not found in the environment. Please install it by running:"
    echo "uv pip install -e .[runner] --resolution=lowest"
    exit 1
fi

# --- Environment Variables ---
export PROJECT_ID="${PROJECT_ID:-cloud-tpu-multipod-dev}"
export CLUSTER_NAME="${CLUSTER_NAME:-v5p-128-bodaborg-europe-west4-b}"
export ZONE="${ZONE:-europe-west4}"
export RESERVATION="${RESERVATION:-cloudtpu-20240716121201-595617744}"
export BASE_OUTPUT_DIRECTORY="${BASE_OUTPUT_DIRECTORY:-gs://chriszuo-maxtext-logs}"
export DATASET_PATH="${DATASET_PATH:-gs://chriszuo-maxtext-datasets}"
export DOCKER_IMAGE="${DOCKER_IMAGE:-gcr.io/tpu-prod-env-multipod/maxtext_jax_stable:2026-06-04}"
export TPU_TYPE="${TPU_TYPE:-v5p-128}"
export NUM_SLICES="${NUM_SLICES:-2}"
export WORKLOAD_NAME="${WORKLOAD_NAME:-$(whoami)-dp-v5p-$(date +%Y%m%d-%H%M%S)}"

# --- Hyperparameters ---
export MODEL_NAME="${MODEL_NAME:-qwen3-30b-a3b}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8.0}"
export MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-2048}"
export TRAINING_STEPS="${TRAINING_STEPS:-100}"

# --- Variable Validation ---
if [ -z "$PROJECT_ID" ] || [ -z "$CLUSTER_NAME" ] || [ -z "$ZONE" ]; then
    echo "Error: PROJECT_ID, CLUSTER_NAME, or ZONE is not set."
    exit 1
fi

if [ -z "$BASE_OUTPUT_DIRECTORY" ] || [ -z "$DATASET_PATH" ]; then
    echo "Error: BASE_OUTPUT_DIRECTORY or DATASET_PATH is not set."
    exit 1
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
steps=$TRAINING_STEPS"

# Workload Creation
echo "Submitting DP job to XPK..."
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
