#!/bin/bash
# Script to launch DeepSeek 671b benchmark on a Pathways/xpk cluster

# --- Configuration (Adjust these) ---
export CLUSTER=next-devx-1
export CLUSTER_NAME=next-devx-1
export PROJECT=tpu-prod-env-automated
export ZONE=us-central1-c
export TPU_TYPE=tpu7x-128
NUM_SLICES=1
DOCKER_IMAGE="gcr.io/tpu-prod-env-multipod/mohit-dsv3-build:20260406" # Ensure this image has pathwaysutils installed

WORKLOAD_NAME="dsv3-$RANDOM"

# The command to execute inside the cluster
# We use --run_vllm_only=True to verify vLLM initialization directly
# and --additional_config to force the 'megablox' backend for faster compilation at scale.
COMMAND="python3 src/maxtext/integration/tunix/weight_mapping/bench_weight_sync.py \
  --model_name=deepseek3-671b \
  --vllm_model_id=deepseek-ai/DeepSeek-V3 \
  --rand_init=True \
  --ici_fsdp_parallelism=1 \
  --ici_tensor_parallelism=128 \
  --rollout_tensor_parallelism=128 \
  --run_vllm_only=True"

echo "Submitting xpk workload: ${WORKLOAD_NAME}"
echo "Command: ${COMMAND}"
echo "Cluster: ${CLUSTER_NAME} | Project: ${PROJECT} | Zone: ${ZONE}"

xpk workload create-pathways \
  --workload="${WORKLOAD_NAME}" \
  --cluster="${CLUSTER}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --tpu-type="${TPU_TYPE}" \
  --num-slices=${NUM_SLICES} \
  --docker-image="${DOCKER_IMAGE}" \
  --command="pip show jax; pip show libtpu; MODEL_IMPL_TYPE=auto NEW_MODEL_DESIGN=True ${COMMAND}"
