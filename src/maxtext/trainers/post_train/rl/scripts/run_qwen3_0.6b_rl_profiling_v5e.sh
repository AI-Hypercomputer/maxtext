#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Launches a small Qwen3-0.6B GRPO run on a Pathways-disaggregated v5litepod-8
# (trainer on one worker, vLLM sampler on the other) via XPK, with targeted
# RL profiling and managed ML Diagnostics enabled. It is the smoke recipe for
# verifying that
#   * `actor_invocation_N` and `rollout_invocation_N` traces are uploaded, and
#   * Tunix scalars reach ML Diagnostics (Cloud Monitoring) and TensorBoard.
#
# Required environment variables:
#   PROJECT_ID, CLUSTER_NAME, ZONE, BASE_OUTPUT_DIRECTORY (a gs:// path),
#   MAXTEXT_CKPT_PATH (a gs:// checkpoint items path), DOCKER_IMAGE, WORKLOAD_NAME
# Optional environment variables:
#   REGION (defaults to the region of ZONE), TPU_TYPE (v5litepod-8), NUM_SLICES (1),
#   NUM_BATCHES (10), HF_TOKEN, PATHWAYS_MAX_NUM_HOSTS (1000).
#
# Notes:
#   * `GKE_DIAGON_IDENTIFIER` / `GKE_DIAGON_METADATA` bind the run to the
#     JobSet in the ML Diagnostics console; without them no run appears.
#   * Pathways traces cover worker 0 only unless `PATHWAYS_MAX_NUM_HOSTS` is
#     exported in the job's environment; the sampler lives on worker 1 here, so
#     the recipe exports it (default 1000 = every worker).
#   * `managed_mldiagnostics_on_demand_profiling=False` is required together
#     with `profiler=xplane`; the config rejects the combination otherwise.

set -euo pipefail

: "${PROJECT_ID:?set PROJECT_ID}"
: "${CLUSTER_NAME:?set CLUSTER_NAME}"
: "${ZONE:?set ZONE}"
: "${BASE_OUTPUT_DIRECTORY:?set BASE_OUTPUT_DIRECTORY to a gs:// path}"
: "${MAXTEXT_CKPT_PATH:?set MAXTEXT_CKPT_PATH to a gs:// checkpoint items path}"
: "${DOCKER_IMAGE:?set DOCKER_IMAGE}"
: "${WORKLOAD_NAME:?set WORKLOAD_NAME}"

# Region defaults to the zone minus its trailing "-<letter>" suffix (us-central1-a -> us-central1).
REGION="${REGION:-${ZONE%-*}}"
TPU_TYPE="${TPU_TYPE:-v5litepod-8}"
NUM_SLICES="${NUM_SLICES:-1}"
NUM_BATCHES="${NUM_BATCHES:-10}"
PATHWAYS_MAX_NUM_HOSTS="${PATHWAYS_MAX_NUM_HOSTS:-1000}"

if ! command -v xpk &> /dev/null; then
  echo "xpk not found on PATH. Install it (pip install xpk) before running this script." >&2
  exit 1
fi

echo "=== Submitting XPK Pathways disaggregated RL profiling workload: ${WORKLOAD_NAME} ==="

MAXTEXT_COMMAND="HF_TOKEN=${HF_TOKEN:-} \
GOOGLE_CLOUD_REGION=${REGION} \
CLOUDSDK_COMPUTE_REGION=${REGION} \
JAX_PLATFORMS=proxy,cpu \
JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 \
ENABLE_PATHWAYS_PERSISTENCE=1 \
PATHWAYS_MAX_NUM_HOSTS=${PATHWAYS_MAX_NUM_HOSTS} \
NEW_MODEL_DESIGN=1 \
JAX_RANDOM_WEIGHTS=0 \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
SKIP_JAX_PRECOMPILE=1 \
TPU_MIN_LOG_LEVEL=0 \
TF_CPP_MIN_LOG_LEVEL=0 \
GKE_DIAGON_IDENTIFIER='{\"metadata.name\":\"'${WORKLOAD_NAME}'\",\"metadata.kind\":\"JobSet\",\"clustername\":\"projects/'${PROJECT_ID}'/locations/'${REGION}'/clusters/'${CLUSTER_NAME}'\",\"namespace\":\"default\"}' \
GKE_DIAGON_METADATA='{\"creation-timestamp\":\"'$(date -u +%Y-%m-%dT%H:%M:%SZ)'\"}' \
python3 -m maxtext.trainers.post_train.rl.train_rl \
  model_name=qwen3-0.6b \
  tokenizer_path=Qwen/Qwen3-0.6B \
  load_parameters_path=${MAXTEXT_CKPT_PATH} \
  run_name=${WORKLOAD_NAME} \
  base_output_directory=${BASE_OUTPUT_DIRECTORY} \
  managed_mldiagnostics=True \
  managed_mldiagnostics_on_demand_profiling=False \
  managed_mldiagnostics_region=${REGION} \
  profiler=xplane \
  rl.profiler_start_invocation=2 \
  rl.profiler_num_invocations=1 \
  async_scheduling=True \
  chips_per_vm=4 \
  num_batches=${NUM_BATCHES} \
  num_test_batches=2 \
  eval_interval=10 \
  log_period=1 \
  batch_size=8 \
  train_micro_batch_size=4 \
  rollout_micro_batch_size=4 \
  rollout_data_parallelism=2 \
  rollout_tensor_parallelism=2 \
  rl.num_generations=4 \
  rl.grpo_beta=0.05 \
  rl.grpo_epsilon=0.2 \
  gradient_clipping_threshold=1.0 \
  learning_rate=1e-5 \
  dataset_name=openai/gsm8k \
  hf_name=main \
  train_split=train \
  eval_split=test \
  chat_template_path=maxtext/examples/chat_templates/gsm8k_rl.json \
  max_target_length=1024 \
  max_prefill_predict_length=256 \
  enable_dp_attention=True \
  hbm_utilization_vllm=0.75 \
  max_num_seqs=64 \
  max_num_batched_tokens=2048 \
  allow_split_physical_axes=True \
  enable_tunix_perf_metrics=True \
  enable_checkpointing=True \
  checkpoint_period=10 \
  max_num_checkpoints_to_keep=2 \
  checkpoint_storage_use_ocdbt=False \
  vllm_hf_overrides='{\"architectures\": [\"MaxTextForCausalLM\"]}' \
  vllm_additional_config='{\"maxtext_config\": {\"model_name\": \"qwen3-0.6b\", \"model_call_mode\": \"inference\", \"enable_dp_attention\": false, \"allow_split_physical_axes\": true, \"log_config\": false, \"weight_dtype\": \"bfloat16\"}}'"

xpk workload create-pathways \
  --cluster="${CLUSTER_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --priority=high \
  --max-restarts=0 \
  --tpu-type="${TPU_TYPE}" \
  --num-slices="${NUM_SLICES}" \
  --docker-image="${DOCKER_IMAGE}" \
  --workload="${WORKLOAD_NAME}" \
  --command="${MAXTEXT_COMMAND}"
