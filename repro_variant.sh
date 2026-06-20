#!/bin/bash
# Generalized variant runner for the task-2 XLA-flag A/B.
# Usage: RUN_TAG=<tag> FLAGS_FILE=<path> bash repro_variant.sh
# - MAXTEXT_ARGS are identical to repro_run.sh (the 16.24s baseline).
# - Only the XLA / LIBTPU flags vary, supplied via FLAGS_FILE (one flag per line,
#   '#' comments and blank lines ignored). No code / sharding / kernel changes.

VENV=/usr/local/google/home/surajkolla/VENVS/ubench/bin/activate
[ -f "$VENV" ] && source "$VENV"

set -e
set -o pipefail

: "${RUN_TAG:?set RUN_TAG}"
: "${FLAGS_FILE:?set FLAGS_FILE}"

export PROJECT_ID="cloud-tpu-multipod-dev"
export CLUSTER_NAME="bodaborg-super-xpk-x8p"
export ZONE="us-central1-a"
export BASE_OUTPUT_DIR="gs://ubench-logs"
export ARTIFACT_DIR="gs://ubench-logs/${RUN_TAG}"
export WORKLOAD_IMAGE="${WORKLOAD_IMAGE:-gcr.io/cloud-tpu-multipod-dev/integrate_v2:gmmv2_bf16_log27}"
export WORKLOAD_NAME="${RUN_TAG}"

# Build XLA_FLAGS from FLAGS_FILE (strip comments/blank lines, join with spaces).
XLA_FLAGS=" $(grep -vE '^\s*(#|$)' "$FLAGS_FILE" | sed 's/[[:space:]]*$//' | tr '\n' ' ') "
echo "=== XLA_FLAGS for $RUN_TAG ==="
echo "$XLA_FLAGS" | tr ' ' '\n' | grep -v '^$'
echo "==============================="

MAXTEXT_ARGS="\
model_name=deepseek3-671b \
per_device_batch_size=4.0 \
max_target_length=4096 \
ici_fsdp_parallelism=128 \
ici_expert_parallelism=4 \
ici_data_parallelism=1 \
ici_fsdp_transpose_parallelism=1 \
dcn_data_parallelism=-1 \
dcn_pipeline_parallelism=1 \
ici_pipeline_parallelism=1 \
shard_exp_on_fsdp=False \
use_iota_embed=True \
tokenizer_path=assets/tokenizer.mistral-v3 \
dataset_type=synthetic \
dataset_path=gs://max-datasets-rogue \
opt_type=adamw \
mu_dtype=bfloat16 \
grad_dtype=bfloat16 \
dtype=bfloat16 \
sa_use_fused_bwd_kernel=True \
megablox=True \
sparse_matmul=True \
use_tokamax_gmm=True \
use_tokamax_splash=True \
use_max_logit_estimate=-1 \
cost_estimate_flops_fwd=5000000000000 \
cost_estimate_flops_bwd=5000000000000 \
float32_weight_sum=False \
remat_policy=custom \
allow_split_physical_axes=False \
decoder_layer_input=device \
enable_tpu_profiling_options=True \
async_checkpointing=False \
enable_checkpointing=False \
attention=flash \
sa_block_q=2048 \
sa_block_kv=2048 \
sa_block_kv_compute=2048 \
sa_block_q_dkv=2048 \
sa_block_kv_dkv=2048 \
sa_block_kv_dkv_compute=2048 \
sa_block_kv_dq=2048 \
sa_block_q_dq=2048 \
use_random_routing=True \
use_ring_of_experts=True \
use_custom_sort_vjp=True \
use_ragged_sort=True \
merge_gating_gmm=False \
wi_tile_fwd_batch_seq=256 \
wi_tile_fwd_embed_dim=7168 \
wi_tile_fwd_mlp_dim=1024 \
wi_tile_dlhs_batch_seq=256 \
wi_tile_dlhs_embed_dim=3584 \
wi_tile_dlhs_mlp_dim=2048 \
wi_tile_drhs_batch_seq=512 \
wi_tile_drhs_embed_dim=1792 \
wi_tile_drhs_mlp_dim=2048 \
wo_tile_fwd_batch_seq=512 \
wo_tile_fwd_embed_dim=3584 \
wo_tile_fwd_mlp_dim=2048 \
wo_tile_dlhs_batch_seq=512 \
wo_tile_dlhs_embed_dim=1792 \
wo_tile_dlhs_mlp_dim=2048 \
wo_tile_drhs_batch_seq=512 \
wo_tile_drhs_embed_dim=1792 \
wo_tile_drhs_mlp_dim=2048 \
skip_jax_distributed_system=True \
steps=20 \
base_output_directory=${BASE_OUTPUT_DIR} \
run_name=${WORKLOAD_NAME} \
profiler=xplane \
skip_first_n_steps_for_profiler=5 \
profiler_steps=3 \
${EXTRA_MAXTEXT_ARGS}"

mkdir -p /tmp/ubench_recipe/${WORKLOAD_NAME}/recipe_for_github/k8s

echo "=== Creating XPK Workload: $WORKLOAD_NAME ==="
xpk workload create \
  --cluster=$CLUSTER_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --priority=${PRIORITY:-very-high} \
  --max-restarts=0 \
  --device-type=tpu7x-4x8x8 \
  --num-slices=1 \
  --docker-image="${WORKLOAD_IMAGE}" \
  --enable-debug-logs \
  --no-use-parallel-containers \
  --workload="${WORKLOAD_NAME}" \
  --output-manifest-file=/tmp/ubench_recipe/${WORKLOAD_NAME}/recipe_for_github/k8s/k8s_manifest.yaml \
  --command="set -e && set -o pipefail && export ENABLE_PATHWAYS_PERSISTENCE='1' && \
export LIBTPU_INIT_ARGS='${XLA_FLAGS}' && \
export ARTIFACT_DIR='${ARTIFACT_DIR}' && \
export JAX_PLATFORMS='tpu,cpu' && export ENABLE_PJRT_COMPATIBILITY='true' && \
python3 -m maxtext.trainers.pre_train.train maxtext/configs/base.yml ${MAXTEXT_ARGS} 2>&1 | tee train.log ; \
gcloud storage cp --no-user-output-enabled train.log ${ARTIFACT_DIR}/logs/train-\${TPU_WORKER_ID}.log"
