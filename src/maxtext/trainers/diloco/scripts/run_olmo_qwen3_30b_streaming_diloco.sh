#!/bin/bash
#
# Script to train Qwen3-30B-A3B (MoE) on OLMo dataset with Streaming DiLoCo across TPU slices.
# References:
#   - src/maxtext/trainers/diloco/scripts/run_moe.sh (Qwen3-30B MoE & DiLoCo hyperparameters, SparseCore flags)
#   - MyStuff/scripts/pretrain/run_olmo_streaming_diloco_v5p-128.sh (OLMo grain dataset pipeline & gcsfuse setup)
#

set -euo pipefail

# -------------------------- Cluster & Topology Settings --------------------------
: "${XPK_PROJECT:=cloud-tpu-multipod-dev}"
: "${XPK_ZONE:=europe-west4-b}"
: "${XPK_CLUSTER:=mlperf-v5p}"
: "${XPK_RESERVATION:=cloudtpu-20240716121201-595617744}"
: "${XPK_DEVICE_TYPE:=v5p-128}"
: "${XPK_NUM_SLICES:=2}"
: "${XPK_PRIORITY:=medium}"
: "${XPK_MAX_RESTARTS:=50}"

# -------------------------- Docker & Storage --------------------------
: "${XPK_DOCKER_IMAGE:=gcr.io/tpu-prod-env-multipod/maxtext_jax_stable:2026-07-17}"
: "${BASE_OUTPUT_DIRECTORY:=gs://chriszuo-maxtext-logs}"

# -------------------------- OLMo Data & Tokenizer --------------------------
: "${OLMO_INDEX_PATH:=/tmp/olmo-data/olmo/indices/olmo_index_seq8192.json}"
: "${OLMO_GCS_BASE:=gs://chriszuo-maxtext-datasets}"
: "${OLMO_LOCAL_MOUNT:=/tmp/olmo-data}"

# HuggingFace Token (autodetect if ~/.hf_token.sh exists)
if [ -z "${HF_TOKEN:-}" ] && [ -f "${HOME}/.hf_token.sh" ]; then
  # shellcheck disable=SC1090
  source "${HOME}/.hf_token.sh"
fi
: "${HF_TOKEN:=}"

# -------------------------- MoE Model Configuration --------------------------
: "${MODEL_NAME:=qwen3-30b-a3b}"
: "${TOKENIZER_TYPE:=huggingface}"
: "${TOKENIZER_PATH:=maxtext/assets/tokenizers/qwen3-tokenizer}"
: "${MAX_TARGET_LENGTH:=8192}"

# -------------------------- DiLoCo Hyperparameters --------------------------
: "${ENABLE_STREAMING_DILOCO:=true}"
: "${DILOCO_SYNC_PERIOD:=49}"
: "${DILOCO_OUTER_LR:=0.1}"
: "${DILOCO_OUTER_MOMENTUM:=0.9}"
: "${DILOCO_NUM_FRAGMENTS:=49}"
: "${DILOCO_USE_SEQUENTIAL_LAYERS:=false}"

# -------------------------- Training & Optimizer Hyperparameters --------------------------
: "${RUN_NAME:=jzuo-qwen3-olmo-dlco-$(date +%m%d%H%M)}"
: "${WORKLOAD_NAME:=qw3-olmo-$(date +%m%d%H%M)}"
: "${WARMUP_STEPS:=2000}"
: "${TARGET_GLOBAL_BATCH:=512}"

# Calculate total steps for 1 full epoch of OLMo dataset (24,357,482 sequences / global batch 512 = 47,573 steps, ~199.5B tokens)
: "${TOTAL_INSTANCES:=24357482}"
TOTAL_DATASET_STEPS=$(( TOTAL_INSTANCES / TARGET_GLOBAL_BATCH ))
: "${LR_SCHEDULE_STEPS:=${TOTAL_DATASET_STEPS}}"
: "${STEPS:=${LR_SCHEDULE_STEPS}}"
: "${LEARNING_RATE:=1.0e-4}"
: "${COSINE_FINAL_FRAC:=0.1}"
: "${ADAM_B1:=0.9}"
: "${ADAM_B2:=0.95}"
: "${ADAM_EPS:=1e-8}"
: "${ADAM_WD:=0.1}"
: "${GRAD_CLIP:=1.0}"

# MoE Z-loss & Load Balancing
: "${Z_LOSS:=1.0e-5}"
: "${LOAD_BALANCE_LOSS_WEIGHT:=0.001}"
: "${FLOAT32_GATE_LOGITS:=true}"
: "${DATA_SEED:=42}"
: "${LOAD_FULL_STATE_PATH:=}"
: "${LOAD_PARAMETERS_PATH:=}"


# Determine batch size per device
DEVICE_NUM=$(echo "${XPK_DEVICE_TYPE}" | cut -d'-' -f2)
TOTAL_DEVICES=$(( DEVICE_NUM * XPK_NUM_SLICES ))
: "${PER_DEVICE_BATCH_SIZE:=$(( TARGET_GLOBAL_BATCH / TOTAL_DEVICES ))}"

# Resolve container-side index path
if [[ "${OLMO_INDEX_PATH}" = /* ]]; then
  OLMO_INDEX_PATH_IN_CONTAINER="${OLMO_INDEX_PATH}"
else
  OLMO_INDEX_PATH_IN_CONTAINER="/deps/${OLMO_INDEX_PATH}"
fi

WARMUP_FRAC=$(python3 -c "print(${WARMUP_STEPS}/${LR_SCHEDULE_STEPS})")

# -------------------------- LibTPU & SparseCore Flags --------------------------
LIBTPU_INIT_ARGS=" \
  --xla_tpu_scoped_vmem_limit_kib=65472 \
  --xla_tpu_bf16_emission_mode=NATIVE_EMISSION \
  --xla_tpu_enable_sparse_core_reduce_scatter_v2=true \
  --xla_tpu_enable_sparse_core_collective_offload_all_gather=true \
  --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true \
  --xla_tpu_enable_all_gather_offload_tracing=true \
  --xla_tpu_use_tc_device_shape_on_sc=True \
  --xla_sc_disable_megacore_partitioning=True \
  --xla_tpu_enable_async_collective_fusion_fuse_all_gather=false \
  --xla_enable_async_all_gather=true \
  --xla_tpu_prefer_async_allgather_to_allreduce=true \
  --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true \
  --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true \
  --xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true \
  --xla_tpu_use_single_sparse_core_for_all_gather_offload=true \
  --xla_tpu_enable_concurrent_sparse_core_offloading=true \
  --xla_tpu_aggressive_opt_barrier_removal=true \
  --xla_tpu_enable_offloading_gather_to_sparsecore=true \
  --xla_tpu_sparse_core_all_gather_latency_multiplier=1 \
  --xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3 \
  --xla_tpu_enable_sparse_core_collective_aggregator=true \
  --xla_tpu_enable_latency_hiding_layer_scheduler=true \
  --xla_tpu_scheduler_percent_shared_memory_limit=150 \
  --xla_tpu_enable_layer_scheduler_for_dependent_collectives=true \
  --xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true \
  --xla_tpu_pcie_bandwidth_multiplier=0.03 \
  --xla_tpu_enable_sparse_core_offload_queuing_in_lhs=true \
  --xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=false \
  --xla_tpu_enable_3d_reduce_scatter_decomposer=false "

## -------------------------- Command Execution --------------------------
CMD="set -euo pipefail; mkdir -p /deps/src/src/dependencies/scripts && ( [ -f /deps/src/src/dependencies/scripts/setup_gcsfuse.sh ] || ln -sf /deps/src/dependencies/scripts/setup_gcsfuse.sh /deps/src/src/dependencies/scripts/setup_gcsfuse.sh ); mkdir -p /deps/src/src/maxtext && ( [ -d /deps/src/src/maxtext/configs ] || ln -sf /deps/src/maxtext/configs /deps/src/src/maxtext/configs ); if [[ \"\${MOUNT_GCSFUSE:-1}\" = \"1\" ]]; then bucket=\"\${GCS_BASE#gs://}\"; bucket=\"\${bucket%/}\"; bash /deps/src/dependencies/scripts/setup_gcsfuse.sh DATASET_GCS_BUCKET=\"\${bucket}\" MOUNT_PATH=\"\${LOCAL_MOUNT}\"; fi; export PYTHONPATH=/app/src:\${PYTHONPATH:-}; cd /app/src && unset XLA_FLAGS; export LIBTPU_INIT_ARGS=\"${LIBTPU_INIT_ARGS}\"; LOAD_PARAM_ARG=\"\"; if [ -n \"\${LOAD_FULL_STATE_PATH}\" ]; then LOAD_PARAM_ARG=\"load_full_state_path=\${LOAD_FULL_STATE_PATH}\"; elif [ -n \"\${LOAD_PARAMETERS_PATH}\" ]; then LOAD_PARAM_ARG=\"load_parameters_path=\${LOAD_PARAMETERS_PATH}\"; fi; python3 -m maxtext.trainers.pre_train.train maxtext/configs/base.yml run_name=\"\${RUN_NAME}\" save_config_to_gcs=true base_output_directory=\"\${OUTPUT_DIR}\" \${LOAD_PARAM_ARG} dataset_type=olmo_grain olmo_index_path=\"\${INDEX_PATH}\" olmo_path_remap_from=\"\${GCS_BASE}\" olmo_path_remap_to=\"\${LOCAL_MOUNT}\" olmo_apply_ngram_filter=True data_shuffle_seed=\"\${DATA_SEED}\" model_name=\"\${MODEL_NAME}\" tokenizer_type=\"\${TOKENIZER_TYPE}\" tokenizer_path=\"\${TOKENIZER_PATH}\" per_device_batch_size=\"\${PER_DEVICE_BATCH_SIZE}\" max_target_length=\"\${MAX_TARGET_LENGTH}\" learning_rate=\"\${LEARNING_RATE}\" learning_rate_schedule_steps=\"\${LR_SCHEDULE_STEPS}\" learning_rate_final_fraction=\"\${COSINE_FINAL_FRAC}\" warmup_steps_fraction=\"\${WARMUP_FRAC}\" adam_b1=\"\${ADAM_B1}\" adam_b2=\"\${ADAM_B2}\" adam_eps=\"\${ADAM_EPS}\" adam_weight_decay=\"\${ADAM_WD}\" gradient_clipping_threshold=\"\${GRAD_CLIP}\" z_loss_multiplier=\"\${Z_LOSS}\" load_balance_loss_weight=\"\${LOAD_BALANCE_LOSS_WEIGHT}\" float32_gate_logits=\"\${FLOAT32_GATE_LOGITS}\" enable_diloco=true enable_streaming_diloco=\"\${ENABLE_STREAMING_DILOCO}\" pure_nnx=true dcn_diloco_parallelism=\"\${XPK_NUM_SLICES}\" diloco_sync_period=\"\${DILOCO_SYNC_PERIOD}\" diloco_outer_lr=\"\${DILOCO_OUTER_LR}\" diloco_outer_momentum=\"\${DILOCO_OUTER_MOMENTUM}\" num_diloco_fragments=\"\${DILOCO_NUM_FRAGMENTS}\" use_sequential_layers=\"\${DILOCO_USE_SEQUENTIAL_LAYERS}\" jax_distributed_initialization_timeout=1200 steps=\"\${STEPS}\""

echo "Submitting Qwen3-30B-A3B + OLMo DiLoCo training workload using XPK..."
echo "Workload Name: ${WORKLOAD_NAME}"
echo "Run Name:      ${RUN_NAME}"
echo "Model:         ${MODEL_NAME}"
echo "Dataset:       olmo_grain (${OLMO_INDEX_PATH_IN_CONTAINER})"
echo "Topology:      ${XPK_DEVICE_TYPE} x ${XPK_NUM_SLICES} slices"

xpk workload create \
  --cluster "${XPK_CLUSTER}" \
  --workload "${WORKLOAD_NAME}" \
  --project "${XPK_PROJECT}" \
  --zone "${XPK_ZONE}" \
  --tpu-type "${XPK_DEVICE_TYPE}" \
  --num-slices "${XPK_NUM_SLICES}" \
  --priority "${XPK_PRIORITY}" \
  --max-restarts "${XPK_MAX_RESTARTS}" \
  --reservation "${XPK_RESERVATION}" \
  --base-docker-image "${XPK_DOCKER_IMAGE}" \
  --script-dir "$(pwd)" \
  --command "export HF_TOKEN='${HF_TOKEN}'; export INDEX_PATH='${OLMO_INDEX_PATH_IN_CONTAINER}'; export GCS_BASE='${OLMO_GCS_BASE}'; export LOCAL_MOUNT='${OLMO_LOCAL_MOUNT}'; export OUTPUT_DIR='${BASE_OUTPUT_DIRECTORY}'; export RUN_NAME='${RUN_NAME}'; export LOAD_FULL_STATE_PATH='${LOAD_FULL_STATE_PATH}'; export LOAD_PARAMETERS_PATH='${LOAD_PARAMETERS_PATH}'; export MODEL_NAME='${MODEL_NAME}'; export TOKENIZER_TYPE='${TOKENIZER_TYPE}'; export TOKENIZER_PATH='${TOKENIZER_PATH}'; export MAX_TARGET_LENGTH='${MAX_TARGET_LENGTH}'; export STEPS='${STEPS}'; export WARMUP_STEPS='${WARMUP_STEPS}'; export LR_SCHEDULE_STEPS='${LR_SCHEDULE_STEPS}'; export LEARNING_RATE='${LEARNING_RATE}'; export COSINE_FINAL_FRAC='${COSINE_FINAL_FRAC}'; export ADAM_B1='${ADAM_B1}'; export ADAM_B2='${ADAM_B2}'; export ADAM_EPS='${ADAM_EPS}'; export ADAM_WD='${ADAM_WD}'; export GRAD_CLIP='${GRAD_CLIP}'; export Z_LOSS='${Z_LOSS}'; export LOAD_BALANCE_LOSS_WEIGHT='${LOAD_BALANCE_LOSS_WEIGHT}'; export FLOAT32_GATE_LOGITS='${FLOAT32_GATE_LOGITS}'; export ENABLE_STREAMING_DILOCO='${ENABLE_STREAMING_DILOCO}'; export DILOCO_SYNC_PERIOD='${DILOCO_SYNC_PERIOD}'; export DILOCO_OUTER_LR='${DILOCO_OUTER_LR}'; export DILOCO_OUTER_MOMENTUM='${DILOCO_OUTER_MOMENTUM}'; export DILOCO_NUM_FRAGMENTS='${DILOCO_NUM_FRAGMENTS}'; export DILOCO_USE_SEQUENTIAL_LAYERS='${DILOCO_USE_SEQUENTIAL_LAYERS}'; export DATA_SEED='${DATA_SEED}'; export MOUNT_GCSFUSE=1; export XPK_NUM_SLICES='${XPK_NUM_SLICES}'; export PER_DEVICE_BATCH_SIZE='${PER_DEVICE_BATCH_SIZE}'; export WARMUP_FRAC='${WARMUP_FRAC}'; ${CMD}"


echo "Qwen3-30B-A3B + OLMo DiLoCo workload submission complete!"
