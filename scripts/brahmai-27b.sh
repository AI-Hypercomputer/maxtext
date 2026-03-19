#!/usr/bin/env bash

# ======================================================================
# brahmai-27b.sh — MaxText pre-training script for BrahmaI 27B
#
# Environment variables are defined in deploy.sh and exported before
# this script runs. The following vars MUST be set in the environment:
#
#   From deploy.sh (Lustre section):
#     MOUNTPOINT    — e.g. /lustre-data
#
#   Secrets (set externally, never committed):
#     HF_TOKEN      — Hugging Face access token
#
# Run-specific overrides can be set here if needed.
# ======================================================================

# ======================================================================
# Run-Specific Variables
# ======================================================================
export MOUNTPOINT="${MOUNTPOINT:-/lustre-data}"  # Inherited from deploy.sh; fallback for standalone runs
export RUN_NAME="brahmai-27b-test"
export MODEL_NAME="brahmai-27b"
export TOKENIZER="src/maxtext/brahmai_tokenizer_v2"
export DATASET_PATH="${MOUNTPOINT}/english_dclm"
export BASE_OUTPUT_DIR="${MOUNTPOINT}/${MODEL_NAME}/checkpoints"

# XLA / TPU tuning flags
export TENSORSTORE_NUM_THREADS=4
export LIBTPU_INIT_ARGS='--xla_tpu_scoped_vmem_limit_kib=98304 --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1 --xla_tpu_assign_all_reduce_scatter_layout=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true --xla_tpu_host_transfer_overlap_limit=24 --xla_tpu_aggressive_opt_barrier_removal=ENABLED --xla_lhs_prioritize_async_depth_over_stall=ENABLED --xla_latency_hiding_scheduler_rerun=2'

# ======================================================================
# Echo Variables
# ======================================================================
echo ""
echo "======================================================================"
echo "  brahmai-27b.sh — ENVIRONMENT"
echo "======================================================================"
echo "  RUN_NAME               = ${RUN_NAME}"
echo "  MODEL_NAME             = ${MODEL_NAME}"
echo "  TOKENIZER              = ${TOKENIZER}"
echo "  DATASET_PATH           = ${DATASET_PATH}"
echo "  BASE_OUTPUT_DIR        = ${BASE_OUTPUT_DIR}"
echo "  MOUNTPOINT             = ${MOUNTPOINT}"
echo "  TENSORSTORE_NUM_THREADS= ${TENSORSTORE_NUM_THREADS}"
echo "  HF_TOKEN               = ${HF_TOKEN:+(set)}"
echo "======================================================================"
echo ""

# ======================================================================
# Test Lustre Storage
# ======================================================================
echo 'Creating directories...'
mkdir -p "${MOUNTPOINT}/datasets/c4/" "${BASE_OUTPUT_DIR}"

echo 'Testing lustre storage...'
if [ "${JOB_COMPLETION_INDEX:-0}" = "0" ]; then
  if [ ! -f "${MOUNTPOINT}/datasets/c4/sync_complete" ]; then
    echo 'Head Pod: Cleaning up corrupted files from previous attempt...'
    rm -rf "${MOUNTPOINT}/datasets/c4/*" && \
      echo 'Head Pod: Syncing dataset to Lustre preserving TFDS structure...'
    touch "${MOUNTPOINT}/datasets/c4/sync_complete"
  else
    echo 'Head Pod: Dataset already synced. Skipping download.'
  fi
else
  echo 'Worker Pod: Waiting for Head Pod to finish downloading dataset...'
  while [ ! -f "${MOUNTPOINT}/datasets/c4/sync_complete" ]; do sleep 5; done
fi
echo 'Dataset sync confirmed'

# ======================================================================
# Start Pre-training
# ======================================================================
echo 'Starting MaxText Training...'

python3 -m maxtext.trainers.pre_train.train maxtext/configs/base.yml \
  model_name="$MODEL_NAME" \
  run_name="$RUN_NAME" \
  dataset_type='grain' \
  grain_train_files="${DATASET_PATH}/*.arrayrecord*" \
  grain_packing_type="best_fit" \
  grain_eval_files="${DATASET_PATH}/*.arrayrecord*" \
  base_output_directory="${BASE_OUTPUT_DIR}" \
  tokenizer_path="${TOKENIZER}" \
  tokenizer_type="huggingface" \
  grain_worker_count=8 \
  grain_prefetch_buffer_size=20 \
  per_device_batch_size=2 \
  max_target_length=4096 \
  attention='flash' \
  ici_fsdp_parallelism=-1 \
  remat_policy='full' \
  decoder_layer_input='offload' \
  query_proj='device' \
  scan_layers=True \
  checkpoint_period=5000 \
  log_period=50 \
  profiler="xplane" \
  skip_first_n_steps_for_profiler=10 \
  profiler_steps=5 \
  enable_checkpointing=True \
  async_checkpointing=True \
  upload_all_profiler_results=True
