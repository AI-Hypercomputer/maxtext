#!/bin/bash
# Launch qwen3-30b-a3b-base distillation on TPU v7x.
# Usage: bash scripts/distillation/distill_qwen3_30b_base.sh [submit|monitor|resume_until_done]
#
# Set DISTILL_GCS_BUCKET to your own GCS bucket and XPK_BASE_IMAGE to your own
# image before running — the placeholders below will not work as-is. Everything
# else has a working default. Example:
#
  # DISTILL_GCS_BUCKET=gs://ajkv-distillation \ß
  # XPK_BASE_IMAGE=gcr.io/cloud-tpu-multipod-dev/maxtext_base_image:latest \
  #   bash scripts/distillation/distill_qwen3_30b_base.sh submit
set -euo pipefail

MODE="${1:-submit}"
REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO_ROOT"

# Your GCS bucket: run outputs and staged YAML land here.
export DISTILL_GCS_BUCKET="${DISTILL_GCS_BUCKET:-gs://ajkv-distillation}"

export XPK_WORKLOAD="${XPK_WORKLOAD:-q30b-base-$(date +%Y%m%d-%H%M)}"
export XPK_RUN_NAME="${XPK_RUN_NAME:-qwen3_30b_base}"
export XPK_CLUSTER="${XPK_CLUSTER:-bodaborg-super-xpk-x8p}"
export XPK_PROJECT="${XPK_PROJECT:-cloud-tpu-multipod-dev}"
export XPK_ZONE="${XPK_ZONE:-us-central1}"
export XPK_DEVICE_TYPE="${XPK_DEVICE_TYPE:-tpu7x-4x4x4}"
export XPK_BASE_OUTPUT_DIR="${XPK_BASE_OUTPUT_DIR:-${DISTILL_GCS_BUCKET}/distillation}"
export XPK_BASE_IMAGE="${XPK_BASE_IMAGE:-gcr.io/cloud-tpu-multipod-dev/maxtext_base_image:latest}"
export XPK_PRIORITY="${XPK_PRIORITY:-high}"

export XPK_USE_GCSFUSE=1
export XPK_DATASET_BUCKET="${XPK_DATASET_BUCKET:-maxtext-dataset}"
export XPK_DATASET_SUBPATH="${XPK_DATASET_SUBPATH:-array-record/climbmix/*.arrayrecord}"

LOCAL_YAML="src/maxtext/configs/post_train/distillation.yml"
export XPK_DISTILL_CONFIG="${XPK_DISTILL_CONFIG:-$LOCAL_YAML}"
export XPK_YAML_GCS="${XPK_YAML_GCS:-${DISTILL_GCS_BUCKET}/distill-configs/distillation.yml}"

export DISTILL_ALPHA="${DISTILL_ALPHA:-0.6}"
export DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-1.0}"
export DISTILL_BETA="${DISTILL_BETA:-0.0}"
export DISTILL_TEACHER_TOP_K="${DISTILL_TEACHER_TOP_K:-32}"
export DISTILL_LAYER_INDICES="${DISTILL_LAYER_INDICES:-[0,1,2,3,4,5,6,7]}"

# XLA flags tuned for ~20% MFU.
export XPK_LIBTPU_INIT_ARGS="${XPK_LIBTPU_INIT_ARGS:---xla_tpu_scoped_vmem_limit_kib=61440 \
--xla_tpu_enable_all_experimental_scheduler_features=true \
--xla_tpu_enable_scheduler_memory_pressure_tracking=true \
--xla_tpu_host_transfer_overlap_limit=24 \
--xla_tpu_aggressive_opt_barrier_removal=ENABLED \
--xla_lhs_prioritize_async_depth_over_stall=ENABLED \
--xla_tpu_enable_ag_backward_pipelining=true \
--xla_should_allow_loop_variant_parameter_in_chain=ENABLED \
--xla_should_add_loop_invariant_op_in_chain=ENABLED \
--xla_max_concurrent_host_send_recv=100 \
--xla_tpu_scheduler_percent_shared_memory_limit=100 \
--xla_latency_hiding_scheduler_rerun=2}"

if [ "$MODE" = "submit" ] || [ "$MODE" = "resume_until_done" ]; then
  gcloud storage cp "$LOCAL_YAML" "$XPK_YAML_GCS"
fi

exec bash src/maxtext/trainers/post_train/distillation/scripts/run_distill_xpk.sh "$MODE"
