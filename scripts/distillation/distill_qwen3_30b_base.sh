#!/bin/bash
# Launch qwen3-30b-a3b-base distillation on TPU v7x via Cluster Toolkit (gcluster).
# Usage: bash scripts/distillation/distill_qwen3_30b_base.sh [submit|monitor|cleanup|resume_until_done]
#
# Set DISTILL_GCS_BUCKET to your own GCS bucket and DOCKER_IMAGE to your own
# image before running — the placeholders below will not work as-is. Everything
# else has a working default. Example:
#
#   DISTILL_GCS_BUCKET=gs://your-bucket \
#   DOCKER_IMAGE=gcr.io/your-project/maxtext_base_image:tag \
#     bash scripts/distillation/distill_qwen3_30b_base.sh submit
set -euo pipefail

MODE="${1:-submit}"
REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO_ROOT"

# Your GCS bucket: run outputs and staged YAML land here.
export DISTILL_GCS_BUCKET="${DISTILL_GCS_BUCKET:-gs://YOUR-BUCKET}"

export CTK_WORKLOAD="${CTK_WORKLOAD:-q30b-base-$(date +%Y%m%d-%H%M)}"
export RUN_NAME="${RUN_NAME:-qwen3_30b_base}"
export GKE_CLUSTER="${GKE_CLUSTER:-bodaborg-super-xpk-x8p}"
export PROJECT_ID="${PROJECT_ID:-cloud-tpu-multipod-dev}"
export LOCATION="${LOCATION:-us-central1}"
export COMPUTE_TYPE="${COMPUTE_TYPE:-tpu7x-standard-4t}"
export TOPOLOGY="${TOPOLOGY:-4x4x4}"
export BASE_OUTPUT_DIRECTORY="${BASE_OUTPUT_DIRECTORY:-${DISTILL_GCS_BUCKET}/distillation}"
export DOCKER_IMAGE="${DOCKER_IMAGE:-gcr.io/cloud-tpu-multipod-dev/maxtext_base_image:agagik-distill}"
export CTK_PRIORITY="${CTK_PRIORITY:-high}"

export CTK_USE_GCSFUSE=1
export CTK_DATASET_BUCKET="${CTK_DATASET_BUCKET:-maxtext-dataset}"
export CTK_DATASET_SUBPATH="${CTK_DATASET_SUBPATH:-array-record/climbmix/*.arrayrecord}"

LOCAL_YAML="src/maxtext/configs/post_train/distillation_qwen3_30b_base.yml"
export CTK_DISTILL_CONFIG="${CTK_DISTILL_CONFIG:-$LOCAL_YAML}"
export CTK_YAML_GCS="${CTK_YAML_GCS:-${DISTILL_GCS_BUCKET}/distill-configs/distillation_qwen3_30b_base.yml}"

export DISTILL_ALPHA="${DISTILL_ALPHA:-0.6}"
export DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-1.0}"
export DISTILL_BETA="${DISTILL_BETA:-1.0}"
export DISTILL_LAYER_INDICES="${DISTILL_LAYER_INDICES:-[0,1,2,3,4,5,6,7]}"

# XLA flags tuned for ~20% MFU. With the megablox tiles, SEQ_MINOR q-layout,
# and context=device now set in the configs: ~26% MFU (default config) /
# ~24% (pdbs=8 + activation offload variant).
export CTK_LIBTPU_INIT_ARGS="${CTK_LIBTPU_INIT_ARGS:---xla_tpu_scoped_vmem_limit_kib=61440 \
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

if [ "$MODE" = "submit" ]; then
  gcloud storage cp "$CTK_DISTILL_CONFIG" "$CTK_YAML_GCS"
fi

exec bash src/maxtext/trainers/post_train/distillation/scripts/run_distill_ctk.sh "$@"
