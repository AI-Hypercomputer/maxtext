#!/bin/bash
# Launch gpt-oss-20b distillation on TPU v7x via Cluster Toolkit (gcluster).
# Usage: bash scripts/distillation/distill_gpt_oss_20b.sh [submit|monitor|cleanup|resume_until_done]
#
# Set DISTILL_GCS_BUCKET to your own GCS bucket and DOCKER_IMAGE to your own
# image before running — the placeholders below will not work as-is. Everything
# else has a working default. Example:
#
#   DISTILL_GCS_BUCKET=gs://your-bucket \
#   DOCKER_IMAGE=gcr.io/your-project/maxtext_base_image:tag \
#     bash scripts/distillation/distill_gpt_oss_20b.sh submit
set -euo pipefail

MODE="${1:-submit}"
REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO_ROOT"

# Your GCS bucket: run outputs, staged YAML, and tokenizer files all land here.
export DISTILL_GCS_BUCKET="${DISTILL_GCS_BUCKET:-gs://YOUR-BUCKET}"

export CTK_WORKLOAD="${CTK_WORKLOAD:-goss-base-$(date +%Y%m%d-%H%M)}"
export RUN_NAME="${RUN_NAME:-gpt_oss_20b_base}"
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

# Stage HF tokenizer files (not in the image for gpt-oss).
export CTK_TOKENIZER_GCS="${CTK_TOKENIZER_GCS:-${DISTILL_GCS_BUCKET}/distill-configs/tokenizer-gpt-oss-20b/}"
export CTK_TOKENIZER_LOCAL="${CTK_TOKENIZER_LOCAL:-/deps/src/maxtext/assets/tokenizers/gpt-oss-20b-tokenizer}"

LOCAL_YAML="src/maxtext/configs/post_train/distillation_gpt_oss_20b.yml"
export CTK_DISTILL_CONFIG="${CTK_DISTILL_CONFIG:-$LOCAL_YAML}"
export CTK_YAML_GCS="${CTK_YAML_GCS:-${DISTILL_GCS_BUCKET}/distill-configs/distillation_gpt_oss_20b.yml}"

# distill_beta=0: decoder feature loss is broken on gpt-oss.
export DISTILL_ALPHA="${DISTILL_ALPHA:-0.5}"
export DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-1.0}"
export DISTILL_BETA="${DISTILL_BETA:-0}"
export DISTILL_LAYER_INDICES="${DISTILL_LAYER_INDICES:-[]}"

# XLA flags tuned for ~17% MFU (~19% with context=device, the splash/megablox
# tile sizes, and the dp2 x fsdp64 mesh, all set in the config).
# sparse_core_collective_aggregator is required by latency_hiding_layer_scheduler.
export CTK_LIBTPU_INIT_ARGS="${CTK_LIBTPU_INIT_ARGS:---xla_tpu_scoped_vmem_limit_kib=65536 \
--xla_tpu_impure_enable_packed_bf16_math_ops=true \
--xla_tpu_aggressive_opt_barrier_removal=true \
--xla_tpu_enable_sparse_core_collective_aggregator=true \
--xla_tpu_enable_latency_hiding_layer_scheduler=true \
--xla_tpu_enable_layer_scheduler_for_dependent_collectives=true \
--xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=true \
--xla_tpu_scheduler_percent_shared_memory_limit=150 \
--xla_enable_async_all_gather=true \
--xla_tpu_prefer_async_allgather_to_allreduce=true \
--xla_max_concurrent_async_all_gathers=2 \
--xla_max_concurrent_async_reduce_scatters=2 \
--xla_tpu_enable_async_collective_fusion_fuse_all_gather=false}"

if [ "$MODE" = "submit" ]; then
  gcloud storage cp "$CTK_DISTILL_CONFIG" "$CTK_YAML_GCS"
fi

exec bash src/maxtext/trainers/post_train/distillation/scripts/run_distill_ctk.sh "$@"
