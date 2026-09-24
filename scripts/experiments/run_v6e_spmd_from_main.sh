#!/bin/bash
# Build + submit the SPMD streaming DiLoCo baseline from ORIGIN/MAIN.
#
# WHY A SEPARATE SCRIPT:
#   The SPMD DiLoCo implementation on branch chris/dev/non-spmd-sept does not
#   compile on v6e. With ZERO XLA flags it still dies at
#     p_train_step.lower(...).compile(...)
#     INTERNAL: LLO_CHECK failure (llo_region_builder.cc:5268)
#     displacement->ConstantFitsInSingleImmediateWithSignex()
#   (runs v6e-sp-f1 / v6e-sp-g1 / v6e-sp-h1, the last with no flags at all).
#   main has a restructured + merged SPMD implementation
#   (diloco/utils/spmd_diloco_sync.py etc.), so we build the baseline from a
#   detached worktree at origin/main rather than rebasing 73 commits over 575
#   mid-experiment.
#
# Usage: run_v6e_spmd_from_main.sh build | submit [suffix]
set -uo pipefail

WT=MyStuff/tmp/main-wt
CLUSTER=bodaborg-v6e-nap
PROJECT=tpu-prod-env-one-vm
ZONE=southamerica-west1
DEVICE_TYPE=v6e-8
NUM_SLICES=2
TAG="${TAG:-v6e-main-spmd}"
MY_IMAGE="gcr.io/cloud-tpu-multipod-dev/jzuo-runner:${TAG}"
# main pins jax>=0.11.1 / libtpu>=0.0.46 (src/dependencies/requirements/
# generated_requirements/tpu-requirements.txt), whereas this branch pins
# jax>=0.10.2 / libtpu>=0.0.42.1. Building main on the 2026-07-17 base (jax
# 0.10.2) fails at import:
#   ImportError: cannot import name 'must_fuse_call' from
#   'jax.experimental.xla_metadata'
# main's own script uses maxtext_jax_stable:latest, so default to that.
DOCKER_IMAGE_BASE="${DOCKER_IMAGE_BASE:-gcr.io/tpu-prod-env-multipod/maxtext_jax_stable:latest}"

# Match the non-SPMD arms exactly so the comparison is meaningful.
MODEL_NAME=qwen3-8b
PDBS="${PDBS:-4}"
MAX_TARGET_LENGTH=2048
STEPS="${STEPS:-120}"
NFRAG="${NFRAG:-37}"
HSYNC="${HSYNC:-37}"
TAU="${TAU:-5}"
XLA_FLAGS="${XLA_FLAGS:-}"

export CLOUDSDK_CORE_PROJECT="${PROJECT}"
export GOOGLE_CLOUD_PROJECT="${PROJECT}"
export GOOGLE_CLOUD_QUOTA_PROJECT="${PROJECT}"

build() {
  [ -d "${WT}" ] || { echo "ERROR: worktree ${WT} missing"; exit 1; }
  cp .dockerignore "${WT}/.dockerignore" 2>/dev/null || true
  echo "Building ${MY_IMAGE} from origin/main worktree ($(git -C "${WT}" rev-parse --short HEAD)) ..."
  docker build -t "${MY_IMAGE}" -f - "${WT}" <<INNER_EOF
FROM ${DOCKER_IMAGE_BASE}
WORKDIR /app
COPY . .
RUN find /app -name "*.pyc" -delete && find /app -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
INNER_EOF
  [ $? -eq 0 ] || { echo "ERROR: docker build failed"; exit 1; }
  docker push "${MY_IMAGE}" | tail -2
}

submit() {
  local SUFFIX="${1:-m1}"
  local RUNNAME="v6e-spm-${SUFFIX}"
  local MANIFEST="MyStuff/Data/manifests/${RUNNAME}.yaml"
  mkdir -p "$(dirname "${MANIFEST}")"

  local CMD="export PYTHONPATH=/app/src:\$PYTHONPATH && unset XLA_FLAGS \
&& export LIBTPU_INIT_ARGS='${XLA_FLAGS}' && cd /app/src/ \
&& python3 maxtext/trainers/pre_train/train.py \
     maxtext/configs/base.yml \
     run_name=${RUNNAME} \
     base_output_directory=gs://chriszuo-maxtext-logs \
     dataset_type=synthetic \
     model_name=${MODEL_NAME} \
     weight_dtype=bfloat16 \
     dtype=bfloat16 \
     per_device_batch_size=${PDBS} \
     max_target_length=${MAX_TARGET_LENGTH} \
     steps=${STEPS} \
     log_period=1 \
     enable_checkpointing=false \
     enable_diloco=true \
     enable_streaming_diloco=true \
     dcn_diloco_parallelism=${NUM_SLICES} \
     num_diloco_fragments=${NFRAG} \
     diloco_sync_period=${HSYNC} \
     num_communication_overlapping_steps=${TAU} \
     communication_overlapping_alpha=0.0 \
     use_sequential_layers=false \
     diloco_outer_lr=0.1 \
     diloco_outer_momentum=0.9"

  echo "=== ${RUNNAME} from origin/main  image=${MY_IMAGE} ==="
  /usr/local/google/home/jzuo/xpk_venv/bin/xpk workload create --workload "${RUNNAME}" \
    --docker-image "${MY_IMAGE}" --command "${CMD}" \
    --num-slices=${NUM_SLICES} --priority medium \
    --cluster "${CLUSTER}" --tpu-type "${DEVICE_TYPE}" \
    --project "${PROJECT}" --zone "${ZONE}" \
    --dry-run --output-manifest-file "${MANIFEST}"
  [ -s "${MANIFEST}" ] || { echo "ERROR: no manifest produced"; exit 1; }
  gcloud container clusters get-credentials "${CLUSTER}" --location="${ZONE}" \
    --project="${PROJECT}" >/dev/null 2>&1
  kubectl apply -f "${MANIFEST}"
}

case "${1:-}" in
  build) build ;;
  submit) shift; submit "$@" ;;
  *) echo "usage: $0 build | $0 submit [suffix]"; exit 2 ;;
esac
