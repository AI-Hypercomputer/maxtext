#!/bin/bash
# SAME-TREE, SAME-IMAGE SPMD vs non-SPMD streaming DiLoCo on 2 x v6e-8.
#
# Both arms run from ONE docker image built from MyStuff/tmp/port-wt
# (branch chris/dev/nonspmd-on-main = main 51502bad7 + ported threaded DiLoCo).
# The SPMD code (trainers/diloco/diloco.py + diloco/utils/*) in that tree is
# byte-identical to 51502bad7, the commit the SPMD baseline v6e-spm-m2 ran.
#
# Identical workload for both arms (defaults; override MODEL/PDBS/NFRAG/HSYNC/DEVICE_TYPE):
# qwen3-8b, bf16, pdbs=4, seq 2048, synthetic,
# 120 steps, N_frags=37, H=37, tau=5, alpha=0, outer lr 0.1 / momentum 0.9,
# log_period=1, no checkpointing, no tc shaping, no profiler, same XLA flags
# (default: none).
#
# Arms:
#   spmd : McJAX (SPMD's native runtime; SPMD on Pathways is broken, see progress2.md)
#   ns   : Pathways, enable_non_spmd_diloco=true, colocated CPU outer,
#          DILOCO_SHARDED_APPLY=${SHARDED_APPLY:-1}
#
# Usage: run_v6e_portwt.sh build | submit <spmd|ns> <suffix>
set -euo pipefail

WT=MyStuff/tmp/port-wt
CLUSTER=bodaborg-v6e-nap
PROJECT=tpu-prod-env-one-vm
ZONE=southamerica-west1
DEVICE_TYPE="${DEVICE_TYPE:-v6e-8}"
NUM_SLICES=2
TAG="${TAG:-v6e-portwt-1}"
MY_IMAGE="gcr.io/cloud-tpu-multipod-dev/jzuo-runner:${TAG}"
DOCKER_IMAGE_BASE="${DOCKER_IMAGE_BASE:-gcr.io/tpu-prod-env-multipod/maxtext_jax_stable:latest}"
STEPS="${STEPS:-120}"
MODEL="${MODEL:-qwen3-8b}"
PDBS="${PDBS:-4}"
NFRAG="${NFRAG:-37}"   # num decoder layers + 1
HSYNC="${HSYNC:-${NFRAG}}"  # H = NFRAG -> one fragment per step
TAU="${TAU:-5}"
XLA_FLAGS="${XLA_FLAGS:- }"

export CLOUDSDK_CORE_PROJECT="${PROJECT}"
export GOOGLE_CLOUD_PROJECT="${PROJECT}"
export GOOGLE_CLOUD_QUOTA_PROJECT="${PROJECT}"

build() {
  cp .dockerignore "${WT}/.dockerignore" 2>/dev/null || true
  echo "Building ${MY_IMAGE} from ${WT} ($(git -C "${WT}" rev-parse --short HEAD), dirty=$(git -C "${WT}" status --short src tests | wc -l))"
  docker build -t "${MY_IMAGE}" -f - "${WT}" <<INNER_EOF
FROM ${DOCKER_IMAGE_BASE}
WORKDIR /app
COPY . .
RUN find /app -name "*.pyc" -delete && find /app -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
INNER_EOF
  docker push "${MY_IMAGE}" | tail -1
  echo "IMAGE_DIGEST: $(docker inspect --format='{{index .RepoDigests 0}}' "${MY_IMAGE}")"
}

submit() {
  local ARM="$1"; local SUFFIX="$2"
  local RUNNAME="v6e-pw${ARM}-${SUFFIX}"
  local SUB ENVS DILOCO_ARGS
  local COMMON="maxtext/configs/base.yml run_name=${RUNNAME} \
     base_output_directory=gs://chriszuo-maxtext-logs dataset_type=synthetic \
     model_name=${MODEL} weight_dtype=bfloat16 dtype=bfloat16 \
     per_device_batch_size=${PDBS} max_target_length=2048 steps=${STEPS} log_period=1 \
     enable_checkpointing=false enable_diloco=true enable_streaming_diloco=true \
     dcn_diloco_parallelism=${NUM_SLICES} num_diloco_fragments=${NFRAG} diloco_sync_period=${HSYNC} \
     num_communication_overlapping_steps=${TAU} communication_overlapping_alpha=0.0 \
     use_sequential_layers=false diloco_outer_lr=0.1 diloco_outer_momentum=0.9"
  if [[ "${PROFILE:-0}" == "1" ]]; then
    # All 4 worker hosts (2 slices x 2 hosts) x 4 chips. profiler_max_num_hosts
    # is passed explicitly because the topology probe under-counted hosts on
    # v6e Pathways (v6e-nscc-p1 logged "host_groups=2 (via host_id)").
    COMMON="${COMMON} profiler=xplane skip_first_n_steps_for_profiler=${PROF_SKIP:-40} \
     profiler_steps=${PROF_STEPS:-5} profile_cleanly=true upload_all_profiler_results=true \
     enable_tpu_profiling_options=${PROF_TPU_OPTS:-true} tpu_num_chips_to_profile_per_task=4 \
     profiler_max_num_hosts=${PROF_HOSTS:-4}"
  fi
  case "${ARM}" in
    spmd)
      SUB="workload create"
      ENVS="unset XLA_FLAGS"
      DILOCO_ARGS="" ;;
    ns)
      SUB="workload create-pathways"
      ENVS="export JAX_NUM_CPU_DEVICES=8 && export DILOCO_COLOCATED_CPU_OUTER=1 \
&& export DILOCO_SHARDED_APPLY=${SHARDED_APPLY:-1} && export DILOCO_DONATE_APPLY=${DONATE_APPLY:-1} \
&& export DILOCO_SYMMETRIC_OUTER=${SYMMETRIC_OUTER:-1} && export DILOCO_UNPACKED_TRANSFER=${UNPACKED_TRANSFER:-1}"
      DILOCO_ARGS="enable_non_spmd_diloco=true enable_single_controller=true" ;;
    *) echo "arm must be spmd|ns"; exit 2 ;;
  esac
  local CMD="export PYTHONPATH=/app/src:\$PYTHONPATH && ${ENVS} \
&& export LIBTPU_INIT_ARGS='${XLA_FLAGS}' && cd /app/src/ \
&& python3 maxtext/trainers/pre_train/train.py ${COMMON} ${DILOCO_ARGS}; echo EXIT_CODE=\$?"

  local MANIFEST="MyStuff/Data/manifests/${RUNNAME}.yaml"
  mkdir -p "$(dirname "${MANIFEST}")"
  echo "=== ${RUNNAME} arm=${ARM} image=${MY_IMAGE} ${NUM_SLICES}x${DEVICE_TYPE} model=${MODEL} pdbs=${PDBS} nfrag=${NFRAG} H=${HSYNC} tau=${TAU} ==="
  /usr/local/google/home/jzuo/xpk_venv/bin/xpk ${SUB} --workload "${RUNNAME}" \
    --docker-image "${MY_IMAGE}" --command "${CMD}" \
    --num-slices=${NUM_SLICES} --priority medium \
    --cluster "${CLUSTER}" --tpu-type "${DEVICE_TYPE}" \
    --project "${PROJECT}" --zone "${ZONE}" \
    --dry-run --output-manifest-file "${MANIFEST}" >/dev/null
  [ -s "${MANIFEST}" ] || { echo "ERROR: no manifest"; exit 1; }
  python3 MyStuff/scripts/patch_manifest.py "${MANIFEST}" \
    --proxy-mem=70G --rm-mem=16G --head-mem=130G --head-cpu=20 \
    --libtpu-init-args="${XLA_FLAGS}"
  gcloud container clusters get-credentials "${CLUSTER}" --location="${ZONE}" \
    --project="${PROJECT}" >/dev/null 2>&1
  kubectl apply -f "${MANIFEST}"
}

case "${1:-}" in
  build) build ;;
  submit) shift; submit "$@" ;;
  *) echo "usage: $0 build | submit <spmd|ns> <suffix>"; exit 2 ;;
esac
