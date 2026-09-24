#!/bin/bash
# Fair SPMD vs Non-SPMD Streaming DiLoCo comparison on TPU v6e.
#
# FAIRNESS CONTRACT (this is the whole point of this script):
#   * ONE docker image, built once from THIS workspace, shared by every arm
#     -> both arms run byte-identical code. (The old scripts built the SPMD arm
#        from a different directory/branch than the Non-SPMD arm.)
#   * Identical model, dtypes, dataset, batch, steps, log_period, XLA flags.
#   * NO `tc` traffic shaping.
#   * NO profiler during timing runs (profiling is a separate invocation).
#   * Arms differ ONLY in the variable under test:
#       nonspmd  : enable_non_spmd_diloco=true  enable_single_controller=true  (Pathways)
#       spmd_pw  : enable_non_spmd_diloco=false enable_single_controller=true  (Pathways)
#       spmd     : enable_non_spmd_diloco=false enable_single_controller=false (McJAX)
#     `spmd_pw` is the true apples-to-apples arm (same runtime as nonspmd).
#     `spmd` is included because it is SPMD's normal/native runtime.
#
# Usage:
#   ./run_v6e_diloco_fair.sh build                  # build+push the shared image
#   ./run_v6e_diloco_fair.sh submit <arm> [suffix]  # arm = nonspmd | spmd_pw | spmd
#
# Env overrides: TAG, STEPS, PDBS, TAU, NFRAG, HSYNC, PROFILE=1
set -euo pipefail

# ---------------------------------------------------------------- cluster ---
CLUSTER=bodaborg-v6e-nap
PROJECT=tpu-prod-env-one-vm
ZONE=southamerica-west1
DEVICE_TYPE=v6e-8
NUM_SLICES=2

# ------------------------------------------------------------------ image ---
# Overridable so the non-SPMD arms can be built on the SAME jax/libtpu stack as
# the SPMD baseline from main. main pins jax>=0.11.1 / libtpu>=0.0.46 and runs
# on maxtext_jax_stable:latest; this branch pins jax>=0.10.2 / libtpu>=0.0.42.1.
# Comparing across those two stacks is NOT apples-to-apples -- the branch's SPMD
# hit `LLO_CHECK ... ConstantFitsInSingleImmediateWithSignex()` on v6e with the
# old libtpu and compiled fine on the new one, so the stack demonstrably matters
# on this hardware.
DOCKER_IMAGE_BASE="${DOCKER_IMAGE_BASE:-gcr.io/tpu-prod-env-multipod/maxtext_jax_stable:2026-07-17}"
TAG="${TAG:-v6e-fair-20260923}"
MY_IMAGE="gcr.io/cloud-tpu-multipod-dev/jzuo-runner:${TAG}"

# -------------------------------------------------------- shared workload ---
BASE_OUTPUT_DIRECTORY="gs://chriszuo-maxtext-logs"
MODEL_NAME="qwen3-8b"
# v6e-8 = 8 chips/slice. v5p-8 = 4 chips/slice with pdbs=8 -> 32 seq/slice.
# pdbs=4 on v6e keeps the per-slice global batch at 32 seq AND fits 32GB HBM.
PDBS="${PDBS:-4}"
MAX_TARGET_LENGTH=2048
STEPS="${STEPS:-120}"
LOG_PERIOD=1

NFRAG="${NFRAG:-37}"     # 36 decoder layers + 1 flat fragment
HSYNC="${HSYNC:-37}"     # => steps_between_syncs_plus_1 = 1 (one fragment per step)
TAU="${TAU:-5}"
OUTER_LR=0.1
OUTER_MOMENTUM=0.9

# ---------------------------------------------------------------------------
# Performance XLA flags, applied IDENTICALLY to both runtimes.
#
# Two things had to be fixed to make this correct:
#
#  1. `--xla_tpu_enable_latency_hiding_layer_scheduler=true` REQUIRES
#     `--xla_tpu_enable_sparse_core_collective_aggregator=true`. Omitting the
#     latter is a hard failure on McJAX:
#       INVALID_ARGUMENT: Latency hiding layer scheduler requires sparse core
#       collective aggregator to be enabled.
#     (this killed v6e-spmd-e0 at setup_train_loop). Both are present below.
#
#  2. Under Pathways the TPU is owned by the `pathways-worker` container, not
#     the `jax-tpu` container where the workload command exports
#     LIBTPU_INIT_ARGS -- so these flags were previously IGNORED on Pathways
#     and APPLIED on McJAX. `MyStuff/scripts/patch_manifest.py` now injects
#     LIBTPU_INIT_ARGS into the pathways-worker env so both runtimes get them.
#
# Deliberately excluded: the xprof tracing flags (--xla_enable_hlo_trace,
# --xla_xprof_*) which perturb timing; they are added only for profiling runs.
# ---------------------------------------------------------------------------
#  2. Restoring the FULL v5p-tuned flag set (27 flags, heavy on SparseCore
#     offload + `--xla_sc_disable_megacore_partitioning` +
#     `--xla_tpu_use_tc_device_shape_on_sc`) makes the v6e backend fail codegen
#     for the drjax SPMD program:
#       INTERNAL: LLO_CHECK failure (llo_region_builder.cc:5268)
#       displacement->ConstantFitsInSingleImmediateWithSignex()
#                                                      [killed v6e-sp-f1]
#     Those flags were tuned for v5p and are NOT portable to v6e.
#
# Conservative subset below: generic scheduling / async-collective flags with
# no SparseCore codegen involvement. Override with XLA_FLAGS=... to experiment,
# but re-validate that BOTH arms COMPILE before trusting any comparison.
XLA_FLAGS="${XLA_FLAGS:- \
  --xla_tpu_scoped_vmem_limit_kib=65536 \
  --xla_tpu_bf16_emission_mode=NATIVE_EMISSION \
  --xla_enable_async_all_gather=true \
  --xla_tpu_prefer_async_allgather_to_allreduce=true \
  --xla_tpu_aggressive_opt_barrier_removal=true }"

build() {
  echo "Building shared image ${MY_IMAGE} from $(pwd) ..."
  docker build -t "${MY_IMAGE}" -f - . <<INNER_EOF
FROM ${DOCKER_IMAGE_BASE}
WORKDIR /app
COPY . .
RUN find /app -name "*.pyc" -delete && find /app -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
INNER_EOF
  docker push "${MY_IMAGE}"
  echo "Pushed ${MY_IMAGE}"
  echo "IMAGE_DIGEST: $(docker inspect --format='{{index .RepoDigests 0}}' "${MY_IMAGE}" 2>/dev/null || echo unknown)"
}

submit() {
  local ARM="$1"; local SUFFIX="${2:-}"
  local NONSPMD SINGLECTRL PATHWAYS COCPU CODE
  COCPU=0
  # Short codes: xpk requires the workload name to match
  # `[a-z]([-a-z0-9]*[a-z0-9])?` and stay well under the 63-byte JobSet label
  # limit. Truncating a long name can leave a trailing '-' and get rejected.
  case "${ARM}" in
    nonspmd)       NONSPMD=true;  SINGLECTRL=true;  PATHWAYS=1; CODE=ns ;;
    nonspmd_cocpu) NONSPMD=true;  SINGLECTRL=true;  PATHWAYS=1; COCPU=1; CODE=nscc ;;
    spmd_pw)       NONSPMD=false; SINGLECTRL=true;  PATHWAYS=1; CODE=sppw ;;
    spmd)          NONSPMD=false; SINGLECTRL=false; PATHWAYS=0; CODE=sp ;;
    *) echo "unknown arm '${ARM}' (want nonspmd|nonspmd_cocpu|spmd_pw|spmd)"; exit 2 ;;
  esac

  local RUNNAME="v6e-${CODE}${SUFFIX:+-$SUFFIX}"
  RUNNAME="${RUNNAME:0:18}"

  local PROF_ARGS=""
  if [[ "${PROFILE:-0}" == "1" ]]; then
    # PROF_STEPS>1 is the point: previous attempts used profiler_steps=1 and got
    # a window too short to contain a full async DiLoCo cycle. We also set
    # upload_all_profiler_results so every worker uploads its plane, and
    # enable_tpu_profiling_options so the TPU-side knobs are honoured.
    #
    # The host-side `max_num_hosts` bug is fixed in common/profiler.py; without
    # that fix Pathways collects exactly ONE worker host regardless of these.
    PROF_ARGS="profiler=xplane \
               skip_first_n_steps_for_profiler=${PROF_SKIP:-40} \
               profiler_steps=${PROF_STEPS:-5} \
               profile_cleanly=true \
               upload_all_profiler_results=true \
               enable_tpu_profiling_options=true \
               tpu_num_chips_to_profile_per_task=4 \
               profiler_max_num_hosts=${PROF_HOSTS:-4}"
  fi

  # NOTE: tau is passed to BOTH arms. For SPMD it is the in-graph delayed-apply
  # depth (diloco.py:467-476); for Non-SPMD it is the prefetch pipeline depth.
  # Keeping them equal is what makes the algorithms comparable.
  local CMD="export PYTHONPATH=/app/src:\$PYTHONPATH && export JAX_NUM_CPU_DEVICES=8 \
&& export DILOCO_COLOCATED_CPU_OUTER=${COCPU} \
&& export DILOCO_SHARDED_APPLY=${SHARDED_APPLY:-1} && export DILOCO_DONATE_APPLY=${DONATE_APPLY:-1} \
&& export LIBTPU_INIT_ARGS='${XLA_FLAGS}' && cd /app/src/ \
&& python3 maxtext/trainers/pre_train/train.py \
     maxtext/configs/base.yml \
     run_name=${RUNNAME} \
     base_output_directory=${BASE_OUTPUT_DIRECTORY} \
     save_config_to_gcs=true \
     dataset_type=synthetic \
     model_name=${MODEL_NAME} \
     weight_dtype=bfloat16 \
     dtype=bfloat16 \
     per_device_batch_size=${PDBS} \
     max_target_length=${MAX_TARGET_LENGTH} \
     steps=${STEPS} \
     log_period=${LOG_PERIOD} \
     enable_checkpointing=false \
     pure_nnx=true \
     enable_diloco=true \
     enable_streaming_diloco=true \
     enable_non_spmd_diloco=${NONSPMD} \
     enable_single_controller=${SINGLECTRL} \
     dcn_diloco_parallelism=${NUM_SLICES} \
     num_diloco_fragments=${NFRAG} \
     diloco_sync_period=${HSYNC} \
     num_communication_overlapping_steps=${TAU} \
     communication_overlapping_alpha=0.0 \
     use_sequential_layers=false \
     diloco_outer_lr=${OUTER_LR} \
     diloco_outer_momentum=${OUTER_MOMENTUM} \
     ${PROF_ARGS}"

  echo "=========================================================="
  echo " arm=${ARM}  run=${RUNNAME}"
  echo " cluster=${CLUSTER} (${PROJECT}/${ZONE}) ${NUM_SLICES}x${DEVICE_TYPE}"
  echo " image=${MY_IMAGE}"
  echo " pdbs=${PDBS} steps=${STEPS} nfrag=${NFRAG} H=${HSYNC} tau=${TAU}"
  echo " pathways=${PATHWAYS}  tc_shaping=NONE  profiler=${PROFILE:-0}"
  echo "=========================================================="

  local XPK=/usr/local/google/home/jzuo/xpk_venv/bin/xpk
  local SUB="workload create"
  [[ "${PATHWAYS}" == "1" ]] && SUB="workload create-pathways"

  # xpk makes Cloud Resource Manager calls whose "consumer" is the ADC
  # quota project, not --project. On this workstation the ADC quota project is
  # `ais-01-855664`, which does not have cloudresourcemanager.googleapis.com
  # enabled, so xpk dies before doing anything. Override per-invocation rather
  # than mutating the user's global gcloud config.
  export CLOUDSDK_CORE_PROJECT="${PROJECT}"
  export GOOGLE_CLOUD_PROJECT="${PROJECT}"
  export GOOGLE_CLOUD_QUOTA_PROJECT="${PROJECT}"

  xpk_apply "${SUB}" "${RUNNAME}" "${CMD}"
}

# ---------------------------------------------------------------------------
# xpk renders the Pathways head pod with THREE containers, all limits-only
# (so requests == limits), and the two pathways containers are native sidecars
# (initContainers with restartPolicy: Always), which means their resources ADD
# to the pod total:
#     pathways-proxy 100G + pathways-rm 32G + jax-tpu 200G = 332G
# The cpu-np pool here is n2d-standard-64 with allocatable 251523088Ki
# (~239.9 GiB / ~257.6 GB), so the pod can never be scheduled:
#     "0/50 nodes are available: ... 30 Insufficient memory"
# There is no xpk flag for this, so we render with --dry-run and shrink all
# three before applying. New total 70+16+130 = 216G (~201 GiB), leaving ~39 GiB
# of headroom for system daemonsets.
# Measured context: the Non-SPMD client was ~4 GB RSS and SPMD ~32 GB, so these
# limits remain far above observed usage.
# ---------------------------------------------------------------------------
PROXY_MEM="${PROXY_MEM:-70G}"
RM_MEM="${RM_MEM:-16G}"
HEAD_MEM="${HEAD_MEM:-130G}"
HEAD_CPU="${HEAD_CPU:-20}"

xpk_apply() {
  local SUB="$1"; local RUNNAME="$2"; local CMD="$3"
  local XPK=/usr/local/google/home/jzuo/xpk_venv/bin/xpk
  local MANIFEST="MyStuff/Data/manifests/${RUNNAME}.yaml"
  mkdir -p "$(dirname "${MANIFEST}")"

  ${XPK} ${SUB} --workload "${RUNNAME}" \
    --docker-image "${MY_IMAGE}" \
    --command "${CMD}" \
    --num-slices=${NUM_SLICES} \
    --priority medium \
    --cluster "${CLUSTER}" --tpu-type "${DEVICE_TYPE}" \
    --project "${PROJECT}" --zone "${ZONE}" \
    --dry-run --output-manifest-file "${MANIFEST}"

  if [[ ! -s "${MANIFEST}" ]]; then
    echo "ERROR: xpk did not produce ${MANIFEST}"; return 1
  fi

  # NOTE: `|| true` is required. The script runs under `set -euo pipefail`, and
  # grep exits 1 when there are no matches. The McJAX manifest has no
  # `memory:`/`cpu:` limits, so this diagnostic silently aborted the whole
  # submit function before kubectl apply ever ran.
  echo "--- head resources BEFORE patch ---"
  grep -nE "memory:|cpu: " "${MANIFEST}" | head -8 || true
  # NOTE: must use --opt=value form. `--libtpu-init-args "--xla_foo=1"` makes
  # argparse treat the value as a new flag ("expected one argument").
  python3 MyStuff/scripts/patch_manifest.py "${MANIFEST}" \
    --proxy-mem="${PROXY_MEM}" --rm-mem="${RM_MEM}" \
    --head-mem="${HEAD_MEM}" --head-cpu="${HEAD_CPU}" \
    --libtpu-init-args="${XLA_FLAGS}" || { echo "ERROR: manifest patch failed"; return 1; }
  echo "--- verify LIBTPU_INIT_ARGS reached the worker ---"
  grep -c "LIBTPU_INIT_ARGS" "${MANIFEST}" || true

  gcloud container clusters get-credentials "${CLUSTER}" --location="${ZONE}" \
    --project="${PROJECT}" >/dev/null 2>&1
  kubectl apply -f "${MANIFEST}"
  echo "applied ${MANIFEST}"
}

bench() {
  # Runs the TPU -> colocated-CPU-mesh transfer microbenchmark under Pathways.
  local SUFFIX="${1:-}"
  local RUNNAME="v6e-cocpu${SUFFIX:+-$SUFFIX}"
  RUNNAME="${RUNNAME:0:18}"
  local MB="${MB:-360}"

  local CMD="export PYTHONPATH=/app/src:\$PYTHONPATH && export JAX_NUM_CPU_DEVICES=8 \
&& export LIBTPU_INIT_ARGS='${XLA_FLAGS}' && cd /app \
&& python3 MyStuff/scripts/bench_colocated_cpu_transfer.py --mb ${MB} --iters 7 --learners 2 \
; echo BENCH_EXIT_CODE=\$?"

  echo "=== microbenchmark ${RUNNAME}  payload=${MB} MB  image=${MY_IMAGE} ==="
  export CLOUDSDK_CORE_PROJECT="${PROJECT}"
  export GOOGLE_CLOUD_PROJECT="${PROJECT}"
  export GOOGLE_CLOUD_QUOTA_PROJECT="${PROJECT}"
  xpk_apply "workload create-pathways" "${RUNNAME}" "${CMD}"
}

case "${1:-}" in
  build)  build ;;
  submit) shift; submit "$@" ;;
  bench)  shift; bench "$@" ;;
  *) echo "usage: $0 build | $0 submit <nonspmd|spmd_pw|spmd> [suffix] | $0 bench [suffix]"; exit 2 ;;
esac
