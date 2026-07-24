#!/usr/bin/env bash
#
# Builds the current non-SPMD streaming DiLoCo sources and launches a staged
# acceptance test on a Google Cloud Pathways cluster.
#
# Start with:
#   PROJECT_ID=my-project LOCATION=us-east5 \
#   BASE_OUTPUT_DIRECTORY=gs://my-bucket/maxtext \
#   ./scripts/diloco/run_streaming_diloco_gcloud.sh plan tiny
#
# Then submit the phases in order:
#   .../run_streaming_diloco_gcloud.sh submit layout
#   .../run_streaming_diloco_gcloud.sh submit tiny
#   .../run_streaming_diloco_gcloud.sh submit qwen8b

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
WORKLOAD_RUNNER="${SCRIPT_DIR}/run_streaming_diloco_acceptance_workload.sh"

CLUSTER="${CLUSTER:-mlperf-v5p}"
PROJECT_ID="${PROJECT_ID:-${PROJECT:-}}"
LOCATION="${LOCATION:-${ZONE:-}}"
TPU_TYPE="${TPU_TYPE:-v5p-8}"
BASE_OUTPUT_DIRECTORY="${BASE_OUTPUT_DIRECTORY:-}"
BASE_IMAGE="${BASE_IMAGE:-gcr.io/tpu-prod-env-multipod/maxtext_jax_stable:2026-07-17}"
IMAGE_REPOSITORY="${IMAGE_REPOSITORY:-}"
IMAGE="${IMAGE:-}"
RUN_NAME="${RUN_NAME:-}"
MAXTEXT_RUN_NAME="${MAXTEXT_RUN_NAME:-}"
XPK_BIN="${XPK_BIN:-xpk}"
DOCKER_BIN="${DOCKER_BIN:-docker}"
DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"
SKIP_BUILD="${SKIP_BUILD:-0}"
PUSH_IMAGE="${PUSH_IMAGE:-0}"
PRIORITY="${PRIORITY:-medium}"
MEMORY_INTERVAL_SECONDS="${DILOCO_MEMORY_INTERVAL_SECONDS:-5}"
MONITOR_INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-15}"
LOG_WAIT_TIMEOUT="${LOG_WAIT_TIMEOUT:-20m}"
PREFLIGHT_TIMEOUT_SECONDS="${DILOCO_PREFLIGHT_TIMEOUT_SECONDS:-300}"
PHASE_TIMEOUT_SECONDS="${DILOCO_PHASE_TIMEOUT_SECONDS:-}"
TIMEOUT_GRACE_SECONDS="${DILOCO_TIMEOUT_GRACE_SECONDS:-60}"

DEFAULT_V5P_XLA_FLAGS="\
--xla_tpu_scoped_vmem_limit_kib=65536 \
--xla_tpu_bf16_emission_mode=NATIVE_EMISSION \
--xla_tpu_spmd_rng_bit_generator_unsafe=true \
--xla_tpu_enable_sparse_core_reduce_scatter_v2=true \
--xla_tpu_enable_sparse_core_collective_offload_all_gather=true \
--xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true \
--xla_tpu_enable_all_gather_offload_tracing=true \
--xla_tpu_use_tc_device_shape_on_sc=true \
--xla_sc_disable_megacore_partitioning=true \
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
--xla_tpu_enable_3d_reduce_scatter_decomposer=false"
XLA_FLAGS="${XLA_FLAGS-${DEFAULT_V5P_XLA_FLAGS}}"

usage() {
  cat <<'EOF'
Usage:
  run_streaming_diloco_gcloud.sh plan <phase> [safe MaxText key=value overrides ...]
  run_streaming_diloco_gcloud.sh build <phase>
  run_streaming_diloco_gcloud.sh submit <phase> [extra MaxText key=value ...]
  run_streaming_diloco_gcloud.sh status <workload-name>
  run_streaming_diloco_gcloud.sh logs <workload-name>
  run_streaming_diloco_gcloud.sh monitor <workload-name>
  run_streaming_diloco_gcloud.sh delete <workload-name>

Phases:
  layout       One-slice live JAX physical-format regression and colocation probe.
  tiny         Cheap two-slice end-to-end transport/update test; fragment 0 twice.
  qwen8b       Two-slice Qwen3-8B BF16 memory test; 80 steps, fragment 0 twice.
  tiny-save    Tiny checkpoint producer; saves an aligned checkpoint at step 6.
  tiny-resume  Restores tiny-save. Set MAXTEXT_RUN_NAME to the tiny-save run name.

Required environment:
  PROJECT_ID              Google Cloud project.
  LOCATION                GKE cluster region or zone.
  BASE_OUTPUT_DIRECTORY   Writable gs:// path (not needed by layout).

Useful overrides:
  CLUSTER                 Defaults to mlperf-v5p.
  TPU_TYPE                Defaults to v5p-8.
  IMAGE_REPOSITORY        Defaults to gcr.io/$PROJECT_ID/maxtext-diloco-acceptance.
  BASE_IMAGE              Pinned Pathways-compatible MaxText base image.
  RUN_NAME                Exact XPK workload name.
  MAXTEXT_RUN_NAME        MaxText output/checkpoint run name.
  IMAGE                   Exact image URI; required with SKIP_BUILD=1.
  XPK_BIN                 Defaults to xpk.
  XLA_FLAGS               Proxy-server XLA flags; set to an empty string to disable.
  DILOCO_PHASE_TIMEOUT_SECONDS
                           In-container train timeout (phase-specific by default).

Examples:
  PROJECT_ID=p LOCATION=us-east5 BASE_OUTPUT_DIRECTORY=gs://bucket/maxtext \
    ./scripts/diloco/run_streaming_diloco_gcloud.sh submit tiny

  # Reuse one image across the staged phases after the first submit builds it:
  export IMAGE=gcr.io/p/maxtext-diloco-acceptance:streaming-fix
  PROJECT_ID=p LOCATION=us-east5 ./scripts/diloco/run_streaming_diloco_gcloud.sh submit layout
  PROJECT_ID=p LOCATION=us-east5 BASE_OUTPUT_DIRECTORY=gs://bucket/maxtext \
    SKIP_BUILD=1 ./scripts/diloco/run_streaming_diloco_gcloud.sh submit tiny

  PROJECT_ID=p LOCATION=us-east5 BASE_OUTPUT_DIRECTORY=gs://bucket/maxtext \
    ./scripts/diloco/run_streaming_diloco_gcloud.sh submit qwen8b \
    per_device_batch_size=8 max_target_length=2048

Only per_device_batch_size and max_target_length may be overridden. Model,
precision, DiLoCo scheduling, topology, checkpointing, and pass criteria remain
fixed so an acceptance phase cannot silently test something else.

The script packages only src/, the live layout test, and the in-container
runner. It does not send the rest of the working tree or local virtualenvs to
Docker.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

require_value() {
  local name="$1"
  local value="$2"
  [[ -n "${value}" ]] || die "${name} is required"
}

require_command() {
  local command_name="$1"
  command -v "${command_name}" >/dev/null 2>&1 || die "Required command not found: ${command_name}"
}

configure_kubectl_context() {
  require_command gcloud
  require_command kubectl
  gcloud container clusters get-credentials "${CLUSTER}" \
    --project "${PROJECT_ID}" \
    --location "${LOCATION}" >/dev/null
}

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

validate_workload_name() {
  local value="$1"
  [[ "${value}" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ ]] || {
    die "Workload name must be a lowercase DNS label: ${value}"
  }
  (( ${#value} <= 40 )) || die "Workload name must be at most 40 characters: ${value}"
}

validate_maxtext_override() {
  local argument="$1"
  [[ "${argument}" =~ ^[A-Za-z0-9_.-]+=.+$ ]] || {
    die "Extra MaxText arguments must be non-empty key=value tokens; got: ${argument}"
  }
  case "${argument%%=*}" in
    per_device_batch_size|max_target_length) ;;
    *)
      die "Acceptance override is not allowed: ${argument%%=*} (allowed: per_device_batch_size, max_target_length)"
      ;;
  esac
}

cloud_preflight() {
  local phase="$1"
  require_command "${XPK_BIN}"
  require_command gcloud

  local xpk_help
  xpk_help="$("${XPK_BIN}" workload create-pathways --help 2>&1 || true)"
  grep -q -- "--custom-pathways-proxy-server-args" <<<"${xpk_help}" || {
    die "${XPK_BIN} is too old: create-pathways lacks --custom-pathways-proxy-server-args"
  }

  [[ -n "$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null)" ]] || {
    die "No active gcloud account; authenticate before submitting"
  }
  gcloud container clusters describe "${CLUSTER}" \
    --project "${PROJECT_ID}" \
    --location "${LOCATION}" >/dev/null
  configure_kubectl_context

  if [[ "${phase}" != "layout" ]]; then
    local bucket_name
    bucket_name="${BASE_OUTPUT_DIRECTORY#gs://}"
    bucket_name="${bucket_name%%/*}"
    gcloud storage buckets describe "gs://${bucket_name}" --project "${PROJECT_ID}" >/dev/null
  fi
}

docker_preflight() {
  require_command "${DOCKER_BIN}"
  "${DOCKER_BIN}" info >/dev/null
}

build_image() {
  docker_preflight
  (
    local build_context
    build_context="$(mktemp -d /tmp/maxtext-diloco-build.XXXXXX)"
    case "${build_context}" in
      /tmp/maxtext-diloco-build.*) ;;
      *) die "Unexpected temporary build context: ${build_context}" ;;
    esac
    trap 'rm -rf -- "${build_context}"' EXIT

    cp -a "${REPO_ROOT}/src" "${build_context}/src"
    mkdir -p "${build_context}/tests/unit"
    cp "${REPO_ROOT}/tests/unit/pathways_null_layout_repro_test.py" "${build_context}/tests/unit/"
    cp "${WORKLOAD_RUNNER}" "${build_context}/acceptance_workload.sh"

    echo "Building ${IMAGE} from a restricted source context..."
    "${DOCKER_BIN}" build \
      --platform "${DOCKER_PLATFORM}" \
      --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
      --build-arg "MAXTEXT_SOURCE_REVISION=${SOURCE_REVISION}" \
      --label "org.opencontainers.image.revision=${SOURCE_REVISION}" \
      --tag "${IMAGE}" \
      --file - \
      "${build_context}" <<'DOCKERFILE'
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG MAXTEXT_SOURCE_REVISION=unknown
ENV MAXTEXT_SOURCE_REVISION=${MAXTEXT_SOURCE_REVISION}
WORKDIR /app
RUN rm -rf /app/src /app/tests /app/run-diloco-acceptance
COPY src /app/src
COPY tests /app/tests
COPY acceptance_workload.sh /app/run-diloco-acceptance
RUN chmod 0755 /app/run-diloco-acceptance \
    && find /app/src /app/tests -type f -name '*.py[co]' -delete \
    && find /app/src /app/tests -type d -name '__pycache__' -prune -exec rm -rf '{}' +
DOCKERFILE
  )
}

print_observation_commands() {
  local workload_name="$1"
  echo
  echo "Workload submitted: ${workload_name}"
  echo "Status:"
  print_command "${XPK_BIN}" workload list \
    --cluster "${CLUSTER}" --project "${PROJECT_ID}" --zone "${LOCATION}"
  echo "Controller and Pathways logs:"
  print_command kubectl logs -f \
    -l "jobset.sigs.k8s.io/jobset-name=${workload_name}" \
    --all-containers=true --prefix=true --max-log-requests=50 \
    --pod-running-timeout="${LOG_WAIT_TIMEOUT}"
  echo "All-container memory (run repeatedly or use the monitor action):"
  print_command kubectl top pods \
    -l "jobset.sigs.k8s.io/jobset-name=${workload_name}" --containers
  echo "Delete:"
  print_command "${0}" delete "${workload_name}"
  echo
  echo "Cloud Logging query:"
  echo "resource.type=\"k8s_container\""
  echo "resource.labels.project_id=\"${PROJECT_ID}\""
  echo "resource.labels.location=\"${LOCATION}\""
  echo "resource.labels.cluster_name=\"${CLUSTER}\""
  echo "resource.labels.pod_name:\"${workload_name}\""
}

ACTION="${1:-help}"
TARGET="${2:-}"
if [[ "${ACTION}" == "help" || "${ACTION}" == "--help" || "${ACTION}" == "-h" ]]; then
  usage
  exit 0
fi
[[ -n "${TARGET}" ]] || {
  usage >&2
  exit 2
}
shift 2
if [[ "${1:-}" == "--" ]]; then
  shift
fi
EXTRA_MAXTEXT_ARGS=("$@")

require_value PROJECT_ID "${PROJECT_ID}"
require_value LOCATION "${LOCATION}"

case "${ACTION}" in
  status)
    validate_workload_name "${TARGET}"
    require_command "${XPK_BIN}"
    configure_kubectl_context
    "${XPK_BIN}" workload list --cluster "${CLUSTER}" --project "${PROJECT_ID}" --zone "${LOCATION}"
    kubectl get pods -l "jobset.sigs.k8s.io/jobset-name=${TARGET}" -o wide
    exit 0
    ;;
  logs)
    validate_workload_name "${TARGET}"
    configure_kubectl_context
    exec kubectl logs -f \
      -l "jobset.sigs.k8s.io/jobset-name=${TARGET}" \
      --all-containers=true --prefix=true --max-log-requests=50 \
      --pod-running-timeout="${LOG_WAIT_TIMEOUT}"
    ;;
  monitor)
    validate_workload_name "${TARGET}"
    configure_kubectl_context
    while true; do
      date -u +%Y-%m-%dT%H:%M:%SZ
      kubectl get pods -l "jobset.sigs.k8s.io/jobset-name=${TARGET}" -o wide || true
      kubectl top pods -l "jobset.sigs.k8s.io/jobset-name=${TARGET}" --containers || true
      sleep "${MONITOR_INTERVAL_SECONDS}"
    done
    ;;
  delete)
    validate_workload_name "${TARGET}"
    require_command "${XPK_BIN}"
    if [[ "${YES:-0}" != "1" ]]; then
      if [[ ! -t 0 ]]; then
        die "Set YES=1 to delete non-interactively"
      fi
      read -r -p "Delete Pathways workload ${TARGET}? [y/N] " reply
      [[ "${reply}" == "y" || "${reply}" == "Y" ]] || exit 0
    fi
    configure_kubectl_context
    exec "${XPK_BIN}" workload delete \
      --workload "${TARGET}" \
      --cluster "${CLUSTER}" \
      --project "${PROJECT_ID}" \
      --zone "${LOCATION}"
    ;;
  plan|build|submit) ;;
  *)
    usage >&2
    die "Unknown action: ${ACTION}"
    ;;
esac

PHASE="${TARGET}"
case "${PHASE}" in
  layout)
    DEFAULT_NUM_SLICES=1
    PHASE_SLUG=layout
    ;;
  tiny)
    DEFAULT_NUM_SLICES=2
    PHASE_SLUG=tiny
    ;;
  tiny-save)
    DEFAULT_NUM_SLICES=2
    PHASE_SLUG=tsave
    ;;
  tiny-resume)
    DEFAULT_NUM_SLICES=2
    PHASE_SLUG=tresume
    ;;
  qwen8b)
    DEFAULT_NUM_SLICES=2
    PHASE_SLUG=q8b
    ;;
  *)
    usage >&2
    die "Unknown phase: ${PHASE}"
    ;;
esac

NUM_SLICES="${NUM_SLICES:-${DEFAULT_NUM_SLICES}}"
[[ "${NUM_SLICES}" =~ ^[0-9]+$ ]] || die "NUM_SLICES must be an integer"
(( NUM_SLICES == DEFAULT_NUM_SLICES )) || {
  die "${PHASE} acceptance requires NUM_SLICES=${DEFAULT_NUM_SLICES}; got ${NUM_SLICES}"
}

if [[ "${PHASE}" != "layout" ]]; then
  require_value BASE_OUTPUT_DIRECTORY "${BASE_OUTPUT_DIRECTORY}"
  [[ "${BASE_OUTPUT_DIRECTORY}" =~ ^gs://[^/]+(/.*)?$ ]] || {
    die "BASE_OUTPUT_DIRECTORY must contain a GCS bucket, for example gs://bucket/maxtext"
  }
fi
if [[ "${PHASE}" == "layout" && ${#EXTRA_MAXTEXT_ARGS[@]} -ne 0 ]]; then
  die "The layout phase does not accept MaxText key=value overrides"
fi
for argument in "${EXTRA_MAXTEXT_ARGS[@]}"; do
  validate_maxtext_override "${argument}"
done

SOURCE_REVISION="$(
  git -c "safe.directory=${REPO_ROOT}" -C "${REPO_ROOT}" rev-parse --short=12 HEAD 2>/dev/null || echo unknown
)"
if ! git -c "safe.directory=${REPO_ROOT}" -C "${REPO_ROOT}" diff --quiet --ignore-submodules -- 2>/dev/null; then
  SOURCE_REVISION="${SOURCE_REVISION}-dirty"
fi

if [[ -z "${RUN_NAME}" ]]; then
  RUN_NAME="diloco-${PHASE_SLUG}-$(date -u +%m%d-%H%M%S)"
fi
validate_workload_name "${RUN_NAME}"

if [[ -z "${MAXTEXT_RUN_NAME}" ]]; then
  if [[ "${PHASE}" == "tiny-resume" ]]; then
    die "tiny-resume requires MAXTEXT_RUN_NAME from the preceding tiny-save job"
  fi
  MAXTEXT_RUN_NAME="${RUN_NAME}"
fi

if [[ -z "${IMAGE_REPOSITORY}" ]]; then
  IMAGE_REPOSITORY="gcr.io/${PROJECT_ID}/maxtext-diloco-acceptance"
fi
if [[ -z "${IMAGE}" ]]; then
  if [[ "${SKIP_BUILD}" == "1" ]]; then
    die "IMAGE must name an existing pushed image when SKIP_BUILD=1"
  fi
  IMAGE="${IMAGE_REPOSITORY}:${RUN_NAME}"
fi

if [[ -z "${PHASE_TIMEOUT_SECONDS}" ]]; then
  case "${PHASE}" in
    layout) PHASE_TIMEOUT_SECONDS=1800 ;;
    tiny|tiny-save|tiny-resume) PHASE_TIMEOUT_SECONDS=3600 ;;
    qwen8b) PHASE_TIMEOUT_SECONDS=14400 ;;
  esac
fi
for timeout_setting in \
  "DILOCO_PREFLIGHT_TIMEOUT_SECONDS=${PREFLIGHT_TIMEOUT_SECONDS}" \
  "DILOCO_PHASE_TIMEOUT_SECONDS=${PHASE_TIMEOUT_SECONDS}" \
  "DILOCO_TIMEOUT_GRACE_SECONDS=${TIMEOUT_GRACE_SECONDS}"; do
  timeout_value="${timeout_setting#*=}"
  [[ "${timeout_value}" =~ ^[1-9][0-9]*$ ]] || {
    die "${timeout_setting%%=*} must be a positive integer; got ${timeout_value}"
  }
done
[[ "${MEMORY_INTERVAL_SECONDS}" =~ ^[1-9][0-9]*$ ]] || {
  die "DILOCO_MEMORY_INTERVAL_SECONDS must be a positive integer; got ${MEMORY_INTERVAL_SECONDS}"
}

remote_argv=(
  env
  PYTHONUNBUFFERED=1
  JAX_PLATFORMS=proxy,cpu
  JAX_BACKEND_TARGET=grpc://127.0.0.1:29000
  ENABLE_PATHWAYS_PERSISTENCE=1
  ENABLE_PJRT_COMPATIBILITY=true
  "DILOCO_TEST_PHASE=${PHASE}"
  "EXPECTED_SLICES=${NUM_SLICES}"
  "MAXTEXT_RUN_NAME=${MAXTEXT_RUN_NAME}"
  "BASE_OUTPUT_DIRECTORY=${BASE_OUTPUT_DIRECTORY}"
  "DILOCO_MEMORY_INTERVAL_SECONDS=${MEMORY_INTERVAL_SECONDS}"
  "DILOCO_PREFLIGHT_TIMEOUT_SECONDS=${PREFLIGHT_TIMEOUT_SECONDS}"
  "DILOCO_PHASE_TIMEOUT_SECONDS=${PHASE_TIMEOUT_SECONDS}"
  "DILOCO_TIMEOUT_GRACE_SECONDS=${TIMEOUT_GRACE_SECONDS}"
  /app/run-diloco-acceptance
  "${EXTRA_MAXTEXT_ARGS[@]}"
)
printf -v remote_payload '%q ' "${remote_argv[@]}"
printf -v REMOTE_COMMAND 'bash -lc %q' "${remote_payload}"

xpk_command=(
  "${XPK_BIN}"
  workload
  create-pathways
  --workload "${RUN_NAME}"
  --docker-image "${IMAGE}"
  --command "${REMOTE_COMMAND}"
  --num-slices "${NUM_SLICES}"
  --cluster "${CLUSTER}"
  --tpu-type "${TPU_TYPE}"
  --project "${PROJECT_ID}"
  --zone "${LOCATION}"
  --priority "${PRIORITY}"
  --max-restarts 0
  --enable-debug-logs
)
if [[ -n "${XLA_FLAGS}" ]]; then
  xpk_command+=("--custom-pathways-proxy-server-args=${XLA_FLAGS}")
fi

echo "Resolved acceptance test:"
echo "  action=${ACTION}"
echo "  phase=${PHASE}"
echo "  cluster=${CLUSTER}"
echo "  project=${PROJECT_ID}"
echo "  location=${LOCATION}"
echo "  tpu_type=${TPU_TYPE}"
echo "  slices=${NUM_SLICES}"
echo "  workload=${RUN_NAME}"
echo "  maxtext_run=${MAXTEXT_RUN_NAME}"
echo "  image=${IMAGE}"
echo "  base_image=${BASE_IMAGE}"
if [[ "${SKIP_BUILD}" == "1" ]]; then
  echo "  local_source_revision=${SOURCE_REVISION} (not injected into reused image)"
  echo "  tested_image_revision=reported by the container's baked MAXTEXT_SOURCE_REVISION"
else
  echo "  source_revision_to_build=${SOURCE_REVISION}"
fi
echo "  profiler=disabled"
echo "  preflight_timeout_seconds=${PREFLIGHT_TIMEOUT_SECONDS}"
echo "  phase_timeout_seconds=${PHASE_TIMEOUT_SECONDS}"
echo "  checkpointing=$([[ "${PHASE}" == tiny-save || "${PHASE}" == tiny-resume ]] && echo enabled || echo disabled)"
echo
echo "XPK command:"
print_command "${xpk_command[@]}"

if [[ "${ACTION}" == "plan" ]]; then
  echo
  echo "Plan only: no image was built and no cloud resource was changed."
  exit 0
fi

[[ -x "${WORKLOAD_RUNNER}" ]] || die "Workload runner is not executable: ${WORKLOAD_RUNNER}"

if [[ "${ACTION}" == "build" ]]; then
  build_image
  if [[ "${PUSH_IMAGE}" == "1" ]]; then
    echo "Pushing ${IMAGE}..."
    "${DOCKER_BIN}" push "${IMAGE}"
  else
    echo "Built ${IMAGE}; set PUSH_IMAGE=1 to push it."
  fi
  exit 0
fi

cloud_preflight "${PHASE}"
if [[ "${SKIP_BUILD}" != "1" ]]; then
  build_image
  echo "Pushing ${IMAGE}..."
  "${DOCKER_BIN}" push "${IMAGE}"
fi

"${xpk_command[@]}"
print_observation_commands "${RUN_NAME}"

if [[ "${PHASE}" == "tiny-save" ]]; then
  echo "After this workload passes, test restore with the same image and MaxText run:"
  echo "  MAXTEXT_RUN_NAME=${MAXTEXT_RUN_NAME} IMAGE=${IMAGE} SKIP_BUILD=1 \\"
  echo "    PROJECT_ID=${PROJECT_ID} LOCATION=${LOCATION} BASE_OUTPUT_DIRECTORY=${BASE_OUTPUT_DIRECTORY} \\"
  echo "    ${0} submit tiny-resume"
fi
