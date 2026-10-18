#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Deploys the two-region Cluster Toolkit blueprint (Cloud Pathways or McJax),
# runs Qwen3-30B-A3B cross-region Streaming DiLoCo training, streams logs,
# tears down all created cloud resources on exit, and runs a zero-leak audit.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_PROJECT_ID="test-diloco-project"
DEFAULT_ZONE_R1="us-east5-b"
DEFAULT_ZONE_R2="europe-west1-b"
DEFAULT_SPOT_TPU="true"
DEFAULT_RUNTIME="pathways"
DEFAULT_STEPS=60
DEFAULT_JOB_TIMEOUT_SECONDS=14400
# Keep Terraform state, generated modules, and markers in a private per-user
# directory rather than shared /tmp to prevent local directory tampering.
if [[ -n "${XDG_STATE_HOME:-}" ]]; then
  DEFAULT_WORKDIR_ROOT="${XDG_STATE_HOME}/maxtext_cross_region_diloco"
elif [[ -n "${HOME:-}" && "${HOME}" != "/" ]]; then
  DEFAULT_WORKDIR_ROOT="${HOME}/.local/state/maxtext_cross_region_diloco"
else
  DEFAULT_WORKDIR_ROOT="${TMPDIR:-/tmp}/maxtext_cross_region_diloco_$(id -u)"
fi
OWNERSHIP_DIR_NAME=".deployment_ownership"
REQUIRED_GCP_SERVICES=(
  "serviceusage.googleapis.com"
  "compute.googleapis.com"
  "container.googleapis.com"
)

RAW_USER="$(id -un 2>/dev/null || whoami 2>/dev/null || echo "${USER:-user}")"
USER_HASH="$(printf '%s' "${RAW_USER}" | sha256sum | cut -c1-4)"
CLEAN_USER="$(printf '%s' "${RAW_USER}" | tr '[:upper:]' '[:lower:]' | tr '._' '-' | tr -cd 'a-z0-9-' | sed -E 's/^-+//; s/-+$//' | cut -c1-8 | sed -E 's/-+$//')"
if [[ -z "${CLEAN_USER}" ]]; then
  CLEAN_USER="user"
fi
DEFAULT_DEPLOYMENT_NAME="${CLEAN_USER}-${USER_HASH}-pwd"

PROJECT_ID="${PROJECT_ID:-${DEFAULT_PROJECT_ID}}"
GCS_BUCKET="${GCS_BUCKET:-}"
MAXTEXT_IMAGE="${MAXTEXT_IMAGE:-}"
RUNTIME="${RUNTIME:-${DEFAULT_RUNTIME}}"
ZONE_R1="${ZONE_R1:-${DEFAULT_ZONE_R1}}"
ZONE_R2="${ZONE_R2:-${DEFAULT_ZONE_R2}}"
SPOT_TPU="${SPOT_TPU:-${DEFAULT_SPOT_TPU}}"
STEPS="${STEPS:-${DEFAULT_STEPS}}"
RUN_NAME="${RUN_NAME:-qwen3_30b_streaming_diloco_$(date +%Y%m%d_%H%M%S)}"
AUTHORIZED_CIDR="${AUTHORIZED_CIDR:-}"
EXTRA_MAXTEXT_ARGS="${EXTRA_MAXTEXT_ARGS:-}"
EXTRA_GCLUSTER_VARS=()
DEPLOYMENT_NAME="${DEPLOYMENT_NAME:-${DEFAULT_DEPLOYMENT_NAME}}"
WORKDIR_ROOT="${WORKDIR_ROOT:-${DEFAULT_WORKDIR_ROOT}}"
JOB_TIMEOUT_SECONDS="${JOB_TIMEOUT_SECONDS:-${DEFAULT_JOB_TIMEOUT_SECONDS}}"
KEEP_GCS_ARTIFACTS=0
ENABLE_APIS=0
CLEANUP_ONLY=0
FORCE_CLEANUP=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --project-id <id>          GCP Project ID (default: \${PROJECT_ID} or ${DEFAULT_PROJECT_ID})
  --gcs-bucket <gs-uri>      GCS bucket for scratch and logs (default: gs://<project-id>-pathways)
  --maxtext-image <uri>      MaxText image URI (default: us-docker.pkg.dev/<project-id>/maxtext-images/maxtext_jax_stable:latest)
  --runtime <pathways|mcjax> Orchestration runtime: 'pathways' or 'mcjax' (default: ${DEFAULT_RUNTIME})
  --zone-r1 <zone>           Region 1 TPU zone (default: ${DEFAULT_ZONE_R1})
  --zone-r2 <zone>           Region 2 TPU zone (default: ${DEFAULT_ZONE_R2})
  --spot-tpu <bool>          Use Spot VMs for TPU node pools (default: ${DEFAULT_SPOT_TPU})
  --steps <int>              Training steps before teardown (default: ${DEFAULT_STEPS})
  --run-name <name>          Run name inside the GCS bucket
  --authorized-cidr <cidr>   GKE control-plane authorized CIDR (default: auto-detected caller IP/32, fallback 0.0.0.0/0)
  --maxtext-args "<args>"    Extra MaxText CLI key=value overrides (e.g. "profiler=xplane skip_first_n_steps_for_profiler=46")
  --vars "<k=v,...>"         Additional gcluster blueprint variable overrides (repeatable)
  --deployment-name <name>   Cluster Toolkit deployment name (default: ${DEFAULT_DEPLOYMENT_NAME})
  --workdir <dir>            Out-of-tree directory for Terraform state (default: ${DEFAULT_WORKDIR_ROOT})
  --keep-gcs-artifacts       Retain GCS scratch/run objects after exit
  --enable-apis              Enable missing required GCP APIs automatically
  --cleanup                  Run teardown and zero-leak audit for an owned deployment without starting training
  --force-cleanup            Force teardown of --deployment-name even if the local ownership marker is missing
  -h, --help                 Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-id) PROJECT_ID="$2"; shift 2 ;;
    --gcs-bucket) GCS_BUCKET="$2"; shift 2 ;;
    --maxtext-image) MAXTEXT_IMAGE="$2"; shift 2 ;;
    --runtime) RUNTIME="$2"; shift 2 ;;
    --zone-r1) ZONE_R1="$2"; shift 2 ;;
    --zone-r2) ZONE_R2="$2"; shift 2 ;;
    --spot-tpu) SPOT_TPU="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --authorized-cidr) AUTHORIZED_CIDR="$2"; shift 2 ;;
    --maxtext-args) EXTRA_MAXTEXT_ARGS="$2"; shift 2 ;;
    --vars) EXTRA_GCLUSTER_VARS+=(--vars "$2"); shift 2 ;;
    --deployment-name) DEPLOYMENT_NAME="$2"; shift 2 ;;
    --workdir) WORKDIR_ROOT="$2"; shift 2 ;;
    --keep-gcs-artifacts) KEEP_GCS_ARTIFACTS=1; shift 1 ;;
    --enable-apis) ENABLE_APIS=1; shift 1 ;;
    --cleanup) CLEANUP_ONLY=1; shift 1 ;;
    --force-cleanup) CLEANUP_ONLY=1; FORCE_CLEANUP=1; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

for bin in gcluster terraform gcloud kubectl gke-gcloud-auth-plugin; do
  if ! command -v "${bin}" >/dev/null 2>&1; then
    echo "[ERROR] Required CLI '${bin}' was not found on PATH." >&2
    exit 1
  fi
done

case "${SPOT_TPU,,}" in
  true|1) SPOT_TPU="true" ;;
  false|0) SPOT_TPU="false" ;;
  *) echo "[ERROR] Invalid --spot-tpu '${SPOT_TPU}'. Expected true or false." >&2; exit 1 ;;
esac

case "${RUNTIME,,}" in
  pathways) RUNTIME="pathways"; SKIP_WORKLOAD_GROUP="workload_mcjax" ;;
  mcjax) RUNTIME="mcjax"; SKIP_WORKLOAD_GROUP="workload_pathways" ;;
  *) echo "[ERROR] Invalid --runtime '${RUNTIME}'. Expected pathways or mcjax." >&2; exit 1 ;;
esac

if [[ ! "${DEPLOYMENT_NAME}" =~ ^[a-z][a-z0-9-]{2,22}[a-z0-9]$ ]]; then
  echo "[ERROR] Invalid --deployment-name '${DEPLOYMENT_NAME}' (must match ^[a-z][a-z0-9-]{2,22}[a-z0-9]$)." >&2
  exit 1
fi

resolve_system_zone() {
  case "$1" in
    us-east5-b) printf '%s\n' "us-east5-a" ;;
    us-east1-d) printf '%s\n' "us-east1-b" ;;
    *) printf '%s\n' "$1" ;;
  esac
}

resolve_cpu_family() {
  case "$1" in
    us-east5) printf '%s\n' "n4d" ;;
    europe-west1) printf '%s\n' "n2d" ;;
    *) printf '%s\n' "n2" ;;
  esac
}

REGION_R1="${ZONE_R1%-*}"
REGION_R2="${ZONE_R2%-*}"
ZONE_R1_SYSTEM="$(resolve_system_zone "${ZONE_R1}")"
ZONE_R2_SYSTEM="$(resolve_system_zone "${ZONE_R2}")"
CPU_FAMILY_R1="$(resolve_cpu_family "${REGION_R1}")"
CPU_FAMILY_R2="$(resolve_cpu_family "${REGION_R2}")"
SYSTEM_MACHINE_TYPE_R1="${CPU_FAMILY_R1}-standard-4"
HEAD_MACHINE_TYPE_R1="${CPU_FAMILY_R1}-standard-32"
SYSTEM_MACHINE_TYPE_R2="${CPU_FAMILY_R2}-standard-4"

MAXTEXT_IMAGE="${MAXTEXT_IMAGE:-us-docker.pkg.dev/${PROJECT_ID}/maxtext-images/maxtext_jax_stable:latest}"
GCS_BUCKET="${GCS_BUCKET:-gs://${PROJECT_ID}-pathways}"
if [[ "${GCS_BUCKET}" != gs://* ]]; then
  echo "[ERROR] --gcs-bucket must start with gs:// (got '${GCS_BUCKET}')." >&2
  exit 1
fi
GCS_BUCKET="${GCS_BUCKET%/}"
GCS_SCRATCH_LOCATION="${GCS_BUCKET}/${DEPLOYMENT_NAME}"

CLUSTER_R1_NAME="${DEPLOYMENT_NAME}-c1"
CLUSTER_R2_NAME="${DEPLOYMENT_NAME}-c2"
VPC_NAME="${DEPLOYMENT_NAME}-vpc"
SUBNET_R1_NAME="${DEPLOYMENT_NAME}-sub-r1"
SUBNET_R2_NAME="${DEPLOYMENT_NAME}-sub-r2"
FIREWALL_RULE_NAME="${DEPLOYMENT_NAME}-allow-pathways-cross-region"
FIREWALL_IAP_NAME="${VPC_NAME}-fw-allow-iap-ingress"
FIREWALL_INTERNAL_NAME="${VPC_NAME}-fw-allow-internal-traffic"
ROUTER_NAME="${VPC_NAME}-router"
NAT_IPS_NAME="${VPC_NAME}-nat-ips"
GKE_VM_PREFIX="gke-${DEPLOYMENT_NAME:0:16}"

# Reject symlinks, foreign owners, and group/world write permissions on every
# reused state directory so another local user cannot swap Terraform modules.
check_private_dir_node() {
  local target_dir="$1"
  if [[ -L "${target_dir}" ]]; then
    echo "[ERROR] Refusing to use symlink directory '${target_dir}'." >&2
    exit 1
  fi
  if [[ ! -d "${target_dir}" ]]; then
    echo "[ERROR] Expected directory at '${target_dir}'." >&2
    exit 1
  fi
  if [[ ! -O "${target_dir}" ]]; then
    echo "[ERROR] Directory '${target_dir}' is not owned by UID $(id -u)." >&2
    exit 1
  fi
  local mode_octal
  mode_octal="$(stat -c '%a' "${target_dir}" 2>/dev/null || stat -f '%Lp' "${target_dir}" 2>/dev/null || echo "")"
  if [[ -z "${mode_octal}" ]] || (( (8#${mode_octal} & 8#022) != 0 )); then
    echo "[ERROR] Directory '${target_dir}' must not be writable by group or others (mode ${mode_octal:-unknown})." >&2
    exit 1
  fi
}

ensure_private_dir() {
  local target_dir="$1"
  if [[ -L "${target_dir}" ]]; then
    echo "[ERROR] Refusing to use symlink directory '${target_dir}'." >&2
    exit 1
  fi
  if [[ ! -e "${target_dir}" ]]; then
    (umask 077 && mkdir -p "${target_dir}")
  fi
  check_private_dir_node "${target_dir}"
  chmod 0700 "${target_dir}"
}

verify_reused_subdirs() {
  local root_dir="$1"
  if [[ -L "${root_dir}" ]]; then
    echo "[ERROR] Refusing to reuse symlink directory '${root_dir}'." >&2
    exit 1
  fi
  if [[ ! -e "${root_dir}" ]]; then
    return 0
  fi
  check_private_dir_node "${root_dir}"
  local entry
  while IFS= read -r -d '' entry; do
    if [[ -L "${entry}" ]]; then
      if [[ ! -O "${entry}" ]]; then
        echo "[ERROR] Symlink '${entry}' inside '${root_dir}' is not owned by UID $(id -u)." >&2
        exit 1
      fi
    elif [[ -d "${entry}" ]]; then
      check_private_dir_node "${entry}"
    elif [[ ! -O "${entry}" ]]; then
      echo "[ERROR] File '${entry}' inside '${root_dir}' is not owned by UID $(id -u)." >&2
      exit 1
    fi
  done < <(find -P "${root_dir}" -mindepth 1 \( -type d -o ! -user "$(id -u)" \) -print0)
}

ensure_private_dir "${WORKDIR_ROOT}"
WORKDIR_ROOT="$(cd -P "${WORKDIR_ROOT}" && pwd)"
ensure_private_dir "${WORKDIR_ROOT}"
OWNERSHIP_MARKER_DIR="${WORKDIR_ROOT}/${OWNERSHIP_DIR_NAME}"
ensure_private_dir "${OWNERSHIP_MARKER_DIR}"
verify_reused_subdirs "${OWNERSHIP_MARKER_DIR}"
verify_reused_subdirs "${WORKDIR_ROOT}/${DEPLOYMENT_NAME}"
OWNERSHIP_MARKER_FILE="${OWNERSHIP_MARKER_DIR}/${PROJECT_ID}_${DEPLOYMENT_NAME}.marker"
VARS_FILE=""

CREATED_GCS_BUCKET=0
LOG_PID=""
DEPLOY_STARTED=0

GCLUSTER_BIN="$(command -v gcluster)"
export GHPC_MOCK_MACHINE_CONFIG='{"cpus":{"n4-standard-4":{"count":4},"n4-standard-32":{"count":32},"n4d-standard-4":{"count":4},"n4d-standard-32":{"count":32},"n2-standard-4":{"count":4},"n2-standard-32":{"count":32},"n2d-standard-4":{"count":4},"n2d-standard-32":{"count":32}},"tpus":{"ct5p-hightpu-4t":{"count":4},"ct5lp-hightpu-4t":{"count":4}},"gpus":{}}'
CTX_R1="gke_${PROJECT_ID}_${REGION_R1}_${CLUSTER_R1_NAME}"
CTX_R2="gke_${PROJECT_ID}_${REGION_R2}_${CLUSTER_R2_NAME}"

verify_ownership_marker() {
  if [[ -L "${OWNERSHIP_MARKER_FILE}" || ! -f "${OWNERSHIP_MARKER_FILE}" || ! -O "${OWNERSHIP_MARKER_FILE}" ]]; then
    return 1
  fi
  local marker_project marker_deployment
  marker_project="$(grep -E '^PROJECT_ID=' "${OWNERSHIP_MARKER_FILE}" 2>/dev/null | cut -d= -f2- || true)"
  marker_deployment="$(grep -E '^DEPLOYMENT_NAME=' "${OWNERSHIP_MARKER_FILE}" 2>/dev/null | cut -d= -f2- || true)"
  [[ "${marker_project}" == "${PROJECT_ID}" && "${marker_deployment}" == "${DEPLOYMENT_NAME}" ]]
}

load_ownership_marker_metadata() {
  if [[ -L "${OWNERSHIP_MARKER_FILE}" || ! -f "${OWNERSHIP_MARKER_FILE}" || ! -O "${OWNERSHIP_MARKER_FILE}" ]]; then
    return 0
  fi
  local marker_created_bucket marker_runtime marker_r1 marker_z1 marker_z1_sys marker_r2 marker_z2 marker_z2_sys
  marker_created_bucket="$(grep -E '^CREATED_GCS_BUCKET=' "${OWNERSHIP_MARKER_FILE}" 2>/dev/null | cut -d= -f2- || true)"
  marker_runtime="$(grep -E '^RUNTIME=' "${OWNERSHIP_MARKER_FILE}" 2>/dev/null | cut -d= -f2- || true)"
  marker_r1="$(grep -E '^REGION_R1=' "${OWNERSHIP_MARKER_FILE}" 2>/dev/null | cut -d= -f2- || true)"
  marker_z1="$(grep -E '^ZONE_R1=' "${OWNERSHIP_MARKER_FILE}" 2>/dev/null | cut -d= -f2- || true)"
  marker_z1_sys="$(grep -E '^ZONE_R1_SYSTEM=' "${OWNERSHIP_MARKER_FILE}" 2>/dev/null | cut -d= -f2- || true)"
  marker_r2="$(grep -E '^REGION_R2=' "${OWNERSHIP_MARKER_FILE}" 2>/dev/null | cut -d= -f2- || true)"
  marker_z2="$(grep -E '^ZONE_R2=' "${OWNERSHIP_MARKER_FILE}" 2>/dev/null | cut -d= -f2- || true)"
  marker_z2_sys="$(grep -E '^ZONE_R2_SYSTEM=' "${OWNERSHIP_MARKER_FILE}" 2>/dev/null | cut -d= -f2- || true)"
  if [[ "${marker_created_bucket}" == "1" ]]; then
    CREATED_GCS_BUCKET=1
  fi
  if [[ "${marker_runtime}" == "pathways" ]]; then
    RUNTIME="pathways"
    SKIP_WORKLOAD_GROUP="workload_mcjax"
  elif [[ "${marker_runtime}" == "mcjax" ]]; then
    RUNTIME="mcjax"
    SKIP_WORKLOAD_GROUP="workload_pathways"
  fi
  if [[ -n "${marker_r1}" ]]; then
    REGION_R1="${marker_r1}"
    CTX_R1="gke_${PROJECT_ID}_${REGION_R1}_${CLUSTER_R1_NAME}"
  fi
  if [[ -n "${marker_z1}" ]]; then
    ZONE_R1="${marker_z1}"
  fi
  if [[ -n "${marker_z1_sys}" ]]; then
    ZONE_R1_SYSTEM="${marker_z1_sys}"
  fi
  if [[ -n "${marker_r2}" ]]; then
    REGION_R2="${marker_r2}"
    CTX_R2="gke_${PROJECT_ID}_${REGION_R2}_${CLUSTER_R2_NAME}"
  fi
  if [[ -n "${marker_z2}" ]]; then
    ZONE_R2="${marker_z2}"
  fi
  if [[ -n "${marker_z2_sys}" ]]; then
    ZONE_R2_SYSTEM="${marker_z2_sys}"
  fi
}

write_ownership_marker() {
  ensure_private_dir "${OWNERSHIP_MARKER_DIR}"
  if [[ -L "${OWNERSHIP_MARKER_FILE}" ]]; then
    echo "[ERROR] Refusing to overwrite symlink '${OWNERSHIP_MARKER_FILE}'." >&2
    exit 1
  fi
  local tmp_marker
  tmp_marker="$(umask 077 && mktemp "${OWNERSHIP_MARKER_DIR}/.marker.XXXXXX")"
  cat <<EOF > "${tmp_marker}"
PROJECT_ID=${PROJECT_ID}
DEPLOYMENT_NAME=${DEPLOYMENT_NAME}
RUNTIME=${RUNTIME}
REGION_R1=${REGION_R1}
ZONE_R1=${ZONE_R1}
ZONE_R1_SYSTEM=${ZONE_R1_SYSTEM}
REGION_R2=${REGION_R2}
ZONE_R2=${ZONE_R2}
ZONE_R2_SYSTEM=${ZONE_R2_SYSTEM}
CREATED_BY=${RAW_USER}
CREATED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
CREATED_GCS_BUCKET=${CREATED_GCS_BUCKET}
EOF
  mv -f "${tmp_marker}" "${OWNERSHIP_MARKER_FILE}"
}

remove_ownership_marker() {
  if [[ -n "${VARS_FILE}" ]]; then
    rm -f "${VARS_FILE}" 2>/dev/null || true
  fi
  rm -f "${OWNERSHIP_MARKER_FILE}" 2>/dev/null || true
  rmdir "${OWNERSHIP_MARKER_DIR}" 2>/dev/null || true
}

preflight_check_no_conflicting_resources() {
  if verify_ownership_marker; then
    echo "[PREFLIGHT] Found valid ownership marker for '${DEPLOYMENT_NAME}' in project '${PROJECT_ID}'."
    load_ownership_marker_metadata
    return 0
  fi

  echo "[PREFLIGHT] Checking that no conflicting GCP resources exist for '${DEPLOYMENT_NAME}' in project '${PROJECT_ID}'..."
  local conflicts=()
  if gcloud container clusters describe "${CLUSTER_R1_NAME}" --region="${REGION_R1}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    conflicts+=("GKE Cluster: ${CLUSTER_R1_NAME} (${REGION_R1})")
  fi
  if gcloud container clusters describe "${CLUSTER_R2_NAME}" --region="${REGION_R2}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    conflicts+=("GKE Cluster: ${CLUSTER_R2_NAME} (${REGION_R2})")
  fi
  if gcloud compute networks describe "${VPC_NAME}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    conflicts+=("VPC Network: ${VPC_NAME}")
  fi
  if gcloud compute networks subnets describe "${SUBNET_R1_NAME}" --region="${REGION_R1}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    conflicts+=("Subnetwork: ${SUBNET_R1_NAME} (${REGION_R1})")
  fi
  if gcloud compute networks subnets describe "${SUBNET_R2_NAME}" --region="${REGION_R2}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    conflicts+=("Subnetwork: ${SUBNET_R2_NAME} (${REGION_R2})")
  fi
  for fw_name in "${FIREWALL_RULE_NAME}" "${FIREWALL_IAP_NAME}" "${FIREWALL_INTERNAL_NAME}"; do
    if gcloud compute firewall-rules describe "${fw_name}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
      conflicts+=("Firewall Rule: ${fw_name}")
    fi
  done
  if gcloud compute routers describe "${ROUTER_NAME}" --region="${REGION_R1}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    conflicts+=("Cloud Router: ${ROUTER_NAME} (${REGION_R1})")
  fi
  if gcloud compute routers describe "${ROUTER_NAME}" --region="${REGION_R2}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    conflicts+=("Cloud Router: ${ROUTER_NAME} (${REGION_R2})")
  fi
  local existing_nat_ips
  existing_nat_ips="$(gcloud compute addresses list --filter="name ~ '^${NAT_IPS_NAME}(-.*)?$'" --format="value(name)" --project="${PROJECT_ID}" 2>/dev/null || true)"
  if [[ -n "${existing_nat_ips}" ]]; then
    conflicts+=("NAT IP Address(es): $(echo "${existing_nat_ips}" | tr '\n' ' ')")
  fi

  if [[ "${#conflicts[@]}" -gt 0 ]]; then
    echo "[ERROR] Pre-existing GCP resources match '${DEPLOYMENT_NAME}' in '${PROJECT_ID}' without '${OWNERSHIP_MARKER_FILE}':" >&2
    for item in "${conflicts[@]}"; do
      echo "  - ${item}" >&2
    done
    echo "Specify a distinct --deployment-name, or run with '--cleanup --force-cleanup' if these belong to your interrupted run." >&2
    exit 1
  fi
}

if [[ "${CLEANUP_ONLY}" -eq 1 && "${FORCE_CLEANUP}" -ne 1 ]] && ! verify_ownership_marker; then
  echo "[ERROR] Refusing to tear down '${DEPLOYMENT_NAME}' without ownership marker '${OWNERSHIP_MARKER_FILE}'." >&2
  echo "Re-run with '--cleanup --force-cleanup' if you created this deployment and deleted the local marker." >&2
  exit 1
fi

wait_for_running_gke_operations() {
  for cluster_info in "${CLUSTER_R1_NAME}:${REGION_R1}" "${CLUSTER_R2_NAME}:${REGION_R2}"; do
    local c_name="${cluster_info%%:*}"
    local c_region="${cluster_info##*:}"
    local attempt
    for attempt in 1 2 3; do
      local running_ops
      running_ops="$(gcloud container operations list \
        --location="${c_region}" \
        --project="${PROJECT_ID}" \
        --filter="targetLink ~ '/clusters/${c_name}(/|$)' AND status = 'RUNNING'" \
        --format="value(name)" \
        --limit=20 \
        --quiet 2>/dev/null || true)"
      if [[ -z "${running_ops}" ]]; then
        break
      fi
      while read -r op_name; do
        local op_id="${op_name##*/}"
        if [[ -n "${op_id}" ]]; then
          echo "[TEARDOWN] Waiting for active GKE operation '${op_id}' on ${c_name} (${c_region}) (pass ${attempt}/3)..."
          gcloud container operations wait "${op_id}" --location="${c_region}" --project="${PROJECT_ID}" --quiet || true
        fi
      done <<< "${running_ops}"
    done
  done
}

destroy_gcluster_deployment() {
  local dep_dir="${WORKDIR_ROOT}/${DEPLOYMENT_NAME}"
  if [[ ! -d "${dep_dir}" ]]; then
    return 0
  fi

  cd "${WORKDIR_ROOT}"
  mkdir -p "${dep_dir}/networking" "${dep_dir}/cluster_r1" "${dep_dir}/cluster_r2" "${dep_dir}/workload_pathways" "${dep_dir}/workload_mcjax"
  "${GCLUSTER_BIN}" export-outputs "${dep_dir}/cluster_r1" 2>/dev/null || true
  "${GCLUSTER_BIN}" export-outputs "${dep_dir}/cluster_r2" 2>/dev/null || true

  if [[ ! -f "${dep_dir}/.ghpc/artifacts/cluster_r1_outputs.tfvars" ]] || \
     [[ ! -f "${dep_dir}/.ghpc/artifacts/cluster_r2_outputs.tfvars" ]] || \
     ! grep -q "cluster_ca_certificate" "${dep_dir}/.ghpc/artifacts/cluster_r1_outputs.tfvars" 2>/dev/null || \
     ! grep -q "cluster_ca_certificate" "${dep_dir}/.ghpc/artifacts/cluster_r2_outputs.tfvars" 2>/dev/null; then
    echo "[TEARDOWN] Cluster output artifacts incomplete in ${dep_dir}. Skipping workload groups..."
    "${GCLUSTER_BIN}" destroy "${DEPLOYMENT_NAME}" --skip "workload_pathways,workload_mcjax" --auto-approve || true
    return 0
  fi

  local inactive_groups=""
  local has_pw_state=0 has_mcjax_state=0
  if [[ -f "${dep_dir}/workload_pathways/terraform.tfstate" ]]; then
    has_pw_state=1
  fi
  if [[ -f "${dep_dir}/workload_mcjax/terraform.tfstate" ]]; then
    has_mcjax_state=1
  fi
  if [[ "${has_pw_state}" -eq 0 && "${has_mcjax_state}" -eq 0 ]]; then
    inactive_groups="workload_pathways,workload_mcjax"
  elif [[ "${has_pw_state}" -eq 1 && "${has_mcjax_state}" -eq 0 ]]; then
    inactive_groups="workload_mcjax"
  elif [[ "${has_pw_state}" -eq 0 && "${has_mcjax_state}" -eq 1 ]]; then
    inactive_groups="workload_pathways"
  fi

  local destroy_ok=0
  if [[ -n "${inactive_groups}" ]]; then
    if "${GCLUSTER_BIN}" destroy "${DEPLOYMENT_NAME}" --skip "${inactive_groups}" --auto-approve; then
      destroy_ok=1
    fi
  else
    if "${GCLUSTER_BIN}" destroy "${DEPLOYMENT_NAME}" --auto-approve; then
      destroy_ok=1
    fi
  fi
  if [[ "${destroy_ok}" -ne 1 ]]; then
    echo "[TEARDOWN] Retrying gcluster destroy with '--skip workload_pathways,workload_mcjax'..."
    "${GCLUSTER_BIN}" destroy "${DEPLOYMENT_NAME}" --skip "workload_pathways,workload_mcjax" --auto-approve || true
  fi
}

teardown() {
  echo ""
  echo "========================================================================"
  echo "[TEARDOWN] Tearing down deployment '${DEPLOYMENT_NAME}' in project '${PROJECT_ID}'..."
  echo "========================================================================"
  if [[ -n "${LOG_PID}" ]] && kill -0 "${LOG_PID}" 2>/dev/null; then
    kill "${LOG_PID}" 2>/dev/null || true
  fi

  if [[ "${FORCE_CLEANUP}" -ne 1 ]] && ! verify_ownership_marker; then
    echo "[ERROR] Refusing to tear down '${DEPLOYMENT_NAME}' without ownership marker '${OWNERSHIP_MARKER_FILE}'." >&2
    return 1
  fi

  load_ownership_marker_metadata

  if gcloud container clusters describe "${CLUSTER_R1_NAME}" --region="${REGION_R1}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    gcloud container clusters get-credentials "${CLUSTER_R1_NAME}" --region="${REGION_R1}" --project="${PROJECT_ID}" >/dev/null 2>&1 || true
    kubectl --context="${CTX_R1}" delete svc pathways-rm-ilb mcjax-coord-ilb --ignore-not-found --timeout=60s 2>/dev/null || true
    for _ in {1..12}; do
      local remaining_fr
      remaining_fr="$(gcloud compute forwarding-rules list --filter="subnetwork ~ '/subnetworks/${SUBNET_R1_NAME}$'" --format="value(name)" --project="${PROJECT_ID}" 2>/dev/null || true)"
      if [[ -z "${remaining_fr}" ]]; then
        break
      fi
      sleep 5
    done
  fi

  wait_for_running_gke_operations
  destroy_gcluster_deployment

  wait_for_running_gke_operations
  for cluster_info in "${CLUSTER_R1_NAME}:${REGION_R1}" "${CLUSTER_R2_NAME}:${REGION_R2}"; do
    local c_name="${cluster_info%%:*}"
    local c_region="${cluster_info##*:}"
    if gcloud container clusters describe "${c_name}" --region="${c_region}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
      local cluster_label
      cluster_label="$(gcloud container clusters describe "${c_name}" --region="${c_region}" --project="${PROJECT_ID}" --format="value(resourceLabels.ghpc_deployment)" 2>/dev/null || true)"
      if [[ -n "${cluster_label}" && "${cluster_label}" != "${DEPLOYMENT_NAME}" ]]; then
        echo "[ERROR] Cluster '${c_name}' has label ghpc_deployment='${cluster_label}' (expected '${DEPLOYMENT_NAME}'). Skipping." >&2
        continue
      fi
      gcloud container clusters delete "${c_name}" --region="${c_region}" --project="${PROJECT_ID}" --quiet || true
    fi
  done

  if [[ -d "${WORKDIR_ROOT}/${DEPLOYMENT_NAME}" ]]; then
    cd "${WORKDIR_ROOT}"
    "${GCLUSTER_BIN}" destroy "${DEPLOYMENT_NAME}" --skip "workload_pathways,workload_mcjax,cluster_r1,cluster_r2" --auto-approve || true
  fi

  local vpc_self_link_regex="/projects/${PROJECT_ID}/global/networks/${VPC_NAME}$"
  local subnet_self_link_regex="/subnetworks/${DEPLOYMENT_NAME}-sub-(r1|r2)$"
  if gcloud compute networks describe "${VPC_NAME}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    echo "[TEARDOWN] Purging remaining VPC resources for '${VPC_NAME}' directly via gcloud..."
    while read -r fr_name fr_reg; do
      if [[ -n "${fr_name}" && -n "${fr_reg}" ]]; then
        gcloud compute forwarding-rules delete "${fr_name}" --region="${fr_reg}" --project="${PROJECT_ID}" --quiet 2>/dev/null || true
      fi
    done < <(gcloud compute forwarding-rules list --filter="subnetwork ~ '${subnet_self_link_regex}' OR network ~ '${vpc_self_link_regex}'" --format="value(name,region)" --project="${PROJECT_ID}" 2>/dev/null || true)

    local hc_urls=()
    while read -r bs_name bs_reg bs_hcs; do
      if [[ -n "${bs_name}" ]]; then
        if [[ -n "${bs_hcs}" ]]; then
          hc_urls+=("${bs_hcs}")
        fi
        if [[ -n "${bs_reg}" ]]; then
          gcloud compute backend-services delete "${bs_name}" --region="${bs_reg}" --project="${PROJECT_ID}" --quiet 2>/dev/null || true
        else
          gcloud compute backend-services delete "${bs_name}" --global --project="${PROJECT_ID}" --quiet 2>/dev/null || true
        fi
      fi
    done < <(gcloud compute backend-services list --filter="network ~ '${vpc_self_link_regex}'" --format="value(name,region,healthChecks)" --project="${PROJECT_ID}" 2>/dev/null || true)

    for hc_list in "${hc_urls[@]}"; do
      for hc_url in ${hc_list//;/ }; do
        local hc_name="${hc_url##*/}"
        if [[ -n "${hc_name}" ]]; then
          gcloud compute health-checks delete "${hc_name}" --global --project="${PROJECT_ID}" --quiet 2>/dev/null || \
          gcloud compute health-checks delete "${hc_name}" --region="${REGION_R1}" --project="${PROJECT_ID}" --quiet 2>/dev/null || \
          gcloud compute health-checks delete "${hc_name}" --region="${REGION_R2}" --project="${PROJECT_ID}" --quiet 2>/dev/null || true
        fi
      done
    done

    while read -r neg_name neg_zone; do
      if [[ -n "${neg_name}" && -n "${neg_zone}" ]]; then
        gcloud compute network-endpoint-groups delete "${neg_name}" --zone="${neg_zone}" --project="${PROJECT_ID}" --quiet 2>/dev/null || true
      fi
    done < <(gcloud compute network-endpoint-groups list --filter="network ~ '${vpc_self_link_regex}'" --format="value(name,zone)" --project="${PROJECT_ID}" 2>/dev/null || true)

    local fw_rules
    fw_rules="$(gcloud compute firewall-rules list --filter="network ~ '${vpc_self_link_regex}'" --format="value(name)" --project="${PROJECT_ID}" 2>/dev/null || true)"
    if [[ -n "${fw_rules}" ]]; then
      echo "${fw_rules}" | xargs -r gcloud compute firewall-rules delete --quiet --project="${PROJECT_ID}" 2>/dev/null || true
    fi

    while read -r r_name r_reg; do
      if [[ -n "${r_name}" && -n "${r_reg}" ]]; then
        gcloud compute routers delete "${r_name}" --region="${r_reg}" --project="${PROJECT_ID}" --quiet 2>/dev/null || true
      fi
    done < <(gcloud compute routers list --filter="network ~ '${vpc_self_link_regex}'" --format="value(name,region)" --project="${PROJECT_ID}" 2>/dev/null || true)

    gcloud compute networks subnets delete "${SUBNET_R1_NAME}" --region="${REGION_R1}" --project="${PROJECT_ID}" --quiet 2>/dev/null || true
    gcloud compute networks subnets delete "${SUBNET_R2_NAME}" --region="${REGION_R2}" --project="${PROJECT_ID}" --quiet 2>/dev/null || true

    while read -r p_line; do
      for p_name in ${p_line//;/ }; do
        if [[ -n "${p_name}" ]]; then
          gcloud compute networks peerings delete "${p_name}" --network="${VPC_NAME}" --project="${PROJECT_ID}" --quiet 2>/dev/null || true
        fi
      done
    done < <(gcloud compute networks peerings list --network="${VPC_NAME}" --format="value(peerings[].name)" --project="${PROJECT_ID}" 2>/dev/null || true)

    gcloud compute networks delete "${VPC_NAME}" --quiet --project="${PROJECT_ID}" 2>/dev/null || true
  fi

  while read -r ip_name ip_reg; do
    if [[ -n "${ip_name}" && -n "${ip_reg}" ]]; then
      gcloud compute addresses delete "${ip_name}" --region="${ip_reg}" --project="${PROJECT_ID}" --quiet 2>/dev/null || true
    fi
  done < <(gcloud compute addresses list --filter="name ~ '^${NAT_IPS_NAME}(-.*)?$'" --format="value(name,region)" --project="${PROJECT_ID}" 2>/dev/null || true)

  if [[ "${KEEP_GCS_ARTIFACTS}" -eq 1 ]]; then
    echo "[TEARDOWN] Preserving GCS artifacts at ${GCS_SCRATCH_LOCATION} (--keep-gcs-artifacts set)."
  else
    echo "[TEARDOWN] Cleaning up GCS scratch and run objects under ${GCS_SCRATCH_LOCATION}..."
    gcloud storage rm --recursive "${GCS_SCRATCH_LOCATION}/**" "${GCS_SCRATCH_LOCATION}" 2>/dev/null || true
    if [[ "${CREATED_GCS_BUCKET}" -eq 1 ]]; then
      gcloud storage buckets delete "${GCS_BUCKET}" --project="${PROJECT_ID}" --quiet 2>/dev/null || true
    fi
  fi

  echo ""
  echo "========================================================================"
  echo "[AUDIT] Checking zero remaining '${DEPLOYMENT_NAME}' resources in project ${PROJECT_ID}..."
  echo "========================================================================"
  local audit_leaks=()
  local out

  out="$(gcloud container clusters list --filter="name ~ '^${DEPLOYMENT_NAME}-(c1|c2)$'" --format="value(name)" --project="${PROJECT_ID}" 2>/dev/null || true)"
  if [[ -n "${out}" ]]; then audit_leaks+=("GKE Clusters: ${out}"); fi

  out="$(gcloud compute instances list --filter="name ~ '^${GKE_VM_PREFIX}' AND networkInterfaces.network ~ '${vpc_self_link_regex}'" --format="value(name)" --project="${PROJECT_ID}" 2>/dev/null || true)"
  if [[ -n "${out}" ]]; then audit_leaks+=("GCE Instances: ${out}"); fi

  out="$(gcloud compute disks list --filter="name ~ '^${GKE_VM_PREFIX}' AND (zone ~ '${ZONE_R1}$' OR zone ~ '${ZONE_R1_SYSTEM}$' OR zone ~ '${ZONE_R2}$' OR zone ~ '${ZONE_R2_SYSTEM}$')" --format="value(name)" --project="${PROJECT_ID}" 2>/dev/null || true)"
  if [[ -n "${out}" ]]; then audit_leaks+=("GCE Disks: ${out}"); fi

  out="$(gcloud compute forwarding-rules list --filter="subnetwork ~ '${subnet_self_link_regex}' OR network ~ '${vpc_self_link_regex}'" --format="value(name)" --project="${PROJECT_ID}" 2>/dev/null || true)"
  if [[ -n "${out}" ]]; then audit_leaks+=("Forwarding Rules: ${out}"); fi

  out="$(gcloud compute backend-services list --filter="network ~ '${vpc_self_link_regex}'" --format="value(name)" --project="${PROJECT_ID}" 2>/dev/null || true)"
  if [[ -n "${out}" ]]; then audit_leaks+=("Backend Services: ${out}"); fi

  out="$(gcloud compute networks subnets list --filter="name ~ '^${DEPLOYMENT_NAME}-sub-(r1|r2)$'" --format="value(name)" --project="${PROJECT_ID}" 2>/dev/null || true)"
  if [[ -n "${out}" ]]; then audit_leaks+=("Subnetworks: ${out}"); fi

  out="$(gcloud compute routers list --filter="network ~ '${vpc_self_link_regex}' OR name = '${ROUTER_NAME}'" --format="value(name)" --project="${PROJECT_ID}" 2>/dev/null || true)"
  if [[ -n "${out}" ]]; then audit_leaks+=("Cloud Routers: ${out}"); fi

  out="$(gcloud compute firewall-rules list --filter="network ~ '${vpc_self_link_regex}' OR name = '${FIREWALL_RULE_NAME}'" --format="value(name)" --project="${PROJECT_ID}" 2>/dev/null || true)"
  if [[ -n "${out}" ]]; then audit_leaks+=("Firewall Rules: ${out}"); fi

  out="$(gcloud compute addresses list --filter="name ~ '^${NAT_IPS_NAME}(-.*)?$'" --format="value(name,region)" --project="${PROJECT_ID}" 2>/dev/null || true)"
  if [[ -n "${out}" ]]; then audit_leaks+=("NAT IP Addresses: ${out}"); fi

  out="$(gcloud compute networks list --filter="name = '${VPC_NAME}'" --format="value(name)" --project="${PROJECT_ID}" 2>/dev/null || true)"
  if [[ -n "${out}" ]]; then audit_leaks+=("VPC Networks: ${out}"); fi

  if [[ "${#audit_leaks[@]}" -gt 0 ]]; then
    echo "[ERROR] Scoped resource audit detected remaining resources for '${DEPLOYMENT_NAME}':" >&2
    for leak in "${audit_leaks[@]}"; do
      echo "  - ${leak}" >&2
    done
    echo "Retaining ownership marker '${OWNERSHIP_MARKER_FILE}' so '--cleanup' can be re-run." >&2
    return 1
  fi

  rm -rf "${WORKDIR_ROOT:?}/${DEPLOYMENT_NAME}" 2>/dev/null || true
  remove_ownership_marker
  echo "[AUDIT] Scoped resource check passed: zero '${DEPLOYMENT_NAME}' resources remain in project ${PROJECT_ID}."
  echo "========================================================================"
}

cleanup_on_exit() {
  local exit_code=$?
  if [[ -n "${VARS_FILE}" ]]; then
    rm -f "${VARS_FILE}" 2>/dev/null || true
  fi
  if [[ "${DEPLOY_STARTED}" -eq 1 ]]; then
    if ! teardown; then
      exit_code=1
    fi
  fi
  exit "${exit_code}"
}

if [[ "${CLEANUP_ONLY}" -eq 1 ]]; then
  teardown
  exit $?
fi

trap cleanup_on_exit EXIT INT TERM

preflight_check_no_conflicting_resources

if [[ -z "${AUTHORIZED_CIDR}" ]]; then
  detected_ip=""
  if command -v python3 >/dev/null 2>&1; then
    detected_ip="$(python3 - <<'PYEOF' 2>/dev/null || true
import urllib.request
with urllib.request.urlopen("https://checkip.amazonaws.com", timeout=5) as resp:
  print(resp.read().decode("utf-8").strip())
PYEOF
    )"
  fi
  if [[ "${detected_ip}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    AUTHORIZED_CIDR="${detected_ip}/32"
    echo "[PREFLIGHT] Restricted GKE master_authorized_networks to caller IPv4: ${AUTHORIZED_CIDR}"
  else
    AUTHORIZED_CIDR="0.0.0.0/0"
    echo "[WARNING] Could not auto-detect caller IPv4; falling back to authorized_cidr=${AUTHORIZED_CIDR} (pass --authorized-cidr <cidr> to restrict)."
  fi
fi

echo "========================================================================"
echo "Starting Cross-Region DiLoCo Run (${STEPS} steps)"
echo "  Project ID:      ${PROJECT_ID}"
echo "  Deployment:      ${DEPLOYMENT_NAME} (${WORKDIR_ROOT}/${DEPLOYMENT_NAME})"
echo "  Runtime:         ${RUNTIME}"
echo "  Region 1:        ${REGION_R1} (TPU: ${ZONE_R1}, CPU: ${ZONE_R1_SYSTEM}, spot_tpu=${SPOT_TPU})"
echo "  Region 2:        ${REGION_R2} (TPU: ${ZONE_R2}, CPU: ${ZONE_R2_SYSTEM}, spot_tpu=${SPOT_TPU})"
echo "  Authorized CIDR: ${AUTHORIZED_CIDR}"
echo "  GCS Scratch:     ${GCS_SCRATCH_LOCATION}"
echo "  MaxText Image:   ${MAXTEXT_IMAGE}"
echo "  Run Name:        ${RUN_NAME}"
if [[ -n "${EXTRA_MAXTEXT_ARGS}" ]]; then
  echo "  MaxText Args:    ${EXTRA_MAXTEXT_ARGS}"
fi
echo "========================================================================"

echo "[1/5] Checking required GCP services (Service Usage, Compute, GKE)..."
if ! enabled_services="$(gcloud services list --enabled --project="${PROJECT_ID}" --format="value(config.name)")"; then
  echo "[ERROR] Failed to query enabled GCP services in project '${PROJECT_ID}'." >&2
  exit 1
fi
missing_services=()
for svc in "${REQUIRED_GCP_SERVICES[@]}"; do
  if ! printf '%s\n' "${enabled_services}" | grep -qx "${svc}"; then
    missing_services+=("${svc}")
  fi
done
if [[ "${#missing_services[@]}" -gt 0 ]]; then
  if [[ "${ENABLE_APIS}" -eq 1 ]]; then
    echo "[1/5] Enabling missing GCP services: ${missing_services[*]}..."
    gcloud services enable "${missing_services[@]}" --project="${PROJECT_ID}"
  else
    echo "[ERROR] Missing enabled GCP services in '${PROJECT_ID}': ${missing_services[*]}" >&2
    echo "Re-run with '--enable-apis' or enable them manually." >&2
    exit 1
  fi
fi

write_ownership_marker
DEPLOY_STARTED=1

if ! gcloud storage buckets describe "${GCS_BUCKET}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "[2/5] Creating temporary GCS bucket ${GCS_BUCKET}..."
  CREATED_GCS_BUCKET=1
  write_ownership_marker
  gcloud storage buckets create "${GCS_BUCKET}" --project="${PROJECT_ID}" --location=us --uniform-bucket-level-access
else
  echo "[2/5] Reusing GCS bucket ${GCS_BUCKET} (scoped to ${GCS_SCRATCH_LOCATION})."
fi

echo "[3/5] Deploying cross-region infrastructure and workloads via gcluster..."
verify_reused_subdirs "${WORKDIR_ROOT}"
verify_reused_subdirs "${WORKDIR_ROOT}/${DEPLOYMENT_NAME}"
if [[ -d "${WORKDIR_ROOT}/${DEPLOYMENT_NAME}" && ! -d "${WORKDIR_ROOT}/${DEPLOYMENT_NAME}/.ghpc" ]]; then
  rm -rf "${WORKDIR_ROOT:?}/${DEPLOYMENT_NAME}"
fi

ESCAPED_MAXTEXT_ARGS="${EXTRA_MAXTEXT_ARGS//\\/\\\\}"
ESCAPED_MAXTEXT_ARGS="${ESCAPED_MAXTEXT_ARGS//\"/\\\"}"
VARS_FILE="$(umask 077 && mktemp "${WORKDIR_ROOT}/${DEPLOYMENT_NAME}_vars.XXXXXX.yaml")"
cat <<EOF > "${VARS_FILE}"
vars:
  project_id: "${PROJECT_ID}"
  deployment_name: "${DEPLOYMENT_NAME}"
  authorized_cidr: "${AUTHORIZED_CIDR}"
  region: "${REGION_R1}"
  zone: "${ZONE_R1}"
  region_r1: "${REGION_R1}"
  zone_r1: "${ZONE_R1}"
  zone_r1_system: "${ZONE_R1_SYSTEM}"
  system_machine_type_r1: "${SYSTEM_MACHINE_TYPE_R1}"
  head_machine_type_r1: "${HEAD_MACHINE_TYPE_R1}"
  region_r2: "${REGION_R2}"
  zone_r2: "${ZONE_R2}"
  zone_r2_system: "${ZONE_R2_SYSTEM}"
  system_machine_type_r2: "${SYSTEM_MACHINE_TYPE_R2}"
  spot_tpu: ${SPOT_TPU}
  gcs_scratch_location: "${GCS_SCRATCH_LOCATION}"
  maxtext_image: "${MAXTEXT_IMAGE}"
  run_name: "${RUN_NAME}"
  steps: ${STEPS}
  extra_maxtext_args: "${ESCAPED_MAXTEXT_ARGS}"
EOF

report_failed_gke_operations() {
  for cluster_info in "${CLUSTER_R1_NAME}:${REGION_R1}" "${CLUSTER_R2_NAME}:${REGION_R2}"; do
    local c_name="${cluster_info%%:*}"
    local c_region="${cluster_info##*:}"
    gcloud container operations list \
      --location="${c_region}" \
      --project="${PROJECT_ID}" \
      --filter="targetLink ~ '/clusters/${c_name}(/|$)' AND (statusMessage != '' OR error.message != '')" \
      --format="table(name,operationType,status,targetLink.basename():label=TARGET,statusMessage,error.message)" \
      --quiet >&2 2>/dev/null || true
  done
}

if ! "${GCLUSTER_BIN}" deploy "${SCRIPT_DIR}/deployment.yaml" \
  -o "${WORKDIR_ROOT}" \
  -d "${VARS_FILE}" \
  "${EXTRA_GCLUSTER_VARS[@]}" \
  --skip "${SKIP_WORKLOAD_GROUP}" \
  --force \
  --auto-approve; then
  echo "[ERROR] gcluster deploy failed for '${DEPLOYMENT_NAME}'." >&2
  report_failed_gke_operations
  exit 1
fi

echo "[4/5] Fetching GKE cluster credentials..."
gcloud container clusters get-credentials "${CLUSTER_R1_NAME}" --region="${REGION_R1}" --project="${PROJECT_ID}"
gcloud container clusters get-credentials "${CLUSTER_R2_NAME}" --region="${REGION_R2}" --project="${PROJECT_ID}"

dump_workload_diagnostics() {
  echo "=== Cluster 1 (${CTX_R1}) nodes and pods ===" >&2
  kubectl --context="${CTX_R1}" get nodes,pods -o wide >&2 2>&1 || true
  echo "=== Cluster 2 (${CTX_R2}) nodes and pods ===" >&2
  kubectl --context="${CTX_R2}" get nodes,pods -o wide >&2 2>&1 || true
  if [[ "${RUNTIME}" == "pathways" ]]; then
    kubectl --context="${CTX_R1}" describe pod -l app=pathways-head >&2 2>&1 || true
    for c in pathways-rm pathways-proxy maxtext-trainer; do
      echo "=== pathways-head (${c}) logs ===" >&2
      kubectl --context="${CTX_R1}" logs job/pathways-head -c "${c}" --tail=80 >&2 2>&1 || true
    done
    kubectl --context="${CTX_R1}" logs job/pathways-worker-slice-0 --all-pods=true --tail=50 >&2 2>&1 || true
    kubectl --context="${CTX_R2}" logs job/pathways-worker-slice-1 --all-pods=true --tail=50 >&2 2>&1 || true
  else
    kubectl --context="${CTX_R1}" describe pod -l job-name=maxtext-worker-slice-0 >&2 2>&1 || true
    kubectl --context="${CTX_R1}" logs job/maxtext-worker-slice-0 -c maxtext-worker --all-pods=true --tail=80 >&2 2>&1 || true
    kubectl --context="${CTX_R2}" describe pod -l job-name=maxtext-worker-slice-1 >&2 2>&1 || true
    kubectl --context="${CTX_R2}" logs job/maxtext-worker-slice-1 -c maxtext-worker --all-pods=true --tail=80 >&2 2>&1 || true
  fi
}

has_fatal_waiting_container() {
  local ctx="$1" selector="$2"
  local reasons
  reasons="$(kubectl --context="${ctx}" get pods -l "${selector}" -o jsonpath='{range .items[*]}{range .status.initContainerStatuses[*]}{.state.waiting.reason}{"\n"}{end}{range .status.containerStatuses[*]}{.state.waiting.reason}{"\n"}{end}{end}' 2>/dev/null || true)"
  printf '%s\n' "${reasons}" | grep -qE "^(CrashLoopBackOff|ErrImagePull|ImagePullBackOff|CreateContainerConfigError|InvalidImageName)$"
}

if [[ "${RUNTIME}" == "pathways" ]]; then
  echo "[5/5] Waiting for pathways-head Pod to start in ${CTX_R1}..."
  wait_seconds=0
  until kubectl --context="${CTX_R1}" get pods -l app=pathways-head -o jsonpath='{.items[0].status.phase}' 2>/dev/null | grep -qE "Running|Succeeded"; do
    if kubectl --context="${CTX_R1}" get job pathways-head -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null | grep -q "True" || \
       kubectl --context="${CTX_R1}" get pods -l app=pathways-head -o jsonpath='{.items[0].status.phase}' 2>/dev/null | grep -q "Failed" || \
       has_fatal_waiting_container "${CTX_R1}" "app=pathways-head"; then
      echo "[ERROR] pathways-head failed during startup." >&2
      dump_workload_diagnostics
      exit 1
    fi
    if (( wait_seconds >= JOB_TIMEOUT_SECONDS )); then
      echo "[ERROR] Timed out after ${JOB_TIMEOUT_SECONDS}s waiting for pathways-head to start." >&2
      dump_workload_diagnostics
      exit 1
    fi
    sleep 10
    wait_seconds=$((wait_seconds + 10))
  done

  echo "Streaming MaxText training logs (${STEPS} steps)..."
  kubectl --context="${CTX_R1}" logs -f job/pathways-head -c maxtext-trainer &
  LOG_PID=$!

  elapsed_job_seconds=0
  job_status="Running"
  while (( elapsed_job_seconds < JOB_TIMEOUT_SECONDS )); do
    if kubectl --context="${CTX_R1}" get job pathways-head -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null | grep -q "True"; then
      job_status="Complete"; break
    fi
    if kubectl --context="${CTX_R1}" get job pathways-head -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null | grep -q "True"; then
      job_status="Failed"; break
    fi
    if [[ -n "${LOG_PID}" ]] && ! kill -0 "${LOG_PID}" 2>/dev/null; then
      kubectl --context="${CTX_R1}" logs -f job/pathways-head -c maxtext-trainer 2>/dev/null &
      LOG_PID=$!
    fi
    sleep 15
    elapsed_job_seconds=$((elapsed_job_seconds + 15))
  done

  if [[ "${job_status}" != "Complete" ]]; then
    echo "[ERROR] Job pathways-head ended with status '${job_status}'." >&2
    dump_workload_diagnostics
    exit 1
  fi
  echo "[SUCCESS] Completed ${STEPS} steps of cross-region DiLoCo training (Pathways)."
else
  echo "[5/5] Waiting for McJax Slice 0 Pod 0 to start in ${CTX_R1}..."
  mcjax_pod0_selector="job-name=maxtext-worker-slice-0,batch.kubernetes.io/job-completion-index=0"
  wait_seconds=0
  until kubectl --context="${CTX_R1}" get pods -l "${mcjax_pod0_selector}" -o jsonpath='{.items[0].status.phase}' 2>/dev/null | grep -qE "Running|Succeeded"; do
    for slice_ctx_job in "${CTX_R1}:maxtext-worker-slice-0" "${CTX_R2}:maxtext-worker-slice-1"; do
      s_ctx="${slice_ctx_job%%:*}"
      s_job="${slice_ctx_job##*:}"
      if kubectl --context="${s_ctx}" get job "${s_job}" -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null | grep -q "True" || \
         kubectl --context="${s_ctx}" get pods -l "job-name=${s_job}" -o jsonpath='{.items[*].status.phase}' 2>/dev/null | grep -q "Failed" || \
         has_fatal_waiting_container "${s_ctx}" "job-name=${s_job}"; then
        echo "[ERROR] McJax Job '${s_job}' in ${s_ctx} failed during startup." >&2
        dump_workload_diagnostics
        exit 1
      fi
    done
    if (( wait_seconds >= JOB_TIMEOUT_SECONDS )); then
      echo "[ERROR] Timed out after ${JOB_TIMEOUT_SECONDS}s waiting for McJax Slice 0 Pod 0 to start." >&2
      dump_workload_diagnostics
      exit 1
    fi
    sleep 10
    wait_seconds=$((wait_seconds + 10))
  done

  echo "Streaming McJax Slice 0 Pod 0 training logs (${STEPS} steps)..."
  kubectl --context="${CTX_R1}" logs -f -l "${mcjax_pod0_selector}" -c maxtext-worker &
  LOG_PID=$!

  elapsed_job_seconds=0
  job_status="Running"
  while (( elapsed_job_seconds < JOB_TIMEOUT_SECONDS )); do
    slice0_complete="$(kubectl --context="${CTX_R1}" get job maxtext-worker-slice-0 -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null || true)"
    slice1_complete="$(kubectl --context="${CTX_R2}" get job maxtext-worker-slice-1 -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null || true)"
    if [[ "${slice0_complete}" == "True" && "${slice1_complete}" == "True" ]]; then
      job_status="Complete"; break
    fi
    if kubectl --context="${CTX_R1}" get job maxtext-worker-slice-0 -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null | grep -q "True" || \
       kubectl --context="${CTX_R2}" get job maxtext-worker-slice-1 -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null | grep -q "True"; then
      job_status="Failed"; break
    fi
    if [[ -n "${LOG_PID}" ]] && ! kill -0 "${LOG_PID}" 2>/dev/null; then
      kubectl --context="${CTX_R1}" logs -f -l "${mcjax_pod0_selector}" -c maxtext-worker 2>/dev/null &
      LOG_PID=$!
    fi
    sleep 15
    elapsed_job_seconds=$((elapsed_job_seconds + 15))
  done

  if [[ "${job_status}" != "Complete" ]]; then
    echo "[ERROR] McJax slice Jobs ended with status '${job_status}'." >&2
    dump_workload_diagnostics
    exit 1
  fi
  echo "[SUCCESS] Completed ${STEPS} steps of cross-region DiLoCo training (McJax)."
fi
