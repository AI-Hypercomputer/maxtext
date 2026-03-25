#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# PROJECT VARIABLES
# ==============================================================================
export PROJECT_ID="zenteiq-lxp-1722918338008"
export REGION="asia-south1"
export ZONE="asia-south1-b"
export CLUSTER_NAME="brahmai-v6e-lustre"
export RESERVATION_NAME="cloudtpu-20260317070000-2123307204"
export NETWORK_NAME="zenteiq-tpu-vpc"

# ==============================================================================
# DEVICE VARIABLES
# ==============================================================================
export DEVICE="tpu"
export TPU_TYPE="v6e-16"
export NUM_SLICES="2"

# ==============================================================================
# VPC VARIABLES
# ==============================================================================
export IP_RANGE_NAME="lustre-peering-range"
export SOURCE_RANGES="10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"

# ==============================================================================
# LUSTRE VARIABLES  (keep in sync with lustre-manifest.yaml and brahmai-27b.sh)
# ==============================================================================
export STORAGE_NAME="ziq-lustre"
export STORAGE_THROUGHPUT="1000"    # 1,000 MBps/TiB performance tier
export STORAGE_CAPACITY="36000"     # Minimum required capacity in GiB
export STORAGE_FS="ziqfs"
export LUSTRE_IP=""                 # IP Address from lustre mountPoint (populated by lustre-manifest-generate)
export MOUNTPOINT="/lustre-data"
export MANIFEST="./lustre.yaml"

# ==============================================================================
# WORKLOAD / DOCKER VARIABLES
# ==============================================================================
export REPOSITORY="asia-south1-docker.pkg.dev/zenteiq-lxp-1722918338008/maxtext-training"
export DOCKER_IMAGE="${REPOSITORY}/brahmai:$(git rev-parse --short HEAD)"

# ==============================================================================
# ECHO ALL ENVIRONMENT VARIABLES
# ==============================================================================
echo_vars() {
  echo ""
  echo "======================================================================"
  echo "  ENVIRONMENT VARIABLES"
  echo "======================================================================"
  echo "  [Project]"
  echo "    PROJECT_ID               = ${PROJECT_ID}"
  echo "    REGION                   = ${REGION}"
  echo "    ZONE                     = ${ZONE}"
  echo "    CLUSTER_NAME             = ${CLUSTER_NAME}"
  echo "    RESERVATION_NAME         = ${RESERVATION_NAME}"
  echo "    NETWORK_NAME             = ${NETWORK_NAME}"
  echo ""
  echo "  [Device]"
  echo "    DEVICE                   = ${DEVICE}"
  echo "    TPU_TYPE                 = ${TPU_TYPE}"
  echo "    NUM_SLICES               = ${NUM_SLICES}"
  echo "    GPU_TYPE                 = ${GPU_TYPE}"
  echo "    NUM_NODES                = ${NUM_NODES}"
  echo ""
  echo "  [VM]"
  echo "    VM_NAME                  = ${VM_NAME}"
  echo "    TPU_VERSION              = ${TPU_VERSION}"
  echo ""
  echo "  [VPC]"
  echo "    IP_RANGE_NAME            = ${IP_RANGE_NAME}"
  echo "    SOURCE_RANGES            = ${SOURCE_RANGES}"
  echo ""
  echo "  [Lustre]"
  echo "    STORAGE_NAME             = ${STORAGE_NAME}"
  echo "    STORAGE_THROUGHPUT       = ${STORAGE_THROUGHPUT}"
  echo "    STORAGE_CAPACITY         = ${STORAGE_CAPACITY}"
  echo "    STORAGE_FS               = ${STORAGE_FS}"
  echo "    LUSTRE_IP                = ${LUSTRE_IP}"
  echo "    MOUNTPOINT               = ${MOUNTPOINT}"
  echo "    MANIFEST                 = ${MANIFEST}"
  echo ""
  echo "  [Workload / Docker]"
  echo "    REPOSITORY               = ${REPOSITORY}"
  echo "    DOCKER_IMAGE             = ${DOCKER_IMAGE}"
  echo "    BASE_OUTPUT_DIR          = ${BASE_OUTPUT_DIR}"
  echo "    DATASET_PATH             = ${DATASET_PATH}"
  echo "======================================================================"
  echo ""
}

# ==============================================================================
# AUTH
# ==============================================================================
login() {
  echo "[login] Authenticating with gcloud..."
  gcloud auth login
  gcloud auth application-default login
  gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
}

setup() {
  echo "[setup] Configuring gcloud project and zone..."
  gcloud config set project "${PROJECT_ID}"
  gcloud config set compute/zone "${ZONE}"
}

# ==============================================================================
# VPC
# ==============================================================================
vpc-ip() {
  echo "[vpc-ip] Creating VPC peering IP range '${IP_RANGE_NAME}'..."
  gcloud compute addresses create "${IP_RANGE_NAME}" \
    --global \
    --purpose=VPC_PEERING \
    --prefix-length=20 \
    --network="${NETWORK_NAME}" \
    --project="${PROJECT_ID}"
}

vpc-peering() {
  echo "[vpc-peering] Connecting VPC peering for '${NETWORK_NAME}'..."
  gcloud services vpc-peerings connect \
    --network="${NETWORK_NAME}" \
    --ranges="${IP_RANGE_NAME}" \
    --service=servicenetworking.googleapis.com \
    --project="${PROJECT_ID}"
}

vpc-firewall() {
  echo "[vpc-firewall] Creating Lustre firewall rule..."
  gcloud compute firewall-rules create allow-lustre-all-internal \
    --allow=tcp:988,tcp:6988 \
    --network="${NETWORK_NAME}" \
    --source-ranges="${SOURCE_RANGES}" \
    --project="${PROJECT_ID}"
}

# ==============================================================================
# LUSTRE
# ==============================================================================
lustre-create() {
  echo "[lustre-create] Creating Lustre instance '${STORAGE_NAME}'..."
  gcloud lustre instances create "${STORAGE_NAME}" \
    --per-unit-storage-throughput="${STORAGE_THROUGHPUT}" \
    --capacity-gib="${STORAGE_CAPACITY}" \
    --filesystem="${STORAGE_FS}" \
    --location="${ZONE}" \
    --network="projects/${PROJECT_ID}/global/networks/${NETWORK_NAME}" \
    --project="${PROJECT_ID}" \
    --async
}

lustre-list() {
  echo "[lustre-list] Describing Lustre instance '${STORAGE_NAME}'..."
  gcloud lustre instances describe "${STORAGE_NAME}" \
    --location="${ZONE}" \
    --project="${PROJECT_ID}" \
    --format="value(mountPoint)"
}

lustre-automount() {
  echo "[lustre-automount] Attaching Lustre storage to cluster '${CLUSTER_NAME}'..."
  xpk storage attach "${STORAGE_NAME}" \
    --cluster="${CLUSTER_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --type=lustre \
    --mount-point="${MOUNTPOINT}" \
    --readonly=false \
    --auto-mount=true \
    --manifest="${MANIFEST}"
}

# Regenerates lustre-manifest.yaml from current env vars (keeps yaml in sync).
# If LUSTRE_IP is not already set, fetches it automatically from gcloud.
lustre-manifest-generate() {
  echo "[lustre-manifest-generate] Writing lustre-manifest.yaml from current env vars..."

  if [[ -z "${LUSTRE_IP:-}" ]]; then
    echo "[lustre-manifest-generate] LUSTRE_IP not set — fetching from gcloud..."
    local mount_point
    mount_point=$(gcloud lustre instances describe "${STORAGE_NAME}" \
      --location="${ZONE}" \
      --project="${PROJECT_ID}" \
      --format="value(mountPoint)" 2>/dev/null)

    if [[ -z "$mount_point" ]]; then
      echo "Error: could not retrieve mountPoint for '${STORAGE_NAME}'. Is the instance ready?" >&2
      return 1
    fi

    # mountPoint format is "<IP>@tcp:/<filesystem>" — extract the IP part
    LUSTRE_IP="${mount_point%%@*}"

    if [[ -z "$LUSTRE_IP" ]]; then
      echo "Error: could not parse IP from mountPoint '${mount_point}'" >&2
      return 1
    fi

    echo "[lustre-manifest-generate] Resolved LUSTRE_IP=${LUSTRE_IP}"
  fi

  cat > "${MANIFEST}" <<EOF
apiVersion: v1
kind: PersistentVolume
metadata:
  name: xpk-lustre-pv
spec:
  storageClassName: ""
  capacity:
    storage: "${STORAGE_CAPACITY}Gi"
  accessModes:
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Retain
  volumeMode: Filesystem
  claimRef:
    namespace: default
    name: xpk-lustre-pvc
  csi:
    driver: lustre.csi.storage.gke.io
    volumeHandle: "projects/${PROJECT_ID}/locations/${ZONE}/instances/${STORAGE_NAME}"
    volumeAttributes:
      ip: ${LUSTRE_IP} # IP Address from lustre mountPoint
      filesystem: ${STORAGE_FS}
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: xpk-lustre-pvc
  namespace: default
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: ""
  volumeName: xpk-lustre-pv
  resources:
    requests:
      storage: "${STORAGE_CAPACITY}Gi"
EOF
  echo "[lustre-manifest-generate] Written to ${MANIFEST}"
}

lustre-status() {
  echo "[lustre-status] Checking state of Lustre instance '${STORAGE_NAME}'..."
  gcloud lustre instances describe "${STORAGE_NAME}" \
    --location="${ZONE}" \
    --project="${PROJECT_ID}" \
    --format="table(name,state,capacityGib,network,mountPoint)"
}

# Runs a one-shot pod in the cluster that performs write/read/throughput checks
# on the Lustre mountpoint, then exits. Requires the PVC to already be mounted.
lustre-test() {
  echo "[lustre-test] Submitting Lustre I/O test pod to cluster '${CLUSTER_NAME}'..."
  kubectl run lustre-io-test \
    --image=busybox \
    --restart=Never \
    --rm \
    --attach \
    --overrides="{
      \"spec\": {
        \"containers\": [{
          \"name\": \"lustre-io-test\",
          \"image\": \"busybox\",
          \"command\": [\"sh\", \"-c\",
            \"echo '--- Write test ---' &&
             dd if=/dev/zero of=${MOUNTPOINT}/test_write bs=1M count=512 oflag=direct &&
             echo '--- Read test ---' &&
             dd if=${MOUNTPOINT}/test_write of=/dev/null bs=1M iflag=direct &&
             echo '--- Metadata test ---' &&
             mkdir -p ${MOUNTPOINT}/.meta_test &&
             for i in \$(seq 1 100); do touch ${MOUNTPOINT}/.meta_test/file_\$i; done &&
             ls ${MOUNTPOINT}/.meta_test | wc -l &&
             echo '--- Cleanup ---' &&
             rm -f ${MOUNTPOINT}/test_write &&
             rm -rf ${MOUNTPOINT}/.meta_test &&
             echo 'Lustre I/O test complete.'\"],
          \"volumeMounts\": [{
            \"name\": \"lustre-vol\",
            \"mountPath\": \"${MOUNTPOINT}\"
          }]
        }],
        \"volumes\": [{
          \"name\": \"lustre-vol\",
          \"persistentVolumeClaim\": { \"claimName\": \"xpk-lustre-pvc\" }
        }],
        \"restartPolicy\": \"Never\"
      }
    }"
}

# ==============================================================================
# CLUSTER
# ==============================================================================
cluster-list() {
  xpk cluster list
}
cluster-create() {
  echo "[cluster-create] Creating XPK cluster '${CLUSTER_NAME}'..."
  xpk cluster create \
    --cluster "${CLUSTER_NAME}" \
    --tpu-type="${TPU_TYPE}" \
    --num-slices="${NUM_SLICES}" \
    --reservation="${RESERVATION_NAME}" \
    --zone="${ZONE}" \
    --project="${PROJECT_ID}" \
    --enable-lustre-csi-driver \
    --skip-validation \
    --custom-cluster-arguments="--network=${NETWORK_NAME} --release-channel=None"
}

# TOPOLOGY = 4x4, MACHINE_TYPE = ct6e-standard-4t
cluster-create-manual() {
  local TOPOLOGY="${1:?Usage: deploy.sh cluster-create-manual <TOPOLOGY> <MACHINE_TYPE>}"
  local MACHINE_TYPE="${2:?Usage: deploy.sh cluster-create-manual <TOPOLOGY> <MACHINE_TYPE>}"
  echo "[cluster-create-manual] Creating node pool (topology=${TOPOLOGY}, machine=${MACHINE_TYPE}) and adapting cluster..."

  gcloud beta container node-pools create "${DEVICE}-${TPU_TYPE}" \
    --cluster="${CLUSTER_NAME}" \
    --location="${REGION}" \
    --node-locations="${ZONE}" \
    --machine-type="${MACHINE_TYPE}" \
    --tpu-topology="${TOPOLOGY}" \
    --reservation-affinity=specific \
    --reservation="${RESERVATION_NAME}" \
    --image-type=cos_containerd \
    --project="${PROJECT_ID}" \
    --scopes="storage-full,cloud-platform"

  xpk cluster adapt \
    --cluster="${CLUSTER_NAME}" \
    --tpu-type="${TPU_TYPE}" \
    --num-slices="${NUM_SLICES}" \
    --reservation="${RESERVATION_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --enable-lustre-csi-driver
}

cluster-enable-lustre() {
  echo "[cluster-enable-lustre] Enabling Lustre CSI driver addon on cluster '${CLUSTER_NAME}'..."
  gcloud container clusters update "${CLUSTER_NAME}" \
    --location="${REGION}" \
    --project="${PROJECT_ID}" \
    --update-addons=LustreCsiDriver=ENABLED
}

cluster-delete() {
  local CLUSTER="${1:?Usage: deploy.sh cluster-delete <CLUSTER_NAME>}"
  echo "[cluster-delete] Deleting cluster '${CLUSTER}'..."
  xpk cluster delete --cluster "${CLUSTER}"
}

# ==============================================================================
# WORKLOAD
# ==============================================================================
workload-setup() {
  echo "[workload-setup] Setting up workspace and dependencies..."
  uv venv --python 3.12 --seed && source .venv/bin/activate && pip install xpk
}

workload-build() {
  echo "[workload-build] Building Docker image '${DOCKER_IMAGE}'..."
  exprt MAXTEXT_DIR="src/"
  docker build -t "${DOCKER_IMAGE}" -f "${MAXTEXT_DIR}/dependencies/dockerfiles/maxtext_tpu_dependencies.Dockerfile" "${MAXTEXT_DIR}"
}

workload-push() {
  echo "[workload-push] Creating artifact registry (if needed) and pushing image '${DOCKER_IMAGE}'..."
  # gcloud artifacts repositories create $REPOSITORY \
  #   --repository-format=docker \
  #   --location="${REGION}" \
  #   --description="Docker repository for MaxText TPU training images" \
  #   --project="${PROJECT_ID}" 2>/dev/null || true
  docker push "${DOCKER_IMAGE}"
}

workload-deploy() {
  echo "[workload-deploy] Deploying workload '${NAME}' using script '${SCRIPT}' and image '${DOCKER_IMAGE}'..."
  xpk workload create \
    --cluster "${CLUSTER_NAME}" \
    --workload "${NAME}" \
    --tpu-type="${TPU_TYPE}" \
    --reservation="${RESERVATION_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --docker-image="${DOCKER_IMAGE}" \
    --skip-validation \
    --command="$(cat SCRIPT)"
}

workload-list() {
  echo "[workload-list] Listing workloads for cluster '${CLUSTER_NAME}'..."
  xpk workload list --cluster "${CLUSTER_NAME}"
}

workload-delete() {
  local WORKLOAD="${1:?Usage: deploy.sh workload-delete <WORKLOAD>}"
  echo "[workload-delete] Deleting workload '${WORKLOAD}'..."
  xpk workload delete --cluster "${CLUSTER_NAME}" --workload "${WORKLOAD}"
}

workload-logs() {
  local WORKLOAD="${1:?Usage: deploy.sh workload-logs <WORKLOAD>}"
  echo "[workload-logs] Streaming logs for workload '${WORKLOAD}'..."
  local POD_NAME
  POD_NAME=$(kubectl get pods \
    -l "xpk.google.com/workload=${WORKLOAD},batch.kubernetes.io/job-completion-index=0" \
    -o name | head -n 1)

  if [[ -z "${POD_NAME}" ]]; then
    echo "Error: no pods found for workload '${WORKLOAD}'. Is it running?" >&2
    return 1
  fi

  echo "[workload-logs] Attaching to pod ${POD_NAME} container jax-tpu..."
  kubectl logs "${POD_NAME}" -c jax-tpu -f
}

workload-status() {
  local WORKLOAD="${1:?Usage: deploy.sh workload-status <WORKLOAD>}"
  echo "[workload-status] Pod status for workload '${WORKLOAD}'..."
  kubectl get pods \
    -n default \
    -l "xpk.google.com/workload=${WORKLOAD}" \
    -o wide
  echo ""
  echo "[workload-status] JobSet conditions..."
  kubectl get jobset "${WORKLOAD}" -n default -o jsonpath='{.status.conditions}' | python3 -m json.tool 2>/dev/null || \
    kubectl get jobset "${WORKLOAD}" -n default
}

workload-events() {
  local WORKLOAD="${1:?Usage: deploy.sh workload-events <WORKLOAD>}"
  echo "[workload-events] Recent events for workload '${WORKLOAD}'..."
  kubectl get events \
    -n default \
    --field-selector "involvedObject.name=${WORKLOAD}" \
    --sort-by='.lastTimestamp'
}

# ==============================================================================
# AUTO PIPELINES
# Composite commands that chain individual steps end-to-end.
# Usage: ./deploy.sh auto <pipeline>
# ==============================================================================

# auto vpc: vpc-ip → vpc-peering → vpc-firewall
auto-vpc() {
  echo ""
  echo "======================================================================"
  echo "  AUTO: vpc pipeline"
  echo "======================================================================"
  echo ""
  echo "--- Step 1/3: vpc-ip ---"
  vpc-ip
  echo ""
  echo "--- Step 2/3: vpc-peering ---"
  vpc-peering
  echo ""
  echo "--- Step 3/3: vpc-firewall ---"
  vpc-firewall
  echo ""
  echo "======================================================================"
  echo "  AUTO: vpc pipeline complete"
  echo "======================================================================"
}

# auto lustre: lustre-create → lustre-manifest-generate → lustre-automount
# Note: lustre-create runs with --async; the manifest-generate step will wait
# for the instance to be ready before fetching the IP.
auto-lustre() {
  echo ""
  echo "======================================================================"
  echo "  AUTO: lustre pipeline"
  echo "======================================================================"
  echo ""
  echo "--- Step 1/3: lustre-create ---"
  lustre-create
  echo ""
  echo "--- Step 2/3: lustre-manifest-generate ---"
  echo "  Waiting for Lustre instance to become READY..."
  until gcloud lustre instances describe "${STORAGE_NAME}" \
      --location="${ZONE}" --project="${PROJECT_ID}" \
      --format="value(state)" 2>/dev/null | grep -q "^ACTIVE$"; do
    echo "  ... not ready yet, retrying in 30s"
    sleep 30
  done
  lustre-manifest-generate
  echo ""
  echo "--- Step 3/3: lustre-automount ---"
  lustre-automount
  echo ""
  echo "======================================================================"
  echo "  AUTO: lustre pipeline complete"
  echo "======================================================================"
}

# auto cluster: cluster-create → lustre-automount
auto-cluster() {
  echo ""
  echo "======================================================================"
  echo "  AUTO: cluster pipeline"
  echo "======================================================================"
  echo ""
  echo "--- Step 1/2: cluster-create ---"
  cluster-create
  echo ""
  echo "--- Step 2/2: lustre-automount ---"
  lustre-automount
  echo ""
  echo "======================================================================"
  echo "  AUTO: cluster pipeline complete"
  echo "======================================================================"
}

# auto workload: workload-setup → workload-build → workload-push → workload-deploy → workload-status
# Workload name is derived from the script filename.
auto-workload() {
  local SCRIPT="${1:?Usage: deploy.sh auto workload <SCRIPT_PATH>}"

  local NAME
  NAME="$(basename "${SCRIPT}" | sed 's/\.[^.]*$//')"

  echo ""
  echo "======================================================================"
  echo "  AUTO: workload pipeline  (name=${NAME}, script=${SCRIPT})"
  echo "======================================================================"
  echo ""
  echo "--- Step 1/5: workload-setup ---"
  workload-setup
  echo ""
  echo "--- Step 2/5: workload-build ---"
  workload-build "${SCRIPT}"
  echo ""
  echo "--- Step 3/5: workload-push ---"
  workload-push "${SCRIPT}"
  echo ""
  echo "--- Step 4/5: workload-deploy ---"
  workload-deploy "${SCRIPT}"
  echo ""
  echo "--- Step 5/5: workload-status ---"
  workload-status "${NAME}"
  echo ""
  echo "======================================================================"
  echo "  AUTO: workload pipeline complete"
  echo "======================================================================"
}

# ==============================================================================
# HELP / USAGE
# ==============================================================================
usage() {
  echo ""
  echo "Usage: ./deploy.sh <command> [args]"
  echo ""
  echo "  Auth:"
  echo "    login                                        Authenticate with gcloud"
  echo "    setup                                        Set project and zone in gcloud config"
  echo ""
  echo "  VPC:"
  echo "    vpc-ip                                       Create VPC peering IP range"
  echo "    vpc-peering                                  Connect VPC peering"
  echo "    vpc-firewall                                 Create Lustre firewall rules"
  echo ""
  echo "  Lustre:"
  echo "    lustre-create                                Create Lustre storage instance"
  echo "    lustre-list                                  Describe Lustre instance (shows mountPoint)"
  echo "    lustre-automount                             Attach Lustre storage to cluster"
  echo "    lustre-manifest-generate                     Regenerate lustre-manifest.yaml from env vars"
  echo "                                                 (fetches LUSTRE_IP from gcloud automatically)"
  echo "    lustre-status                                Show state and details of Lustre instance"
  echo "    lustre-test                                  Run write/read/metadata I/O test pod on mountpoint"
  echo ""
  echo "  Cluster:"
  echo "    cluster-list                                 List XPK Clusters"
  echo "    cluster-create                               Create XPK cluster"
  echo "    cluster-create-manual <TOPOLOGY> <MACHINE>   Manually create node pool + adapt cluster"
  echo "    cluster-enable-lustre                        Enable Lustre CSI driver on existing cluster"
  echo "    cluster-delete        <CLUSTER>              Delete a cluster"
  echo ""
  echo "  Workload:"
  echo "    workload-setup                               Clone repo and install deps"
  echo "    workload-build  [SCRIPT]                     Build Docker image (name derived from script, or default)"
  echo "    workload-push   [SCRIPT]                     Push image (name derived from script, or default)"
  echo "    workload-deploy  <SCRIPT>                    Deploy a cluster workload"
  echo "                                                 Workload name derived from script filename"
  echo "    workload-list                                List all workloads in cluster"
  echo "    workload-delete       <WORKLOAD>             Delete a workload"
  echo "    workload-logs         <WORKLOAD>             Stream live logs (head pod, jax-tpu container)"
  echo "    workload-status       <WORKLOAD>             Show pod states and JobSet conditions"
  echo "    workload-events       <WORKLOAD>             Show recent Kubernetes events for a workload"
  echo ""
  echo "  Auto pipelines:"
  echo "    auto vpc                                     vpc-ip → vpc-peering → vpc-firewall"
  echo "    auto lustre                                  lustre-create → manifest-generate → automount"
  echo "    auto cluster                                 cluster-create → lustre-automount"
  echo "    auto workload <SCRIPT>                    setup → build → push → deploy → status"
  echo ""
}

# ==============================================================================
# MAIN DISPATCHER
# ==============================================================================
COMMAND="${1:-help}"

# Always echo vars for any real command (not help)
if [[ "$COMMAND" != "help" && "$COMMAND" != "--help" && "$COMMAND" != "-h" ]]; then
  echo_vars
fi

case "$COMMAND" in
  login)                    login ;;
  setup)                    setup ;;
  vpc-ip)                   vpc-ip ;;
  vpc-peering)              vpc-peering ;;
  vpc-firewall)             vpc-firewall ;;
  lustre-create)            lustre-create ;;
  lustre-list)              lustre-list ;;
  lustre-automount)         lustre-automount ;;
  lustre-manifest-generate) lustre-manifest-generate ;;
  lustre-status)            lustre-status ;;
  lustre-test)              lustre-test ;;
  cluster-list)             cluster-list ;;
  cluster-create)           cluster-create ;;
  cluster-create-manual)    cluster-create-manual "${2:-}" "${3:-}" ;;
  cluster-enable-lustre)    cluster-enable-lustre ;;
  cluster-delete)           cluster-delete "${2:-}" ;;
  workload-setup)           workload-setup ;;
  workload-build)           workload-build "${2:-}" ;;
  workload-push)            workload-push "${2:-}" ;;
  workload-deploy)          workload-deploy "${2:-}" ;;
  workload-list)            workload-list ;;
  workload-delete)          workload-delete "${2:-}" ;;
  workload-logs)            workload-logs "${2:-}" ;;
  workload-status)          workload-status "${2:-}" ;;
  workload-events)          workload-events "${2:-}" ;;
  auto)
    case "${2:-}" in
      vpc)      auto-vpc ;;
      lustre)   auto-lustre ;;
      cluster)  auto-cluster ;;
      workload) auto-workload "${3:-}" ;;
      *)
        echo "Error: unknown auto pipeline '${2:-}'. Available: vpc, lustre, cluster, workload"
        usage
        exit 1
        ;;
    esac
    ;;
  help|--help|-h)           usage ;;
  *)
    echo "Error: Unknown command '${COMMAND}'"
    usage
    exit 1
    ;;
esac
