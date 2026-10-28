#!/usr/bin/env bash
# ==============================================================================
# Run Single-Slice & Multi-Slice Benchmark on GKE TPU Cluster (bodaborg-v5p-nap)
#
# Supports:
#   - Single-Slice (`--num-slices-list 1`): Deploys 1 GKE TPU v5p-8 node (4 chips, ~270 GB RAM)
#   - Multi-Slice (`--num-slices-list 2` or `1 2`): Deploys 2 GKE TPU v5p-8 nodes:
#       * Node 1 (Slice 1): Runs `benchmark_ckpt_load_time.py --remote-slice-daemon --daemon-port 55555`
#       * Node 0 (Slice 0): Runs `benchmark_ckpt_load_time.py` with `REMOTE_SLICE1_ADDR=<slice1_ip>:55555`
#   - Automatically cleans up all Kubernetes Jobs, Services, ConfigMaps, and Secrets on exit.
#
# Usage Examples:
#   # Run Tier 1 & Tier 2 (mixtral-8x7b, llama3.1-70b, gpt-oss-120b) across 1 & 2 slices:
#   ./checkpoint_load_benchmark/scripts/run_multi_slice_cluster.sh \
#     --models mixtral-8x7b llama3.1-70b gpt-oss-120b \
#     --num-slices-list 1 2
#
#   # Run Tier 3 DeepSeek OOM test suite (284B - 1600B) across 1 & 2 slices:
#   ./checkpoint_load_benchmark/scripts/run_multi_slice_cluster.sh \
#     --models deepseek4-284b deepseek3-671b deepseek3.2-671b deepseek4-1600b \
#     --num-slices-list 1 2 \
#     --stage-sparse-oom
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

MODELS=("${TIER1_MODELS[@]}" "${TIER2_MODELS[@]}")
SCAN_MODES=("${DEFAULT_SCAN_MODES[@]}")
NUM_SLICES_LIST=(1 2)
DCN_AXIS="${DEFAULT_DCN_AXIS}"
HARDWARE="cpu"
WEIGHT_DTYPE="${DEFAULT_WEIGHT_DTYPE}"
GCS_BASE="${DEFAULT_GCS_BASE}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
CLUSTER_NAME="${DEFAULT_CLUSTER_NAME}"
CLUSTER_ZONE="${DEFAULT_CLUSTER_ZONE}"
CLUSTER_PROJECT="${DEFAULT_CLUSTER_PROJECT}"
CONTAINER_IMAGE="${DEFAULT_CONTAINER_IMAGE}"
GIT_BRANCH="${GIT_BRANCH:-docs/checkpoint-load-benchmark}"
JOB_NAME="bench-ckpt-$(date +%H%M%S)"
STAGE_SPARSE_DEEPSEEK_OOM="0"
KEEP_RESOURCES="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)
      shift
      MODELS=()
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        MODELS+=("$1")
        shift
      done
      ;;
    --scan-modes)
      shift
      SCAN_MODES=()
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        SCAN_MODES+=("$1")
        shift
      done
      ;;
    --num-slices-list)
      shift
      NUM_SLICES_LIST=()
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        NUM_SLICES_LIST+=("$1")
        shift
      done
      ;;
    --dcn-parallelism-axis)
      DCN_AXIS="$2"
      shift 2
      ;;
    --hardware)
      HARDWARE="$2"
      shift 2
      ;;
    --weight-dtype)
      WEIGHT_DTYPE="$2"
      shift 2
      ;;
    --gcs-base)
      GCS_BASE="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --cluster-name)
      CLUSTER_NAME="$2"
      shift 2
      ;;
    --cluster-zone)
      CLUSTER_ZONE="$2"
      shift 2
      ;;
    --cluster-project)
      CLUSTER_PROJECT="$2"
      shift 2
      ;;
    --job-name)
      JOB_NAME="$2"
      shift 2
      ;;
    --stage-sparse-oom)
      STAGE_SPARSE_DEEPSEEK_OOM="1"
      shift
      ;;
    --keep-cluster-resources)
      KEEP_RESOURCES="true"
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

NEEDS_MULTISLICE="false"
for s in "${NUM_SLICES_LIST[@]}"; do
  if [[ "${s}" -gt 1 ]]; then
    NEEDS_MULTISLICE="true"
  fi
done

print_banner "GKE TPU Cluster Benchmark Runner (${CLUSTER_NAME})"
echo " Cluster           : ${CLUSTER_NAME} (${CLUSTER_ZONE}, project=${CLUSTER_PROJECT})"
echo " Job Name          : ${JOB_NAME}"
echo " Models            : ${MODELS[*]}"
echo " Scan Modes        : ${SCAN_MODES[*]}"
echo " Num Slices List   : ${NUM_SLICES_LIST[*]} (multi-slice=${NEEDS_MULTISLICE})"
echo " DCN Axis          : ${DCN_AXIS}"
echo " Hardware          : ${HARDWARE}"
echo " Sparse OOM Mode   : ${STAGE_SPARSE_DEEPSEEK_OOM}"
echo " GCS Base          : ${GCS_BASE}"
echo " Run ID            : ${RUN_ID}"
echo "================================================================================"

cleanup_cluster() {
  if [[ "${KEEP_RESOURCES}" == "true" ]]; then
    echo "[Cluster] Keeping cluster resources (--keep-cluster-resources enabled)."
    return
  fi
  echo "[Cluster] Cleaning up Kubernetes resources for ${JOB_NAME}..."
  kubectl delete job "${JOB_NAME}" "${JOB_NAME}-slice1" --ignore-not-found=true >/dev/null 2>&1 || true
  kubectl delete service "${JOB_NAME}-svc" --ignore-not-found=true >/dev/null 2>&1 || true
  kubectl delete configmap "${JOB_NAME}-script" --ignore-not-found=true >/dev/null 2>&1 || true
}
trap cleanup_cluster EXIT

# 1. Configure kubectl credentials
gcloud container clusters get-credentials "${CLUSTER_NAME}" \
  --zone "${CLUSTER_ZONE}" \
  --project "${CLUSTER_PROJECT}"

# 2. Ensure HF token secret and benchmark script ConfigMap exist
if [[ -n "${HF_TOKEN:-}" ]]; then
  kubectl create secret generic hf-token-secret \
    --from-literal=token="${HF_TOKEN}" \
    --dry-run=client -o yaml | kubectl apply -f - >/dev/null
fi

kubectl create configmap "${JOB_NAME}-script" \
  --from-file=benchmark_ckpt_load_time.py="${BENCHMARK_SCRIPT}" \
  --dry-run=client -o yaml | kubectl apply -f - >/dev/null

SLICE1_IP=""
if [[ "${NEEDS_MULTISLICE}" == "true" ]]; then
  echo "[Cluster] Launching Slice 1 companion daemon job (${JOB_NAME}-slice1)..."
  cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}-slice1
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 3600
  template:
    metadata:
      labels:
        bench-group: ${JOB_NAME}
        bench-role: slice1
    spec:
      restartPolicy: Never
      nodeSelector:
        cloud.google.com/gke-tpu-accelerator: tpu-v5p-slice
        cloud.google.com/gke-tpu-topology: 2x2x1
      volumes:
        - name: dshm
          emptyDir:
            medium: Memory
            sizeLimit: 180Gi
        - name: bench-script
          configMap:
            name: ${JOB_NAME}-script
      containers:
        - name: slice1-daemon
          image: ${CONTAINER_IMAGE}
          env:
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-token-secret
                  key: token
            - name: STAGE_SPARSE_DEEPSEEK_OOM
              value: "${STAGE_SPARSE_DEEPSEEK_OOM}"
          volumeMounts:
            - name: dshm
              mountPath: /mnt/shm
            - name: bench-script
              mountPath: /opt/bench
          resources:
            requests:
              google.com/tpu: 4
            limits:
              google.com/tpu: 4
          command:
            - bash
            - -c
            - |
              set -euo pipefail
              mkdir -p /mnt/shm/workspace /mnt/shm/tmp
              cd /mnt/shm/workspace
              git clone --depth 1 --branch "${GIT_BRANCH}" https://github.com/AI-Hypercomputer/maxtext.git .
              cp /opt/bench/benchmark_ckpt_load_time.py /mnt/shm/workspace/benchmark_ckpt_load_time.py
              pip install --no-cache-dir --upgrade "orbax-checkpoint>=0.12.7"
              export PYTHONPATH="/mnt/shm/workspace/src:\${PYTHONPATH:-}"
              python3 /mnt/shm/workspace/benchmark_ckpt_load_time.py --remote-slice-daemon --daemon-port 55555
EOF

  echo "[Cluster] Waiting for Slice 1 pod to be Running and listening on port 55555..."
  for _ in $(seq 1 120); do
    SLICE1_IP="$(kubectl get pods -l "job-name=${JOB_NAME}-slice1" -o jsonpath='{.items[0].status.podIP}' 2>/dev/null || true)"
    SLICE1_PHASE="$(kubectl get pods -l "job-name=${JOB_NAME}-slice1" -o jsonpath='{.items[0].status.phase}' 2>/dev/null || true)"
    if [[ "${SLICE1_PHASE}" == "Running" && -n "${SLICE1_IP}" ]]; then
      if kubectl logs "job/${JOB_NAME}-slice1" 2>/dev/null | grep -q "Listening on 0.0.0.0:55555"; then
        echo "[Cluster] Slice 1 daemon ready at ${SLICE1_IP}:55555"
        break
      fi
    fi
    sleep 5
  done
  if [[ -z "${SLICE1_IP}" ]]; then
    echo "ERROR: Timed out waiting for Slice 1 daemon pod." >&2
    exit 1
  fi
fi

MODELS_STR="${MODELS[*]}"
SCAN_MODES_STR="${SCAN_MODES[*]}"
NUM_SLICES_STR="${NUM_SLICES_LIST[*]}"
REMOTE_ADDR_ENV=""
if [[ -n "${SLICE1_IP}" ]]; then
  REMOTE_ADDR_ENV="${SLICE1_IP}:55555"
fi

echo "[Cluster] Launching Coordinator benchmark job (${JOB_NAME})..."
cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 3600
  template:
    metadata:
      labels:
        bench-group: ${JOB_NAME}
        bench-role: coordinator
    spec:
      restartPolicy: Never
      nodeSelector:
        cloud.google.com/gke-tpu-accelerator: tpu-v5p-slice
        cloud.google.com/gke-tpu-topology: 2x2x1
      volumes:
        - name: dshm
          emptyDir:
            medium: Memory
            sizeLimit: 180Gi
        - name: bench-script
          configMap:
            name: ${JOB_NAME}-script
      containers:
        - name: coordinator
          image: ${CONTAINER_IMAGE}
          env:
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-token-secret
                  key: token
            - name: REMOTE_SLICE1_ADDR
              value: "${REMOTE_ADDR_ENV}"
            - name: STAGE_SPARSE_DEEPSEEK_OOM
              value: "${STAGE_SPARSE_DEEPSEEK_OOM}"
          volumeMounts:
            - name: dshm
              mountPath: /mnt/shm
            - name: bench-script
              mountPath: /opt/bench
          resources:
            requests:
              google.com/tpu: 4
            limits:
              google.com/tpu: 4
          command:
            - bash
            - -c
            - |
              set -euo pipefail
              mkdir -p /mnt/shm/workspace /mnt/shm/tmp
              cd /mnt/shm/workspace
              git clone --depth 1 --branch "${GIT_BRANCH}" https://github.com/AI-Hypercomputer/maxtext.git .
              cp /opt/bench/benchmark_ckpt_load_time.py /mnt/shm/workspace/benchmark_ckpt_load_time.py
              pip install --no-cache-dir --upgrade "orbax-checkpoint>=0.12.7"
              export PYTHONPATH="/mnt/shm/workspace/src:\${PYTHONPATH:-}"
              python3 /mnt/shm/workspace/benchmark_ckpt_load_time.py \
                --models ${MODELS_STR} \
                --scan-modes ${SCAN_MODES_STR} \
                --num-slices-list ${NUM_SLICES_STR} \
                --dcn-parallelism-axis "${DCN_AXIS}" \
                --hardware "${HARDWARE}" \
                --weight-dtype "${WEIGHT_DTYPE}" \
                --gcs-base "${GCS_BASE}" \
                --run-id "${RUN_ID}" \
                --local-work-dir "/mnt/shm/tmp/benchmark_${RUN_ID}" \
                --csv-report "/mnt/shm/tmp/benchmark_${RUN_ID}/summary.csv" \
                --reuse-model-hf-cache \
                --skip-completed \
                --allow-step-failure
EOF

echo "[Cluster] Waiting for Coordinator pod (${JOB_NAME}) to start..."
for _ in $(seq 1 120); do
  COORD_PHASE="$(kubectl get pods -l "job-name=${JOB_NAME}" -o jsonpath='{.items[0].status.phase}' 2>/dev/null || true)"
  if [[ "${COORD_PHASE}" == "Running" || "${COORD_PHASE}" == "Succeeded" || "${COORD_PHASE}" == "Failed" ]]; then
    break
  fi
  sleep 5
done

echo "[Cluster] Streaming Coordinator logs for ${JOB_NAME}..."
kubectl logs -f "job/${JOB_NAME}" || true

echo "[Cluster] Syncing results from ${GCS_BASE}/run_${RUN_ID}/logs/ to /tmp/benchmark_ckpt_load_time_${RUN_ID}/logs/ ..."
mkdir -p "/tmp/benchmark_ckpt_load_time_${RUN_ID}/logs"
gcloud storage cp -r "${GCS_BASE}/run_${RUN_ID}/logs/*" "/tmp/benchmark_ckpt_load_time_${RUN_ID}/logs/" || true
echo "[Cluster] Done!"
