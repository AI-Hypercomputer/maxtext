#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Script to launch multi-slice weight transfer benchmark on GKE cluster via XPK.

set -e

export PATH="/usr/local/google/home/mohitkhatwani/max_venv/bin:$PATH"

# Configurable environment variables with sensible defaults
CLUSTER_NAME="${CLUSTER_NAME:-my-gke-cluster}"
DOCKER_IMAGE="${DOCKER_IMAGE:-}"
BASE_DOCKER_IMAGE="${BASE_DOCKER_IMAGE:-}"
DEVICE_TYPE="${DEVICE_TYPE:-v5p-8}"
NUM_SLICES="${NUM_SLICES:-2}"
WORKLOAD_NAME="${WORKLOAD_NAME:-weight-transfer-$(date +%Y%m%d-%H%M%S)}"

WEIGHT_SIZE_MB="${WEIGHT_SIZE_MB:-1024}"
NUM_LAYERS="${NUM_LAYERS:-12}"
ITERATIONS="${ITERATIONS:-10}"
PROFILE_DIR="${PROFILE_DIR:-}"
USE_BASE_DOCKER_IMAGE="${USE_BASE_DOCKER_IMAGE:-}"
SCRIPT_DIR="${SCRIPT_DIR:-/usr/local/google/home/mohitkhatwani/maxtext}"

DEFAULT_IMAGE="gcr.io/cloud-tpu-multipod-dev/mohitkhatwani-rl:raiden"

# Optional cluster parameters
PROJECT_ID="${PROJECT_ID:-}"
ZONE="${ZONE:-}"
RESERVATION="${RESERVATION:-}"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --docker-image=*)
      DOCKER_IMAGE="${1#*=}"
      USE_BASE_DOCKER_IMAGE="false"
      shift
      ;;
    --docker-image)
      DOCKER_IMAGE="$2"
      USE_BASE_DOCKER_IMAGE="false"
      shift 2
      ;;
    --base-docker-image=*)
      BASE_DOCKER_IMAGE="${1#*=}"
      USE_BASE_DOCKER_IMAGE="true"
      shift
      ;;
    --base-docker-image)
      BASE_DOCKER_IMAGE="$2"
      USE_BASE_DOCKER_IMAGE="true"
      shift 2
      ;;
    --script-dir=*)
      SCRIPT_DIR="${1#*=}"
      shift
      ;;
    --script-dir)
      SCRIPT_DIR="$2"
      shift 2
      ;;
    --cluster-name=*|--cluster=*)
      CLUSTER_NAME="${1#*=}"
      shift
      ;;
    --cluster-name|--cluster)
      CLUSTER_NAME="$2"
      shift 2
      ;;
    --device-type=*)
      DEVICE_TYPE="${1#*=}"
      shift
      ;;
    --device-type)
      DEVICE_TYPE="$2"
      shift 2
      ;;
    --num-slices=*)
      NUM_SLICES="${1#*=}"
      shift
      ;;
    --num-slices)
      NUM_SLICES="$2"
      shift 2
      ;;
    --workload-name=*|--workload=*)
      WORKLOAD_NAME="${1#*=}"
      shift
      ;;
    --workload-name|--workload)
      WORKLOAD_NAME="$2"
      shift 2
      ;;
    --weight-size-mb=*)
      WEIGHT_SIZE_MB="${1#*=}"
      shift
      ;;
    --weight-size-mb)
      WEIGHT_SIZE_MB="$2"
      shift 2
      ;;
    --num-layers=*)
      NUM_LAYERS="${1#*=}"
      shift
      ;;
    --num-layers)
      NUM_LAYERS="$2"
      shift 2
      ;;
    --iterations=*)
      ITERATIONS="${1#*=}"
      shift
      ;;
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --profile-dir=*)
      PROFILE_DIR="${1#*=}"
      shift
      ;;
    --profile-dir)
      PROFILE_DIR="$2"
      shift 2
      ;;
    --project-id=*|--project=*)
      PROJECT_ID="${1#*=}"
      shift
      ;;
    --project-id|--project)
      PROJECT_ID="$2"
      shift 2
      ;;
    --zone=*)
      ZONE="${1#*=}"
      shift
      ;;
    --zone)
      ZONE="$2"
      shift 2
      ;;
    --reservation=*)
      RESERVATION="${1#*=}"
      shift
      ;;
    --reservation)
      RESERVATION="$2"
      shift 2
      ;;
    -h|--help)
      cat << 'EOF'
Usage: run_weight_transfer_xpk.sh [options]

Options:
  --docker-image IMAGE      Use pre-built Docker image for XPK (--docker-image)
  --base-docker-image IMAGE Use base Docker image for XPK with script directory upload (--base-docker-image)
  --script-dir DIR          Script directory to upload with base docker image (default: /usr/local/google/home/mohitkhatwani/maxtext)
  --cluster NAME            XPK cluster name (default: my-gke-cluster)
  --device-type TYPE        Device type (default: v5p-8)
  --num-slices SLICES       Number of slices (default: 2)
  --workload NAME           Workload name
  --weight-size-mb MB       Weight size in MB (default: 1024)
  --num-layers LAYERS       Number of layers (default: 12)
  --iterations ITERS        Number of iterations (default: 10)
  --profile-dir DIR         Profile directory (optional)
  --project PROJECT         GCP project ID
  --zone ZONE               GCP zone
  --reservation RESERVATION GCP reservation
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

# Resolve docker image mode and values
if [ -n "${BASE_DOCKER_IMAGE}" ]; then
  USE_BASE_DOCKER_IMAGE="true"
  IMAGE_VAL="${BASE_DOCKER_IMAGE}"
elif [ -n "${DOCKER_IMAGE}" ]; then
  if [ "${USE_BASE_DOCKER_IMAGE}" = "true" ]; then
    IMAGE_VAL="${DOCKER_IMAGE}"
  else
    USE_BASE_DOCKER_IMAGE="false"
    IMAGE_VAL="${DOCKER_IMAGE}"
  fi
else
  # Neither BASE_DOCKER_IMAGE nor DOCKER_IMAGE specified
  if [ "${USE_BASE_DOCKER_IMAGE}" = "false" ]; then
    IMAGE_VAL="${DEFAULT_IMAGE}"
  else
    USE_BASE_DOCKER_IMAGE="true"
    IMAGE_VAL="${DEFAULT_IMAGE}"
  fi
fi

# Configure docker image flags for XPK
DOCKER_ARGS=()
if [ "${USE_BASE_DOCKER_IMAGE}" = "true" ]; then
  DOCKER_ARGS=(--base-docker-image="${IMAGE_VAL}" --script-dir="${SCRIPT_DIR}")
else
  DOCKER_ARGS=(--docker-image="${IMAGE_VAL}")
fi

# Optional cluster parameters
OPTIONAL_ARGS=()
if [ -n "${PROJECT_ID}" ]; then
  OPTIONAL_ARGS+=(--project="${PROJECT_ID}")
fi
if [ -n "${ZONE}" ]; then
  OPTIONAL_ARGS+=(--zone="${ZONE}")
fi
if [ -n "${RESERVATION}" ]; then
  OPTIONAL_ARGS+=(--reservation="${RESERVATION}")
fi

# Construct command to run inside container
COMMAND="export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python && PYTHONPATH=/app/src:/app:\$PYTHONPATH python3 /app/src/maxtext/experimental/weight_transfer/transfer_weights_raiden.py --weight_size_mb=${WEIGHT_SIZE_MB} --num_layers=${NUM_LAYERS} --iterations=${ITERATIONS}"

if [ -n "${PROFILE_DIR}" ]; then
  COMMAND="${COMMAND} --profile_dir=${PROFILE_DIR}"
fi

echo "========================================================="
echo "Launching Weight Transfer Workload via XPK"
echo "========================================================="
echo "Workload Name : ${WORKLOAD_NAME}"
echo "Cluster Name  : ${CLUSTER_NAME}"
if [ "${USE_BASE_DOCKER_IMAGE}" = "true" ]; then
  echo "Base Image    : ${IMAGE_VAL}"
  echo "Script Dir    : ${SCRIPT_DIR}"
else
  echo "Docker Image  : ${IMAGE_VAL}"
fi
echo "Device Type   : ${DEVICE_TYPE}"
echo "Num Slices    : ${NUM_SLICES}"
echo "Weight Size   : ${WEIGHT_SIZE_MB} MB"
echo "Num Layers    : ${NUM_LAYERS}"
echo "Iterations    : ${ITERATIONS}"
if [ -n "${PROFILE_DIR}" ]; then
  echo "Profile Dir   : ${PROFILE_DIR}"
fi
echo "Command       : ${COMMAND}"
echo "========================================================="

xpk workload create \
  --cluster="${CLUSTER_NAME}" \
  --device-type="${DEVICE_TYPE}" \
  --num-slices="${NUM_SLICES}" \
  "${DOCKER_ARGS[@]}" \
  --workload="${WORKLOAD_NAME}" \
  "${OPTIONAL_ARGS[@]}" \
  --command="${COMMAND}"
