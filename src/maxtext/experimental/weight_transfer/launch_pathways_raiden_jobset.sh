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

set -eo pipefail

JOBSET_NAME="${1:-r2s-$(date +%H%M%S)}"
CLUSTER="${2:-auto-v5p-8-bodaborg}"
REGION="${3:-europe-west4}"
PROJECT="${4:-cloud-tpu-multipod-dev}"
DOCKER_TAG="${5:-gcr.io/cloud-tpu-multipod-dev/mohitkhatwani-rl:raiden-bench-latest}"
SKIP_BUILD="${SKIP_BUILD:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAXTEXT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "=========================================================================================="
echo "[TPU Raiden Pathways 2-Slice Workflow Launcher]"
echo "  JobSet Name : ${JOBSET_NAME}"
echo "  Cluster     : ${CLUSTER} (${REGION})"
echo "  Project     : ${PROJECT}"
echo "  Docker Image: ${DOCKER_TAG}"
echo "=========================================================================================="

# 1. Build and Push Benchmark Docker Image (unless SKIP_BUILD=true)
if [ "${SKIP_BUILD}" != "true" ]; then
    echo ""
    echo "1. Building TPU Raiden Pathways Benchmark Docker Image..."
    docker build \
        -f "${SCRIPT_DIR}/Dockerfile.raiden_bench" \
        -t "${DOCKER_TAG}" \
        "${MAXTEXT_ROOT}"

    echo ""
    echo "2. Pushing Docker Image to Google Container Registry..."
    docker push "${DOCKER_TAG}"
else
    echo "Skipping Docker build and push (SKIP_BUILD=true)."
fi

# 2. Activate python virtual environment if present
if [ -f "/usr/local/google/home/mohitkhatwani/max_venv/bin/activate" ]; then
    source /usr/local/google/home/mohitkhatwani/max_venv/bin/activate
fi

export PYTHONPATH="${MAXTEXT_ROOT}:${MAXTEXT_ROOT}/src:${PYTHONPATH}"

# 3. Deploy JobSet and Monitor Pods
echo ""
echo "3. Launching JobSet on GKE Cluster..."
python3 "${SCRIPT_DIR}/launch_pathways_raiden_jobset.py" \
    --jobset_name="${JOBSET_NAME}" \
    --cluster="${CLUSTER}" \
    --region="${REGION}" \
    --project="${PROJECT}"
