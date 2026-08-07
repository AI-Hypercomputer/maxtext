#!/bin/bash
set -e

# Setup Python VENV paths
export PATH="/usr/local/google/home/jzuo/maxtext/maxtext_venv/bin:/usr/local/google/home/jzuo/xpk_venv/bin:$PATH"

CLUSTER="${CLUSTER:-mlperf-v5p}"
PROJECT="${PROJECT:-cloud-tpu-multipod-dev}"
ZONE="${ZONE:-europe-west4}"

NUM_SLICES="${NUM_SLICES:-2}"
DEVICE_TYPE="${DEVICE_TYPE:-v5p-8}"

WORKLOAD_NAME="${WORKLOAD_NAME:-repro-stack-$(date +%m%d%H%M)}"
DOCKER_IMAGE_BASE="${DOCKER_IMAGE_BASE:-gcr.io/tpu-prod-env-multipod/maxtext_jax_stable:2026-07-17}"
MY_IMAGE="gcr.io/${PROJECT}/$(whoami)-runner:${WORKLOAD_NAME}"

CMD="export PYTHONPATH=/app/src:/app:\$PYTHONPATH && cd /app && (python3 tests/unit/stack_across_meshes_repro_test.py || python3 MyStuff/stack_across_meshes_repro_test.py)"

echo "Building docker image: ${MY_IMAGE}"
docker build -t "${MY_IMAGE}" -f - . << 'EOF'
FROM gcr.io/tpu-prod-env-multipod/maxtext_jax_stable:2026-07-17
WORKDIR /app
COPY . .
RUN find /app -name "*.pyc" -delete && find /app -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
EOF

echo "Pushing docker image: ${MY_IMAGE}"
docker push "${MY_IMAGE}"

echo "Submitting reproduction test workload to XPK via Pathways..."
xpk workload create-pathways --workload "${WORKLOAD_NAME}" \
  --docker-image "${MY_IMAGE}" \
  --command "${CMD}" \
  --num-slices="${NUM_SLICES}" \
  --enable-debug-logs \
  --cluster "${CLUSTER}" --tpu-type "${DEVICE_TYPE}" --project "${PROJECT}" --zone "${ZONE}"

echo "Workload submission complete! Workload name: ${WORKLOAD_NAME}"
