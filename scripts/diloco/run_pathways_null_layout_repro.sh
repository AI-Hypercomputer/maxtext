#!/bin/bash
set -e

# Launches tests/unit/pathways_null_layout_repro_test.py as a single-slice, single-host
# Pathways workload, to reproduce the device_put null-layout crash for real (not mocked).
#
# Usage: edit CLUSTER/PROJECT/ZONE/DEVICE_TYPE below to match a cluster from
# ~/.claude/CLAUDE.md, then run this script. It builds+pushes your local working tree as
# a docker image and submits it via `xpk workload create-pathways`.

# cluster -- pick a single-host TPU type on a cluster you have access to
CLUSTER=bodaborg-super-xpk-v54
PROJECT=cloud-tpu-multipod-dev
ZONE=us-central1
DEVICE_TYPE=v5p-8 # single host, 4 chips -- enough for the diloco/model=2x1 mesh in the test

RUNNAME="pw-null-layout-repro-$(date +%d%H%M)"
DOCKER_IMAGE_BASE="gcr.io/tpu-prod-env-multipod/maxtext_jax_stable:latest"
MY_IMAGE="gcr.io/${PROJECT}/pw-null-layout-repro:${RUNNAME}"

CMD="export PYTHONPATH=/app/src:\$PYTHONPATH && export RUN_PATHWAYS_REPRO=1 && cd /app/src/ && \
python3 -m pytest tests/unit/pathways_null_layout_repro_test.py -v -s"

echo "Building docker image containing local changes..."
docker build -t "${MY_IMAGE}" -f - . <<EOF
FROM ${DOCKER_IMAGE_BASE}
WORKDIR /app
COPY . .
RUN find /app -name "*.pyc" -delete && find /app -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
EOF

echo "Pushing image ${MY_IMAGE}..."
docker push "${MY_IMAGE}"

echo "Creating workload: ${RUNNAME}"
xpk workload create-pathways --workload "${RUNNAME}" \
  --docker-image "${MY_IMAGE}" \
  --command "${CMD}" \
  --num-slices=1 \
  --cluster "${CLUSTER}" --tpu-type "${DEVICE_TYPE}" --project "${PROJECT}" --zone "${ZONE}"

echo "Tail logs with: xpk workload list --cluster ${CLUSTER} --project ${PROJECT} --zone ${ZONE}"
echo "or: kubectl logs -f -l jobset.sigs.k8s.io/jobset-name=${RUNNAME}"
