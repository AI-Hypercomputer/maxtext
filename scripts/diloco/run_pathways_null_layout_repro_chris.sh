#!/bin/bash
set -e

PROJECT="cloud-tpu-multipod-dev"
RUNNAME="pwr-$(date +%m%d%H%M)"
MY_IMAGE="gcr.io/${PROJECT}/pw-null-layout-repro:latest"
DOCKER_IMAGE_BASE="gcr.io/tpu-prod-env-multipod/maxtext_jax_stable:latest"

# 1. Build local changes into Docker image (rebuilds and updates latest tag)
echo "Building image ${MY_IMAGE}..."
docker build -t "${MY_IMAGE}" -f - . <<EOF
FROM ${DOCKER_IMAGE_BASE}
WORKDIR /app
COPY . .
EOF

# 2. Push updated image to registry
echo "Pushing image..."
docker push "${MY_IMAGE}"

# 3. Update workload name in YAML (must be < 20 chars to avoid 63-byte K8s label truncation), then apply
echo "Submitting PathwaysJob ${RUNNAME}..."
sed -i "s|name: pw.*|name: ${RUNNAME}|g" scripts/diloco/run_pathways_null_layout_repro.yaml
kubectl apply -f scripts/diloco/run_pathways_null_layout_repro.yaml
