#!/bin/bash
export PROJECT_ID="cloud-tpu-multipod-dev"
export IMAGE_TAG="gcr.io/${PROJECT_ID}/deepseek3-prefetch:$(date +%Y%m%d-%H%M%S)"
gcloud auth configure-docker gcr.io --quiet
docker build -t "${IMAGE_TAG}" -f Dockerfile.prefetch .
docker push "${IMAGE_TAG}"
echo "BUILT_IMAGE_TAG=${IMAGE_TAG}"
