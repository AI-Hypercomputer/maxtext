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

# Submit a data preparation K8s Job on the cluster.
#
# Converts a HuggingFace dataset to MaxText Grain ArrayRecord format
# and uploads shards to GCS. Skips if data already exists.
#
# Required env vars (set by Cloud Build or manually):
#   DATASET       - "synthetic" or "glaive"
#   BUCKET_NAME   - GCS bucket name
#   PROJECT_ID    - GCP project ID
#   NUM_SHARDS    - number of ArrayRecord shards
#   HF_DATASET    - HuggingFace dataset path
#   IMAGE         - container image for the data prep job
#
# Usage:
#   # Via Cloud Build (env vars set automatically):
#   bash deploy/jobs/data-prep/submit.sh
#
#   # Standalone:
#   DATASET=glaive BUCKET_NAME=my-bucket PROJECT_ID=my-project \
#   NUM_SHARDS=8 HF_DATASET=hiyouga/glaive-function-calling-v2-sharegpt \
#   IMAGE=us-central1-docker.pkg.dev/my-project/maxtext/maxtext-runner:latest \
#   bash deploy/jobs/data-prep/submit.sh

set -euo pipefail

# Skip if synthetic
if [ "${DATASET}" != "glaive" ]; then
  echo "=== Using synthetic data, skipping data prep ==="
  exit 0
fi

# Skip if data already exists
MARKER=$(printf "gs://${BUCKET_NAME}/data/glaive-fc-v2/train.array_record-00000-of-%05d" "${NUM_SHARDS}")
if gcloud storage ls "${MARKER}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "=== Data already prepared, skipping ==="
  gcloud storage ls "gs://${BUCKET_NAME}/data/glaive-fc-v2/" --project="${PROJECT_ID}"
  exit 0
fi

# Create ConfigMap from prepare_data.py
echo "=== Creating ConfigMap with prepare_data.py ==="
kubectl delete configmap data-prep-script --ignore-not-found=true 2>/dev/null
kubectl create configmap data-prep-script --from-file=prepare_data.py=deploy/jobs/data-prep/prepare_data.py

# Render the Job manifest
echo "=== Rendering data prep Job manifest ==="
sed \
  -e "s|IMAGE_PLACEHOLDER|${IMAGE}|g" \
  -e "s|HF_DATASET_PLACEHOLDER|${HF_DATASET}|g" \
  -e "s|BUCKET_PLACEHOLDER|${BUCKET_NAME}|g" \
  -e "s|NUM_SHARDS_PLACEHOLDER|${NUM_SHARDS}|g" \
  deploy/jobs/data-prep/manifest.yaml > /tmp/data-prep-job.yaml

# Submit the Job
echo "=== Submitting data prep job ==="
kubectl delete job data-prep-glaive --ignore-not-found=true 2>/dev/null
kubectl apply -f /tmp/data-prep-job.yaml
echo "=== Data prep job submitted ==="
kubectl get job data-prep-glaive
