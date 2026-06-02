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

# Deploy the elastic training JobSet on the cluster.
#
# Renders the training manifest with dataset-specific flags
# and applies it to the cluster.
#
# Required env vars (set by Cloud Build or manually):
#   DATASET       - "synthetic" or "glaive"
#   BUCKET_NAME   - GCS bucket name
#   IMAGE         - container image for training
#   RUN_NAME      - training run name
#
# Prerequisites:
#   K8s secret "hf-token" must exist (created by Terraform)
#
# Usage:
#   # Via Cloud Build (env vars set automatically):
#   bash deploy/jobs/training/submit.sh
#
#   # Standalone:
#   DATASET=glaive BUCKET_NAME=my-bucket RUN_NAME=elastic-qwen3-0.6b \
#   IMAGE=us-central1-docker.pkg.dev/my-project/maxtext/maxtext-runner:latest \
#   bash deploy/jobs/training/submit.sh

set -euo pipefail

# Clean up previous run output
echo "=== Cleaning GCS output for run: ${RUN_NAME} ==="
gcloud storage rm -r "gs://${BUCKET_NAME}/output/${RUN_NAME}/" 2>/dev/null || true
gcloud storage rm -r "gs://${BUCKET_NAME}/pathways/" 2>/dev/null || true

# Delete existing job if present
echo "=== Deploying training job (dataset=${DATASET}) ==="
kubectl delete jobset pw-elastic --ignore-not-found=true --wait=true 2>/dev/null || true
sleep 3

# Create ConfigMap from train.sh
echo "=== Creating ConfigMap with train.sh ==="
kubectl delete configmap train-script --ignore-not-found=true 2>/dev/null
kubectl create configmap train-script --from-file=train.sh=deploy/jobs/training/train.sh

# Render the manifest
sed \
  -e "s|IMAGE_PLACEHOLDER|${IMAGE}|g" \
  -e "s|BUCKET_PLACEHOLDER|${BUCKET_NAME}|g" \
  -e "s|RUN_NAME_PLACEHOLDER|${RUN_NAME}|g" \
  -e "s|DATASET_PLACEHOLDER|${DATASET}|g" \
  -e "s|DEBUGGING_PLACEHOLDER|${DEBUGGING:-false}|g" \
  deploy/jobs/training/jobset.yaml > /tmp/training-job.yaml

# Apply
kubectl apply -f /tmp/training-job.yaml

echo ""
echo "=== JobSet deployed ==="
echo "Monitor with:"
echo "  kubectl logs -f \$(kubectl get pods -l job-name=pw-elastic-pathways-head-0 -o name | head -1) -c main"
