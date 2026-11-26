#!/bin/bash
set -e

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Ensure temporary Dockerfile in root is cleaned up on exit
trap "rm -f ./Dockerfile .gcloudignore.tmp" EXIT
PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-tpu-prod-env-multipod}"
REGION="${GOOGLE_CLOUD_REGION:-us-central1}"
REPO_NAME="${AGENT_REPO_NAME:-maxtext-agent-repo}"
IMAGE_NAME="${AGENT_IMAGE_NAME:-overwatch-sidecar}"
JOB_NAME="${AGENT_JOB_NAME:-maxtext-validation-job}"

echo "1. Configuring GCP Project..."
gcloud config set project $PROJECT_ID

echo "2. Building Docker Image remotely via Google Cloud Build..."
# Navigate to the root of the git repository securely regardless of where you ran this from
cd "$(git rev-parse --show-toplevel)"

# Copy Dockerfile to root temporarily so Cloud Build finds it easily
cp src/maxtext/experimental/agent/ckpt_validation_pipeline/agent_sidecar/Dockerfile ./Dockerfile

# Create a temporary ignore file to allow .git folder upload, bypassing the root .dockerignore
cp .dockerignore .gcloudignore.tmp || touch .gcloudignore.tmp
sed -i 's/^.git/#.git/' .gcloudignore.tmp

# Submit build to Google Cloud Build (bypasses need for local Docker)
gcloud builds submit --tag $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:latest --project $PROJECT_ID --ignore-file=.gcloudignore.tmp .

# Clean up
rm ./Dockerfile .gcloudignore.tmp

echo "3. Image built and pushed successfully by Cloud Build."

echo "4. Deploying to Google Cloud Run Jobs..."
# We use 'jobs deploy' instead of 'run deploy' (which is for Services)
gcloud run jobs deploy $JOB_NAME \
  --image $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:latest \
  --region $REGION \
  --service-account=ml-auto-solutions@$PROJECT_ID.iam.gserviceaccount.com \
  --memory=16Gi \
  --cpu=4 \
  --task-timeout=3h \
  --update-env-vars=PYTHONUNBUFFERED=1

echo "Deployment Complete! The Overwatch Agent is now deployed as a Serverless Job and is triggered exclusively by Airflow on failure."
