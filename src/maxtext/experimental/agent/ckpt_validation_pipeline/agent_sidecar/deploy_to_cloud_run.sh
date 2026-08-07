#!/bin/bash
set -e

PROJECT_ID="tpu-prod-env-multipod"
REGION="us-central1"
REPO_NAME="maxtext-agent-repo"
IMAGE_NAME="overwatch-sidecar"
JOB_NAME="maxtext-validation-job"

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
