#!/bin/bash
set -e

PROJECT_ID="tpu-prod-env-multipod"
REGION="us-central1"
REPO_NAME="maxtext-agent-repo"
IMAGE_NAME="overwatch-sidecar"
JOB_NAME="maxtext-validation-job"

echo "1. Configuring GCP Project..."
gcloud config set project $PROJECT_ID

echo "2. Building Docker Image..."
# We build the image locally. Assuming this is run from maxtext/src/maxtext/experimental/agent/ckpt_validation_pipeline/agent_sidecar/
cd ../../../../../../ # Go to root of project
docker build -t $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:latest -f maxtext/src/maxtext/experimental/agent/ckpt_validation_pipeline/agent_sidecar/Dockerfile .

echo "3. Pushing to Artifact Registry..."
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:latest

echo "4. Deploying to Google Cloud Run Jobs..."
# We use 'jobs deploy' instead of 'run deploy' (which is for Services)
gcloud run jobs deploy $JOB_NAME \
  --image $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:latest \
  --region $REGION \
  --service-account=ml-auto-solutions@$PROJECT_ID.iam.gserviceaccount.com \
  --set-env-vars=PYTHONUNBUFFERED=1

echo "5. Checking/Creating Cloud Scheduler Trigger..."
# Trigger every 15 minutes (or 5 minutes, depending on preference)
# We use '|| true' on create and 'update' to ensure idempotency if it already exists
gcloud scheduler jobs create http $JOB_NAME-trigger \
  --location $REGION \
  --schedule "*/15 * * * *" \
  --uri="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${JOB_NAME}:run" \
  --http-method POST \
  --oauth-service-account-email=ml-auto-solutions@$PROJECT_ID.iam.gserviceaccount.com 2>/dev/null || \
gcloud scheduler jobs update http $JOB_NAME-trigger \
  --location $REGION \
  --schedule "*/15 * * * *" \
  --uri="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${JOB_NAME}:run" \
  --http-method POST \
  --oauth-service-account-email=ml-auto-solutions@$PROJECT_ID.iam.gserviceaccount.com

echo "Deployment Complete! The Overwatch Agent is now scheduled as a Serverless Job."
