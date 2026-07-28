#!/bin/bash
set -e

PROJECT_ID="cloud-ml-auto-solutions"
REGION="us-central1"
REPO_NAME="maxtext-agent-repo"
IMAGE_NAME="overwatch-sidecar"
SERVICE_NAME="maxtext-validation-sidecar"

echo "1. Configuring GCP Project..."
gcloud config set project $PROJECT_ID

echo "2. Building Docker Image..."
# We build the image locally. Assuming this is run from maxtext/src/maxtext/experimental/agent/ckpt_validation_pipeline/agent_sidecar/
cd ../../../../../../ # Go to root of project if needed, or adjust build context
docker build -t $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:latest -f maxtext/src/maxtext/experimental/agent/ckpt_validation_pipeline/agent_sidecar/Dockerfile .

echo "3. Pushing to Artifact Registry..."
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:latest

echo "4. Deploying to Google Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:latest \
  --region $REGION \
  --platform managed \
  --no-allow-unauthenticated \
  --service-account=ml-auto-solutions@$PROJECT_ID.iam.gserviceaccount.com \
  --set-env-vars=PYTHONUNBUFFERED=1

echo "Deployment Complete! The Overwatch Sidecar is now running 24/7 on Cloud Run."
