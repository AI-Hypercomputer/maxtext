#!/bin/bash
# Helper script to rebuild the custom DiLoCo Docker image and run the 2-slice test on GKE.
set -e

CLUSTER="wl-v5p-8-2"
REGION="us-east5"
PROJECT="cloud-tpu-multipod-dev"
IMAGE_TAG="us-east5-docker.pkg.dev/cloud-tpu-multipod-dev/maxtext-images/diloco-mohit:latest"
JOB_NAME="diloco-test-2slice"

echo "======================================================================"
echo "Step 1: Rebuilding and pushing custom Docker image..."
echo "======================================================================"
# gcloud builds submit is recommended as it runs fast in Cloud Build without local push bandwidth limits
gcloud builds submit --tag "$IMAGE_TAG" . --project="$PROJECT"

echo "======================================================================"
echo "Step 2: Authenticating to cluster $CLUSTER and submitting job..."
echo "======================================================================"
gcloud container clusters get-credentials "$CLUSTER" --region "$REGION" --project "$PROJECT"

# Delete any previous job instance
kubectl delete pathwaysjob "$JOB_NAME" --ignore-not-found=true

# Apply updated deployment spec
kubectl apply -f tests/diloco_pathwaysjob_wl.yaml

echo "======================================================================"
echo "Step 3: Waiting for pod to start and streaming logs..."
echo "======================================================================"
sleep 5
while ! kubectl get pods -l jobset.sigs.k8s.io/replicatedjob-name=pathways-head | grep -q "Running"; do
  echo "Waiting for pathways-head pod to enter Running state..."
  sleep 5
done

echo "=== Pod is Running! Streaming live execution logs ==="
kubectl logs -f -l jobset.sigs.k8s.io/replicatedjob-name=pathways-head -c jax-tpu
