#!bin/bash

function usage() {
  echo "error: $1"
  echo "Usage: $0 [--project [--cluster] [--region] [--nodepool] [--num_workers]"
  echo "  --num_workers        Number of VMs in a TPU nodepool"
  echo ""
  echo "Example for a v5p-128 nodepool: $0 --project tpu-test-project --cluster tpu-cluster --region asia-east1 --num_workers 16"
  exit 1
}

while [[ "$#" > 0 ]]; do case $1 in
  -n|--project) GKE_PROJECT="$2"; shift;shift;;
  -s|--cluster) GKE_CLUSTER="$2";shift;shift;;
  -r|--region)  GKE_REGION="$2";shift;shift;;
  --nodepool)   NODEPOOL="$2";shift;shift;;
  --num_workers)   NUM_WORKERS="$2";shift;shift;;
  --libtpu_args)   LIBTPU_ARGS="$2";shift;shift;;
  *) usage "Unknown parameter passed: $1"; shift; shift;;
esac; done

if [ -z "$GKE_PROJECT" ]; then usage "--project not set"; fi;
if [ -z "$GKE_CLUSTER" ]; then usage "--cluster not set"; fi;
if [ -z "$GKE_REGION" ]; then usage "--region not set"; fi;
if [ -z "$NODEPOOL" ]; then usage "--nodepool not set"; fi;
if [ -z "$NUM_WORKERS" ]; then usage "--num_workers not set"; fi;


TPU_TOPOLOGY=$(gcloud container node-pools describe $NODEPOOL --region=$GKE_REGION --cluster=$GKE_CLUSTER --project=$GKE_PROJECT | grep "tpuTopology" | awk '{print $2}')
if [ -z "$TPU_TOPOLOGY" ]; then exit; fi;
TPU_ACCELERATOR=$(gcloud container node-pools describe $NODEPOOL --region=$GKE_REGION --cluster=$GKE_CLUSTER --project=$GKE_PROJECT | grep "goog-gke-accelerator-type" | awk '{print $2}')
if [ -z "$TPU_ACCELERATOR" ]; then exit; fi;

UUID=$(uuidgen)
export JOB_NAME="${UUID:0:5}-maxtest"
export DOCKER_IMAGE="us-docker.pkg.dev/cloud-tpu-images-public/tpu/healthscan:latest"
export NODEPOOL
export TPU_TOPOLOGY
export TPU_ACCELERATOR
export GKE_PROJECT
export GKE_REGION
export GKE_CLUSTER
export LIBTPU_ARGS

export MEMORY_PER_HOST="407Gi"
export TPU_CHIPS_PER_HOST=4
export COMPLETIONS=$NUM_WORKERS # Number of VMs in the nodepool (v6e -> 2 VMs for v6e-8, v5p -> 1 VM for a v5p-8)

YAML_VARS='$JOB_NAME $DOCKER_IMAGE $NODEPOOL $TPU_TOPOLOGY $TPU_ACCELERATOR $COMPLETIONS $MEMORY_PER_HOST $TPU_CHIPS_PER_HOST $GKE_PROJECT $GKE_REGION $GKE_CLUSTER $LIBTPU_ARGS'

envsubst "${YAML_VARS}" < maxtest.yaml.template > maxtest.yaml

# --- Execution ---
gcloud container clusters get-credentials $GKE_CLUSTER --region=$GKE_REGION --project=$GKE_PROJECT
echo "Applying generated configuration to the cluster:"
STATUS=$(kubectl apply -f maxtest.yaml)
echo $STATUS

if [[ $STATUS =~ "unchanged"|"created" ]]; then
  echo ""
  echo "Job started, visit https://console.cloud.google.com/logs/query;query=resource.labels.pod_name%3D~%22${JOB_NAME}%22;duration=PT1H?project=${GKE_PROJECT} to view logs"
  echo "or run: 'kubectl logs job.batch/${JOB_NAME}'";
fi
