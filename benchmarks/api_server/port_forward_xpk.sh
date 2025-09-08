#!/bin/bash
#
# This script automates finding the correct pod in a MaxText server workload
# and establishes a port-forward connection to it.
#
# Usage:
# bash port_forward_xpk.sh job_name=<job_name> project=<project> zone=<zone> cluster=<cluster> [namespace=<namespace>]

set -e # Exit immediately if a command exits with a non-zero status.

# --- Argument Parsing ---
NAMESPACE="default" # Default namespace

for arg in "$@"
do
    case $arg in
        job_name=*) 
        JOB_NAME="${arg#*=}"
        shift
        ;;
        project=*) 
        PROJECT="${arg#*=}"
        shift
        ;;
        zone=*) 
        ZONE="${arg#*=}"
        shift
        ;;
        cluster=*) 
        CLUSTER="${arg#*=}"
        shift
        ;;
        namespace=*) 
        NAMESPACE="${arg#*=}"
        shift
        ;;
    esac
done

# --- Validate Arguments ---
if [ -z "$JOB_NAME" ] || [ -z "$PROJECT" ] || [ -z "$ZONE" ] || [ -z "$CLUSTER" ]; then
    echo "Usage: $0 job_name=<job_name> project=<project> zone=<zone> cluster=<cluster> [namespace=<namespace>]"
    exit 1
fi

echo "--- Configuration ---"
echo "Project:   $PROJECT"
echo "Zone:      $ZONE"
echo "Cluster:   $CLUSTER"
echo "Job Name:  $JOB_NAME"
echo "Namespace: $NAMESPACE"
echo "---------------------"

# --- Get GKE Credentials ---
echo "Fetching cluster credentials..."
gcloud container clusters get-credentials "$CLUSTER" --zone "$ZONE" --project "$PROJECT" > /dev/null

# --- Find the Server Pod ---
echo "Searching for pods in namespace '$NAMESPACE' associated with job '$JOB_NAME'..."
PODS=$(kubectl get pods -n "$NAMESPACE" -o jsonpath='{.items[*].metadata.name}' | tr " " "\n" | grep "$JOB_NAME")

if [ -z "$PODS" ]; then
    echo "Error: No pods found for job name '$JOB_NAME' in namespace '$NAMESPACE'."
    exit 1
fi

SERVER_POD=""
for pod in $PODS; do
    echo "Checking logs for pod: $pod..."
    # Use grep -q for a silent check. The command succeeds if the pattern is found.
    if kubectl logs "$pod" -n "$NAMESPACE" | grep -q "Uvicorn running on http://0.0.0.0:8000"; then
        echo "Found server running in pod: $pod"
        SERVER_POD=$pod
        break # Exit the loop once the server pod is found
    fi
done

# --- Establish Port Forwarding ---
if [ -n "$SERVER_POD" ]; then
    echo "Establishing port-forward from localhost:8000 to $SERVER_POD:8000 in namespace '$NAMESPACE'..."
    echo "You can now send requests to http://localhost:8000"
    kubectl port-forward "pod/$SERVER_POD" -n "$NAMESPACE" 8000:8000
else
    echo "Error: Could not find a pod running the Uvicorn server for job '$JOB_NAME' in namespace '$NAMESPACE'."
    exit 1
fi