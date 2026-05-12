<!--
 Copyright 2023–2025 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

(run-pathways)=

# Via Pathways

This guide provides a comprehensive walkthrough for running MaxText workloads on a Google Kubernetes Engine (GKE) cluster using Pathways. Pathways acts as a powerful orchestrator for large-scale JAX jobs on AI Hypercomputer infrastructure.

This document assumes you have already created a Pathways GKE cluster using `xpk`. If you haven't, follow the instructions at the [Google Cloud Pathways & XPK documentation](https://cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/create-gke-cluster#xpk).

We will cover two primary modes of operation:

- **Batch workload**: Ideal for long-running, non-interactive training jobs.
- **Headless workload**: Ideal for interactive development, debugging, and running code from a local machine or CPU VM.

## 1. Prerequisites

Before you can run a MaxText workload, you must complete the following setup steps.

1. **Install XPK and its dependencies**. Ensure that the `xpk` command-line tool is installed.

2. **Create a GKE cluster** configured for Pathways.

3. **Build and upload a MaxText Docker image** to your project's Artifact Registry. For instructions on building and uploading the MaxText Docker image, please refer to the [official documentation](../../build_maxtext).

## 2. Environment configuration

The following commands use placeholder variables. Before running them, set these environment variables in your shell.

```bash
# -- Google Cloud Configuration --
# Your GCP project ID. Find it on the [Cloud Console Dashboard](https://console.cloud.google.com/home/dashboard).
export PROJECT_ID=<GCP project ID>

# The GCP location (listed as "Location" in the UI) and name of your
# TPU-enabled GKE cluster. Both can be found on the
# [Cloud Console](https://console.cloud.google.com/kubernetes/list).
export ZONE=<GCP location> # e.g., 'us-central1'
export GKE_CLUSTER=<cluster name>

# -- Workload Configuration --
# An arbitrary string to identify this specific run.
# Note: Kubernetes requires workload names to be valid DNS labels (lowercase, no underscores or periods).
export RUN_NAME="maxtext-run-$(date +%Y%m%d-%H%M%S)"

# For a full list of MaxText-supported TPU types, see: `src/maxtext/utils/accelerator_to_spec_map.py`. To see the TPU type
# of your cluster:

# 1. Connect to the cluster (required for kubectl commands later):
# gcloud container clusters get-credentials ${GKE_CLUSTER?} --location ${ZONE?} --project ${PROJECT_ID?}

# 2. Find your TPU type (e.g., 'v5p-128') by checking the accelerator labels on your nodes:
# kubectl get nodes -l cloud.google.com/gke-tpu-accelerator -o jsonpath='{.items[*].metadata.labels.cloud\.google\.com/gke-tpu-accelerator}' | tr ' ' '\n' | sort -u
export TPU_TYPE="v5p-8" # Or your desired TPU type, e.g., v5e-4
export NUM_SLICES=1 # Number of TPU slices for your job

# -- MaxText & Storage Configuration --
# Use a GCS bucket you own to store logs and checkpoints. Ideally in the same
# region as your TPUs to minimize latency and costs.
# You can list your buckets and their locations in the
# [Cloud Console](https://console.cloud.google.com/storage/browser).
export BASE_OUTPUT_DIRECTORY=<gcs bucket path> # e.g., gs://my-bucket/maxtext-runs

# The Docker image you pushed in the prerequisite step
export CLOUD_IMAGE_NAME=<image name>
export DOCKER_IMAGE="gcr.io/${PROJECT_ID?}/${CLOUD_IMAGE_NAME?}"
```

## 3. Running a batch workload

A batch workload runs entirely within the GKE cluster. You submit the job definition, and Pathways manages its execution.

### Submit the batch workload

Use the `xpk workload create-pathways` command to start the job.

```bash
xpk workload create-pathways \
  --workload=${RUN_NAME?} \
  --cluster=${GKE_CLUSTER?} \
  --num-slices=${NUM_SLICES?} \
  --tpu-type=${TPU_TYPE?} \
  --project=${PROJECT_ID?} \
  --zone=${ZONE?} \
  --docker-image=${DOCKER_IMAGE?} \
  --command="python3 -m maxtext.trainers.pre_train.train \
    base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    per_device_batch_size=1 \
    enable_checkpointing=false \
    dataset_type=synthetic \
    enable_single_controller=True \
    run_name=${RUN_NAME?}-pathways-batch"
```

### Verify the workload

You can check the status of your running workloads with the `xpk workload list` command.

```bash
xpk workload list --cluster=${GKE_CLUSTER?} --project=${PROJECT_ID?} --zone=${ZONE?}
```

## 4. Running a headless (interactive) workload

A headless workload reserves TPUs on the cluster and sets up a controller, but the Python script itself runs on a separate machine, like a local laptop or a Compute Engine VM. This is useful for rapid development and debugging. The headless mode refers to launching the Pathways backend services, such as resource manager and IFRT proxy, without a predefined user-workload container.

### Step 1: Start the headless service

This command reserves the TPUs and starts the Pathways head service on the cluster. It will wait until the resources are ready.

```bash
xpk workload create-pathways \
  --headless \
  --workload=${RUN_NAME?} \
  --num-slices=${NUM_SLICES?} \
  --tpu-type=${TPU_TYPE?} \
  --project=${PROJECT_ID?} \
  --zone=${ZONE?} \
  --cluster=${GKE_CLUSTER?}
```

### Step 2: Connect to the cluster via port forwarding

On the machine where you will run your Python script, open a **new terminal** and create a secure tunnel to the cluster's Pathways controller.

This command forwards local port 29000 to the controller pod in the cluster. It runs in the background.

```bash
kubectl port-forward \
  "$(kubectl get pods -o name | grep ${RUN_NAME?}-pathways-head)" \
  29000:29000 &> /dev/null &
```

### Step 3: Run your MaxText script locally

With the port forward active, you can now run your MaxText script. The JAX environment variables direct it to connect to the TPUs through the tunnel.

```bash
# Set these environment variables to tell JAX how to connect to the TPUs
export JAX_PLATFORMS=proxy
export JAX_BACKEND_TARGET=grpc://127.0.0.1:29000

# Run the training script
python3 -m maxtext.trainers.pre_train.train \
  base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
  per_device_batch_size=1 \
  enable_checkpointing=false \
  dataset_type=synthetic \
  enable_single_controller=True \
  run_name=${RUN_NAME?}-pathways-headless
```

The output streams directly to your terminal, just as if you were running on a local accelerator.

## Troubleshooting

- **Permission denied errors for Cloud Storage bucket**: Check that the service account used by your GKE nodes has "Storage Object Admin" permissions on your GCS bucket.
- **`Image not found` or `ImagePullBackOff`**:
  - Verify your `DOCKER_IMAGE` variable is correct.
  - Ensure you have successfully pushed the image to your project's Artifact Registry.
  - Check that your GKE cluster has permissions to pull from the registry.
- **`kubectl port-forward` fails**:
  - Confirm that the pod from Step 1 is running (`kubectl get pods`). The name should match `${RUN_NAME?}-pathways-head-0`.
  - Ensure you are authenticated with `kubectl` and have the correct context set for your GKE cluster.
- Make sure you import `pathwaysutils` package and call `pathwaysutils.initialize()` in your script when running the workload.

## More information

For more advanced configurations and a deeper dive into the Pathways architecture, see the official [Pathways on Cloud documentation](https://cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/pathways-intro).
