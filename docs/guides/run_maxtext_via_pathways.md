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
# Guide: Running MaxText via Pathways

This guide provides a comprehensive walkthrough for running MaxText workloads on a Google Kubernetes Engine (GKE) cluster using Pathways. Pathways acts as a powerful orchestrator for large-scale JAX jobs on AI Hypercomputer infrastructure.

This document assumes you have already created a Pathways GKE cluster using `xpk`. If you haven't, follow the instructions at the [Google Cloud Pathways & XPK documentation](https://cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/create-gke-cluster#xpk).

We will cover two primary modes of operation:
*   **Batch workload**: Ideal for long-running, non-interactive training jobs.
*   **Headless workload**: Ideal for interactive development, debugging, and running code from a local machine or CPU VM.

## 1. Prerequisites

Before you can run a MaxText workload, you must complete the following setup steps.

1.  **Install XPK and its dependencies**. Ensure that the `xpk` command-line tool is installed.
2.  **Create a GKE cluster** configured for Pathways.
3.  **Build and upload a MaxText Docker image** to your project's Artifact Registry.

    ```bash
    # Step 1: Build the Docker image for a TPU device
    # This image contains MaxText and its dependencies.
    bash docker_build_dependency_image.sh DEVICE=tpu MODE=jax_ai_image BASEIMAGE=us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:latest

    # Step 2: Configure Docker to authenticate with Google Cloud
    gcloud auth configure-docker

    # Step 3: Upload the image to your project's registry
    # Replace `$USER_runner` with your desired image name.
    bash docker_upload_runner.sh CLOUD_IMAGE_NAME=$USER_runner
    ```

## 2. Environment configuration

The following commands use placeholder variables. Before running them, set these environment variables in your shell.

```bash
# -- Google Cloud Configuration --
export PROJECT="your-gcp-project-id"
export ZONE="your-gcp-zone"
export CLUSTER="your-gke-cluster-name"

# -- Workload Configuration --
export WORKLOAD_NAME="maxtext-job-$(date +%Y%m%d-%H%M%S)"
export TPU_TYPE="v5p-8" # Or your desired TPU type, e.g., v5e-4
export WORKLOAD_NODEPOOL_COUNT=1 # Number of TPU slices for your job

# -- MaxText & Storage Configuration --
export BUCKET_NAME="your-gcs-bucket-name"
export RUN_NAME="maxtext-run-1"
# The Docker image you pushed in the prerequisite step
export DOCKER_IMAGE="gcr.io/${PROJECT}/${USER}_runner"
```

## 3. Running a Batch workload

A batch workload runs entirely within the GKE cluster. You submit the job definition, and Pathways manages its execution.

### Submit the Batch Workload

Use the `xpk workload create-pathways` command to start the job.

```bash
xpk workload create-pathways \
  --workload=$WORKLOAD_NAME \
  --cluster=$CLUSTER \
  --num-slices=$WORKLOAD_NODEPOOL_COUNT \
  --tpu-type=$TPU_TYPE \
  --project=$PROJECT \
  --zone=$ZONE \
  --docker-image=${DOCKER_IMAGE} \
  --command="python3 -m MaxText.train src/MaxText/configs/base.yml \
    base_output_directory=gs://${BUCKET_NAME} \
    per_device_batch_size=1 \
    enable_checkpointing=false \
    dataset_type=synthetic \
    enable_single_controller=True \
    run_name=${RUN_NAME}-pathways-batch"
```

### Verify the workload

You can check the status of your running workloads with the `xpk workload list` command.

```bash
xpk workload list --cluster=$CLUSTER --project=$PROJECT --zone=$ZONE
```

## 4. Running a headless (interactive) workload

A headless workload reserves TPUs on the cluster and sets up a controller, but the Python script itself runs on a separate machine, like a local laptop or a Compute Engine VM. This is useful for rapid development and debugging. The headless mode refers to launching the Pathways backend services, such as resource manager and IFRT proxy, without a predefined user-workload container.

### Step 1: Start the headless service

This command reserves the TPUs and starts the Pathways head service on the cluster. It will wait until the resources are ready.

```bash
xpk workload create-pathways \
  --headless \
  --workload=${WORKLOAD_NAME} \
  --num-slices=${WORKLOAD_NODEPOOL_COUNT} \
  --tpu-type=${TPU_TYPE} \
  --project=${PROJECT} \
  --zone=${ZONE} \
  --cluster=${CLUSTER}
```

### Step 2: Connect to the cluster via port forwarding

On the machine where you will run your Python script, open a **new terminal** and create a secure tunnel to the cluster's Pathways controller.

```bash
# This command forwards local port 29000 to the controller pod in the cluster.
# It runs in the background.
kubectl port-forward \
  "$(kubectl get pods -o name | grep ${WORKLOAD_NAME}-pathways-head)" \
  29000:29000 &> /dev/null &
```

### Step 3: Run Your MaxText script locally

With the port forward active, you can now run your MaxText script. The JAX environment variables direct it to connect to the TPUs through the tunnel.

```bash
# Set these environment variables to tell JAX how to connect to the TPUs
export JAX_PLATFORMS=proxy
export JAX_BACKEND_TARGET=grpc://127.0.0.1:29000

# Run the training script
python3 -m MaxText.train src/MaxText/configs/base.yml \
  base_output_directory=gs://${BUCKET_NAME} \
  per_device_batch_size=1 \
  enable_checkpointing=false \
  dataset_type=synthetic \
  enable_single_controller=True \
  run_name=${RUN_NAME}-pathways-headless
```
The output streams directly to your terminal, just as if you were running on a local accelerator.

## Troubleshooting

*   **Permission Denied errors for Cloud Storage Bucket**: Check that the service account used by your GKE nodes has "Storage Object Admin" permissions on your GCS bucket.
*   **`Image not found` or `ImagePullBackOff`**:
    *   Verify your `DOCKER_IMAGE` variable is correct.
    *   Ensure you have successfully pushed the image to your project's Artifact Registry.
    *   Check that your GKE cluster has permissions to pull from the registry.
*   **`kubectl port-forward` fails**:
    *   Confirm that the pod from Step 1 is running (`kubectl get pods`). The name should match `${WORKLOAD_NAME}-pathways-head-0`.
    *   Ensure you are authenticated with `kubectl` and have the correct context set for your GKE cluster.
* Make sure you import pathwaysutils package and call `pathwaysutils.initialize()` in your script when running the workload

## More information

For more advanced configurations and a deeper dive into the Pathways architecture, see the official [Pathways on Cloud documentation](https://cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/pathways-intro).
