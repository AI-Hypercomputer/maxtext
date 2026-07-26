<!--
 Copyright 2023–2026 Google LLC

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

(run-cluster-toolkit)=

# At scale with Cluster Toolkit (`gcluster`)

This guide provides the recommended workflow for running MaxText on Google Kubernetes Engine (GKE) using **Cluster Toolkit's `gcluster` CLI**. For a complete reference on Cluster Toolkit and `gcluster`, please see the [official Cluster Toolkit repository](https://github.com/GoogleCloudPlatform/cluster-toolkit) and [Google Cloud documentation](https://cloud.google.com/cluster-toolkit/docs/overview).

## Overview of the workflow

The process involves two main stages:

1. **Packaging or Layering the MaxText Application:** You can either prebuild a MaxText Docker image or use `gcluster`'s built-in Crane integration (`--build-context .`) to automatically build and layer your local MaxText working directory on top of a base image on-the-fly.
2. **Orchestrating Workload Deployment:** `gcluster job submit` automatically discovers your GKE cluster's accelerator topology, verifies Kueue and JobSet health, generates the required Kubernetes resources (`JobSet`, `ClusterQueue`, `LocalQueue`, `ResourceFlavor`), and deploys your multi-host JAX workload.

```none
+--------------------------+      +--------------------+      +-------------------+
|                          |      |                    |      |                   |
| Your Development Machine +------>  Artifact Registry +------>  GKE Cluster      |
| (with gcluster CLI)      |      | (Stores your image)|      |(with Accelerators)|
|                          |      |                    |      |                   |
| 1. Local MaxText Code    |      | 2. gcluster builds |      | 3. gcluster runs  |
|    (--build-context .)   |      |    & pushes image  |      |    multi-host job |
+--------------------------+      +--------------------+      +-------------------+
```

______________________________________________________________________

## 1. Prerequisites

Before you begin, ensure you have the necessary client tools installed and permissions configured.

### Required tools

- **Python >= 3.12** with `pip` and `venv`.
- **Google Cloud CLI (`gcloud`):** Install it from [here](https://cloud.google.com/sdk/docs/install) and run `gcloud init`.
- **kubectl & GKE auth plugin:** Required for communicating with GKE clusters:
  ```bash
  gcloud components install kubectl gke-gcloud-auth-plugin
  ```
- **gcluster CLI:** Follow the official [Cluster Toolkit CLI Installation Guide](https://cloud.google.com/cluster-toolkit/docs/install-cli) (or download the binary from the [releases page](https://github.com/GoogleCloudPlatform/cluster-toolkit/releases)) and ensure `gcluster` is in your `$PATH`.
- **Docker Credentials:** Configure Docker authentication for your target Artifact Registry region:
  ```bash
  gcloud auth configure-docker <region>-docker.pkg.dev --quiet
  ```

### GCP permissions and APIs

Make sure the required Google Cloud APIs are enabled on your project:

```bash
gcloud services enable \
  container.googleapis.com \
  artifactregistry.googleapis.com \
  storage.googleapis.com
```

Your Google Cloud user account needs the following IAM roles in your target project:

- Artifact Registry Writer
- Kubernetes Engine Admin / Developer
- Storage Admin (for accessing GCS buckets for checkpoints and datasets)
- Logging / Monitoring Viewer

______________________________________________________________________

## 2. One-time environment setup

First, ensure your local `kubectl` is authenticated with your target GKE cluster:

```bash
gcloud container clusters get-credentials <GKE_CLUSTER_NAME> \
  --region <GCP_REGION_OR_ZONE> \
  --project <GCP_PROJECT_ID>
```

Next, configure your default Google Cloud project, target GKE cluster name, and cluster location in `gcluster`:

```bash
gcluster job config set project <GCP_PROJECT_ID>
gcluster job config set cluster <GKE_CLUSTER_NAME>
gcluster job config set location <GCP_REGION_OR_ZONE> # e.g., europe-west4 or us-east5-a
```

When `gcluster` runs, it inspects your GKE cluster and verifies that **Kueue** and **JobSet** custom resource definitions and webhooks are healthy. If Kueue is missing on your cluster, `gcluster` can prompt to install it automatically.

______________________________________________________________________

## 3. Step-by-Step MaxText TPU Smoke Test (Multi-Host)

To verify that your GKE cluster, Kueue admission queues, and multi-host JAX ICI interconnects are fully functional before launching large production runs, you can run a lightweight synthetic smoke test using MaxText's `default` dummy model (`model_name=default`).

```bash
cd /path/to/maxtext

gcluster job submit \
  --base-image us-east5-docker.pkg.dev/cloud-tpu-multipod-dev/maxtext-images/maxtext_base:latest \
  --build-context . \
  --command "python3 -m maxtext.trainers.pre_train.train run_name=maxtext-multihost-smoke-test steps=5 dataset_type=synthetic model_name=default enable_checkpointing=False" \
  --name maxtext-multihost-smoke-test \
  --compute-type ct5p-hightpu-4t \
  --topology 4x4x4
```

______________________________________________________________________

## 4. Run a production MaxText job on GCS datasets

This section assumes you have an existing GKE cluster with TPU or GPU node pools.

### Step 1: Export target image repository

Specify which Artifact Registry repository in your target project and region should store the container image built by `gcluster`. If you do not already have a Docker repository in Artifact Registry, create one:

```bash
gcloud artifacts repositories create maxtext-images \
  --repository-format=docker \
  --location=<GCP_REGION> \
  --description="MaxText container images"
```

For more details, see the official [Artifact Registry documentation](https://cloud.google.com/artifact-registry/docs/docker/store-docker-container-images). Once created, export the repository name:

```bash
export GCLUSTER_IMAGE_REPO=maxtext-images
```

### Step 2: Submit a training workload

Navigate to the root directory of your local MaxText repository and submit the job using `gcluster job submit`.

Passing `--build-context .` packages your local MaxText source directory on top of a lightweight base image (`--base-image`) and builds/pushes the container image on-the-fly. This allows you to test local code modifications rapidly without manually building a Docker container image.

*(Note: If you already have a fully pre-built custom container image containing your code pushed to Artifact Registry, you can specify `--image <FULL_IMAGE_URI>` directly instead of `--base-image` and `--build-context`.)*

```bash
cd /path/to/maxtext

gcluster job submit \
  --base-image us-east5-docker.pkg.dev/cloud-tpu-multipod-dev/maxtext-images/maxtext_base:latest \
  --build-context . \
  --command "python3 -m maxtext.trainers.pre_train.train run_name=maxtext-test base_output_directory=gs://<MY_BUCKET>/output dataset_path=gs://<MY_DATASET>/ steps=100" \
  --name maxtext-test \
  --compute-type ct5p-hightpu-4t \
  --topology 4x4x4
```

#### Key Flags:

- `--compute-type`: The accelerator machine flavor or accelerator type (e.g., `ct5p-hightpu-4t` for TPU v5p, or `ct6e-hightpu-4t` for TPU v6e).
- `--topology`: The physical 3D mesh topology required for multi-host TPU slices (e.g., `4x4x4` allocates 64 chips across 16 physical nodes; `2x2x1` allocates 4 chips on 1 node).
- `--build-context`: Path to your local build directory (`.`), allowing rapid iterative testing of local MaxText changes.

______________________________________________________________________

## 5. Running with Ahead-of-Time (AOT) Compilation

Ahead-of-Time (AOT) compilation can significantly reduce the startup time of your training job by pre-compiling the JAX training step for a specific hardware topology. With `gcluster`, you can generate the compiled pickle locally and include it via `--build-context .`.

### Step 1: Generate the AOT artifact

Note that running `train_compile.py` locally requires a Python environment with MaxText and JAX installed. Follow the [MaxText Getting Started Guide](https://github.com/AI-Hypercomputer/maxtext#getting-started) to set up your local environment before running compilation.

Run `train_compile.py` locally to create the compiled artifact for the target TPU topology:

```bash
export TPU_TYPE="your-tpu-type" # e.g. "v5p-128"
export NUM_SLICES=1
export PER_DEVICE_BATCH_SIZE=1

python3 -m maxtext.trainers.pre_train.train_compile \
  compile_topology=${TPU_TYPE} \
  compile_topology_num_slices=${NUM_SLICES} \
  compiled_trainstep_file=maxtext_${TPU_TYPE}_aot.pickle \
  per_device_batch_size=${PER_DEVICE_BATCH_SIZE}
```

This creates `maxtext_${TPU_TYPE}_aot.pickle` in your MaxText repository root directory.

### Step 2: Submit the workload referencing the AOT artifact

When you run `gcluster job submit` with `--build-context .`, the generated pickle file is automatically packaged into the container image. Pass `compiled_trainstep_file=maxtext_${TPU_TYPE}_aot.pickle` to tell MaxText to load the pre-compiled step:

```bash
gcluster job submit \
  --base-image us-east5-docker.pkg.dev/cloud-tpu-multipod-dev/maxtext-images/maxtext_base:latest \
  --build-context . \
  --command "python3 -m maxtext.trainers.pre_train.train run_name=maxtext-aot-test base_output_directory=gs://<MY_BUCKET>/output dataset_path=gs://<MY_DATASET>/ steps=100 per_device_batch_size=1 compiled_trainstep_file=maxtext_${TPU_TYPE}_aot.pickle" \
  --name maxtext-aot-test \
  --compute-type ct5p-hightpu-4t \
  --topology 4x4x4
```

______________________________________________________________________

## 6. Monitoring and managing workloads

### List Jobs on the Cluster

View active or queued workloads in the cluster:

```bash
gcluster job list
```

### Check Pod Status

List all pods belonging to your submitted workload:

```bash
kubectl get pods -l gcluster.google.com/workload=<JOB_NAME>
```

### View Workload Logs

Stream logs from the leader container using `gcluster` or `kubectl`:

```bash
gcluster job logs <JOB_NAME>
# Or using kubectl:
kubectl logs -f -l gcluster.google.com/workload=<JOB_NAME>,jobset.sigs.k8s.io/job-index=0
```

Or use the Google Cloud Console Logs Explorer link printed by `gcluster job submit`.

### Cancel or Delete a Job

To terminate a running workload or clean up completed JobSet resources:

```bash
gcluster job cancel <JOB_NAME>
```

Or delete the Kubernetes JobSet directly:

```bash
kubectl delete jobset <JOB_NAME>
```
