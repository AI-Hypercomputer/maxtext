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

(run-xpk)=

# At scale with XPK

This guide provides the recommended workflow for running MaxText on Google Kubernetes Engine (GKE) using the **Accelerated Processing Kit (XPK)**. For a complete reference on XPK, please see the [official XPK repository](https://github.com/AI-Hypercomputer/xpk).

## Overview of the workflow

The process involves two main stages. First, you will package the MaxText application and its dependencies into a self-contained Docker image. This is done on your local machine or any environment where Docker is installed. Second, you will use the XPK command-line tool to orchestrate the deployment of this image as a training job on a GKE cluster equipped with accelerators (TPUs or GPUs).

XPK abstracts away the complexity of cluster management and job submission, handling tasks like uploading your Docker image to Artifact Registry and scheduling the workload on the cluster.

```none
+--------------------------+      +--------------------+      +-------------------+
|                          |      |                    |      |                   |
| Your Development Machine +------>  Artifact Registry +------>  GKE Cluster      |
| (anywhere with Docker)   |      | (Stores your image)|      |(with Accelerators)|
|                          |      |                    |      |                   |
| 1. Build MaxText Docker  |      | 2. XPK uploads     |      | 3. XPK runs job   |
|    Image                 |      |    image for you   |      |    using the image|
+--------------------------+      +--------------------+      +-------------------+
```

______________________________________________________________________

## 1. Prerequisites

Before you begin, you must have the necessary tools installed and permissions configured.

### Required tools

- **Python >= 3.12** with `pip` and `venv`.

- **Google Cloud CLI (`gcloud`):** Install it from [here](https://cloud.google.com/sdk/docs/install) and then run `gcloud init`.

- **kubectl:** The Kubernetes command-line tool.

- **Docker:** Follow the [installation instructions](https://docs.docker.com/engine/install/) and [follow the steps to configure sudoless Docker](https://docs.docker.com/engine/install/linux-postinstall/).

### GCP permissions

Your Google Cloud user account needs the following IAM roles for the project you're using:

- Artifact Registry Writer

- Compute Admin

- Kubernetes Engine Admin

- Logging Admin

- Monitoring Admin

- Service Account User

- Storage Admin

- Vertex AI Administrator

______________________________________________________________________

## 2. One-time environment setup

These commands configure your local environment to connect to Google Cloud services.

1. **Authenticate gcloud**

   ```
   gcloud auth login
   ```

2. **Install GKE auth plugin**

   ```
   sudo apt-get update && sudo apt-get install google-cloud-sdk-gke-gcloud-auth-plugin
   ```

3. **Configure Docker credentials**

   ```
   gcloud auth configure-docker
   ```

______________________________________________________________________

## 3. Build the MaxText Docker image

For instructions on building the MaxText Docker image, please refer to the [official documentation](https://maxtext.readthedocs.io/en/latest/build_maxtext.html).

______________________________________________________________________

## 4. Run your first MaxText job

This section assumes you have an existing GKE cluster with either TPU or GPU nodes.

```{important}
This guide focuses on submitting workloads to an existing cluster. Cluster creation and management is a separate topic. For a comprehensive guide on all `xpk` commands, including `xpk cluster create`, please refer to the **[official XPK documentation](https://github.com/AI-Hypercomputer/xpk)**.
```

1. **Set your configuration**

   Set up the following environment variables to configure your training run. Replace
   placeholders with your actual values.

   ```bash
   # -- Google Cloud Configuration --
   # Your GCP project ID. Find it on the [Cloud Console Dashboard](https://console.cloud.google.com/home/dashboard).
   # If you've already set it in your local config, you can retrieve it via:
   # gcloud config get-value project
   export PROJECT_ID=<GCP project ID>

   # The GCP location (listed as "Location" in the UI) and name of your
   # TPU-enabled (or GPU-enabled) GKE cluster. Both can be found on the
   # [Cloud Console](https://console.cloud.google.com/kubernetes/list).
   export ZONE=<GCP location> # e.g., 'us-central1' or 'us-central1-a'
   export GKE_CLUSTER=<cluster name>

   # -- Workload Configuration --
   # An arbitrary string to identify this specific run.
   # Note: Kubernetes requires workload names to be valid DNS labels (lowercase, no underscores or periods).
   export RUN_NAME="maxtext-run-$(date +%Y%m%d-%H%M%S)"

   # Number of TPU slices (for TPU clusters) or number of nodes (for GPU clusters)
   export NUM_SLICES=1

   # -- MaxText & Storage Configuration --
   # Use a GCS bucket you own to store logs and checkpoints. Ideally in the same
   # region as your TPUs to minimize latency and costs.
   # You can list your buckets and their locations in the
   # [Cloud Console](https://console.cloud.google.com/storage/browser).
   export BASE_OUTPUT_DIRECTORY=<gcs bucket path> # e.g., gs://my-bucket/maxtext-runs
   export DATASET_PATH="gs://your-dataset-bucket/"
   ```

2. **Configure gcloud CLI**

   ```bash
   gcloud config set project ${PROJECT_ID?}
   gcloud config set compute/zone ${ZONE?}
   ```

### A Note on multi-slice and multi-node runs

The examples below run on a single TPU slice (`--num-slices=1`) or a small number of GPU nodes (`--num-nodes=2`). To scale your job to a larger, multi-host configuration, you simply increase the `NUM_SLICES` value.

For instance, to run a job across **four TPU slices**, you would change `export NUM_SLICES=1` to `export NUM_SLICES=4`. This tells XPK to allocate four `v5litepod-256` slices and orchestrate the training job across all of them as a single workload. Similarly, for GPUs, you would increase the value.

3. **Create the workload (run the job)**
   - **On your TPU cluster:**

     ```bash
     xpk workload create \
       --cluster ${GKE_CLUSTER?} \
       --workload ${RUN_NAME?} \
       --base-docker-image maxtext_base_image \
       --tpu-type v5litepod-256 \
       --num-slices ${NUM_SLICES?} \
       --command "python3 -m maxtext.trainers.pre_train.train run_name=${RUN_NAME?} base_output_directory=${BASE_OUTPUT_DIRECTORY?} dataset_path=${DATASET_PATH?} steps=100"
     ```

   - **On your GPU cluster:**

     ```bash
     xpk workload create \
       --cluster ${GKE_CLUSTER?} \
       --workload ${RUN_NAME?} \
       --base-docker-image maxtext_base_image \
       --device-type h100-80gb-8 \
       --num-nodes ${NUM_SLICES?} \
       --command "python3 -m maxtext.trainers.pre_train.train run_name=${RUN_NAME?} base_output_directory=${BASE_OUTPUT_DIRECTORY?} dataset_path=${DATASET_PATH?} steps=100"
     ```

______________________________________________________________________

## 6. Running with Ahead-of-Time (AOT) Compilation

Ahead-of-Time (AOT) compilation can significantly reduce the startup time of your training job by pre-compiling the JAX training step for a specific hardware topology. The workflow involves generating a compiled artifact, including it in your Docker image, and then telling MaxText to use it at runtime.

### Step 1: Generate the AOT artifact

First, run `train_compile.py` script to create the compiled artifact for the specified TPU topology.

```bash
export WORKLOAD_NAME="${USER}-aot-job"
export TPU_TYPE="your-tpu-type" # e.g. "v5p-32"
export NUM_SLICES=your-num-slices # e.g. 1
export PER_DEVICE_BATCH_SIZE=your-batch-size # e.g. 1
export USER=your-username

python3 -m maxtext.trainers.pre_train.train_compile \
  compile_topology=${TPU_TYPE} \
  compile_topology_num_slices=${NUM_SLICES} \
  compiled_trainstep_file=maxtext_${TPU_TYPE}_aot.pickle \
  per_device_batch_size=${PER_DEVICE_BATCH_SIZE}
```

This will create a file named `maxtext_${TPU_TYPE}_aot.pickle` in your MaxText root directory.

### Step 2: Re-build and upload your Docker image

The AOT artifact must be included in your Docker image. The `docker_upload_runner.sh` script automatically copies the contents of your MaxText directory into the image:

```bash
export CLOUD_IMAGE_NAME="${USER}-maxtext-aot-runner"

bash src/dependencies/scripts/docker_upload_runner.sh CLOUD_IMAGE_NAME=${CLOUD_IMAGE_NAME}
```

### Step 3: Create the XPK workload with the AOT artifact

Now, submit the workload as before, but add the `compiled_trainstep_file` argument to the command and use the custom Docker Image we built in step 2. This tells MaxText to load the pre-compiled step instead of compiling it at runtime.

```bash
export BASE_OUTPUT_DIR=your-output-dir # e.g. gs://your-output-bucket/
export DATASET_PATH=your-dataset-path # e.g. gs://your-dataset-bucket/
export CLUSTER_NAME=your-cluster-name
export PROJECT_ID=your-project-id

xpk workload create \
  --cluster ${CLUSTER_NAME} \
  --workload ${WORKLOAD_NAME} \
  --docker-image "gcr.io/${PROJECT_ID}/${CLOUD_IMAGE_NAME}" \
  --tpu-type ${TPU_TYPE} \
  --num-slices ${NUM_SLICES} \
  --command "python3 -m maxtext.trainers.pre_train.train \
    run_name=${WORKLOAD_NAME} \
    base_output_directory=${BASE_OUTPUT_DIR} \
    dataset_path=${DATASET_PATH} \
    steps=100 \
    per_device_batch_size=${PER_DEVICE_BATCH_SIZE} \
    compiled_trainstep_file=/deps/maxtext_${TPU_TYPE}_aot.pickle"
```

Your job will now start faster by skipping the JAX compilation step on the cluster.

______________________________________________________________________

## 7. Managing and monitoring your job

- **View logs in real-time:** The easiest way to see the output of your training job is through the Google Cloud Console.

- 1. Navigate to the **Kubernetes Engine** section.

- 2. Go to **Workloads**.

  3. Find your workload (e.g., `${RUN_NAME?}`) and click on it.

- 4. Select the **Logs** tab to view the container logs.

- **List your jobs:**

  ```bash
  xpk workload list --cluster ${GKE_CLUSTER?}
  ```

- **Analyze output:** Checkpoints and other artifacts will be saved to the Google Cloud Storage bucket you specified in `BASE_OUTPUT_DIRECTORY`.

- **Delete a job:**

  ```bash
  xpk workload delete --cluster ${GKE_CLUSTER?} --workload ${RUN_NAME?}
  ```
