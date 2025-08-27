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

Running MaxText at Scale with XPK
=================================

This guide provides the recommended workflow for running MaxText on Google Kubernetes Engine (GKE) using the **Accelerated Processing Kit (XPK)**. For a complete reference on XPK, please see the [official XPK repository](https://github.com/AI-Hypercomputer/xpk).

### Overview of the Workflow

The process involves two main stages. First, you will package the MaxText application and its dependencies into a self-contained Docker image. This is done on your local machine or any environment where Docker is installed. Second, you will use the XPK command-line tool to orchestrate the deployment of this image as a training job on a GKE cluster equipped with accelerators (TPUs or GPUs).

XPK abstracts away the complexity of cluster management and job submission, handling tasks like uploading your Docker image to Artifact Registry and scheduling the workload on the cluster.

```
+--------------------------+      +--------------------+      +-------------------+
|                          |      |                    |      |                   |
| Your Development Machine +------>  Artifact Registry +------>  GKE Cluster      |
| (anywhere with Docker)   |      | (Stores your image)|      |(with Accelerators)|
|                          |      |                    |      |                   |
| 1. Build MaxText Docker  |      | 2. XPK uploads     |      | 3. XPK runs job   |
|    Image                 |      |    image for you   |      |    using the image|
+--------------------------+      +--------------------+      +-------------------+

```

* * * * *

1\. Prerequisites
-----------------

Before you begin, you must have the necessary tools installed and permissions configured.

### Required Tools

-   **Python >= 3.12** with `pip` and `venv`.

-   **Google Cloud CLI (`gcloud`):** Install it from [here](https://cloud.google.com/sdk/docs/install) and then run `gcloud init`.

-   **kubectl:** The Kubernetes command-line tool.

-   **Docker:** Follow the [installation instructions](https://docs.docker.com/engine/install/) and complete the [post-install steps](https://docs.docker.com/engine/install/linux-postinstall/) to run Docker without `sudo`.

### GCP Permissions

Your Google Cloud user account needs the following IAM roles for the project you're using:

-   Artifact Registry Writer

-   Compute Admin

-   Kubernetes Engine Admin

-   Logging Admin

-   Monitoring Admin

-   Service Account User

-   Storage Admin

-   Vertex AI Administrator

* * * * *

2\. One-Time Environment Setup
------------------------------

These commands configure your local environment to connect to Google Cloud services.

1.  **Authenticate gcloud**


    ```
    gcloud auth login

    ```

2.  **Install GKE Auth Plugin**


    ```
    sudo apt-get update && sudo apt-get install google-cloud-sdk-gke-gcloud-auth-plugin

    ```

3.  **Configure Docker Credentials**


    ```
    gcloud auth configure-docker

    ```

* * * * *

3\. Install XPK
---------------

It is best practice to install XPK in a dedicated Python virtual environment.


```
# Create a virtual environment (only needs to be done once)
python3 -m venv ~/xpk_venv

# Activate the virtual environment (do this every time you open a new terminal)
source ~/xpk_venv/bin/activate

# Install XPK
pip install xpk

```

* * * * *

4\. Build the MaxText Docker Image
----------------------------------

A recommended approach for running MaxText is to build your image from a **JAX AI Image**, which ensures all core libraries are version-matched and stable.

1.  **Clone the MaxText Repository**


    ```
    git clone https://github.com/google/maxtext.git
    cd maxtext

    ```

2.  **Build the Image for your target hardware (TPU or GPU)** This script creates a local Docker image named `maxtext_base_image`. You can find a full list of available base images in the [JAX AI Images documentation](https://cloud.google.com/ai-hypercomputer/docs/images).

    -   **For TPUs:**


        ```
        bash docker_build_dependency_image.sh DEVICE=tpu MODE=jax_ai_image BASEIMAGE=us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.5.2-rev2

        ```

    -   **For GPUs:**


        ```
        bash docker_build_dependency_image.sh DEVICE=gpu MODE=jax_ai_image BASEIMAGE=us-central1-docker.pkg.dev/deeplearning-images/jax-ai-image/gpu:jax0.5.1-cuda_dl25.02-rev1

        ```

* * * * *

5\. Run Your First MaxText Job
------------------------------

This section assumes you have an existing GKE cluster with either TPU or GPU nodes.

> **IMPORTANT NOTE**
>
> This guide focuses on submitting workloads to an existing cluster. Cluster creation and management is a separate topic. For a comprehensive guide on all `xpk` commands, including `xpk cluster create`, please refer to the **[official XPK documentation](https://github.com/AI-Hypercomputer/xpk)**.

1.  **Set Your Configuration**


    ```
    export PROJECT_ID="your-gcp-project-id"
    export ZONE="your-gcp-zone" # e.g., us-central1-a
    export CLUSTER_NAME="your-existing-cluster-name"
    export BASE_OUTPUT_DIR="gs://your-output-bucket/"
    export DATASET_PATH="gs://your-dataset-bucket/"

    ```

2.  **Configure gcloud CLI**


    ```
    gcloud config set project $PROJECT_ID
    gcloud config set compute/zone $ZONE

    ```

#### A Note on Multi-Slice and Multi-Node Runs

The examples below run on a single TPU slice (`--num-slices=1`) or a small number of GPU nodes (`--num-nodes=2`). To scale your job to a larger, multi-host configuration, you simply increase these values.

For instance, to run a job across **four TPU slices**, you would change `--num-slices=1` to `--num-slices=4`. This tells XPK to allocate four `v5litepod-256` slices and orchestrate the training job across all of them as a single workload. Similarly, for GPUs, you would increase the `--num-nodes` value.

3.  **Create the Workload (Run the Job)**

    -   **On your TPU Cluster:**


        ```
        xpk workload create\
          --cluster ${CLUSTER_NAME}\
          --workload ${USER}-tpu-job\
          --base-docker-image maxtext_base_image\
          --tpu-type v5litepod-256\
          --num-slices 1\
          --command "python3 -m MaxText.train MaxText/configs/base.yml run_name=${USER}-tpu-job base_output_directory=${BASE_OUTPUT_DIR} dataset_path=${DATASET_PATH} steps=100"

        ```

    -   **On your GPU Cluster:**


        ```
        xpk workload create\
          --cluster ${CLUSTER_NAME}\
          --workload ${USER}-gpu-job\
          --base-docker-image maxtext_base_image\
          --device-type h100-80gb-8\
          --num-nodes 2\
          --command "python3 -m MaxText.train MaxText/configs/base.yml run_name=${USER}-gpu-job base_output_directory=${BASE_OUTPUT_DIR} dataset_path=${DATASET_PATH} steps=100"

        ```

* * * * *

6\. Managing and Monitoring Your Job
------------------------------------

-   **View Logs in Real-Time:** The easiest way to see the output of your training job is through the Google Cloud Console.

    1.  Navigate to the **Kubernetes Engine** section.

    2.  Go to **Workloads**.

    3.  Find your workload (e.g., `${USER}-tpu-job`) and click on it.

    4.  Select the **Logs** tab to view the container logs.

-   **List Your Jobs:**


    ```
    xpk workload list --cluster ${CLUSTER_NAME}

    ```

-   **Analyze Output:** Checkpoints and other artifacts will be saved to the Google Cloud Storage bucket you specified in `BASE_OUTPUT_DIR`.

-   **Delete a Job:**


    ```
    xpk workload delete --cluster ${CLUSTER_NAME} --workload <your-workload-name>
    ```