# Run MaxText with `gcluster`

This guide provides the recommended workflow for running MaxText on Google Kubernetes Engine (GKE) using `gcluster`. `gcluster` is a command-line tool provided by the Cluster Toolkit to simplify cluster management and job submission for high-performance workloads.

## Overview of the workflow

The process involves two main stages. First, you will package the MaxText application and its dependencies into a self-contained Docker image. Second, you will use the `gcluster job submit` command to orchestrate the deployment of this image as a training job on a GKE cluster.

`gcluster` abstracts away the complexity of cluster management and job submission, handling tasks like creating the required Kubernetes manifests (JobSet) and integrating with Kueue for job scheduling.

```text
+--------------------------+      +--------------------+      +-------------------+
|                          |      |                    |      |                   |
| Your Development Machine +------>  Artifact Registry +------>  GKE Cluster      |
| (anywhere with Docker)   |      | (Stores your image)|      |(with Accelerators)|
|                          |      |                    |      |                   |
| 1. Build MaxText Docker  |      | 2. Uploads image   |      | 3. gcluster runs  |
|    Image                 |      |                    |      |    job using image|
+--------------------------+      +--------------------+      +-------------------+
```

---

## 1. Prerequisites

Before you begin, you must have the necessary tools installed and permissions configured.

### Required tools

*   **gcluster:** Follow the [installation guide](https://docs.cloud.google.com/cluster-toolkit/docs/setup/configure-environment#install).
*   **Python >= 3.12** with `pip` and `venv`.
*   **Google Cloud CLI (`gcloud`):** Install it and run `gcloud init`.
*   **kubectl:** The Kubernetes command-line tool.
*   **Docker:** Follow the installation instructions and configure sudoless Docker.

### Cluster Creation
Before submitting jobs, you need a GKE cluster. If not already available you can create one via `gcluster`.
*   To create a v6e cluster, refer to the [cluster creation guide](https://github.com/GoogleCloudPlatform/cluster-toolkit/blob/main/examples/gke-tpu-v6e/README.md).

### GCP permissions

Your Google Cloud user account needs the necessary IAM roles for the project you’re using:

*   Artifact Registry Writer
*   Compute Admin
*   Kubernetes Engine Admin
*   Logging Admin
*   Monitoring Admin
*   Service Account User
*   Storage Admin
*   Vertex AI Administrator

---

## 2. One-time environment setup

These commands configure your local environment to connect to Google Cloud services.

1.  **Authenticate gcloud**
    ```bash
    gcloud auth login
    ```

2.  **Install GKE auth plugin**
    ```bash
    sudo apt-get update && sudo apt-get install google-cloud-sdk-gke-gcloud-auth-plugin
    ```

3.  **Configure Docker credentials**
    ```bash
    gcloud auth configure-docker
    ```

---

## 3. Build the MaxText Docker image

For instructions on building the MaxText Docker image, please refer to the [MaxText documentation](https://maxtext.readthedocs.io/en/latest/build_maxtext.html).

---

## 4. Run your first MaxText job

This section assumes you have an existing GKE cluster.

### Command Mapping from XPK

If you are familiar with XPK, here is how common `xpk` flags map to `gcluster`:

| XPK Flag | GCluster Equivalent | Description |
| :--- | :--- | :--- |
| `--workload <name>` | `--name <name>` | Unique name for the job. |
| `--cluster <name>` | `--cluster <name>` | Target cluster name. |
| `--device-type <type>` | `--compute-type <type>` | TPU type (e.g., `v6e-8`). |
| `--num-slices <n>` | `--num-slices <n>` | Number of slices. |
| `--command <cmd>` | `--command <cmd>` | Command to run in container. |

1.  **Set your configuration**

    Set up the following environment variables to configure your training run. Replace placeholders with approved values.

    ```bash
    # -- Google Cloud Configuration --
    export PROJECT_ID=<GCP project ID>
    export ZONE=<GCP location> # e.g., 'us-central1' or 'us-central1-a'
    export GKE_CLUSTER=<cluster name>

    # -- Workload Configuration --
    export RUN_NAME="maxtext-run-$(date +%Y%m%d-%H%M%S)"

    # -- MaxText & Storage Configuration --
    export BASE_OUTPUT_DIRECTORY=<gcs bucket path> # e.g., gs://my-bucket/maxtext-runs
    export DATASET_PATH="gs://your-dataset-bucket/"
    export IMAGE_NAME=<full image path> # e.g., us-docker.pkg.dev/my-project/my-repo/maxtext-runner:latest
    ```

2.  **Configure gcluster defaults (Optional)**
    To avoid passing the project, cluster, and location flags with every subsequent command, you can configure them as persistent defaults. These are saved in your local `~/.config/gcluster/` folder, so you only need to set them once:
    ```bash
    gcluster job config set project ${PROJECT_ID}
    gcluster job config set cluster ${GKE_CLUSTER}
    gcluster job config set location ${ZONE}
    ```

3.  **Submit the job**

    Use `gcluster job submit` to submit your job. Assuming you set defaults in the previous step, you can completely omit the `--project`, `--cluster`, and `--location` flags.

    ```bash
    gcluster job submit \
        --name ${RUN_NAME} \
        --image ${IMAGE_NAME} \
        --compute-type v6e-8 \
        --num-slices 1 \
        --priority medium \
        --command "python3 -m maxtext.trainers.pre_train.train maxtext/configs/base.yml run_name=${RUN_NAME} base_output_directory=${BASE_OUTPUT_DIRECTORY} dataset_path=${DATASET_PATH} dataset_type=synthetic steps=30"
    ```

    ### Understanding the `gcluster` flags:
    *   **`--name`:** A unique label inside the cluster for this training run.
    *   **`--image`:** The URI of your built docker runner in Artifact Registry.
    *   **`--compute-type`:** The type of TPU or GPU accelerator slice to use (e.g. `v6e-8` indicates one physical v6e board/host with 8 chips).
    *   **`--num-slices`:** The number of TPU nodes or slices to allocate. Multi-slice scaling is managed automatically behind the scenes.
    *   **`--priority`:** Job queuing priority (integrates directly with the cluster's Kueue scheduler to manage machine slots).
    *   **`--command`:** The training command to spawn in the container entry point.

    *Note: `gcluster` automatically calculates the number of nodes per slice for TPUs based on the provided topology or computing type. You do not need to specify `--num-nodes` for TPU jobs.*

---

## 5. Running with Ahead-of-Time (AOT) Compilation

Ahead-of-Time (AOT) compilation can significantly reduce the startup time of your training job by pre-compiling the JAX training step.

### Step 1: Generate the AOT artifact
Run the `train_compile.py` script to create the compiled artifact for the specified TPU topology.

```bash
python3 -m maxtext.trainers.pre_train.train_compile \
    compile_topology=v6e-8 \
    compile_topology_num_slices=1 \
    compiled_trainstep_file=maxtext_v6e_8_aot.pickle \
    per_device_batch_size=2
```

### Step 2: Re-build and upload your Docker image
Include the generated `maxtext_v6e_8_aot.pickle` file in your Docker image. If you are using a Dockerfile, you can add a line like this:
```dockerfile
COPY maxtext_v6e_8_aot.pickle /app/
```
For more details on building images with AOT, refer to the [MaxText documentation](https://maxtext.readthedocs.io/en/latest/build_maxtext.html).

### Step 3: Submit the job with the AOT artifact
Submit the training run using the custom image containing the pre-compiled graph. Align the topology and slice arguments with the compilation parameters from Step 1, and specify the path to the loaded pickle file within the container (`/app/maxtext_v6e_8_aot.pickle`):

```bash
gcluster job submit \
    --name ${RUN_NAME}-aot \
    --image ${IMAGE_NAME_WITH_AOT} \
    --compute-type v6e-8 \
    --num-slices 1 \
    --command "python3 -m maxtext.trainers.pre_train.train maxtext/configs/base.yml run_name=${RUN_NAME}-aot base_output_directory=${BASE_OUTPUT_DIRECTORY} dataset_path=${DATASET_PATH} dataset_type=synthetic steps=30 compiled_trainstep_file=/app/maxtext_v6e_8_aot.pickle"
```

---

## 6. Managing and monitoring your job

`gcluster` simplifies routine management tasks by handling Kubernetes orchestration underneath. Assuming your defaults are configured, you do not need to supply cluster metadata arguments to query or alter workloads:

*   **List your jobs:**
    Retrieve status lists, active resource allocations, and scheduler queue placements of all your runs:
    ```bash
    gcluster job list
    ```

*   **View logs:**
    Stream standard console outputs (stdout/stderr) of the container without typing complex kubectl queries:
    ```bash
    gcluster job logs ${RUN_NAME}
    ```
    *Tip: Pass the `-f` flag to dynamically follow (stream) logs as their training steps execute.*

*   **Cancel a job:**
    Tear down active workloads and release cluster resources back into Kueue:
    ```bash
    gcluster job cancel ${RUN_NAME}
    ```
