<!--
 # Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
 -->

# SFT on multi-host TPUs

Supervised fine-tuning (SFT) is a process where a pre-trained large language model is fine-tuned on a labeled dataset to adapt the model to perform better on specific tasks.

This tutorial demonstrates step-by-step instructions for setting up the multi-host TPU environment and then training the model on the Hugging Face dataset using SFT. In this tutorial we use a multi-host TPU such as `v6e-256`.

We use [Tunix](https://github.com/google/tunix), a JAX-based library designed for post-training tasks, to perform SFT.

Let's get started!

## Prerequisites

Before starting, ensure you have:

- Access to a Google Cloud Project with TPU quotas.
- **IAM Roles** required:
  - **Kubernetes Engine Developer** (`roles/container.developer`) to submit and manage workloads on GKE.
  - **Artifact Registry Writer** (`roles/artifactregistry.writer`) to upload Docker images.
  - **Storage Admin** (`roles/storage.admin`) or **Storage Object Admin** (`roles/storage.objectAdmin`) combined with **Storage Legacy Bucket Reader** (`roles/storage.legacyBucketReader`) on your GCS bucket to read/write checkpoints and logs. (Note: A bucket-level read permission like `storage.buckets.get` is required by JAX/TensorStore to verify bucket existence and metadata; using `roles/storage.objectAdmin` alone will cause a misleading "bucket not found" error).
- A Hugging Face account with an access token for downloading models.
- Cluster Toolkit installed and configured. Follow [Running MaxText with Cluster Toolkit](../../run_maxtext/run_maxtext_via_cluster_toolkit.md) for `gcluster` setup.
  - **Important:** Modern GKE clusters require the GKE auth plugin. If you encounter `gke-gcloud-auth-plugin not found` when running `kubectl` commands, you must install it locally (e.g., `sudo apt-get install google-cloud-cli-gke-gcloud-auth-plugin` for `apt` installations, or `gcloud components install gke-gcloud-auth-plugin` for standalone archive installations).
- A GKE cluster configured for Cluster Toolkit, including healthy Kueue and JobSet components.
- **Docker** installed and configured for sudoless use. Follow the steps to [configure sudoless Docker](https://docs.docker.com/engine/install/linux-postinstall/).

## Build and upload MaxText Docker image

For instructions on building and uploading the MaxText Docker image with post-training dependencies, please refer to the [official documentation](build-docker).

## Configure GKE cluster with Cluster Toolkit

Configure access to the target cluster with `gcloud`, then configure the project, cluster, and location with `gcluster` as described in [Running MaxText with Cluster Toolkit](../../run_maxtext/run_maxtext_via_cluster_toolkit.md):

```bash
gcloud container clusters get-credentials ${GKE_CLUSTER?} \
  --zone ${ZONE?} \
  --project ${PROJECT_ID?}
gcluster job config set project ${PROJECT_ID?}
gcluster job config set cluster ${GKE_CLUSTER?}
gcluster job config set location ${ZONE?}
```

## Environment configuration

Set up the following environment variables to configure your training run. Replace
placeholders with your actual values.

```bash
# -- Model configuration --
# The MaxText model name. See `src/maxtext/configs/types.py` for `ModelName` for a
# full list of supported models.
export MODEL=<MODEL_NAME> # e.g., deepseek3-671b

# Your Hugging Face access token. Required to download gated models like Llama.
# You can generate one at https://huggingface.co/settings/tokens.
export HF_TOKEN=<HF_TOKEN>

# -- MaxText configuration --
# Use a GCS bucket you own to store logs and checkpoints. Ideally in the same
# region as your TPUs to minimize latency and costs.
# You can list your buckets and their locations in the
# [Cloud Console](https://console.cloud.google.com/storage/browser) or via
# `gcloud storage buckets list --format="table(name, location)"`.
export BASE_OUTPUT_DIRECTORY=<GCS_BUCKET> # e.g., gs://my-bucket/maxtext-runs

# An arbitrary string to identify this specific run.
# Note: Workload names cannot exceed 28 characters and must be valid DNS labels (lowercase alphanumeric and hyphens).
export RUN_NAME="sft-$(date +%m%d%H%M%S)"

# -- Workload configuration --
# Your GCP project ID. Find it on the [Cloud Console Dashboard](https://console.cloud.google.com/home/dashboard).
# If you've already set it in your local config, you can retrieve it via:
# gcloud config get-value project
export PROJECT_ID=<PROJECT_ID>

# The GCP location (listed as "Location" in the UI) and name of your
# TPU-enabled GKE cluster. Both can be found on the
# [Cloud Console](https://console.cloud.google.com/kubernetes/list).
export ZONE=<ZONE> # e.g., 'us-central1'
export GKE_CLUSTER=<CLUSTER_NAME>

# For a full list of MaxText-supported TPU types, see: `src/maxtext/utils/accelerator_to_spec_map.py`. To see the TPU type
# of your cluster:

# 1. Connect to the cluster (required for kubectl commands later):
# gcloud container clusters get-credentials ${GKE_CLUSTER?} --zone ${ZONE?} --project ${PROJECT_ID?}

# 2. Find your TPU type (e.g., 'v5p-128') by checking the accelerator labels on your nodes:
# kubectl get nodes -l cloud.google.com/gke-tpu-accelerator -o jsonpath='{.items[*].metadata.labels.cloud\.google\.com/gke-tpu-accelerator}' | tr ' ' '\n' | sort -u
export TPU_TYPE=<TPU_TYPE>
export NUM_SLICES=<NUM_SLICES>

# Cluster Toolkit workload placement. See the Cluster Toolkit guide for the
# compute type and topology matching your TPU slice.
export COMPUTE_TYPE=<COMPUTE_TYPE>
export TOPOLOGY=<TOPOLOGY>

# The Docker image you pushed in the prerequisite step
export CLOUD_IMAGE_NAME=<IMAGE_NAME>
export DOCKER_IMAGE="gcr.io/${PROJECT_ID?}/${CLOUD_IMAGE_NAME?}"

# -- Fine-Tuning configuration --
export STEPS=<STEPS> # e.g., 1000

# -- Dataset configuration --
export DATASET_NAME=<DATASET_NAME> # e.g., HuggingFaceH4/ultrachat_200k
export TRAIN_SPLIT=<TRAIN_SPLIT> # e.g., train_sft
export TRAIN_DATA_COLUMNS=<DATA_COLUMNS> # e.g., ['messages']
```

## Get MaxText model checkpoint

This section explains how to prepare your model checkpoint for use with MaxText. You have two options: using an existing MaxText checkpoint or converting a Hugging Face checkpoint.

### Option 1: Using an existing MaxText checkpoint

If you already have a MaxText-compatible model checkpoint, simply set the following environment variable and move on to the next section.

```bash
export MAXTEXT_CKPT_PATH=<CKPT_PATH> # e.g., gs://my-bucket/my-model-checkpoint/0/items
```

**Note:** Make sure that `MAXTEXT_CKPT_PATH` has the checkpoints created using the correct storage flags:

```
export USE_PATHWAYS=0  # Set to 1 for Pathways, 0 for McJAX.
checkpoint_storage_use_zarr3=$((1 - USE_PATHWAYS))
checkpoint_storage_use_ocdbt=$((1 - USE_PATHWAYS))
```

### Option 2: Converting a Hugging Face checkpoint

Refer the steps in [Hugging Face to MaxText](hf-to-maxtext) to convert a hugging face checkpoint to MaxText. Make sure you have correct checkpoint files converted and saved. Similar as Option 1, you can set the following environment and move on.

```bash
export MAXTEXT_CKPT_PATH=<CKPT_PATH> # gs://my-bucket/my-checkpoint-directory/0/items
```

> [!IMPORTANT]
> **Automatic `scan_layers` Resolution:**
> MaxText automatically loads `scan_layers` from the checkpoint's saved metadata when resuming (via `load_parameters_path`) if you do not explicitly specify it on the command-line.
>
> - You do not need to manually supply `scan_layers=False` (or `scan_layers=True`) when loading checkpoints; MaxText will configure this automatically.
> - If you do explicitly provide a `scan_layers` argument, it must match the checkpoint's saved setting or a `ValueError` mismatch error will be raised.
>   See the [Checkpoints concept guide](../../reference/core_concepts/checkpoints.md) for more details.

## Submit workload on GKE cluster

This section provides the command to run SFT on a GKE cluster.

### SFT with Multi-Controller JAX (McJAX)

```bash
gcluster job submit \
--image=${DOCKER_IMAGE?} \
--command "python3 -m maxtext.trainers.post_train.sft.train_sft run_name=${RUN_NAME?} base_output_directory=${BASE_OUTPUT_DIRECTORY?} model_name=${MODEL?} load_parameters_path=${MAXTEXT_CKPT_PATH?} hf_access_token=${HF_TOKEN?} per_device_batch_size=1 steps=${STEPS?} profiler=xplane hf_path=${DATASET_NAME?} train_split=${TRAIN_SPLIT?} train_data_columns=${TRAIN_DATA_COLUMNS?}" \
--name=${RUN_NAME?} \
--compute-type=${COMPUTE_TYPE?} \
--topology=${TOPOLOGY?}
```

Once the fine-tuning is completed, you can access your model checkpoints at `${BASE_OUTPUT_DIRECTORY}/${RUN_NAME}/checkpoints`.

### SFT with Pathways

To submit an SFT workload with Pathways using Cluster Toolkit, use `gcluster job submit` with the `--pathways` flag:

```bash
export USE_PATHWAYS=1

gcluster job submit \
  --image=${DOCKER_IMAGE?} \
  --name=${RUN_NAME?} \
  --pathways \
  --compute-type=${COMPUTE_TYPE?} \
  --topology=${TOPOLOGY?} \
  --num-slices=${NUM_SLICES:-1} \
  --pathways-gcs-location=${BASE_OUTPUT_DIRECTORY?} \
  --command="python3 -m maxtext.trainers.post_train.sft.train_sft \
    run_name=${RUN_NAME?} \
    base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    model_name=${MODEL?} \
    load_parameters_path=${MAXTEXT_CKPT_PATH?} \
    hf_access_token=${HF_TOKEN?} \
    per_device_batch_size=1 \
    steps=${STEPS?} \
    profiler=xplane \
    checkpoint_storage_use_zarr3=$((1 - USE_PATHWAYS)) \
    checkpoint_storage_use_ocdbt=$((1 - USE_PATHWAYS)) \
    enable_single_controller=True"
```

Once the fine-tuning is completed, you can access your model checkpoints at `${BASE_OUTPUT_DIRECTORY}/${RUN_NAME}/checkpoints`.

## Monitor and clean up
 
```bash
gcluster job list
# Note: For Pathways workloads (> 5 pods), specify --main-only=false to retrieve logs from all pods:
gcluster job logs ${RUN_NAME?} --main-only=false
gcluster job cancel ${RUN_NAME?}
```

You can also inspect the Kubernetes resources directly:

```bash
kubectl get jobset -l gcluster.google.com/workload=${RUN_NAME?}
# In Pathways workloads, use the jobset-name label to select all pods (both pathways-head and worker pods):
kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${RUN_NAME?}
```
