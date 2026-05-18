<!--
 Copyright 2023-2026 Google LLC

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

# Reinforcement Learning with Qwen3-30b-a3b on Multi-Host TPUs

This tutorial provides step-by-step instructions for setting up the environment
and training the Qwen3-30b-a3b model on the [OpenMathInstruct-2 dataset](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) on Ironwood GKE cluster with `tpu7x-128` nodes.

## Prerequisites

Before starting, ensure you have:

- Access to a Google Cloud Project with TPU quotas.
- A Hugging Face account with an access token for downloading models.
- Permissions for Google Artifact Registry (Artifact Registry Writer role).
- Prerequisites for XPK installed (follow [official documentation](https://github.com/AI-Hypercomputer/xpk/blob/main/docs/installation.md#1-prerequisites)).
- A Pathways-ready GKE cluster (see [create GKE cluster](https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/create-gke-cluster)).
- **Docker** installed and configured for sudoless use. Follow the steps to [configure sudoless Docker](https://docs.docker.com/engine/install/linux-postinstall/).

## Build and Upload MaxText Docker Image

For instructions on building and uploading the MaxText Docker image with post-training dependencies, please refer to the [official documentation](../../build_maxtext.md).

## Setup Environment Variables

Set up the following environment variables to configure your training run. Replace
placeholders with your actual values.

```bash
# Your GCP project ID.
# If you've already set it in your local config, you can retrieve it via:
# gcloud config get-value project
export PROJECT_ID=<PROJECT_ID>

# The name of your GKE cluster.
export CLUSTER_NAME=<CLUSTER_NAME>

# The GCP location of your GKE cluster.
export ZONE=<ZONE> # e.g., 'us-central1' or 'us-central1-a'

# Use a GCS bucket you own to store logs and checkpoints.
export BASE_OUTPUT_DIRECTORY=<GCS_BUCKET> # e.g., gs://my-bucket/maxtext-runs

# The Docker image you pushed in the previous step
export CLOUD_IMAGE_NAME=<IMAGE_NAME>
export DOCKER_IMAGE="gcr.io/${PROJECT_ID?}/${CLOUD_IMAGE_NAME?}"
```

# Clone MaxText Repository

If you haven't already, clone the MaxText repository to your local machine:

```bash
git clone https://github.com/AI-Hypercomputer/maxtext.git
cd maxtext
```

## Get Your MaxText Compatible Model Checkpoint

### Option 1: Using an existing MaxText checkpoint

If you already have a MaxText-compatible model checkpoint, simply set the
following environment variable and move on to the next section.

```bash
export MAXTEXT_CKPT_PATH=<CKPT_PATH> # e.g., gs://my-bucket/my-model-checkpoint/0/items
```

### Option 2: Converting from a Hugging Face checkpoint

> **Note:** Converting the 30B model requires approximately 62 GB of free disk space to download its safetensors. Please verify you have sufficient space before running the conversion script.

```bash
# Optional: If you run out of disk space when downloading Hugging Face safetensors,
# customize your "HF_HOME" to redirect the cache to a larger or mounted disk (e.g., on a TPU VM).
# export HF_HOME="/dev/shm/huggingface_tmp"

# Create and activate a virtual environment
uv venv --python 3.12 --seed tpu_venv
source tpu_venv/bin/activate
uv pip install -e .[tpu] --resolution=lowest

# Authenticate with Hugging Face
hf auth login

# Run the conversion script to convert the Hugging Face checkpoint to MaxText format
bash scripts/run_qwen3_30b_hf_to_maxtext.sh

# Deactivate the virtual environment
deactivate
rm -rf tpu_venv
```

## Run RL Workload

### Submit your workload

```bash
# Create and activate a virtual environment
uv venv --python 3.12 --seed runner_venv
source runner_venv/bin/activate
uv pip install -e .[runner] --resolution=lowest

# Run the RL training script on your cluster
bash scripts/run_qwen3_30b_rl.sh

# Deactivate the virtual environment
deactivate
rm -rf runner_venv
```

### Monitor your workload

To monitor your job's progress, you can use `kubectl` to check the `Jobset` status and stream logs directly from the pods.

```bash
kubectl get jobset -n default ${WORKLOAD_NAME}

# List pods to find the specific name
kubectl get pods | grep ${WORKLOAD_NAME}

# stream the logs from the running pod (replace <POD_NAME> with the name you found)
kubectl logs -f <POD_NAME>
```

Alternatively, after running the bash script, you will also get a link to the Google Cloud Console to view your workload logs. Follow the link to view logs and monitor your workload's progress in the Cloud Console.

## Convert Checkpoint to Hugging Face Format

After training, you may want to convert your MaxText checkpoint back to Hugging Face format. Use the following script to perform the conversion:

```bash
# Create and activate a virtual environment
uv venv --python 3.12 --seed tpu_venv
source tpu_venv/bin/activate
uv pip install -e .[tpu] --resolution=lowest

# Authenticate with Hugging Face
hf auth login

# Run the conversion script to convert the MaxText checkpoint back to Hugging Face format 
bash scripts/run_qwen3_30b_maxtext_to_hf.sh

# Deactivate the virtual environment
deactivate
rm -rf tpu_venv
```
