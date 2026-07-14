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

# Reinforcement Learning with GPT-OSS 20B on Multi-Host TPUs

This tutorial provides step-by-step instructions for setting up the environment
and training the GPT-OSS 20B model on the [GSM8K dataset](https://huggingface.co/datasets/openai/gsm8k) on an Viperfish GKE cluster with `v5p-64` nodes.

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

## Clone MaxText Repository

If you haven't already, clone the MaxText repository to your local machine:

```bash
git clone https://github.com/AI-Hypercomputer/maxtext.git
cd maxtext
```

## Authenticate with Hugging Face

To download the `gpt-oss-20b` model checkpoint from Hugging Face, you need to authenticate using your Hugging Face account credentials. Run the following command and follow the prompts to log in:

```bash
hf auth login
```

## Get Your MaxText Compatible Model Checkpoint

### Option 1: Using an existing MaxText checkpoint

If you already have a MaxText-compatible model checkpoint, simply set the
following environment variable and move on to the next section.

```bash
export MAXTEXT_CKPT_PATH=<CKPT_PATH> # e.g., gs://my-bucket/my-model-checkpoint/0/items
```

### Option 2: Converting from a Hugging Face checkpoint

> **Note:** Converting the 20B model requires approximately 40 GB of free disk space to download its safetensors. Please verify you have sufficient space before running the conversion.

```bash
# Create and activate a virtual environment
uv venv --python 3.12 --seed tpu_venv
source tpu_venv/bin/activate
uv pip install -e .[tpu] --resolution=lowest

# Install torch for conversion
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# Download Hugging Face weights to a local directory
huggingface-cli download openai/gpt-oss-20b --local-dir ./gpt-oss-20b-hf

# Set up the MaxText checkpoint path environment variable where the converted checkpoint will be saved
export MAXTEXT_CKPT_PATH="${BASE_OUTPUT_DIRECTORY}/checkpoints/gpt-oss-20b-converted/0/items"

# Run the conversion script to convert the Hugging Face checkpoint to MaxText format
python3 -m maxtext.checkpoint_conversion.standalone_scripts.convert_gpt_oss_ckpt \
    --base-model-path ./gpt-oss-20b-hf \
    --maxtext-model-path ${MAXTEXT_CKPT_PATH} \
    --model-size gpt-oss-20b \
    --use-ocdbt=False --use-zarr3=False

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
bash scripts/run_gptoss_20b_rl.sh

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

### Monitor RL Metrics

During RL training, you can monitor key metrics to track model convergence, reward trends, and hardware performance.

To enable Tunix-managed metrics measurement, set `enable_tunix_perf_metrics` to `true` in `src/maxtext/configs/post_train/rl.yml`. Note that this flag is already set to `True` by default in the [scripts/run_gptoss_20b_rl.sh](../../../scripts/run_gptoss_20b_rl.sh) script for this tutorial workload. When enabled, Tunix automatically collects and uploads these metrics to TensorBoard.

For a complete list of collected metrics, see the [Tunix Metrics Documentation](https://tunix.readthedocs.io/en/latest/metrics.html). Key metrics to monitor include:

- **Model Quality & Reward Metrics:**
  - `rewards/mean`: The average reward across the batch (crucial for tracking learning progress).
  - `score/mean`: The average raw score from the reward model before applying the KL penalty.
- **Rollout & Generation Metrics:**
  - `rollout_time`: How long each rollout step takes.
  - `completions/mean_length`: The average token length of generated completions.
  - `actor_dequeue_time`: The time spent waiting for data from the rollout workers (relevant when async rollout is enabled).
- **Performance & Efficiency Metrics:**
  - `step_time_sec`: The execution time for a single training step.

## Convert Checkpoint to Hugging Face Format

After training, you may want to convert your MaxText checkpoint back to Hugging Face format. Use the following command to perform the conversion:

```bash
# Create and activate a virtual environment
uv venv --python 3.12 --seed tpu_venv
source tpu_venv/bin/activate
uv pip install -e .[tpu] --resolution=lowest

# Install torch for conversion
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# Define the output path for the converted checkpoint
export HF_CKPT_OUTPUT_DIR="${BASE_OUTPUT_DIRECTORY}/checkpoints/gpt-oss-20b-hf-converted"

# Run the conversion script to convert the MaxText checkpoint back to Hugging Face format 
python3 -m maxtext.checkpoint_conversion.to_huggingface \
    src/maxtext/configs/base.yml \
    model_name=gpt-oss-20b \
    load_parameters_path="${MAXTEXT_CKPT_PATH}" \
    base_output_directory="${HF_CKPT_OUTPUT_DIR}" \
    scan_layers=True \
    weight_dtype=bfloat16 hardware=cpu skip_jax_distributed_system=True

# Deactivate the virtual environment
deactivate
rm -rf tpu_venv
```
