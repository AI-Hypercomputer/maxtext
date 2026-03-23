<!--
 Copyright 2024 Google LLC

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

# Reinforcement Learning on Multi-Host TPUs

This tutorial provides step-by-step instructions for setting up the environment
and training the Llama3.1 70B-IT model on the GSM8K math reasoning dataset using
[Pathways for orchestration](https://cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/pathways-intro)
on multi-host TPU-VMs, such as `v5p-128`.

We utilize two RL algorithms, implemented via the Tunix library, to enhance the
model's reasoning capabilities:

- **Group Relative Policy Optimization (GRPO)**: GRPO is an RL algorithm
  designed to enhance the reasoning abilities of LLMs. It is a variant of
  Proximal Policy Optimization (PPO) that reduces memory usage by eliminating
  the need for a separate value function model. GRPO works by generating
  multiple responses for a given prompt, evaluating these responses using a
  reward model, and then calculating a relative advantage based on the group's
  performance to update the policy.

- **Group Sequence Policy Optimization (GSPO)**: GSPO is an RL algorithm that
  improves training efficiency and performance of LLMs by using sequence-level
  importance ratios and operations. GSPO defines the importance ratio based on
  sequence likelihood and performs sequence-level clipping, rewarding, and
  optimization.

For efficient model inference and response generation during this process, we
rely on the vLLM library.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Build and Upload MaxText Docker Image](#build-and-upload-maxtext-docker-image)
- [Setup Environment Variables](#setup-environment-variables)
- [Get Your Model Checkpoint](#get-your-model-checkpoint)
- [Submit your RL workload via Pathways](#submit-your-rl-workload-via-pathways)
- [Managing Workloads](#managing-workloads)
- [Troubleshooting](#troubleshooting)

## Prerequisites

Before starting, ensure you have:

- Access to a Google Cloud Project with TPU quotas.
- A Hugging Face account with an access token for downloading models.
- Permissions for Google Artifact Registry (Artifact Registry Writer role).
- Prerequisites for XPK installed (follow [official documentation](https://github.com/AI-Hypercomputer/xpk/blob/main/docs/installation.md#1-prerequisites)).
- A Pathways-ready GKE cluster (see [create GKE cluster](https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/create-gke-cluster)).
- **Docker** installed and configured for sudoless use. Follow the steps to [configure sudoless Docker](https://docs.docker.com/engine/install/linux-postinstall/).

## Build and upload MaxText Docker image

For instructions on building and uploading the MaxText Docker image with post-training dependencies, please refer to the [official documentation](https://maxtext.readthedocs.io/en/latest/build_maxtext.html).

## Setup Environment Variables

Set up the following environment variables. Replace placeholders with your
actual values.

```bash
# -- Model configuration --
export MODEL=<MaxText Model> # e.g. 'llama3.1-70b-Instruct'
export HF_TOKEN=<Hugging Face access token>

# -- MaxText configuration --
export BASE_OUTPUT_DIRECTORY=<output directory to store run logs> # e.g., gs://my-bucket/my-output-directory
export WORKLOAD=<Name for this run> # e.g., llama-3-70b-grpo
export MAXTEXT_CKPT_PATH=${BASE_OUTPUT_DIRECTORY?}/${WORKLOAD?}/0/items

# -- Workload configuration --
export TPU_TYPE=<TPU Type> # e.g., 'v5p-128'
export TPU_CLUSTER=<cluster name>
export PROJECT_ID=<GCP project ID>
export ZONE=<GCP zone>
```

## Get Your Model Checkpoint

### Option 1: Using an existing MaxText checkpoint

If you already have a MaxText-compatible model checkpoint, simply set the
following environment variable and move on to the next section.

```bash
export MAXTEXT_CKPT_PATH=<gcs path for MaxText checkpoint> # e.g., gs://my-bucket/my-model-checkpoint/0/items
```

### Option 2: Converting from a Hugging Face checkpoint

Refer the steps in [Hugging Face to MaxText](../../guides/checkpointing_solutions/convert_checkpoint.md#hugging-face-to-maxtext) to convert a hugging face checkpoint to MaxText. Make sure you have correct checkpoint files converted and saved. Similar as Option 1, you can set the following environment and move on.

```bash
export MAXTEXT_CKPT_PATH=<gcs path for MaxText checkpoint> # e.g., gs://my-bucket/my-model-checkpoint/0/items
```

## Submit your RL workload via Pathways

See the **Troubleshooting** section for concise instructions on how to retry or
resume a failed workload.

Ensure you have a Pathways-ready GKE cluster (as mentioned in Prerequisites) and
submit the `train_rl.py` script via XPK.

> **Note:** XPK v0.14.0+ automatically discovers your cluster's location from
> GCP. You don't need to specify `--zone` in the commands below. If using an
> older XPK version, add `--zone=<zone>` to the workload commands.

### Submit GRPO workload

```bash
xpk workload create-pathways --workload ${WORKLOAD?} \
--docker-image gcr.io/${PROJECT_ID?}/${CLOUD_IMAGE_NAME?} --cluster ${TPU_CLUSTER?} \
--tpu-type=${TPU_TYPE?} --num-slices=1 \
--project=${PROJECT_ID?} --priority=high \
--zone=${ZONE?} \
--command "HF_TOKEN=${HF_TOKEN?} TF_CPP_MIN_LOG_LEVEL=0 JAX_PLATFORMS=proxy JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 ENABLE_PATHWAYS_PERSISTENCE='1' \
python3 -m maxtext.trainers.post_train.rl.train_rl \
  model_name=${MODEL?} \
  load_parameters_path=${MAXTEXT_CKPT_PATH?} \
  run_name=${WORKLOAD?} \
  base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
  hf_access_token=${HF_TOKEN?}"
```

### Submit GSPO workload

```bash
xpk workload create-pathways --workload ${WORKLOAD?} \
--docker-image gcr.io/${PROJECT_ID?}/${CLOUD_IMAGE_NAME?} --cluster ${TPU_CLUSTER?} \
--tpu-type=${TPU_TYPE?} --num-slices=1 \
--project=${PROJECT_ID?} --priority=high \
--zone=${ZONE?} \
--command "HF_TOKEN=${HF_TOKEN?} TF_CPP_MIN_LOG_LEVEL=0 JAX_PLATFORMS=proxy JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 ENABLE_PATHWAYS_PERSISTENCE='1' \
python3 -m maxtext.trainers.post_train.rl.train_rl \
  model_name=${MODEL?} \
  load_parameters_path=${MAXTEXT_CKPT_PATH?} \
  run_name=${WORKLOAD?} \
  base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
  hf_access_token=${HF_TOKEN?} \
  loss_algo=gspo-token"
```

## Managing Workloads

- **Monitor workload status**: Check Pathways job status: `kubectl get pathwaysjob`. Check pod status: `kubectl get pods`.
- **Delete a workload**: To remove a failed or unwanted Pathways job, use XPK:
  ```bash
  xpk workload delete \
      --workload ${WORKLOAD?} \
      --cluster ${TPU_CLUSTER?} \
      --project ${PROJECT_ID?}
  ```
  In case the job still lingers on, you can use
  `kubectl get pods` to obtain the name of the pod and then run: `kubectl delete pod <pod-name>`.

## Troubleshooting

- **Authentication Issues**: Ensure your `HF_TOKEN` environment variable is
  set correctly and has access to the required models.
- **Resource Quotas**: Verify you have sufficient TPU quotas in your GCP
  project.
- **Docker Build Failures**: Check that all dependencies are correctly
  installed and authentication is configured.
- **Workload Failures**: Review the logs for specific error messages and
  ensure all environment variables are properly set.
- **Workload retry / resume**:
  - **Retry (fresh run)**: Use a unique workload name to avoid overwriting
    outputs: `export WORKLOAD=${WORKLOAD}-retry1 export MAXTEXT_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/${WORKLOAD}/0/items`. Then
    submit the XPK workload. If "workload already exists" error occurs, pick
    a new name or list jobs: `kubectl get pathwaysjob`.
  - **Resume from checkpoint**: Keep the same `WORKLOAD` and set the
    checkpoint path: `export load_parameters_path=${MAXTEXT_CKPT_PATH}/checkpoint-0000`. Then submit
    the workload again.
  - **Tip**: Verify the checkpoint exists in GCS with read access before
    resuming.

For more detailed troubleshooting, refer to the
[MaxText documentation](https://maxtext.readthedocs.io) and
[XPK documentation](https://github.com/AI-Hypercomputer/xpk).
