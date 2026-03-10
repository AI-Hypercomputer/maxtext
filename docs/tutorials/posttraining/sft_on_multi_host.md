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
- A Hugging Face account with an access token for downloading models.
- Permissions for Google Artifact Registry (Artifact Registry Writer role).
- Prerequisites for XPK installed (follow [official documentation](https://github.com/AI-Hypercomputer/xpk/blob/main/docs/installation.md#1-prerequisites)).
- A Pathways-ready GKE cluster (see [create GKE cluster](https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/create-gke-cluster)).
- **Docker** installed and configured for sudoless use. Follow the steps to [configure sudoless Docker](https://docs.docker.com/engine/install/linux-postinstall/).

## Build and upload MaxText Docker image

For instructions on building and uploading the MaxText Docker image with post-training dependencies, please refer to the [official documentation](https://maxtext.readthedocs.io/en/latest/build_maxtext.html).

## Create GKE cluster

Use a pathways ready GKE cluster as described [here](https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/create-gke-cluster).

## Environment configuration

```bash
# -- Google Cloud Configuration --
export PROJECT=<Google Cloud Project ID>
export CLUSTER_NAME=<Name of GKE Cluster>
export ZONE=<GKE Cluster Zone>

# -- Workload Configuration --
export WORKLOAD_NAME=<Name of Workload> # e.g., sft-$(date +%s)
export TPU_TYPE=<TPU Type> # e.g., v6e-256
export TPU_SLICE=<number of slices>

# -- MaxText Configuration --
export OUTPUT_PATH=<GCS Path for Output/Logs> # e.g., gs://my-bucket/my-output-directory
export STEPS=<Fine-Tuning Steps> # e.g., 1000
export HF_TOKEN=<Hugging Face Access Token>

# -- Model Configuration --
export MODEL_NAME=<Model Name> # e.g., deepseek3-671b

# -- Dataset configuration --
export DATASET_NAME=<Hugging Face Dataset Name> # e.g., HuggingFaceH4/ultrachat_200k
export TRAIN_SPLIT=<Data Split for Train> # e.g., train_sft
export TRAIN_DATA_COLUMNS=<Data Columns to Train on> # e.g., ['messages']
```

## Get MaxText model checkpoint

This section explains how to prepare your model checkpoint for use with MaxText. You have two options: using an existing MaxText checkpoint or converting a Hugging Face checkpoint.

### Option 1: Using an existing MaxText checkpoint

If you already have a MaxText-compatible model checkpoint, simply set the following environment variable and move on to the next section.

```bash
export MAXTEXT_CKPT_PATH=<gcs path for MaxText checkpoint> # e.g., gs://my-bucket/my-model-checkpoint/0/items
```

**Note:** Make sure that `MAXTEXT_CKPT_PATH` has the checkpoints created using the correct storage flags:

```
export USE_PATHWAYS=0  # Set to 1 for Pathways, 0 for McJAX.
checkpoint_storage_use_zarr3=$((1 - USE_PATHWAYS))
checkpoint_storage_use_ocdbt=$((1 - USE_PATHWAYS))
```

### Option 2: Converting a Hugging Face checkpoint

Refer the steps in [Hugging Face to MaxText](../../guides/checkpointing_solutions/convert_checkpoint.md#hugging-face-to-maxtext) to convert a hugging face checkpoint to MaxText. Make sure you have correct checkpoint files converted and saved. Similar as Option 1, you can set the following environment and move on.

```bash
export MAXTEXT_CKPT_PATH=<gcs path for MaxText checkpoint> # gs://my-bucket/my-checkpoint-directory/0/items
```

## Submit workload on GKE cluster

This section provides the command to run SFT on a GKE cluster.

### SFT with Multi-Controller JAX (McJAX)

```bash
xpk workload create \
--cluster=${CLUSTER_NAME?} \
--project=${PROJECT?} \
--zone=${ZONE?} \
--docker-image=gcr.io/${PROJECT_ID?}/${CLOUD_IMAGE_NAME?} \
--workload=${WORKLOAD_NAME?} \
--tpu-type=${TPU_TYPE?} \
--num-slices=${TPU_SLICE?} \
--command "python3 -m maxtext.trainers.post_train.sft.train_sft src/maxtext/configs/post_train/sft.yml run_name=${WORKLOAD_NAME?} base_output_directory=${OUTPUT_PATH?} model_name=${MODEL_NAME?} load_parameters_path=${MODEL_CHECKPOINT_PATH?} hf_access_token=${HF_TOKEN?} per_device_batch_size=1 steps=${STEPS?} profiler=xplane hf_path=${DATASET_NAME?} train_split=${TRAIN_SPLIT?} train_data_columns=${TRAIN_DATA_COLUMNS?}"
```

Once the fine-tuning is completed, you can access your model checkpoints at `$OUTPUT_PATH/$WORKLOAD_NAME/checkpoints`.

### SFT with Pathways

```bash
export USE_PATHWAYS=1

xpk workload create-pathways \
--cluster=${CLUSTER_NAME?} \
--project=${PROJECT?} \
--zone=${ZONE?} \
--docker-image=gcr.io/${PROJECT_ID?}/${CLOUD_IMAGE_NAME?} \
--workload=${WORKLOAD_NAME?} \
--tpu-type=${TPU_TYPE?} \
--num-slices=${TPU_SLICE?} \
--command="JAX_PLATFORMS=proxy JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 ENABLE_PATHWAYS_PERSISTENCE=1 python3 -m maxtext.trainers.post_train.sft.train_sft src/maxtext/configs/post_train/sft.yml run_name=${WORKLOAD_NAME?} base_output_directory=${OUTPUT_PATH?} model_name=${MODEL_NAME?} load_parameters_path=${MODEL_CHECKPOINT_PATH?} hf_access_token=${HF_TOKEN?} per_device_batch_size=1 steps=${STEPS?} profiler=xplane checkpoint_storage_use_zarr3=False checkpoint_storage_use_ocdbt=False enable_single_controller=True"
```

Once the fine-tuning is completed, you can access your model checkpoints at `$OUTPUT_PATH/$WORKLOAD_NAME/checkpoints`.
