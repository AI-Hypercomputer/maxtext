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

# LoRA Fine-tuning on multi-host TPUs

**Low-Rank Adaptation (LoRA)** is a Parameter-Efficient Fine-Tuning (PEFT) technique designed to optimize large language models while minimizing resource consumption.

Unlike traditional full-parameter fine-tuning, LoRA:

- **Freezes the pre-trained model weights**, preserving the original knowledge.
- **Injects trainable rank decomposition matrices** into the Transformer layers.

This tutorial provides step-by-step instructions for setting up the multi-host TPU environment and performing LoRA fine-tuning on a Hugging Face dataset using MaxText. In this tutorial we use a multi-host TPU such as `v6e-256`.

We use [Tunix](https://github.com/google/tunix), a JAX-based library, to power these post-training tasks.

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

Set up the following environment variables to configure your training run. Replace placeholders with your actual values.

```bash
# -- Model configuration --
# The MaxText model name. See `src/maxtext/configs/types.py` for `ModelName` for a
# full list of supported models.
export MODEL=<MODEL_NAME> # e.g., 'gemma4-26b'

# Your Hugging Face access token. Required to download gated models like Gemma.
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
# We recommend to include the model, user, and timestamp.
# Note: Kubernetes requires workload names to be valid DNS labels (lowercase, no underscores or periods).
export RUN_NAME=<RUN_NAME>

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
# gcloud container clusters get-credentials ${GKE_CLUSTER?} --location ${ZONE?} --project ${PROJECT_ID?}

# 2. Find your TPU type (e.g., 'v6e-256') by checking the accelerator labels on your nodes:
# kubectl get nodes -l cloud.google.com/gke-tpu-accelerator -o jsonpath='{.items[*].metadata.labels.cloud\.google\.com/gke-tpu-accelerator}' | tr ' ' '\n' | sort -u
export TPU_TYPE=<TPU_TYPE>
export NUM_SLICES=<NUM_SLICES>

# The Docker image you pushed in the prerequisite step
export CLOUD_IMAGE_NAME=<IMAGE_NAME>
export DOCKER_IMAGE="gcr.io/${PROJECT_ID?}/${CLOUD_IMAGE_NAME?}"

# -- Fine-Tuning configuration --
export STEPS=<STEPS> # e.g., 1000
export PER_DEVICE_BATCH_SIZE=<BATCH_SIZE_PER_DEVICE> # e.g., 1
export LORA_RANK=<LORA_RANK> # e.g., 16
export LORA_ALPHA=<LORA_ALPHA> # e.g., 32.0
export LEARNING_RATE=<LEARNING_RATE> # e.g., 3e-6
export MAX_TARGET_LENGTH=<MAX_TARGET_LENGTH> # e.g., 1024

# -- Dataset configuration --
export DATASET_NAME=<DATASET_NAME> # e.g., openai/gsm8k
export TRAIN_SPLIT=<TRAIN_SPLIT> # e.g., train
export HF_DATA_DIR=<DATASET_PATH> # e.g., main
export TRAIN_DATA_COLUMNS=<DATA_COLUMNS> # e.g., ['question','answer']
export CHAT_TEMPLATE_PATH=<TEMPLATE_PATH> # e.g., maxtext/examples/chat_templates/math_qa.json

# -- LoRA Conversion configuration (Optional) --
export HF_LORA_ADAPTER_PATH=<HF_LORA_ADAPTER_PATH> # e.g., 'username/adapter-name'
```

## Customizing Trainable Layers (Optional)

By default, MaxText determines which layers to apply LoRA to based on the model's architecture by reading `src/maxtext/configs/post_train/lora_module_path.yml`.

If you need to fine-tune specific components (e.g., targeting only Attention layers to optimize memory usage), you can override these defaults through the following hierarchy:

### Configuration Hierarchy

1. **Command Line Argument**: Pass the `lora_module_path` argument directly in your training command.
2. **Task-Specific Config (`sft.yml`)**: Define the `lora_module_path` parameter in `src/maxtext/configs/post_train/sft.yml`.
3. **Global Defaults**: Automatic detection via the model-to-regex mapping defined in `lora_module_path.yml`.

## Get MaxText model checkpoint

This section explains how to prepare your model checkpoint for use with MaxText. You have two options: using an existing MaxText checkpoint or converting a Hugging Face checkpoint.

### Option 1: Using an existing MaxText checkpoint

If you already have a MaxText-compatible model checkpoint, simply set the following environment variable and move on to the next section.

```bash
export MAXTEXT_CKPT_PATH=<CKPT_PATH> # e.g., gs://my-bucket/my-model-checkpoint/0/items
```

**Note:** Make sure that `MAXTEXT_CKPT_PATH` has the checkpoints created using the correct storage flags:

```bash
export USE_PATHWAYS=0  # Set to 1 for Pathways, 0 for McJAX.
checkpoint_storage_use_zarr3=$((1 - USE_PATHWAYS))
checkpoint_storage_use_ocdbt=$((1 - USE_PATHWAYS))
```

### Option 2: Converting a Hugging Face checkpoint

Refer to the steps in [Hugging Face to MaxText](https://maxtext.readthedocs.io/en/maxtext-v0.2.1/guides/checkpointing_solutions/convert_checkpoint.html#hugging-face-to-maxtext) to convert a hugging face checkpoint to MaxText. Make sure you have correct checkpoint files converted and saved. Similar as Option 1, you can set the following environment and move on.

```bash
export MAXTEXT_CKPT_PATH=<CKPT_PATH> # gs://my-bucket/my-checkpoint-directory/0/items
```

## Submit workload on GKE cluster

This section provides the command to run LoRA Fine-Tuning on a GKE cluster.

### Run a Fresh LoRA Fine-Tuning on Hugging Face Dataset

#### LoRA with Multi-Controller JAX (McJAX)

```bash
xpk workload create \
--cluster=${GKE_CLUSTER?} \
--project=${PROJECT_ID?} \
--zone=${ZONE?} \
--docker-image=${DOCKER_IMAGE?} \
--workload=${RUN_NAME?} \
--tpu-type=${TPU_TYPE?} \
--num-slices=${NUM_SLICES?} \
--command="python3 -m maxtext.trainers.post_train.sft.train_sft run_name=${RUN_NAME?} base_output_directory=${BASE_OUTPUT_DIRECTORY?} model_name=${MODEL?} load_parameters_path=${MAXTEXT_CKPT_PATH?} hf_access_token=${HF_TOKEN?} hf_path=${DATASET_NAME?} train_split=${TRAIN_SPLIT?} hf_data_dir=${HF_DATA_DIR?} train_data_columns=${TRAIN_DATA_COLUMNS?} steps=${STEPS?} per_device_batch_size=${PER_DEVICE_BATCH_SIZE?} max_target_length=${MAX_TARGET_LENGTH?} learning_rate=${LEARNING_RATE?} chat_template_path=${CHAT_TEMPLATE_PATH?} enable_nnx=True pure_nnx_decoder=True lora.enable_lora=True lora.lora_rank=${LORA_RANK?} lora.lora_alpha=${LORA_ALPHA?}"
```

Once the fine-tuning is completed, you can access your model checkpoints at `${BASE_OUTPUT_DIRECTORY}/${RUN_NAME/checkpoints`.

#### LoRA with Pathways

```bash
export USE_PATHWAYS=1

xpk workload create-pathways \
--cluster=${GKE_CLUSTER?} \
--project=${PROJECT_ID?} \
--zone=${ZONE?} \
--docker-image=${DOCKER_IMAGE?} \
--workload=${RUN_NAME?} \
--tpu-type=${TPU_TYPE?} \
--num-slices=${NUM_SLICES?} \
--command="JAX_PLATFORMS=proxy JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 ENABLE_PATHWAYS_PERSISTENCE=1 python3 -m maxtext.trainers.post_train.sft.train_sft run_name=${RUN_NAME?} base_output_directory=${BASE_OUTPUT_DIRECTORY?} model_name=${MODEL?} load_parameters_path=${MAXTEXT_CKPT_PATH?} hf_access_token=${HF_TOKEN?} hf_path=${DATASET_NAME?} train_split=${TRAIN_SPLIT?} hf_data_dir=${HF_DATA_DIR?} train_data_columns=${TRAIN_DATA_COLUMNS?} steps=${STEPS?} per_device_batch_size=${PER_DEVICE_BATCH_SIZE?} max_target_length=${MAX_TARGET_LENGTH?} learning_rate=${LEARNING_RATE?} chat_template_path=${CHAT_TEMPLATE_PATH?} enable_nnx=True pure_nnx_decoder=True lora.enable_lora=True lora.lora_rank=${LORA_RANK?} lora.lora_alpha=${LORA_ALPHA?} checkpoint_storage_use_zarr3=$((1 - USE_PATHWAYS)) checkpoint_storage_use_ocdbt=$((1 - USE_PATHWAYS)) enable_single_controller=True"
```

Once the fine-tuning is completed, you can access your model checkpoints at `${BASE_OUTPUT_DIRECTORY}/${RUN_NAME}/checkpoints`.

### (Optional) Resume from a previous LoRA checkpoint

If you want to resume training from a previous run or further fine-tune an existing LoRA adapter, you can specify the LoRA checkpoint path.

#### Step 1: Convert HF LoRA adapter to MaxText format with Multi-Controller JAX (McJAX)

If your LoRA adapter is currently in Hugging Face format, you must convert it to MaxText format before it can be loaded. Use the integrated conversion utility:

```sh
xpk workload create \
--cluster=${GKE_CLUSTER?} \
--project=${PROJECT_ID?} \
--zone=${ZONE?} \
--docker-image=${DOCKER_IMAGE?} \
--workload=${RUN_NAME?} \
--tpu-type=${TPU_TYPE?} \
--num-slices=${NUM_SLICES?} \
--command="python3 -m maxtext.checkpoint_conversion.to_maxtext model_name=${MODEL?} hf_lora_adapter_path=${HF_LORA_ADAPTER_PATH?} base_output_directory=${BASE_OUTPUT_DIRECTORY?}/converted_adapter hf_access_token=${HF_TOKEN?} hardware=cpu skip_jax_distributed_system=True"
```

#### Step 2: Set the restore path

Point `LORA_RESTORE_PATH` to the converted MaxText adapter directory (the directory containing the `0/items` or Orbax files).

- **load_parameters_path**: Points to the frozen base model weights (the original model).
- **lora_restore_path**: Points to the previous LoRA adapter weights you wish to load.

```sh
export LORA_RESTORE_PATH=<LORA_RESTORE_PATH> # e.g., gs://my-bucket/run-1/checkpoints/0/items or /path/to/run-1/checkpoints/0/items
```

#### Step 3-1: Run LoRA Fine-Tuning with the Restore Path through Multi-Controller JAX (McJAX)

Once your environment variables and checkpoints are ready, you can start the LoRA fine-tuning process.

Execute the following command to begin training:

```sh
xpk workload create \
--cluster=${GKE_CLUSTER?} \
--project=${PROJECT_ID?} \
--zone=${ZONE?} \
--docker-image=${DOCKER_IMAGE?} \
--workload=${RUN_NAME?} \
--tpu-type=${TPU_TYPE?} \
--num-slices=${NUM_SLICES?} \
--command="python3 -m maxtext.trainers.post_train.sft.train_sft run_name=${RUN_NAME?} base_output_directory=${BASE_OUTPUT_DIRECTORY?} model_name=${MODEL?} load_parameters_path=${MAXTEXT_CKPT_PATH?} hf_access_token=${HF_TOKEN?} hf_path=${DATASET_NAME?} train_split=${TRAIN_SPLIT?} hf_data_dir=${HF_DATA_DIR?} train_data_columns=${TRAIN_DATA_COLUMNS?} steps=${STEPS?} per_device_batch_size=${PER_DEVICE_BATCH_SIZE?} max_target_length=${MAX_TARGET_LENGTH?} lora.lora_restore_path=${LORA_RESTORE_PATH?} learning_rate=${LEARNING_RATE?} chat_template_path=${CHAT_TEMPLATE_PATH?} enable_nnx=True pure_nnx_decoder=True lora.enable_lora=True lora.lora_rank=${LORA_RANK?} lora.lora_alpha=${LORA_ALPHA?}"
```

#### Step 3-2: Run LoRA Fine-Tuning with the Restore Path through Pathways

```bash
export USE_PATHWAYS=1

xpk workload create-pathways \
--cluster=${GKE_CLUSTER?} \
--project=${PROJECT_ID?} \
--zone=${ZONE?} \
--docker-image=${DOCKER_IMAGE?} \
--workload=${RUN_NAME?} \
--tpu-type=${TPU_TYPE?} \
--num-slices=${NUM_SLICES?} \
--command="JAX_PLATFORMS=proxy JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 ENABLE_PATHWAYS_PERSISTENCE=1 python3 -m maxtext.trainers.post_train.sft.train_sft run_name=${RUN_NAME?} base_output_directory=${BASE_OUTPUT_DIRECTORY?} model_name=${MODEL?} load_parameters_path=${MAXTEXT_CKPT_PATH?} hf_access_token=${HF_TOKEN?} hf_path=${DATASET_NAME?} train_split=${TRAIN_SPLIT?} hf_data_dir=${HF_DATA_DIR?} train_data_columns=${TRAIN_DATA_COLUMNS?} steps=${STEPS?} per_device_batch_size=${PER_DEVICE_BATCH_SIZE?} max_target_length=${MAX_TARGET_LENGTH?} lora.lora_restore_path=${LORA_RESTORE_PATH?} learning_rate=${LEARNING_RATE?} chat_template_path=${CHAT_TEMPLATE_PATH?} enable_nnx=True pure_nnx_decoder=True lora.enable_lora=True lora.lora_rank=${LORA_RANK?} lora.lora_alpha=${LORA_ALPHA?} checkpoint_storage_use_zarr3=$((1 - USE_PATHWAYS)) checkpoint_storage_use_ocdbt=$((1 - USE_PATHWAYS)) enable_single_controller=True"
```

Your fine-tuned model checkpoints will be saved here: `$BASE_OUTPUT_DIRECTORY/$RUN_NAME/checkpoints`.

## (Optional) Convert Fine-tuned LoRA to Hugging Face Format with Multi-Controller JAX (McJAX)

After completing the fine-tuning process, your LoRA weights are stored in MaxText/Orbax format. To use these weights with the Hugging Face ecosystem (e.g., for inference or sharing), convert them back using the `to_huggingface.py` script.

```sh
xpk workload create \
--cluster=${GKE_CLUSTER?} \
--project=${PROJECT_ID?} \
--zone=${ZONE?} \
--docker-image=${DOCKER_IMAGE?} \
--workload="${RUN_NAME?}-to-hf" \
--tpu-type=${TPU_TYPE?} \
--num-slices=1 \
--command="python3 -m maxtext.checkpoint_conversion.to_huggingface \
    model_name=${MODEL?} \
    lora.lora_restore_path=${BASE_OUTPUT_DIRECTORY?}/${RUN_NAME?}/checkpoints/<STEPS>/model_params \
    base_output_directory=${BASE_OUTPUT_DIRECTORY?}/hf_lora_adapter \
    hf_access_token=${HF_TOKEN?}"
    
```

- `lora.lora_restore_path`: Point this to the specific checkpoint directory (e.g., `.../checkpoints/1000/items`) that you want to export.
- `base_output_directory`: The local or GCS directory where the Hugging Face `adapter_model.safetensors` and `adapter_config.json` will be saved.
- `lora.lora_rank` / `lora.lora_alpha`: Must match the values used during the training phase to ensure the `adapter_config.json` is generated correctly.

## A Note on Multi-Host Resharding

When running LoRA fine-tuning in a **multi-host environment** (e.g., a TPU pod with 64 hosts managing 256 TPUs, such as Pathways or McJAX), special care must be taken when resharding arrays.

In a single-host environment, the host has a global view of all devices, so a standard `jax.device_put` can easily distribute slices of data to all local TPUs. However, in a multi-host setup:

- **Addressability:** A host only has a local view of its directly attached devices and cannot push data directly to TPUs managed by other hosts.
- **Memory Constraints:** If every host tries to load the entire weight matrix into RAM just to extract its local piece, the host CPUs will run out of memory (OOM).

To solve this, MaxText uses `jax.make_array_from_callback` for a "safe reshard." Instead of pushing data *to* the devices, this flips the paradigm. It creates a global `jax.Array` construct where each host locally executes a callback (`lambda idx: val[idx]`) to load **only the specific slice** of the data that its attached TPUs need. This completely bypasses cross-host `device_put` limitations and prevents OOMs since each host only indexes what it requires.
