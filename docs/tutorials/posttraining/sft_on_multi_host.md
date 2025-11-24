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

# Supervised Fine-Tuning (SFT) on Multi-Host TPUs
Supervised fine-tuning (SFT) is a process where a pre-trained large language model is fine-tuned on a labeled dataset to adapt the model to perform better on specific tasks.

This tutorial demonstrates step-by-step instructions for setting up the multi-host TPU environment and then training the model on the Hugging Face dataset using SFT. In this tutorial we use a multi-host TPU such as `v6e-256`.

We use [Tunix](https://github.com/google/tunix), a JAX-based library designed for post-training tasks, to perform SFT.

Let's get started!

## 1. Build and upload MaxText Docker image
This section guides you through cloning the MaxText repository, building MaxText Docker image with dependencies, and uploading the docker image to your project's Artifact Registry.

### 1.1. Clone the MaxText repository
```bash
git clone https://github.com/google/maxtext.git
cd maxtext
```

### 1.2. Build MaxText Docker image
```bash
bash dependencies/scripts/docker_build_dependency_image.sh MODE=post-training
```
This creates a local Docker image named `maxtext_base_image`.

### 1.3. Upload the Docker image to Artifact Registry
```bash
# Replace `$USER_runner` with your desired image name
export DOCKER_IMAGE_NAME=${USER}_runner
bash dependencies/scripts/docker_upload_runner.sh CLOUD_IMAGE_NAME=$DOCKER_IMAGE_NAME
```
The `docker_upload_runner.sh` script uploads your Docker image to Artifact Registry.

## 2. Install XPK
Install XPK by following the instructions in the [official documentation](https://github.com/AI-Hypercomputer/xpk?tab=readme-ov-file#installation-via-pip).

## 3. Create GKE cluster
If you don't already have a GKE cluster, create one by following the [XPK cluster creation guide](https://github.com/AI-Hypercomputer/xpk?tab=readme-ov-file#cluster-create). Ensure the cluster is Pathways-compatible when running SFT with Pathways.

## 4. Environment configuration
```bash
# -- Google Cloud Configuration --
export PROJECT=<Google Cloud Project ID>
export CLUSTER_NAME=<Name of GKE Cluster>
export ZONE=<GKE Cluster Zone>

# -- Workload Configuration --
export WORKLOAD_NAME=<Name of Workload> # e.g., $(date +%Y-%m-%d-%H-%M-%S)
export TPU_TYPE=<TPU Type> # e.g., v6e-256
export TPU_SLICE=1
export DOCKER_IMAGE="gcr.io/${PROJECT}/${DOCKER_IMAGE_NAME}"

# -- MaxText Configuration --
export OUTPUT_PATH=<GCS Path for Output/Logs> # e.g., gs://my-bucket/my-output-directory
export STEPS=<Fine-Tuning Steps> # e.g., 1000
export HF_TOKEN=<Hugging Face Access Token>

# -- Model Configuration --
export MODEL_NAME=<Model Name> # e.g., deepseek3-671b
export TOKENIZER_PATH=<Model Tokenizer> # e.g., deepseek-ai/DeepSeek-V3

# -- Dataset configuration --
export DATASET_NAME=<Hugging Face Dataset Name> # e.g., HuggingFaceH4/ultrachat_200k
export TRAIN_SPLIT=<Data Split for Train> # e.g., train_sft
export TRAIN_DATA_COLUMNS=<Data Columns to Train on> # e.g., ['messages']
```

## 5. Get MaxText model checkpoint
This section explains how to prepare your model checkpoint for use with MaxText. You have two options: using an existing MaxText checkpoint or converting a Hugging Face checkpoint.

### Option 1: Using an existing MaxText checkpoint
If you already have a MaxText-compatible model checkpoint, simply set the following environment variable and move on to the next section.

```bash
export MODEL_CHECKPOINT_PATH=<gcs path for MaxText checkpoint> # e.g., gs://my-bucket/my-model-checkpoint/0/items
```
**Note:** Make sure that `MODEL_CHECKPOINT_PATH` has the checkpoints created using the correct storage flags:
* **For SFT with McJAX:** `checkpoint_storage_use_zarr3=True` and `checkpoint_storage_use_ocdbt=True`.
* **For SFT with Pathways:** `checkpoint_storage_use_zarr3=False` and `checkpoint_storage_use_ocdbt=False`.

### Option 2: Converting a Hugging Face checkpoint
If your model checkpoint is from Hugging Face, you need to run a conversion script to make it MaxText-compatible.

1. **Set the Output Path:** First, define where the new MaxText checkpoint will be saved.

```bash
export MODEL_CHECKPOINT_PATH=${OUTPUT_PATH}/${WORKLOAD_NAME}/maxtext-checkpoint/0/items
```

2. **Run the Conversion Script:** Execute the following command that downloads the specified Hugging Face model and converts its weights into the MaxText format. The conversion script only supports official versions of models from Hugging Face. To see the specific models and versions currently supported for conversion, please refer to the `HF_IDS` dictionary in the MaxText utility file [here](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/MaxText/utils/ckpt_conversion/utils/utils.py).

```bash
USE_ZARR3=<Flag to use zarr3> # True to run SFT with McJAX, False to run SFT with Pathways
USE_OCDBT=<Flag to use ocdbt> # True to run SFT with McJAX, False to run SFT with Pathways

xpk workload create \
--cluster=${CLUSTER_NAME} \
--project=${PROJECT} \
--zone=${ZONE} \
--docker-image=${DOCKER_IMAGE} \
--workload=ckpt-${WORKLOAD_NAME} \
--tpu-type=${TPU_TYPE} \
--num-slices=${TPU_SLICE} \
--command "python3 -m MaxText.utils.ckpt_conversion.to_maxtext src/MaxText/configs/base.yml model_name=$MODEL_NAME hf_access_token=$HF_TOKEN base_output_directory=$OUTPUT_PATH/$WORKLOAD_NAME/maxtext-checkpoint scan_layers=True checkpoint_storage_use_zarr3=$USE_ZARR3 checkpoint_storage_use_ocdbt=$USE_OCDBT"
```

## 6. Submit workload on GKE cluster
This section provides the command to run SFT on a GKE cluster.

### 6.1. SFT with Multi-Controller JAX (McJAX)
```bash
xpk workload create \
--cluster=${CLUSTER_NAME} \
--project=${PROJECT} \
--zone=${ZONE} \
--docker-image=${DOCKER_IMAGE} \
--workload=${WORKLOAD_NAME} \
--tpu-type=${TPU_TYPE} \
--num-slices=${TPU_SLICE} \
--command "python3 -m MaxText.sft.sft_trainer src/MaxText/configs/sft.yml run_name=$WORKLOAD_NAME base_output_directory=$OUTPUT_PATH model_name=$MODEL_NAME load_parameters_path=$MODEL_CHECKPOINT_PATH hf_access_token=$HF_TOKEN tokenizer_path=$TOKENIZER_PATH per_device_batch_size=1 steps=$STEPS profiler=xplane hf_path=$DATASET_NAME train_split=$TRAIN_SPLIT train_data_columns=$TRAIN_DATA_COLUMNS"
```
Once the fine-tuning is completed, you can access your model checkpoints at `$OUTPUT_PATH/$WORKLOAD_NAME/checkpoints`.

### 6.2. SFT with Pathways
```bash
xpk workload create-pathways \
--cluster=${CLUSTER_NAME} \
--project=${PROJECT} \
--zone=${ZONE} \
--docker-image=${DOCKER_IMAGE} \
--workload=${WORKLOAD_NAME} \
--tpu-type=${TPU_TYPE} \
--num-slices=${TPU_SLICE} \
--command="JAX_PLATFORMS=proxy JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 ENABLE_PATHWAYS_PERSISTENCE=1 python3 -m MaxText.sft.sft_trainer src/MaxText/configs/sft.yml run_name=$WORKLOAD_NAME base_output_directory=$OUTPUT_PATH model_name=$MODEL_NAME load_parameters_path=$MODEL_CHECKPOINT_PATH hf_access_token=$HF_TOKEN tokenizer_path=$TOKENIZER_PATH per_device_batch_size=1 steps=$STEPS profiler=xplane checkpoint_storage_use_zarr3=False checkpoint_storage_use_ocdbt=False enable_single_controller=True"
```

Once the fine-tuning is completed, you can access your model checkpoints at `$OUTPUT_PATH/$WORKLOAD_NAME/checkpoints`.
