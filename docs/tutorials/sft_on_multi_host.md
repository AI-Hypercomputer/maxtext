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

# Supervised Fine-Tuning (SFT) with Deepseek-V3 Model
This guide provides step by step instructions to run SFT with Deepseek-V3 model on TPU v6e-256. Deepseek-V3 is a Mixture-of-Experts (MoE) language model with 671B parameters.

## 1. Build and Upload MaxText Docker Image
This section guides you through cloning the MaxText repository, building MaxText Docker image with dependencies, and uploading the docker image to your project's Artifact Registry.

### 1.1. Clone the MaxText Repository
```bash
git clone https://github.com/google/maxtext.git
cd maxtext
```

### 1.2. Build MaxText Docker Image
```bash
bash docker_build_dependency_image.sh MODE=jax_ai_image BASEIMAGE=us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:latest
```
This creates a local Docker image named `maxtext_base_image`.

### 1.3. Upload the Docker Image to Artifact Registry 
```bash
# Replace `$USER_runner` with your desired image name
export DOCKER_IMAGE_NAME=${USER}_runner
bash docker_upload_runner.sh CLOUD_IMAGE_NAME=$DOCKER_IMAGE_NAME
```
The `docker_upload_runner.sh` script uploads your Docker image to Artifact Registry.

## 2. Install XPK
Install XPK by following the instructions in the [official documentation](https://github.com/AI-Hypercomputer/xpk?tab=readme-ov-file#installation-via-pip).

## 3. Create GKE Cluster
If you don't already have a GKE cluster with a `v6e-256` TPU slice available, create one by following the [XPK cluster creation guide](https://github.com/AI-Hypercomputer/xpk?tab=readme-ov-file#cluster-create).

## 4. Environment Configuration
```bash
# -- Google Cloud Configuration --
export PROJECT=<Google Cloud Project ID>
export CLUSTER_NAME=<Name of GKE Cluster>
export ZONE=<GKE Cluster Zone>

# -- Workload Configuration --
export WORKLOAD_NAME="sft-$(date +%Y-%m-%d-%H-%M-%S)" # Or your desired workload name
export TPU_TYPE=v6e-256
export TPU_SLICE=1
export DOCKER_IMAGE="gcr.io/${PROJECT}/${DOCKER_IMAGE_NAME}"

# -- MaxText Configuration --
export OUTPUT_PATH=<GCS Bucket Path for output/logs>
export STEPS=100 # Number of fine-tuning steps to run
export HF_TOKEN=<Hugging Face access token>
export MODEL_CHECKPOINT_PATH=<GCS path to model checkpoint>
```

## 5. Submit Workload on GKE Cluster
This section provides the command to run SFT with Deepseek-v3 model on a v6e-256 GKE cluster.
```bash
xpk workload create \
--cluster=${CLUSTER_NAME} \
--project=${PROJECT} \
--zone=${ZONE} \
--docker-image=${DOCKER_IMAGE} \
--workload=${WORKLOAD_NAME} \
--tpu-type=${TPU_TYPE} \
--num-slices=${TPU_SLICE} \
--command "python3 -m MaxText.sft.sft_trainer MaxText/configs/sft.yml run_name=$WORKLOAD_NAME base_output_directory=$OUTPUT_PATH model_name=deepseek3-671b load_parameters_path=$MODEL_CHECKPOINT_PATH hf_access_token=$HF_TOKEN tokenizer_path=deepseek-ai/DeepSeek-V3 per_device_batch_size=1 steps=$STEPS profiler=xplane megablox=False sparse_matmul=False ici_expert_parallelism=16 ici_fsdp_parallelism=16 weight_dtype=bfloat16 dtype=bfloat16 remat_policy=full decoder_layer_input=offload sa_block_q=2048 sa_block_q_dkv=2048 sa_block_q_dq=2048 opt_type=sgd attention=flash capacity_factor=1.0 max_target_length=2048"
```
Once the fine-tuning is completed, you can access your model checkpoint at `${OUTPUT_PATH}/${WORKLOAD_NAME}/${STEPS}/model_params`.
