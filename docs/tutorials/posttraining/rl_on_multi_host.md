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

# Reinforcement Learning on multi-host TPUs

This tutorial demonstrates step-by-step instructions for setting up the environment and then training the Llama3.1 70B-IT model on the GSM8K math reasoning dataset using [Pathways for orchestration](https://cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/pathways-intro) on multi-host TPU-VMs such as `v5p-128`.

We utilize two RL algorithms, implemented via the Tunix library, to enhance the model's reasoning capabilities:

* **Group Relative Policy Optimization (GRPO)**: GRPO is an RL algorithm designed to enhance the reasoning abilities of LLMs. It is a variant of Proximal Policy Optimization (PPO) that reduces memory usage by eliminating the need for a separate value function model. GRPO works by generating multiple responses for a given prompt, evaluating these responses using a reward model, and then calculating a relative advantage based on the group's performance to update the policy.

* **Group Sequence Policy Optimization (GSPO)**: GSPO is an RL algorithm that improves training efficiency and performance of LLMs by using sequence-level importance ratios and operations. GSPO defines the importance ratio based on sequence likelihood and performs sequence-level clipping, rewarding, and optimization.

For efficient model inference and response generation during this process, we rely on the vLLM library.

Let's get started!

## Create virtual environment and Install MaxText dependencies
Follow instructions in [Install MaxText](../../install_maxtext.md), but 
recommend creating the virtual environment outside the `maxtext` directory.


## Setup environment variables

Setup following environment variables:

```bash
# -- Model configuration --
export HF_MODEL='llama3.1-70b-Instruct'
export MODEL='llama3.1-70b'
export TOKENIZER='meta-llama/Llama-3.1-70B-Instruct'
export HF_TOKEN=<Hugging Face access token>

# -- MaxText configuration --
export BASE_OUTPUT_DIRECTORY=<output directory to store run logs> # e.g., gs://my-bucket/my-output-directory
export RUN_NAME=llama-3-70b-grpo
export MAXTEXT_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/${RUN_NAME}/0/items

# -- Workload configuration --
export WORKLOAD=${RUN_NAME}
export TPU_TYPE='v5p-128'
export TPU_CLUSTER=<cluster name>
export PROJECT_ID=<GCP project ID>
export ZONE=<zone name>
```

## Get your model checkpoint

You can convert a Hugging Face checkpoint to MaxText format using the `src/MaxText/utils/ckpt_conversion/to_maxtext.py` script. This is useful if you have a pre-trained model from Hugging Face that you want to use with MaxText.

First, ensure you have the necessary dependencies installed. Then, run the conversion script on a CPU machine. For large models, it is recommended to use the `--lazy_load_tensors` flag to reduce memory usage during conversion. \
For example, converting a Llama3.1-70B model scanned checkpoint using `--lazy_load_tensors=true` will use around 200GB of RAM and completes in ~10 mins. This command will download the Hugging Face model and convert it to the MaxText format, saving it to the specified GCS bucket.

```bash
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# using --lazy_load_tensors=true here will reduce the memory usage. eg, Llama3.1-70B conversion takes around 86GB of RAM
python3 -m MaxText.utils.ckpt_conversion.to_maxtext MaxText/configs/base.yml \
    model_name=${HF_MODEL} \
    hf_access_token=${HF_TOKEN} \
    base_output_directory=${BASE_OUTPUT_DIRECTORY}/${RUN_NAME} \
    scan_layers=true checkpoint_storage_use_ocdbt=false checkpoint_storage_use_zarr3=false \
    skip_jax_distributed_system=true --lazy_load_tensors=true
```

## Build and Upload MaxText Docker Image with Tunix, vLLM, tpu-inference dependencies
Before building the Docker image, authenticate to [Google Artifact Registry](https://docs.cloud.google.com/artifact-registry/docs/docker/authentication#gcloud-helper) for permission to push your images and other access.
```bash
# Authenticate your user account for gcloud CLI access
gcloud auth login
# Configure application default credentials for Docker and other tools
gcloud auth application-default login
# Configure Docker credentials and test your access
gcloud auth configure-docker
docker run hello-world
```

You can install the required dependencies using either of the following two options:

### Option 1: Installing stable releases of tunix and vllm-tpu
Run the following bash script to create a docker image with all the dependencies of MaxText, Tunix, vLLM and tpu-inference installed.

In addition to MaxText dependencies, primarily, it installs `vllm-tpu` which is [vllm](https://github.com/vllm-project/vllm) and [tpu-inference](https://github.com/vllm-project/tpu-inference) and thereby providing TPU inference for vLLM, with unified JAX and PyTorch support. This build process takes approximately 10 to 15 minutes.
 
```
bash dependencies/scripts/docker_build_dependency_image.sh MODE=post-training
```

You can also use `bash dependencies/scripts/docker_build_dependency_image.sh MODE=post-training-experimental` to try out new features via experimental dependencies such as improved pathwaysutils resharding API.

### Option 2: Install from locally git cloned repositories

You can also locally git clone [tunix](https://github.com/google/tunix), [tpu-inference](https://github.com/vllm-project/tpu-inference), [vllm](https://github.com/vllm-project/vllm.git) and then use the following command to build a docker image using them: 
```
bash dependencies/scripts/docker_build_dependency_image.sh MODE=post-training POST_TRAINING_SOURCE=local
```

### Upload the dependency docker image along with MaxText code
> **Note:** You will need the [**Artifact Registry Writer**](https://docs.cloud.google.com/artifact-registry/docs/access-control#permissions) role to push Docker images to your project's Artifact Registry and to allow the cluster to pull them during workload execution. If you don't have this permission, contact your project administrator to grant you this role through "Google Cloud Console -> IAM -> Grant access".
```
bash dependencies/scripts/docker_upload_runner.sh CLOUD_IMAGE_NAME=${CLOUD_IMAGE_NAME}
```

## Submit your RL workload via Pathways

Please create a pathways ready GKE cluster as described [here](https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/create-gke-cluster), and you can submit the `train_rl.py` script via XPK. You can install XPK by following the instructions in the [official documentation](https://github.com/AI-Hypercomputer/xpk/blob/main/docs/installation.md). 

### Submit GRPO workload
```
xpk workload create-pathways --workload $WORKLOAD \
--docker-image <path/to/gcr.io> --cluster $TPU_CLUSTER \
--tpu-type=$TPU_TYPE --num-slices=1  --zone=$ZONE \
--project=$PROJECT_ID --priority=high \
--command "TF_CPP_MIN_LOG_LEVEL=0 JAX_PLATFORMS=proxy JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 ENABLE_PATHWAYS_PERSISTENCE='1' \
python3 -m src.MaxText.rl.train_rl src/MaxText/configs/rl.yml \
  model_name=${MODEL} \
  tokenizer_path=${TOKENIZER} \
  load_parameters_path=${MAXTEXT_CKPT_PATH} \
  run_name=${RUN_NAME} \
  base_output_directory=${BASE_OUTPUT_DIRECTORY} \
  hf_access_token=$HF_TOKEN"
```

### Submit GSPO workload
```
xpk workload create-pathways --workload $WORKLOAD \
--docker-image <path/to/gcr.io> --cluster $TPU_CLUSTER \
--tpu-type=$TPU_TYPE --num-slices=1  --zone=$ZONE \
--project=$PROJECT_ID --priority=high \
--command "TF_CPP_MIN_LOG_LEVEL=0 JAX_PLATFORMS=proxy JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 ENABLE_PATHWAYS_PERSISTENCE='1' \
python3 -m src.MaxText.rl.train_rl src/MaxText/configs/rl.yml \
  model_name=${MODEL} \
  tokenizer_path=${TOKENIZER} \
  load_parameters_path=${MAXTEXT_CKPT_PATH} \
  run_name=${RUN_NAME} \
  base_output_directory=${BASE_OUTPUT_DIRECTORY} \
  hf_access_token=$HF_TOKEN \
  loss_algo=gspo-token"
```
