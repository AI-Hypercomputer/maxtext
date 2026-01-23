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
- [Setup Environment Variables](#setup-environment-variables)
- [Get Your Model Checkpoint](#get-your-model-checkpoint)
- [Build and Upload MaxText Docker Image](#build-and-upload-maxtext-docker-image-with-post-training-dependencies)
- [Submit your RL workload via Pathways](#submit-your-rl-workload-via-pathways)
- [Managing Workloads](#managing-workloads)
- [Troubleshooting](#troubleshooting)

## Prerequisites

Before starting, ensure you have:

- Access to a Google Cloud Project with TPU quotas.
- A Hugging Face account with an access token for downloading models.
- Permissions for Google Artifact Registry (Artifact Registry Writer role).
- XPK installed (follow [official documentation](https://github.com/AI-Hypercomputer/xpk/blob/main/docs/installation.md)).
- A Pathways-ready GKE cluster (see [create GKE cluster](https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/create-gke-cluster)).

## Setup Environment Variables

Set up the following environment variables. Replace placeholders with your
actual values.

```bash
# -- Model configuration --
export HF_MODEL=<Hugging Face Model> # e.g. 'llama3.1-70b-Instruct'
export MODEL=<MaxText Model> # e.g. 'llama3.1-70b'
export TOKENIZER=<Tokenizer> # e.g. 'meta-llama/Llama-3.1-70B-Instruct'
export HF_TOKEN=<Hugging Face access token>

# -- MaxText configuration --
export BASE_OUTPUT_DIRECTORY=<output directory to store run logs> # e.g., gs://my-bucket/my-output-directory
export WORKLOAD=<Name for this run> # e.g., llama-3-70b-grpo
export MAXTEXT_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/${WORKLOAD}/0/items

# -- Workload configuration --
export TPU_TYPE=<TPU Type> # e.g., 'v5p-128'
export TPU_CLUSTER=<cluster name>
export PROJECT_ID=<GCP project ID>
export CLOUD_IMAGE_NAME=<your artifact registry image> # Name for the Docker image to be built
```

## Get Your Model Checkpoint

### Option 1: Using an existing MaxText checkpoint

If you already have a MaxText-compatible model checkpoint, simply set the
following environment variable and move on to the next section.

```bash
export MAXTEXT_CKPT_PATH=<gcs path for MaxText checkpoint> # e.g., gs://my-bucket/my-model-checkpoint/0/items
```

### Option 2: Converting from a Hugging Face checkpoint

You can convert a Hugging Face checkpoint to MaxText format using the
`src/MaxText/utils/ckpt_conversion/to_maxtext.py` script. This is useful if you
have a pre-trained model from Hugging Face that you want to use with MaxText.

First, ensure you have the necessary dependencies installed (PyTorch for the
conversion script). Then, run the conversion script on a CPU machine. For large
models, use the `--lazy_load_tensors` flag to reduce memory usage during
conversion.

For example, converting a Llama3.1-70B model with `--lazy_load_tensors=true`
uses around 200GB of RAM and completes in ~10 minutes. This command will
download the Hugging Face model and convert it to the MaxText format, saving it
to the specified GCS bucket.

```bash
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

python3 -m MaxText.utils.ckpt_conversion.to_maxtext MaxText/configs/base.yml \
    model_name=${HF_MODEL} \
    hf_access_token=${HF_TOKEN} \
    base_output_directory=${BASE_OUTPUT_DIRECTORY}/${WORKLOAD} \
    scan_layers=true checkpoint_storage_use_ocdbt=false checkpoint_storage_use_zarr3=false \
    skip_jax_distributed_system=true --lazy_load_tensors=true
```

## Build and upload MaxText Docker image with post-training dependencies

Before building the Docker image, authenticate to
[Google Artifact Registry](https://docs.cloud.google.com/artifact-registry/docs/docker/authentication#gcloud-helper)
for permission to push your images and other access.

```bash
# Authenticate your user account for gcloud CLI access
gcloud auth login

# Configure application default credentials for Docker and other tools
gcloud auth application-default login

# Configure Docker credentials and test your access
gcloud auth configure-docker
docker run hello-world
```

### Option 1: Install stable releases of post-training dependencies

> **Caution:** RL in MaxText is currently broken with stable releases of
> post-training dependencies. We are working on fixing this and recommend
> following
> [Option 2: Install from Git repositories of post-training dependencies](#option-2-install-from-git-repositories-of-post-training-dependencies)
> in the meantime.

Run the following script to create a Docker image with stable releases of
MaxText, [Tunix](https://github.com/google/tunix),
[vLLM](https://github.com/vllm-project/vllm), and
[tpu-inference](https://github.com/vllm-project/tpu-inference) dependencies.
This installs `vllm-tpu` which provides TPU inference for vLLM with unified JAX
and PyTorch support. The build process takes approximately 10-15 minutes.

```bash
bash dependencies/scripts/docker_build_dependency_image.sh WORKFLOW=post-training
```

For experimental features (such as improved pathwaysutils resharding API), use:

```bash
bash dependencies/scripts/docker_build_dependency_image.sh WORKFLOW=post-training-experimental
```

### Option 2: Install from Git repositories of post-training dependencies

You can also locally clone the [tunix](https://github.com/google/tunix),
[tpu-inference](https://github.com/vllm-project/tpu-inference), and
[vllm](https://github.com/vllm-project/vllm.git) repositories and then build the
docker image with these local sources. To get a set of compatible commit IDs for
`maxtext`, `tunix`, `tpu-inference`, and `vllm`, follow these steps:

1. Navigate to the
   [MaxText Package Tests](https://github.com/AI-Hypercomputer/maxtext/actions/workflows/build_and_test_maxtext.yml?query=event%3Aschedule)
   GitHub Actions workflow.

1. Select the latest successful run.

1. Within the workflow run, find and click on the `maxtext_jupyter_notebooks (py312)` job, then expand the `run` job.

1. Locate the `Record Commit IDs` step. The commit SHAs for `maxtext`, `tunix`,
   `tpu-inference`, and `vllm` that were used in that successful run are listed
   in the logs of this step.

1. Prior to installation, ensure that the `maxtext`, `tunix`, `vllm`, and `tpu-inference` repositories are synchronized to the specific commits recorded from the CI logs. For each repository, use the following command to switch to the correct commit: `git checkout <commit_id>`.

**Note:** Clone these repositories as siblings of the `maxtext` directory (e.g.,
in the same parent directory). After cloning, run the build from inside the
`maxtext` repository so it picks up the local sources:

```bash
bash dependencies/scripts/docker_build_dependency_image.sh WORKFLOW=post-training POST_TRAINING_SOURCE=local
```

### Upload the Docker Image

> **Note:** You will need the
> [**Artifact Registry Writer**](https://docs.cloud.google.com/artifact-registry/docs/access-control#permissions)
> role to push Docker images to your project's Artifact Registry. Contact your
> project administrator if you don't have this permission.

```bash
bash dependencies/scripts/docker_upload_runner.sh CLOUD_IMAGE_NAME=${CLOUD_IMAGE_NAME}
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
xpk workload create-pathways --workload $WORKLOAD \
--docker-image gcr.io/$PROJECT_ID/$CLOUD_IMAGE_NAME --cluster $TPU_CLUSTER \
--tpu-type=$TPU_TYPE --num-slices=1 \
--project=$PROJECT_ID --priority=high \
--command "HF_TOKEN=${HF_TOKEN} TF_CPP_MIN_LOG_LEVEL=0 JAX_PLATFORMS=proxy JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 ENABLE_PATHWAYS_PERSISTENCE='1' \
python3 -m src.MaxText.rl.train_rl src/MaxText/configs/rl.yml \
  model_name=${MODEL} \
  tokenizer_path=${TOKENIZER} \
  load_parameters_path=${MAXTEXT_CKPT_PATH} \
  run_name=${WORKLOAD} \
  base_output_directory=${BASE_OUTPUT_DIRECTORY} \
  hf_access_token=${HF_TOKEN}"
```

### Submit GSPO workload

```bash
xpk workload create-pathways --workload $WORKLOAD \
--docker-image gcr.io/$PROJECT_ID/$CLOUD_IMAGE_NAME --cluster $TPU_CLUSTER \
--tpu-type=$TPU_TYPE --num-slices=1 \
--project=$PROJECT_ID --priority=high \
--command "HF_TOKEN=${HF_TOKEN} TF_CPP_MIN_LOG_LEVEL=0 JAX_PLATFORMS=proxy JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 ENABLE_PATHWAYS_PERSISTENCE='1' \
python3 -m src.MaxText.rl.train_rl src/MaxText/configs/rl.yml \
  model_name=${MODEL} \
  tokenizer_path=${TOKENIZER} \
  load_parameters_path=${MAXTEXT_CKPT_PATH} \
  run_name=${WORKLOAD} \
  base_output_directory=${BASE_OUTPUT_DIRECTORY} \
  hf_access_token=${HF_TOKEN} \
  loss_algo=gspo-token"
```

## Managing Workloads

- **Monitor workload status**: Check Pathways job status: `kubectl get pathwaysjob`. Check pod status: `kubectl get pods`.
- **Delete a workload**: To remove a failed or unwanted Pathways job, use XPK:
  ```bash
  xpk workload delete \
      --workload $WORKLOAD \
      --cluster $TPU_CLUSTER \
      --project $PROJECT_ID
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
