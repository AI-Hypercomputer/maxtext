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

# Build and upload MaxText Docker images

This guide covers setting up a MaxText development environment and building container images for TPU and GPU workloads. These images can be used to run MaxText on GKE clusters with TPUs or GPUs, and are also required for running MaxText through XPK.

## Prerequisites

Before starting, ensure you have the following tools installed and configured:

1. Environment Prep: Install and configure all [XPK prerequisites](https://github.com/AI-Hypercomputer/xpk/blob/main/docs/installation.md#1-prerequisites).

2. Docker Permissions: Follow the steps to [configure sudoless Docker](https://docs.docker.com/engine/install/linux-postinstall/) to run Docker without `sudo`.

3. Artifact Registry Access: Authenticate with [Google Artifact Registry](https://docs.cloud.google.com/artifact-registry/docs/docker/authentication#gcloud-helper) for permission to push your images and other access.

4. Authentication & Access: Run the following commands to authenticate your account and configure Docker:

```bash
# Authenticate your user account for gcloud CLI access
gcloud auth login

# Configure application default credentials for Docker and other tools
gcloud auth application-default login

# Configure Docker credentials and test your access
gcloud auth configure-docker
docker run hello-world
```

## Install MaxText

Follow the instructions in the [Installation guide](install_maxtext.md) to install the `maxtext[runner]` extra, which includes all necessary dependencies for building MaxText Docker images and running workloads through XPK. It automatically installs XPK, so you do not need to install it separately to manage your clusters and workloads.

## Build MaxText Docker image

Select the appropriate build commands based on your hardware (`TPU` or `GPU`) and your specific workflow (`pre-training` or `post-training`). Each of these commands will generate a local Docker image named `maxtext_base_image`.

### TPU pre-training Docker image

```bash
# Option 1: Build with the stable versions of dependencies (default)
build_maxtext_docker_image

# Option 2: Build with latest nightly versions of jax/jaxlib
build_maxtext_docker_image MODE=nightly

# Option 3: Build with the specified jax/jaxlib version
build_maxtext_docker_image MODE=nightly JAX_VERSION=$JAX_VERSION
```

### GPU pre-training Docker image

```bash
# Option 1: Build with the stable versions of dependencies (default)
build_maxtext_docker_image DEVICE=gpu

# Option 2: Build with latest nightly versions of jax/jaxlib
build_maxtext_docker_image DEVICE=gpu MODE=nightly

# Option 3: Build with base image as `ghcr.io/nvidia/jax:base-2024-12-04`
build_maxtext_docker_image DEVICE=gpu MODE=pinned

# Option 4: Build with the specified jax/jaxlib version
build_maxtext_docker_image DEVICE=gpu MODE=nightly JAX_VERSION=$JAX_VERSION
```

### TPU post-training Docker image

```bash
# This build process takes approximately 10 to 15 minutes.
build_maxtext_docker_image WORKFLOW=post-training
```

## Upload MaxText Docker image to Artifact Registry

```bash
# Make sure to set `CLOUD_IMAGE_NAME` with your desired image name.
export CLOUD_IMAGE_NAME=<IMAGE_NAME>
upload_maxtext_docker_image CLOUD_IMAGE_NAME=${CLOUD_IMAGE_NAME?}
```

```{note}
You will need the [**Artifact Registry Writer**](https://docs.cloud.google.com/artifact-registry/docs/access-control#permissions) role to push Docker images to your project's Artifact Registry and to allow the cluster to pull them during workload execution. If you don't have this permission, contact your project administrator to grant you this role through "Google Cloud Console -> IAM -> Grant access".
```

## Troubleshooting

1. If you see the following error while building or uploading your Docker image, try adding the listed file path to `.dockerignore`. Do not include the `./` prefix in the `.dockerignore` file:

```bash
ERROR: Found symbolic links with absolute paths in the build context:
./<add_this_value_to_dockerignore>
```
