<!--
 Copyright 2023–2025 Google LLC

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

(run-xpk)=
# Running MaxText with XPK

This document focuses on steps required to setup XPK on TPU VM and assumes you have gone through the [README](https://github.com/google/xpk/blob/main/README.md) to understand XPK basics.

## Steps to setup XPK on TPU VM

* Verify you have these permissions for your account or service account

    Storage Admin \
    Kubernetes Engine Admin

* gcloud is installed on TPUVMs using the snap distribution package. Install kubectl using snap
```shell
sudo apt-get update
sudo apt install snapd
sudo snap install kubectl --classic
```
* Install `gke-gcloud-auth-plugin`
```shell
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

sudo apt update && sudo apt-get install google-cloud-sdk-gke-gcloud-auth-plugin
```

* Authenticate gcloud installation by running this command and following the prompt
```
gcloud auth login
```

* Run this command to configure docker to use docker-credential-gcloud for GCR registries:
```
gcloud auth configure-docker
```

* Test the installation by running
```
docker run hello-world
```

* If getting a permission error, try running
```
sudo usermod -aG docker $USER
```
after which log out and log back in to the machine.

## Build Docker Image for Maxtext

1. Git clone maxtext locally

    ```shell
    git clone https://github.com/AI-Hypercomputer/maxtext.git
    cd maxtext
    ```
2. Build local Maxtext docker image

    This only needs to be rerun when you want to change your dependencies. This image may expire which would require you to rerun the below command

    ```shell
    # Default will pick stable versions of dependencies
    bash docker_build_dependency_image.sh
    ```

    #### Build Maxtext Docker Image with JAX AI Images (Preview)
    We're excited to announce the preview of building Maxtext Docker images using the JAX AI Training Images, available for both TPUs and GPUs. This provides a more reliable and consistent build environment.

    ###### What is JAX AI Images?
    JAX AI Images provides a consistent environment for Maxtext by bundling JAX with core packages like `orbax`, `flax`, and `optax`, along with Google Cloud utilities and other essential tools. These libraries are tested to ensure compatibility, providing a stable foundation for building and running Maxtext and eliminating potential conflicts due to incompatible package versions.

    ###### How to Use It
    Use the `docker_build_dependency_image.sh` script to build your Maxtext Docker image with JAX AI Images. Set MODE to `stable_stack` and specify the desired `BASEIMAGE`. The `DEVICE` variable determines whether to build for TPUs or GPUs.

    ###### For TPUs:
    
    ```
    # Example: bash docker_build_dependency_image.sh DEVICE=tpu MODE=jax_ai_image BASEIMAGE=us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.5.2-rev2
    bash docker_build_dependency_image.sh DEVICE=tpu MODE=stable_stack BASEIMAGE={{JAX_AI_IMAGE_TPU_BASEIMAGE}}
    ```

    You can find a list of available JAX AI base images [here](https://us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu).

    ###### [New] For GPUs:

    ```
    # Example bash docker_build_dependency_image.sh DEVICE=gpu MODE=stable_stack BASEIMAGE=us-central1-docker.pkg.dev/deeplearning-images/jax-ai-image/gpu:jax0.5.1-cuda_dl25.02-rev1
    bash docker_build_dependency_image.sh MODE=jax_ai_image BASEIMAGE={{JAX_AI_IMAGE_TPU_BASEIMAGE}}
    ```

    You can find a list of available JAX AI base images [here](https://us-central1-docker.pkg.dev/deeplearning-images/jax-ai-image/gpu).

    **Important Note:** The JAX AI Image is currently in the experimental phase. We encourage you to try it out and provide feedback.

3. After building the dependency image `maxtext_base_image`, xpk can handle updates to the working directory when running `xpk workload create` and using `--base-docker-image`.

    See details on docker images in xpk here: https://github.com/google/xpk/blob/main/README.md#how-to-add-docker-images-to-a-xpk-workload

    __Using xpk to upload image to your gcp project and run Maxtext__

      ```shell
      gcloud config set project $PROJECT_ID
      gcloud config set compute/zone $ZONE

      # See instructions in README.me to create below buckets.
      BASE_OUTPUT_DIR=gs://output_bucket/
      DATASET_PATH=gs://dataset_bucket/

      # Install xpk
      python3 -m pip install xpk

      # Make sure you are still in the maxtext github root directory when running this command
      xpk workload create \
      --cluster ${CLUSTER_NAME} \
      --base-docker-image maxtext_base_image \
      --workload ${USER}-first-job \
      --tpu-type=v5litepod-256 \
      --num-slices=1  \
      --command "python3 -m MaxText.train MaxText/configs/base.yml run_name=${USER}-first-job base_output_directory=${BASE_OUTPUT_DIR} dataset_path=${DATASET_PATH} steps=100 per_device_batch_size=1"
      ```

      __Using [xpk github repo](https://github.com/google/xpk.git)__

      ```shell
      git clone https://github.com/google/xpk.git

      # Make sure you are still in the maxtext github root directory when running this command
      python3 xpk/xpk.py workload create \
      --cluster ${CLUSTER_NAME} \
      --base-docker-image maxtext_base_image \
      --workload ${USER}-first-job \
      --tpu-type=v5litepod-256 \
      --num-slices=1  \
      --command "python3 -m MaxText.train MaxText/configs/base.yml run_name=${USER}-first-job base_output_directory=${BASE_OUTPUT_DIR} dataset_path=${DATASET_PATH} steps=100 per_device_batch_size=1"
      ```
