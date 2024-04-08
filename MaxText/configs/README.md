<!--
 Copyright 2023 Google LLC

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

# High Performance Model Configs
This directory contains high performance model configurations for different generations of TPU and GPU hardware.

These configurations do 3 things:
* Sets various XLA compiler flags as `LIBTPU_INIT_ARGS` to optimize runtime performance.
* Runs [rto_setup.sh](https://github.com/google/maxtext/blob/main/rto_setup.sh) to optimize communication protocols for network performance. 
(This only needs to be run once on each worker)
* Runs [train.py](https://github.com/google/maxtext/blob/main/MaxText/train.py) with specific hyper-parameters (batch size, etc.)


## Reproduction Instructions

### Create a custom MTU network
1. Create a custom MTU network to optimize network performance and give it firewall rules. If you are unable to complete this step, you may skip it. This step is not necessary, and is only for improving performance when running on a Multislice setup.
     
     Create a network with an MTU of 8896 bytes and set up firewall rules. (Creating a network requires `compute.networks.create` permission in your project)
     ```
     gcloud compute networks create mtu9k --mtu=8896 --project=${PROJECT} --subnet-mode=auto --bgp-routing-mode=regional
     ```
     ```
     gcloud compute firewall-rules create mtu9kfw --network mtu9k --allow tcp,icmp,udp --project=${PROJECT}
     ```
     
     When you create your TPUs, you need to indicate they should be part of this network. 
     
     Here is an example of a queued-resources request on GCE using the `--network` flag (`--network=mtu9k`). 
     ```
     gcloud alpha compute tpus queued-resources create ${QR_ID} --node-prefix=${TPU_NAME} --node-count=${NUM_SLICES} --accelerator_type=${ACCELERATOR_TYPE} --runtime_version=${RUNTIME_VERSION} --network=mtu9k --project=${PROJECT} --zone=${ZONE}
     ```
     Note: If you want to use only one slice, you need to replace node-prefix with node-id, and remove node-count.
     
     Here is an example of creating a GKE cluster with XPK using the `--network` and `--subnetwork` flags (`--network=mtu9k --subnetwork=mtu9k`).
     ```
     export CLUSTER_ARGUMENTS="--network=mtu9k --subnetwork=mtu9k"
     
     python3 xpk/xpk.py cluster create --cluster ${YOUR_CLUSTER_NAME} --tpu-type ${ACCELERATOR_TYPE} --num-slices ${NUM_SLICES} --custom-cluster-arguments="${CLUSTER_ARGUMENTS}"
     ```

### Run model config scripts on TPUs
1. You can run these model configs on the GCE platform using `multihost_runner.py` or `multihost_job.py`, or on the GKE platform using [XPK](https://github.com/google/xpk). Take a look at the [getting_started](https://github.com/google/maxtext/tree/main/getting_started) directory for directions on how to set up your TPUs and use these tools.

2. Here are some example commands to run the model configs:

    Running with `multihost_runner.py` on GCE:
    ```
    python3 multihost_runner.py --TPU_PREFIX=${TPU_PREFIX} --COMMAND="bash setup.sh && bash MaxText/configs/v5p/128b.sh RUN_NAME=${YOUR_RUN_NAME} OUTPUT_PATH=${MAXTEXT_OUTPUT_PATH} DATASET_PATH=${MAXTEXT_DATASET_PATH} PLATFORM=gce"
    ```

    Running with `multihost_job.py` on GCE:
    ```
    python3 multihost_job.py --NUM_SLICES=${NUM_SLICES} --TPU_TYPE=${ACCELERATOR_TYPE} --VERSION=${RUNTIME_VERSION} --RUN_NAME=${RUN_NAME} --BUCKET_NAME=${GCS_BUCKET_NAME} --COMMAND="bash setup.sh && bash MaxText/configs/v5p/128b.sh RUN_NAME=${YOUR_RUN_NAME} OUTPUT_PATH=${MAXTEXT_OUTPUT_PATH} DATASET_PATH=${MAXTEXT_DATASET_PATH} PLATFORM=gce"

    # Add --CQR_EXTRA_ARGS="--network=mtu9k" to the command if you would like to use the custom MTU network.
    ```

    Running with `XPK` on GKE:
    ```
    xpk workload create --cluster ${YOUR_CLUSTER_NAME} --docker-image gcr.io/${PROJECT}/${YOUR_IMAGE_NAME} --workload ${YOUR_RUN_NAME} --tpu-type=${ACCELERATOR_TYPE} --num-slices=${NUM_SLICES} --command "bash MaxText/configs/v5p/128b.sh OUTPUT_PATH=${MAXTEXT_OUTPUT_PATH} DATASET_PATH=${MAXTEXT_DATASET_PATH} PLATFORM=gke"
    ```

    Note: When running these scripts, be sure to specify the `PLATFORM` flag with the correct platform you are running on `"gce"` or `"gke"`.
