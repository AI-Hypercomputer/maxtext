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

# High performance 16B, 32B, 64B, and 128B models on TPU v5e
The following are details on how to achieve up to 66.86% MFU results on TPU v5e for 16B, 32B, 64B, and 128B parameter model configurations. 

| Hardware    | 16B TFLOP/sec/chip | 16B MFU | 32B TFLOP/sec/chip | 32B MFU | 64B TFLOP/sec/chip | 64B MFU | 128B TFLOP/sec/chip | 128B MFU |
| ----------- | -----------------: | ------- | -----------------: | ------- | -----------------: | ------- | ------------------: | -------- |
| 1x v5e-256  | 120                | 61.10%  | 132                | 66.86%  | 118                | 59.90%  | 110                 | 56.06%   |
| 2x v5e-256  | 117                | 59.37%  | 128                | 64.81%  | 112                | 56.66%  | 110                 | 55.82%   |
| 4x v5e-256  | 117                | 59.14%  | 126                | 64.10%  | 110                | 55.85%  | 108                 | 54.93%   |
| 8x v5e-256  | 115                | 58.27%  | 125                | 63.67%  | 108                | 54.96%  | 104                 | 52.93%   |
| 16x v5e-256 | 111                | 56.56%  | 123                | 62.26%  | 105                | 53.29%  | 100                 | 50.86%   |
| 32x v5e-256 | 108                | 54.65%  | 119                | 60.40%  | 99                 | 50.18%  | 91                  | 46.25%   |

We have shown full scaling performance results up to 32 v5e pods for these configurations, however they are capable of scaling up to hundreds of v5e pods. The full configurations and scripts can be found in [MaxText/configs/largest_job/](https://github.com/google/maxtext/tree/main/MaxText/configs/largest_job). 

Note that these configurations do 3 things:
* Sets various XLA compiler flags as `LIBTPU_INIT_ARGS` to optimize runtime performance.
* Runs [rto_setup.sh](https://github.com/google/maxtext/blob/main/rto_setup.sh) or [gke_rto_setup.sh](https://github.com/google/maxtext/blob/main/gke_rto_setup.sh) to optimize communication protocols for network performance. 
(This only needs to be run once on each worker)
* Runs [train.py](https://github.com/google/maxtext/blob/main/MaxText/train.py) with specific hyper-parameters (batch size, etc.)


## Reproduction Instructions
Depending on your organization's set up, these instructions may vary. Here we provide instructions for running on a vanilla GCE setup or a GKE with XPK setup. Feel free to reach out with questions if your organization has a unique setup.

### Running on GCE
1. Create a custom MTU network to optimize network performance and give it firewall rules. If you are unable to complete this step, you may skip it. This step is not necessary, however performance degradation from scaling up the number of v5e pods over DCN will become more noticable.
     
     Create a network with an MTU of 8896 bytes and set up firewall rules. (Creating a network requires `compute.networks.create` permission in your project)
     ```
     gcloud compute networks create mtu9k --mtu=8896 --project=${PROJECT} --subnet-mode=auto --bgp-routing-mode=regional
     ```
     ```
     gcloud compute firewall-rules create mtu9kfw --network mtu9k --allow tcp,icmp,udp --project=${PROJECT}
     ```
     
     When you create your TPUs, you need to indicate they should be part of this network using the `--network` flag (`--network=mtu9k`). Below is an 
     example of a queued-resources request
     ```
     gcloud alpha compute tpus queued-resources create ${QR_ID} --node-prefix=${TPU_NAME} --node-count=${NUM_SLICES} --accelerator_type=${ACCELERATOR_TYPE} --runtime_version=${RUNTIME_VERSION} --network=mtu9k --project=${PROJECT} --zone=${ZONE}
     ```
     Note: If you want to use only one slice, you need to replace node-prefix with node-id, and remove node-count

2. Install MaxText on your runner.
     ```
     # Install maxtext
     git clone git@github.com:google/maxtext.git
     ```


3. Download dataset and set up GCS paths. If you have not downloaded the dataset before, we recommend following [README.md](https://github.com/google/maxtext/blob/main/README.md#getting-started-download-dataset-and-configure).


4. Install MaxText dependencies and run 16b.sh, 32b.sh, 64b.sh, or 128b.sh on __each worker__.
     ```
     bash setup.sh && bash MaxText/configs/largest_job/128b.sh RUN_NAME=${YOUR_RUN_NAME} OUTPUT_PATH=${MAXTEXT_OUTPUT_PATH} DATASET_PATH=${MAXTEXT_DATASET_PATH} PLATFORM=gce
     ```

     We recommend either the orchestration tool [multihost_runner.py](https://github.com/google/maxtext/blob/main/README.md#getting-started-quick-experiments-on-multiple-slices) 
     to quickly get code up and running for fast experimentation or 
     [multihost_job.py](https://github.com/google/maxtext/blob/main/README.md#getting-started-production-jobs-on-multiple-slices) for longer training runs. If you use these tools, 
     then use the above as the input to `--COMMAND`, e.g.:
     ```
     python3 multihost_runner.py --TPU_PREFIX=${TPU_PREFIX} --COMMAND="bash setup.sh && bash MaxText/configs/largest_job/128b.sh RUN_NAME=${YOUR_RUN_NAME} OUTPUT_PATH=${MAXTEXT_OUTPUT_PATH} DATASET_PATH=${MAXTEXT_DATASET_PATH} PLATFORM=gce"
     ```

### Running on GKE with XPK
These instructions show how to run the model configurations using XPK on top of GKE. More information and usage of XPK can be found [here](https://github.com/google/maxtext/tree/main/xpk).
1. Create a custom MTU network to optimize network performance and give it firewall rules. If you are unable to complete this step, you may skip it. This step is not necessary, however performance degradation from scaling up the number of v5e pods over DCN will become more noticable.
     
     Create a network with an MTU of 8896 bytes and set up firewall rules. (Creating a network requires `compute.networks.create` permission in your project)
     ```
     gcloud compute networks create mtu9k --mtu=8896 --project=${PROJECT} --subnet-mode=auto --bgp-routing-mode=regional
     ```
     ```
     gcloud compute firewall-rules create mtu9kfw --network mtu9k --allow tcp,icmp,udp --project=${PROJECT}
     ```

2. Install MaxText on your runner.
     ```
     # Install maxtext
     git clone git@github.com:google/maxtext.git
     ```


3. Download dataset and set up GCS paths. If you have not downloaded the dataset before, we recommend following [README.md](https://github.com/google/maxtext/blob/main/README.md#getting-started-download-dataset-and-configure).

4. Create a GKE cluster with the custom network created in Step 1.
     ```
     python3 xpk/xpk.py cluster create --cluster ${YOUR_CLUSTER_NAME} --num-slices=${NUM_SLICES} --tpu-type=${ACCELERATOR_TYPE} --zone=${ZONE} --project=${PROJECT} --custom-cluster-arguments="--network=mtu9k --subnetwork=mtu9k"
     ```

     If you did not create a custom network in Step 1, create a GKE cluster by running the same command without the `--network` and `--subnetwork` flags.
     ```
     python3 xpk/xpk.py cluster create --cluster ${YOUR_CLUSTER_NAME} --num-slices=${NUM_SLICES} --tpu-type=${ACCELERATOR_TYPE} --zone=${ZONE} --project=${PROJECT}
     ```

5. On your runner, build a MaxText dependency docker image.
     ```
     bash docker_build_dependency_image.sh
     ```

6. Upload the docker image to your GCP project.
     ```
     # Be sure to have your project and compute/zone set
     gcloud config set project ${PROJECT}
     gcloud config set compute/zone ${ZONE}

     # Upload docker image
     bash docker_upload_runner.sh CLOUD_IMAGE_NAME=${YOUR_IMAGE_NAME}
     ```

7. Submit a job to the cluster to run 16b.sh, 32b.sh, 64b.sh, or 128b.sh.
     ```
     python3 /xpk/xpk.py workload create --cluster ${YOUR_CLUSTER_NAME} --docker-image gcr.io/${PROJECT}/${YOUR_IMAGE_NAME} --workload ${YOUR_RUN_NAME} --tpu-type=${ACCELERATOR_TYPE} --num-slices=${NUM_SLICES} --command "bash MaxText/configs/largest_job/128b.sh OUTPUT_PATH=${MAXTEXT_OUTPUT_PATH} DATASET_PATH=${MAXTEXT_DATASET_PATH}"
     ```

## Caveats
We have found that for the smaller models (16B and 32B), a fairly large batch size gives the best MFU (these configurations match a per pod batch size of 
3.1 million tokens for 16B and 2.1 million tokens for 32B). This provides adequate scalability for the 16B and 32B models to converge in 15 days while 
not exceeding the ~8M token global batch size budget. You can also slightly lower the batch size with a fairly modest performance degradataion.