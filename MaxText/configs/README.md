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
* Sets various XLA compiler flags (see [below](/MaxText/configs#xla-flags-used-by-maxtext)) as `LIBTPU_INIT_ARGS` to optimize runtime performance.
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

## XLA flags used by MaxText
Here are some of the most common XLA compiler flags used by MaxText.

| Flag | Type | Notes |
| ---- | ---- | ----- |
| xla_tpu_enable_data_parallel_all_reduce_opt | Boolean (true/false) | Optimization to increase overlap opportunities for DCN (data center networking) all-reduces used for data parallel sharding. <br> **Usage:**  [v5p/32B](/MaxText/configs/v5p/32b.sh) [v5p/64B](/MaxText/configs/v5p/64b.sh) [v5p/128B](/MaxText/configs/v5p/128b.sh) [v5p/256B](/MaxText/configs/v5p/256b.sh) [v5p/512B](/MaxText/configs/v5p/512b.sh) [v5p/1024B](/MaxText/configs/v5p/1024b.sh) [v5e/16B](/MaxText/configs/v5e/16B.sh) [v5e/32B](/MaxText/configs/v5e/32b.sh) [v5e/64B](/MaxText/configs/v5e/64b.sh) [v5e/128B](/MaxText/configs/v5e/128b.sh) [v5e/Llama2-7B](/MaxText/configs/v5e/llama2_7b.sh) [v5e/Llama2-13B](/MaxText/configs/v5e/llama2_13b.sh) [v5e/Llama2-70B](/MaxText/configs/v5e/llama2_70b.sh) [v5e/GPT3-175B](/MaxText/configs/v5e/gpt3_175b.sh) |
| xla_tpu_data_parallel_opt_different_sized_ops | Boolean (true/false) | Enables pipelining of data parallel ops across multiple iterations even if their output sizes doesn't match what can Be saved in place in the stacked variables. Can increase memory pressure.  <br> **Usage:**  [v5p/32B](/MaxText/configs/v5p/32b.sh) [v5p/64B](/MaxText/configs/v5p/64b.sh) [v5p/128B](/MaxText/configs/v5p/128b.sh) [v5p/256B](/MaxText/configs/v5p/256b.sh) [v5p/512B](/MaxText/configs/v5p/512b.sh) [v5p/1024B](/MaxText/configs/v5p/1024b.sh) [v5e/16B](/MaxText/configs/v5e/16B.sh) [v5e/32B](/MaxText/configs/v5e/32b.sh) [v5e/64B](/MaxText/configs/v5e/64b.sh) [v5e/128B](/MaxText/configs/v5e/128b.sh) [v5e/Llama2-7B](/MaxText/configs/v5e/llama2_7b.sh) [v5e/Llama2-13B](/MaxText/configs/v5e/llama2_13b.sh) [v5e/Llama2-70B](/MaxText/configs/v5e/llama2_70b.sh) [v5e/GPT3-175B](/MaxText/configs/v5e/gpt3_175b.sh) |
| xla_tpu_enable_async_collective_fusion | Boolean (true/false) | Enables the pass which fuses async collective communications with compute ops (output/loop-fusion or convolution) that are scheduled between their -start and -done instructions.  <br> **Usage:**  [v5p/32B](/MaxText/configs/v5p/32b.sh) [v5p/64B](/MaxText/configs/v5p/64b.sh) [v5p/128B](/MaxText/configs/v5p/128b.sh) [v5p/256B](/MaxText/configs/v5p/256b.sh) [v5p/512B](/MaxText/configs/v5p/512b.sh) [v5p/1024B](/MaxText/configs/v5p/1024b.sh) [v5e/16B](/MaxText/configs/v5e/16B.sh) [v5e/32B](/MaxText/configs/v5e/32b.sh) [v5e/64B](/MaxText/configs/v5e/64b.sh) [v5e/128B](/MaxText/configs/v5e/128b.sh) [v5e/Llama2-7B](/MaxText/configs/v5e/llama2_7b.sh) [v5e/Llama2-13B](/MaxText/configs/v5e/llama2_13b.sh) [v5e/Llama2-70B](/MaxText/configs/v5e/llama2_70b.sh) [v5e/GPT3-175B](/MaxText/configs/v5e/gpt3_175b.sh) |
| xla_tpu_enable_async_collective_fusion_fuse_all_gather | TristateFlag (true/false/kAuto) | Enables fusing all-gathers within the AsyncCollectiveFusion pass. <br>If set to ``kAuto``, it will be enabled based on the target."<br> **Usage:**  [v5p/32B](/MaxText/configs/v5p/32b.sh) [v5p/64B](/MaxText/configs/v5p/64b.sh) [v5p/128B](/MaxText/configs/v5p/128b.sh) [v5p/256B](/MaxText/configs/v5p/256b.sh) [v5p/512B](/MaxText/configs/v5p/512b.sh) [v5p/1024B](/MaxText/configs/v5p/1024b.sh) [v5e/16B](/MaxText/configs/v5e/16B.sh) [v5e/32B](/MaxText/configs/v5e/32b.sh) [v5e/64B](/MaxText/configs/v5e/64b.sh) [v5e/128B](/MaxText/configs/v5e/128b.sh) [v5e/Llama2-7B](/MaxText/configs/v5e/llama2_7b.sh) [v5e/Llama2-13B](/MaxText/configs/v5e/llama2_13b.sh) [v5e/Llama2-70B](/MaxText/configs/v5e/llama2_70b.sh) [v5e/GPT3-175B](/MaxText/configs/v5e/gpt3_175b.sh) |
| xla_tpu_enable_async_collective_fusion_multiple_steps | Boolean (true/false) | Enables continuing the same async collective in multiple steps (fusions) in the AsyncCollectiveFusion pass. <br> **Usage:**  [v5p/32B](/MaxText/configs/v5p/32b.sh) [v5p/64B](/MaxText/configs/v5p/64b.sh) [v5p/128B](/MaxText/configs/v5p/128b.sh) [v5p/256B](/MaxText/configs/v5p/256b.sh) [v5p/512B](/MaxText/configs/v5p/512b.sh) [v5p/1024B](/MaxText/configs/v5p/1024b.sh) [v5e/16B](/MaxText/configs/v5e/16B.sh) [v5e/32B](/MaxText/configs/v5e/32b.sh) [v5e/64B](/MaxText/configs/v5e/64b.sh) [v5e/128B](/MaxText/configs/v5e/128b.sh) [v5e/Llama2-7B](/MaxText/configs/v5e/llama2_7b.sh) [v5e/Llama2-13B](/MaxText/configs/v5e/llama2_13b.sh) [v5e/Llama2-70B](/MaxText/configs/v5e/llama2_70b.sh) [v5e/GPT3-175B](/MaxText/configs/v5e/gpt3_175b.sh) |
| xla_tpu_overlap_compute_collective_tc | Boolean (true/false) | Enables the overlap of compute and communication on a single TensorCore, i.e., one core equivalent of MegaCore fusion. <br> **Usage:**  [v5p/32B](/MaxText/configs/v5p/32b.sh) [v5p/64B](/MaxText/configs/v5p/64b.sh) [v5p/128B](/MaxText/configs/v5p/128b.sh) [v5p/256B](/MaxText/configs/v5p/256b.sh) [v5p/512B](/MaxText/configs/v5p/512b.sh) [v5p/1024B](/MaxText/configs/v5p/1024b.sh) [v5e/16B](/MaxText/configs/v5e/16B.sh) [v5e/32B](/MaxText/configs/v5e/32b.sh) [v5e/64B](/MaxText/configs/v5e/64b.sh) [v5e/128B](/MaxText/configs/v5e/128b.sh) [v5e/Llama2-7B](/MaxText/configs/v5e/llama2_7b.sh) [v5e/Llama2-13B](/MaxText/configs/v5e/llama2_13b.sh) [v5e/Llama2-70B](/MaxText/configs/v5e/llama2_70b.sh) [v5e/GPT3-175B](/MaxText/configs/v5e/gpt3_175b.sh) |
| xla_enable_async_all_gather | TristateFlag (true/false/kAuto) | If set to true, enables async all gather. If ``kAuto``, enables only for platforms that implement async all-gather. The implementation (such as BC-offload or continuation fusion) is chosen based on other flag values.  <br> **Usage:**  [v4/22B](/MaxText/configs/v4/22B.sh) [v4/52B](/MaxText/configs/v4/52B.sh) [v5p/32B](/MaxText/configs/v5p/32b.sh) [v5p/64B](/MaxText/configs/v5p/64b.sh) [v5p/128B](/MaxText/configs/v5p/128b.sh) [v5p/256B](/MaxText/configs/v5p/256b.sh) [v5p/512B](/MaxText/configs/v5p/512b.sh) [v5p/1024B](/MaxText/configs/v5p/1024b.sh) [v5e/16B](/MaxText/configs/v5e/16B.sh) [v5e/32B](/MaxText/configs/v5e/32b.sh) [v5e/64B](/MaxText/configs/v5e/64b.sh) [v5e/128B](/MaxText/configs/v5e/128b.sh) [v5e/Llama2-7B](/MaxText/configs/v5e/llama2_7b.sh) [v5e/Llama2-13B](/MaxText/configs/v5e/llama2_13b.sh) [v5e/Llama2-70B](/MaxText/configs/v5e/llama2_70b.sh) [v5e/GPT3-175B](/MaxText/configs/v5e/gpt3_175b.sh) |
| xla_tpu_spmd_rng_bit_generator_unsafe | Boolean (true/false) | Whether to run RngBitGenerator HLO in a partitioned way, which is unsafe if deterministic results are expected with different shardings on different parts of the computation. <br> **Usage:**  [v5e/GPT3-175B](/MaxText/configs/v5e/gpt3_175b.sh) |
| xla_tpu_megacore_fusion_allow_ags | Boolean (true/false) | Allows fusing all-gathers with convolutions/all-reduces. <br> **Usage:**  [v5p/32B](/MaxText/configs/v5p/32b.sh) [v5p/64B](/MaxText/configs/v5p/64b.sh) [v5p/128B](/MaxText/configs/v5p/128b.sh) [v5p/256B](/MaxText/configs/v5p/256b.sh) [v5p/512B](/MaxText/configs/v5p/512b.sh) [v5p/1024B](/MaxText/configs/v5p/1024b.sh) |
| xla_tpu_enable_ag_backward_pipelining | Boolean (true/false) | Pipelines all-gathers (currently megascale all-gathers) backwards through scan loops. <br> **Usage:**  [v5p/32B](/MaxText/configs/v5p/32b.sh) [v5p/64B](/MaxText/configs/v5p/64b.sh) [v5p/128B](/MaxText/configs/v5p/128b.sh) [v5p/256B](/MaxText/configs/v5p/256b.sh) [v5p/512B](/MaxText/configs/v5p/512b.sh) [v5p/1024B](/MaxText/configs/v5p/1024b.sh) |
| xla_enable_async_collective_permute | TristateFlag (true/false/kAuto) | Rewrites all collective-permute operations to their asynchronous variants.  When set to ``kAuto``, XLA can turn on async collective based on other configurations or conditions automatically. <br> **Usage:**  [v5p/32B](/MaxText/configs/v5p/32b.sh) [v5p/64B](/MaxText/configs/v5p/64b.sh) [v5p/128B](/MaxText/configs/v5p/128b.sh) [v5p/256B](/MaxText/configs/v5p/256b.sh) [v5p/512B](/MaxText/configs/v5p/512b.sh) [v5p/1024B](/MaxText/configs/v5p/1024b.sh) |
| xla_dump_to | String (filepath) | The folder where pre-optimization HLO files and other artifacts will be placed (see [XLA Tools](https://openxla.org/xla/tools)).  <br> **Usage:**  [a3/Llama2-7B 1vm](/MaxText/configs/a3/llama_2_7B/1vm.sh) [a3/Llama2-7B 2vm](/MaxText/configs/a3/llama_2_7B/2vm.sh) [a3/Llama2-7B 4vm](/MaxText/configs/a3/llama_2_7B/4vm.sh) [a3/Llama2-7B 8vm](/MaxText/configs/a3/llama_2_7B/8vm.sh) [a3/Llama2-7B 16vm](/MaxText/configs/a3/llama_2_7B/16vm.sh) |
| xla_gpu_enable_latency_hiding_scheduler | Boolean (true/false) |This flag enables latency hiding schedulers to overlap asynchronous communication with computation efficiently. The default value is False. <br> **Usage:**  [a3/Llama2-7B 1vm](/MaxText/configs/a3/llama_2_7B/1vm.sh) [a3/Llama2-7B 2vm](/MaxText/configs/a3/llama_2_7B/2vm.sh) [a3/Llama2-7B 4vm](/MaxText/configs/a3/llama_2_7B/4vm.sh) [a3/Llama2-7B 8vm](/MaxText/configs/a3/llama_2_7B/8vm.sh) [a3/Llama2-7B 16vm](/MaxText/configs/a3/llama_2_7B/16vm.sh) |
| xla_gpu_enable_triton_gemm | Boolean (true/false) | Use Triton-based matrix multiplication. <br> **Usage:**  [a3/Llama2-7B 1vm](/MaxText/configs/a3/llama_2_7B/1vm.sh) [a3/Llama2-7B 2vm](/MaxText/configs/a3/llama_2_7B/2vm.sh) [a3/Llama2-7B 4vm](/MaxText/configs/a3/llama_2_7B/4vm.sh) [a3/Llama2-7B 8vm](/MaxText/configs/a3/llama_2_7B/8vm.sh) [a3/Llama2-7B 16vm](/MaxText/configs/a3/llama_2_7B/16vm.sh) |
| xla_gpu_graph_level | Flag (0-3) | The legacy flag for setting GPU graph level. Use xla_gpu_enable_command_buffer in new use cases. 0 = off; 1 = capture fusions and memcpys; 2 = capture gemms; 3 = capture convolutions. <br> **Usage:**  [a3/Llama2-7B 1vm](/MaxText/configs/a3/llama_2_7B/1vm.sh) [a3/Llama2-7B 2vm](/MaxText/configs/a3/llama_2_7B/2vm.sh) [a3/Llama2-7B 4vm](/MaxText/configs/a3/llama_2_7B/4vm.sh) [a3/Llama2-7B 8vm](/MaxText/configs/a3/llama_2_7B/8vm.sh) [a3/Llama2-7B 16vm](/MaxText/configs/a3/llama_2_7B/16vm.sh) |
| xla_gpu_all_reduce_combine_threshold_bytes | Integer (bytes) | These flags tune when to combine multiple small AllGather / ReduceScatter / AllReduce into one big AllGather / ReduceScatter / AllReduce to reduce time spent on cross-device communication. For example, for the AllGather / ReduceScatter thresholds on a Transformer-based workload, consider tuning them high enough so as to combine at least a Transformer Layer’s weight AllGather / ReduceScatter. By default, the combine_threshold_bytes is set to 256. <br> **Usage:**  [a3/Llama2-7B 1vm](/MaxText/configs/a3/llama_2_7B/1vm.sh) [a3/Llama2-7B 2vm](/MaxText/configs/a3/llama_2_7B/2vm.sh) [a3/Llama2-7B 4vm](/MaxText/configs/a3/llama_2_7B/4vm.sh) [a3/Llama2-7B 8vm](/MaxText/configs/a3/llama_2_7B/8vm.sh) [a3/Llama2-7B 16vm](/MaxText/configs/a3/llama_2_7B/16vm.sh) |
| xla_gpu_all_gather_combine_threshold_bytes | Integer (bytes) | See xla_gpu_all_reduce_combine_threshold_bytes above. <br> **Usage:**  [a3/Llama2-7B 1vm](/MaxText/configs/a3/llama_2_7B/1vm.sh) [a3/Llama2-7B 2vm](/MaxText/configs/a3/llama_2_7B/2vm.sh) [a3/Llama2-7B 4vm](/MaxText/configs/a3/llama_2_7B/4vm.sh) [a3/Llama2-7B 8vm](/MaxText/configs/a3/llama_2_7B/8vm.sh) [a3/Llama2-7B 16vm](/MaxText/configs/a3/llama_2_7B/16vm.sh) |
| xla_gpu_reduce_scatter_combine_threshold_bytes | Integer (bytes) | See xla_gpu_all_reduce_combine_threshold_bytes above. <br> **Usage:**  [a3/Llama2-7B 1vm](/MaxText/configs/a3/llama_2_7B/1vm.sh) [a3/Llama2-7B 2vm](/MaxText/configs/a3/llama_2_7B/2vm.sh) [a3/Llama2-7B 4vm](/MaxText/configs/a3/llama_2_7B/4vm.sh) [a3/Llama2-7B 8vm](/MaxText/configs/a3/llama_2_7B/8vm.sh) [a3/Llama2-7B 16vm](/MaxText/configs/a3/llama_2_7B/16vm.sh) |
| xla_gpu_enable_pipelined_all_gather | Boolean (true/false) | Enable pipelinling of all-gather instructions. <br> **Usage:**  [a3/Llama2-7B 1vm](/MaxText/configs/a3/llama_2_7B/1vm.sh) [a3/Llama2-7B 2vm](/MaxText/configs/a3/llama_2_7B/2vm.sh) [a3/Llama2-7B 4vm](/MaxText/configs/a3/llama_2_7B/4vm.sh) [a3/Llama2-7B 8vm](/MaxText/configs/a3/llama_2_7B/8vm.sh) [a3/Llama2-7B 16vm](/MaxText/configs/a3/llama_2_7B/16vm.sh) |
| xla_gpu_enable_pipelined_reduce_scatter | Boolean (true/false) | Enable pipelinling of reduce-scatter instructions. <br> **Usage:**  [a3/Llama2-7B 1vm](/MaxText/configs/a3/llama_2_7B/1vm.sh) [a3/Llama2-7B 2vm](/MaxText/configs/a3/llama_2_7B/2vm.sh) [a3/Llama2-7B 4vm](/MaxText/configs/a3/llama_2_7B/4vm.sh) [a3/Llama2-7B 8vm](/MaxText/configs/a3/llama_2_7B/8vm.sh) [a3/Llama2-7B 16vm](/MaxText/configs/a3/llama_2_7B/16vm.sh) |
| xla_gpu_enable_pipelined_all_reduce | Boolean (true/false) | Enable pipelinling of all-reduce instructions. <br> **Usage:**  [a3/Llama2-7B 1vm](/MaxText/configs/a3/llama_2_7B/1vm.sh) [a3/Llama2-7B 2vm](/MaxText/configs/a3/llama_2_7B/2vm.sh) [a3/Llama2-7B 4vm](/MaxText/configs/a3/llama_2_7B/4vm.sh) [a3/Llama2-7B 8vm](/MaxText/configs/a3/llama_2_7B/8vm.sh) [a3/Llama2-7B 16vm](/MaxText/configs/a3/llama_2_7B/16vm.sh) |
| xla_gpu_enable_while_loop_double_buffering | Boolean (true/false) | Enable double-buffering for while loop. <br> **Usage:**  [a3/Llama2-7B 1vm](/MaxText/configs/a3/llama_2_7B/1vm.sh) [a3/Llama2-7B 2vm](/MaxText/configs/a3/llama_2_7B/2vm.sh) [a3/Llama2-7B 4vm](/MaxText/configs/a3/llama_2_7B/4vm.sh) [a3/Llama2-7B 8vm](/MaxText/configs/a3/llama_2_7B/8vm.sh) [a3/Llama2-7B 16vm](/MaxText/configs/a3/llama_2_7B/16vm.sh) |
| xla_gpu_enable_triton_softmax_fusion | Boolean (true/false) | Use Triton-based Softmax fusion. <br> **Usage:**  [a3/Llama2-7B 1vm](/MaxText/configs/a3/llama_2_7B/1vm.sh) [a3/Llama2-7B 2vm](/MaxText/configs/a3/llama_2_7B/2vm.sh) [a3/Llama2-7B 4vm](/MaxText/configs/a3/llama_2_7B/4vm.sh) [a3/Llama2-7B 8vm](/MaxText/configs/a3/llama_2_7B/8vm.sh) [a3/Llama2-7B 16vm](/MaxText/configs/a3/llama_2_7B/16vm.sh) |
| xla_gpu_enable_all_gather_combine_by_dim | Boolean (true/false) | Combine all-gather ops with the same gather dimension or irrespective of their dimension. <br> **Usage:**  [a3/Llama2-7B 1vm](/MaxText/configs/a3/llama_2_7B/1vm.sh) [a3/Llama2-7B 2vm](/MaxText/configs/a3/llama_2_7B/2vm.sh) [a3/Llama2-7B 4vm](/MaxText/configs/a3/llama_2_7B/4vm.sh) [a3/Llama2-7B 8vm](/MaxText/configs/a3/llama_2_7B/8vm.sh) [a3/Llama2-7B 16vm](/MaxText/configs/a3/llama_2_7B/16vm.sh) |
| xla_gpu_enable_reduce_scatter_combine_by_dim | Boolean (true/false) | Combine reduce-scatter ops with the same dimension or irrespective of their dimension. <br> **Usage:**  [a3/Llama2-7B 1vm](/MaxText/configs/a3/llama_2_7B/1vm.sh) [a3/Llama2-7B 2vm](/MaxText/configs/a3/llama_2_7B/2vm.sh) [a3/Llama2-7B 4vm](/MaxText/configs/a3/llama_2_7B/4vm.sh) [a3/Llama2-7B 8vm](/MaxText/configs/a3/llama_2_7B/8vm.sh) [a3/Llama2-7B 16vm](/MaxText/configs/a3/llama_2_7B/16vm.sh) |
| xla_disable_hlo_passes | String (comma-separated list of pass names) | Comma-separated list of HLO passes to be disabled. These names must exactly match the pass name (no whitespace around commas). <br> **Usage:**  [a3/Llama2-7B 1vm](/MaxText/configs/a3/llama_2_7B/1vm.sh) [a3/Llama2-7B 2vm](/MaxText/configs/a3/llama_2_7B/2vm.sh) [a3/Llama2-7B 4vm](/MaxText/configs/a3/llama_2_7B/4vm.sh) [a3/Llama2-7B 8vm](/MaxText/configs/a3/llama_2_7B/8vm.sh) [a3/Llama2-7B 16vm](/MaxText/configs/a3/llama_2_7B/16vm.sh) |

