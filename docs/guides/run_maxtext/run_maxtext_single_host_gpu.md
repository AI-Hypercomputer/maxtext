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

# Via single host GPU

This is a short guide to run Maxtext on GPU. For this current set of instructions the GPUs used are A3-high. This is a single node 8 H100 instruction.

## Create a GPU VM

Follow the instructions to create a3 high or an a3 Mega VM
- https://cloud.google.com/compute/docs/gpus/create-gpu-vm-accelerator-optimized#console
- Add enough disk space to work through the examples (at least 500GB)

Ssh into your host:

```bash
gcloud compute ssh --zone "xxx" "hostname" --project "project name"
```

## Install the CUDA libraries

Install CUDA prior to starting:

- Follow the [instructions](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu) to install CUDA
- Check nvida-smi is working
- Check nvcc

Related NVIDIA Content:

- NVIDIA JAX Session:
- Learn more about Jax on GPUs:
    - https://www.nvidia.com/en-us/on-demand/session/gtc24-s62246/
- NVIDIA JAX Toolbox:
    - https://github.com/NVIDIA/JAX-Toolbox

## Install Docker

Follow the following steps to install docker
https://docs.docker.com/engine/install/debian/

## Install NVIDIA Container Toolkit

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

If you get the NVML Error: Please follow these instructions.

https://stackoverflow.com/questions/72932940/failed-to-initialize-nvml-unknown-error-in-docker-after-few-hours

## Install MaxText

Clone MaxText:

```bash
git clone https://github.com/AI-Hypercomputer/maxtext.git
```

## Build MaxText Docker image

This builds a docker image called `maxtext_base_image`. You can retag to a different name.

1. Check out the code changes:

```bash
cd maxtext
```

2. Run the following commands to build and push the docker image:

```bash
export LOCAL_IMAGE_NAME=<docker_image_name>
sudo bash docker_build_dependency_image.sh DEVICE=gpu
docker tag maxtext_base_image $LOCAL_IMAGE_NAME
docker push $LOCAL_IMAGE_NAME
```

Note that when running `bash docker_build_dependency_image.sh DEVICE=gpu`, it
uses `MODE=stable` by default. If you want to use other modes, you need to
specify it explicitly:

- using nightly mode: `bash docker_build_dependency_image.sh DEVICE=gpu MODE=nightly`
- using pinned mode: `bash docker_build_dependency_image.sh DEVICE=gpu MODE=pinned`

## Test

Test the docker, to see if jax can see all the 8 GPUs

```bash
sudo docker run maxtext_base_image:latest python3 -c "import jax; print(jax.devices())"
```

You should see the following:

```
[CudaDevice(id=0), CudaDevice(id=1), CudaDevice(id=2), CudaDevice(id=3), CudaDevice(id=4), CudaDevice(id=5), CudaDevice(id=6), CudaDevice(id=7)]
```

Note: If you only see CPUDevice, that means there is a issue with NVIDIA Container and you need to stop and fix the issue.

We will Run the next commands from inside the docker for convenience. 

## SSH into the docker

```bash
sudo docker run --runtime=nvidia --gpus all -it maxtext_base_image:latest bash
```

If you do not wish to ssh execute the next set of commands by prepending the following:

```bash
sudo docker run --runtime=nvidia --gpus all -it maxtext_base_image:latest ....
```

### Test a 1B model training

```bash
export JAX_COORDINATOR_ADDRESS=localhost
export JAX_COORDINATOR_PORT=2222
export GPUS_PER_NODE=8
export NODE_RANK=0
export NNODES=1
```

Update script and run the command with synthetic data:

```
base_output_directory: A GCS Bucket 
dataset_type: Synthetic or pass a real bucket
attention:cudnn_flash_te (The default in maxtext is flash. Flash does not work on GPUs)
scan_layers=False 
use_iota_embed=True 
hardware=gpu
per_device_batch_size=12 [Update this to get a better MFU]
Hardware: GPU
```

```bash
python3 -m MaxText.train src/MaxText/configs/base.yml run_name=gpu01 base_output_directory=/deps/output  \
  dataset_type=synthetic enable_checkpointing=True steps=10 attention=cudnn_flash_te scan_layers=False \
  use_iota_embed=True hardware=gpu per_device_batch_size=12
```

### Test a LLama2-7B model training

You can find the optimized running of LLama Models for various host configurations here:

https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/configs/a3/llama_2_7b

`1vm.sh` modified script below:

```bash
echo "Running 1vm.sh"

# Example command to invoke this script via XPK
# python3 xpk/xpk.py workload create --cluster ${CLUSTER_NAME} \
# --workload ${WORKLOAD_NAME} --docker-image=gcr.io/supercomputer-testing/${LOCAL_IMAGE_NAME} \
# --device-type ${DEVICE_TYPE} --num-slices 1 \
# --command "bash src/MaxText/configs/a3/llama_2_7b/1vm.sh"

# Stop execution if any command exits with error
set -e

export OUTPUT_PATH="provide an output path"
export RUN_NAME="llama-2-1vm-$(date +%Y-%m-%d-%H-%M)"

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

export XLA_FLAGS="--xla_dump_to=$OUTPUT_PATH/$RUN_NAME/HLO_dumps/
--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false
 --xla_gpu_enable_command_buffer='' --xla_gpu_enable_highest_priority_async_stream=true
 --xla_gpu_all_reduce_combine_threshold_bytes=134217728 --xla_gpu_all_gather_combine_threshold_bytes=134217728
 --xla_gpu_reduce_scatter_combine_threshold_bytes=67108864 --xla_gpu_enable_pipelined_all_gather=true
 --xla_gpu_enable_pipelined_reduce_scatter=true --xla_gpu_enable_pipelined_all_reduce=true
 --xla_gpu_enable_while_loop_double_buffering=true --xla_gpu_enable_triton_softmax_fusion=false
 --xla_gpu_enable_all_gather_combine_by_dim=false --xla_gpu_enable_reduce_scatter_combine_by_dim=false
 --xla_disable_hlo_passes=rematerialization"


# 1 node, DATA_DP=1, ICI_FSDP=8
python3 -m MaxText.train src/MaxText/configs/models/gpu/llama2_7b.yml run_name=$RUN_NAME dcn_data_parallelism=1 \
  ici_fsdp_parallelism=8 base_output_directory=$OUTPUT_PATH attention=cudnn_flash_te scan_layers=False \
  use_iota_embed=True hardware=gpu
```
