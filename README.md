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


[![Unit Tests](https://github.com/google/maxtext/actions/workflows/UnitTests.yml/badge.svg)](https://github.com/google/maxtext/actions/workflows/UnitTests.yml)

# Overview

MaxText is a **high performance**, **arbitrarily scalable**, **open-source**, **simple**, **easily forkable**, **well-tested**, **batteries included** LLM written in pure Python/Jax and targeting Google Cloud TPUs. MaxText typically achieves 55% to 60% model-flop utilization and scales from single host to very large clusters while staying simple and "optimization-free" thanks to the power of Jax and the XLA compiler.

MaxText aims to be a launching off point for ambitious LLM projects both in research and production. We encourage users to start by experimenting with MaxText out of the box and then fork and modify MaxText to meet their needs.

# Table of Contents

* [Getting Started](#getting-started)
* [Runtime Performance Results](#runtime-performance-results)
* [Comparison To Alternatives](#comparison-to-alternatives)
* [Development](#development)
* [Features and Diagnostics](#features-and-diagnostics)

# Getting Started

There are three recommended patterns for running MaxText. You can run locally, run on a cluster experimentally or spawn a production-style that is managed by Google Compute Engine. We recommend starting with Local Development, moving to Cluster Experimentation for some ad hoc development and ultimately running your long running jobs with Queued Resources.

## Getting Started: Download Dataset and Configure
You need to run these steps once per project prior to any local development or cluster experiments.

1. Create two gcs buckets in your project, one for to downloading and retrieving the dataset and the other for storing the logs.
2. Download the dataset in your gcs bucket
```
bash download_dataset.sh {GCS_PROJECT} {GCS_BUCKET_NAME}
```
3. Set config values for `base_output_directory` and `dataset_path` in `configs/base.yml`. `vocab_relative_path` is relative to `base_output_directory` for loading the tokenizer. MaxText assumes these GCS buckets are created in the same project and that it has permissions to read and write from them. We also recommend reviewing the configurable options in `configs/base.yml`, for instance you may change the `steps` or `logging_period` by either modifying `configs/base.yml` or by passing in `steps` and `logging_period` as additional args to the `train.py` call.

To run maxtext the TPUVMs must have permission to read the gcs bucket. These permissions are granted by service account roles, such as the `STORAGE ADMIN` role. 

## Getting Started: Local Development

Local development is the faster and most convenient way to run MaxText. However, it doesn't scale to multiple hosts.

1. [Create and SSH to the single-host TPU of your choice.](https://cloud.google.com/tpu/docs/users-guide-tpu-vm#creating_a_cloud_tpu_vm_with_gcloud) We recommend a `v4-8`.
2. Clone MaxText onto that TPUVM.
3. Within the root directory of that `git` repo, install dependencies by running:
```
bash setup.sh
```
4. After installation completes, run training with the command:
```
python3 MaxText/train.py MaxText/configs/base.yml run_name=$YOUR_JOB_NAME
```

5. If you want to decode, you can decode as follows.
```
python3 MaxText/decode.py MaxText/configs/base.yml run_name=$YOUR_JOB_NAME
```
Be aware, these decodings will be random. To get high quality decodings you need pass in a checkpoint, typically via the `load_parameters_path` argument.

## Getting Started: Quick Experiments on Multiple Hosts (or Multiple Slices)

This workflow using `multihost_runner.py` is optimized for quick experiments, repeatedly re-using the same TPUs. Because the `multihost_runner.py` script depends on long-lived `ssh` connections, we do not recommend it for any long-running jobs.

We call the `runner` machine the one that `multihost_runner.py` is called from. This script will `ssh` into TPUVM `worker` machines that are found from the `--TPU_PREFIX` flag, and must be different than the runner machine.
If the runner machine is a cloud VM, it must be in the same project as the workers.

The `multihost_runner.py` script:
* Distributes your code by recursively copying the current state of the chosen directory to multiple worker TPUVM.
* Runs the code on the workers
* Logs and monitors the processes' error statuses and brings the logs back to the runner machine.

Although there are several steps below, most are for the initial setup. Once setup you can continually make changes to your code and re-run your code with only step 5.

1. Choose a directory on your runner machine to develop and clone MaxText into. The runner machine can 
either be a TPUVM or not, but it cannot be one of the workers. If your runner machine is a TPUVM, it needs service account roles that grant it permission to create queued resources and ssh into them, such as the `TPU ADMIN` role. Clone MaxText, and cd into the root of the repo.

2. Set your project, zone, and ssh keys.

    Set your gcloud config, see https://cloud.google.com/sdk/gcloud/reference/config for more.
    ```
    PROJECT=<project>
    ```
    ```
    ZONE=<zone>
    ```
    ```
    gcloud config set project $PROJECT
    gcloud config set compute/zone $ZONE
    ```

    Create ssh keys for gcloud, we recommend leaving a blank password (hit enter twice after running the below command). If you are prompted that the the file already exists you can choose not to overwrite by selecting "n".
    ```
    ssh-keygen -f ~/.ssh/google_compute_engine 
    ```
    
3. Create your instances via Queued Resource (QR). 
    Choose names for your TPUs and QR:
    ```
    TPU_PREFIX=$YOUR_TPU_NAME # Use new names when you create new TPUs
    QR_ID=$TPU_PREFIX # Convenient to re-use the node names, but can be different
    ```
    Choose the number of nodes (we use 2 below, but you may customize this and other feature of your TPU(s))
    ```
    NODE_COUNT=2
    ```
    Create a multislice environment of nodes using create queued resources
    ```
    gcloud alpha compute tpus queued-resources create $QR_ID --accelerator-type=v4-8 --runtime-version=tpu-ubuntu2204-base --node-count=$NODE_COUNT --node-prefix=$TPU_PREFIX  --reserved
    ```
    We target the `reserved` pool above, but you may instead target the `on-demand` pool by omitting this flag,
    or target pre-emptible capacity with the `--best-effort` flag, which may be necessary if your reservation is full.
 
    You have to wait for the QR to become `ACTIVE` (as opposed to `ACCEPTED` or `PROVISIONING`) which corresponds to the worker nodes becoming `READY` (as opposed to `CREATING`). This may take a minute or two and can be checked via
    ```
    gcloud alpha compute tpus queued-resources list --filter=$QR_ID 
    ```
4. Install dependencies.
    Install the dependencies of `train.py` on each worker using `multihost_runner.py`:
    ```
    python3 multihost_runner.py --TPU_PREFIX=$TPU_PREFIX --COMMAND="bash setup.sh"
    ```
    If you are running the `multihost_runner.py` script from a TPUVM, you will need to set `--INTERNAL_IP=true`.

5. Run your training job.

    Set a RUN_NAME for your job:
    ```
    RUN_NAME=$YOUR_JOB_NAME # You may set this to any unique name for a fresh run.
    ```
    Set config values for `base_output_directory` and `dataset_path` in `configs/base.yml` if not set already.
    ```
    python3 multihost_runner.py --TPU_PREFIX=$TPU_PREFIX --COMMAND="python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME"
    ```
    If you are running the `multihost_runner.py` script from a TPUVM, you will need to set `--INTERNAL_IP=true`.

6. Clean up TPUs and QR when finished.

     ```
    gcloud alpha compute tpus queued-resources delete $QR_ID --force --async
    ```

    The `--force` flag deletes both the queued resources and the TPU VMs, without it only a `SUSPENDED` queued resource whose TPUs have already been deleted can itself be deleted. We highly recommend the `--async` flag since deleting the TPUs and QR will take a minute or two.

## Getting Started: Production Jobs On Multiple Slices

The workflow using `multihost_job.py` is optimized for long running experiments, providing resiliency against hardware failure and avoiding long running ssh connections. Its latency is much higher than `multihost_runner.py` because it needs to provision new capacity each time. The `multihost_job.py` script ends once the request to create the TPUs is issued. Logs are written both to gcloud in real time and also sent to GCS at the end of the job.

The `multihost_job.py` script:

* Copies your code to your GCS bucket
* Spins up specified TPU VM(s) via CQR
* Directs the TPU's to download then run that code. Because this logic is within the CQR's startup script, if there hardware is interrupted, the job will be rescheduled and resumed.
* Logs to gcloud, and additionally sends the logs to GCS at the job end
* Delete the TPUs and QR at the end of the job.

1. Choose a directory on your runner machine to develop and clone MaxText into. The runner machine can 
either be a TPUVM or not. If your runner machine is a TPUVM, it needs service account roles that grant it permission to create queued resources and has write access to GCS, such as the `TPU ADMIN` and `STORAGE ADMIN` roles. Clone MaxText, and cd into the root of the repo.

2. Set your project, zone. 
    Set your gcloud config, see https://cloud.google.com/sdk/gcloud/reference/config for more.
    ```
    PROJECT=<project>
    ```

    ```
    ZONE=<zone>
    ```

    ```
    gcloud config set project $PROJECT
    gcloud config set compute/zone $ZONE
    ```
    
3. Link to a GCS bucket.
    Create a bucket if you don't already have one, see: https://cloud.google.com/storage/docs/creating-buckets for instructions to create one. Once you've identified your bucket:

    ```
    BUCKET_NAME=<your-bucket>
    ```

4. Run your training job.

    *** IMPORTANT *** `multihost_job` creates a request for new capacity for each run! You cannot use this tool on existing capacity, instead we recommend `multihost_runner` for this purpose.

    Choose the number of nodes (we use 2 below, but you may customize this and other feature of your TPU(s))
    ```
    NODE_COUNT=2
    ```
    ```
    RUN_NAME=$YOUR_JOB_NAME # You may set this to any unique name for a fresh run.
    python3 multihost_job.py --NUM_SLICES=$NODE_COUNT --RUN_NAME=$RUN_NAME --BUCKET_NAME=$BUCKET_NAME --CQR_EXTRA_ARGS="--reserved" --COMMAND="bash setup.sh && python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME"
    ```

    We tell `multihost_job` to target the `reserved` pool by  by including `--reserved` as extra arguments to the CQR request, but you may instead target the `on-demand` pool by removing the `--CQR_EXTRA_ARGS` flag (on-demand is default), or the pre-emptible pool with `--CQR_EXTRA_ARGS="--best-effort"`, which may be necessary if your reservation is full.

5. View the job's logs in cloud logging. 

    The link to your job's cloud logging is printed at the end of `multihost_job` output. Additionally logs are saved to GCS when your job finishes, and this bucket's URL is also printed by `multihost_job`.


# Runtime Performance Results

## TPU v4

For a 22B model. See full run configs in `MaxText/configs/` as `1xv4-128.sh`, `2xv4-128.sh`, `4xv4-128.sh`, and `8xv4-128.sh`.

| Hardware    | TFLOP/sec/chip   |  MFU  |
| ----------- | ---------------- | ----- |
| 1x v4-128   | 156              | 56.7% |
| 2x v4-128   | 152              | 55.2% |
| 4x v4-128   | 149              | 54.3% |
| 8x v4-128   | 146              | 53.2% |

For a 52B model. See full run configs in `MaxText/configs/` as `1xv4-384.sh` and `2xv4-384.sh`.

| Hardware    | TFLOP/sec/chip   |  MFU  |
| ----------- | ---------------- | ----- |
| 1x v4-384   | 154              | 56.0% |
| 2x v4-384   | 162              | 58.9% | # this is quirkily higher than single slice because of choices made by the compiler, not for a fundamental reason.

## TPU v5e

For a 16B model. See full run configs in `MaxText/configs/` as `16b.sh`.

| Hardware    | TFLOP/sec/chip   |  MFU  |
| ----------- | ---------------- | ----- |
| 2x v5e-256  | 114              | 57.8% |

For a 32B model. See full run configs in `MaxText/configs/` as `32b.sh`.

| Hardware    | TFLOP/sec/chip   |  MFU  |
| ----------- | ---------------- | ----- |
| 2x v5e-256  | 113              | 57.3% |

More details on reproducing these 16B and 32B model results on v5e can be found in `v5e_high_performance.md`.


# Comparison to Alternatives

MaxText is heavily inspired by [MinGPT](https://github.com/karpathy/minGPT)/[NanoGPT](https://github.com/karpathy/nanoGPT), elegant standalone GPT implementations written in PyTorch and targeting Nvidia GPUs. MaxText is more complex but has an MFU more than three times the [17%](https://twitter.com/karpathy/status/1613250489097027584?cxt=HHwWgIDUhbixteMsAAAA) reported most recently with that codebase, is massively scalable and implements a key-value cache for efficient auto-regressive decoding.

MaxText is more similar to [Nvidia/Megatron-LM](https://github.com/NVIDIA/Megatron-LM), a very well tuned LLM implementation targeting Nvidia GPUs. The two implementations achieve comparable MFUs. The difference in the codebases highlights the different programming strategies. MaxText is pure Python, relying heavily on the XLA compiler to achieve high performance. By contrast, Megatron-LM is a mix of Python and CUDA, relying on well-optimized CUDA kernels to achieve high performance.

MaxText is also comparable to [Pax](https://github.com/google/paxml). Like Pax, MaxText provides high-performance and scalable implementations of LLMs in Jax. Pax focuses on enabling powerful configuration parameters, enabling developers to change the model by editing config parameters. By contrast, MaxText is a simple, concrete implementation of an LLM that encourages users to extend by forking and directly editing the source code. The right choice depends on your project's priorities.

# Development

Whether you are forking MaxText for your own needs or intending to contribute back to the community, we wanted to offer simple testing recipes.

To run unit tests and lint, simply run:
```
bash unit_test_and_lint.sh
```

The full suite of end-to-end tests is in `end_to_end/`. We run them with a nightly cadence.

# Features and Diagnostics
## Collect Stack Traces
When running a Single Program, Multiple Data (SPMD) job on TPU VMs, the overall process can hang if there is any error or any VM hangs/crashes for some reason. In this scenario, capturing stack traces will help to identify and troubleshoot the issues for the jobs running on TPU VMs.

The following configurations will help to debug a fault or when a program is stuck or hung somewhere by collecting stack traces. Change the parameter values accordingly in `MaxText/configs/base.yml`:
1. Set `collect_stack_trace: True` to enable collection of stack traces on faults or when the program is hung. This setting will periodically dump the traces for the program to help in debugging. To disable this, set `collect_stack_trace: False`.
2. Set `stack_trace_to_cloud: False` to display stack traces on console. `stack_trace_to_cloud: True` will create a temporary file in `/tmp/debugging` in the TPUs to store the stack traces. There is an agent running on TPU VMs that will periodically upload the traces from the temporary directory to cloud logging in the gcp project. You can view the traces in Logs Explorer on Cloud Logging using the following query:
```
logName="projects/<project_name>/logs/tpu.googleapis.com%2Fruntime_monitor"
jsonPayload.verb="stacktraceanalyzer"
```
3. `stack_trace_interval_seconds` signifies the duration in seconds between each stack trace collection event. Setting `stack_trace_interval_seconds: 600` will collect the stack traces every 600 seconds (10 minutes).

Here is the related PyPI package: https://pypi.org/project/cloud-tpu-diagnostics.

## Ahead of Time Compilation (AOT)
To compile your training run ahead of time, we provide a tool `train_compile.py`. This tool allows you to compile the main `train_step` in `train.py` for target hardware (e.g. a large number of v5e devices) without using the target hardware, and instead you may use only a CPU or a single VM from a different family. This compilation helps with two main goals:

* It will flag any out of memory (OOM) information, such as when the `per_device_batch_size` is set too high, with an identical OOM stack trace as if it was compiled on the target hardware.

* The ahead of time compilation can be saved and then loaded for fast startup and restart times on the target hardware.

The tool `train_compile.py` is tightly linked to `train.py` and uses the same configuration file `configs/base.yml`. Although you don't need to run on a TPU, you do need to install `jax[tpu]` in addition to other dependenices, so we recommend running `setup.sh` to install these if you have not already done so. 

### Example AOT 1: Compile ahead of time basics
After installing the dependencies listed above, you are ready to compile ahead of time:
```
# Run the below on a single machine, e.g. a CPU
python3 MaxText/train_compile.py MaxText/configs/base.yml compile_topology=v5e-256 compile_topology_num_slices=2 \
global_parameter_scale=16 per_device_batch_size=4
```

This will compile a 16B parameter MaxText model on 2 v5e pods. 

### Example AOT 2: Save compiled function, then load and run it
Here is an example that saves then loads the compiled `train_step`, starting with the save:

**Step 1: Run AOT and save compiled function**
```
# Run the below on a single machine, e.g. a CPU
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true" 
python3 MaxText/train_compile.py MaxText/configs/base.yml compile_topology=v5e-256 \
compile_topology_num_slices=2 \
compiled_trainstep_file=my_compiled_train.pickle global_parameter_scale=16 \
per_device_batch_size=4 steps=10000 learning_rate=1e-3
```

**Step 2: Run train.py and load the compiled function**

To load the compiled train_step, you just need to pass `compiled_trainstep_file=my_compiled_train.pickle` into `train.py`:
```
# Run the below on each host of the target hardware, e.g. each host on 2 slices of v5e-256
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true" 
python3 MaxText/train.py MaxText/configs/base.yml run_name=example_load_compile \
compiled_trainstep_file=my_compiled_train.pickle \
global_parameter_scale=16  per_device_batch_size=4 steps=10000 learning_rate=1e-3 \
base_output_directory=gs://my-output-bucket dataset_path=gs://my-dataset-bucket
```

In the save step of example 2 above we included exporting the compiler flag `LIBTPU_INIT_ARGS` and `learning_rate` because those affect the compiled object `my_compiled_train.pickle.` The sizes of the model (e.g. `global_parameter_scale`, `max_sequence_length` and `per_device_batch`) are fixed when you initally compile via `compile_train.py`, you will see a size error if you try to run the saved compiled object with different sizes than you compiled with. However a subtle note is that the **learning rate schedule** is also fixed when you run `compile_train` - which is determined by both `steps` and `learning_rate`. The optimizer parameters such as  `adam_b1` are passed only as shaped objects to the compiler - thus their real values are determined when you run `train.py`, not during the compilation. If you do pass in different shapes (e.g. `per_device_batch`), you will get a clear error message reporting that the compiled signature has different expected shapes than what was input. If you attempt to run on different hardware than the compilation targets requested via `compile_topology`, you will get an error saying there is a failure to map the devices from the compiled to your real devices. Using different XLA flags or a LIBTPU than what was compiled will probably run silently with the environment you compiled in without error. However there is no guaranteed behavior in this case; you should run in the same environment you compiled in.  




