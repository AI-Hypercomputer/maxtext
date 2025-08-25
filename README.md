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

# MaxText

[![Unit Tests](https://github.com/google/maxtext/actions/workflows/RunTests.yml/badge.svg)](https://github.com/google/maxtext/actions/workflows/RunTests.yml)

## Overview

MaxText is a **high performance**, **highly scalable**, **open-source** LLM written in pure Python/Jax and targeting Google Cloud TPUs and GPUs for **training** and **inference**. MaxText achieves [high MFUs](#runtime-performance-results) and scales from single host to very large clusters while staying simple and "optimization-free" thanks to the power of Jax and the XLA compiler.

MaxText aims to be a launching off point for ambitious LLM projects both in research and production. We encourage users to start by experimenting with MaxText out of the box and then fork and modify MaxText to meet their needs.

We have used MaxText to [demonstrate high-performance, well-converging training in int8](https://cloud.google.com/blog/products/compute/accurate-quantized-training-aqt-for-tpu-v5e) and [scale training to ~51K chips](https://cloud.google.com/blog/products/compute/the-worlds-largest-distributed-llm-training-job-on-tpu-v5e).

Key supported features:
* TPUs and GPUs
* Training and Inference
* Models: Llama 2, Llama 3, Llama 4, Mistral and Mixtral family, Gemma, Gemma 2, Gemma 3, DeepSeek, Qwen3 Dense and MoE family

## Announcements
* [August 13, 2025] The Qwen3 MoE family of models is now supported. We are starting with Qwen3-235B-A22B-Thinking-2507, Qwen3-30B-A3B and Qwen3-Coder-480B-A35B in addition to our existing Qwen3 Dense family of 0.6B, 4B, 8B, 14B, and 32B models.
* [July 27, 2025] We have updated our TFLOPS/s calculation to account for causal attention, dividing the attention flops in half in this [PR](https://github.com/AI-Hypercomputer/maxtext/pull/1988). Also we account for sliding window and chunked attention reduced attention flops in [PR](https://github.com/AI-Hypercomputer/maxtext/pull/2009) and [PR](https://github.com/AI-Hypercomputer/maxtext/pull/2030). These changes especially impact large sequence configs, since attention flops grow quadratically with sequence length, as explained in this [ReadMe](https://github.com/AI-Hypercomputer/maxtext/blob/main/getting_started/Performance_Metrics.md)
* [July 16, 2025] We will be restructuring the MaxText repository for improved organization and clarity. Please review the [proposed structure](RESTRUCTURE.md) and provide feedback.
* [July 11, 2025] Multi-Token Prediction (MTP) training is now supported! This feature adds an auxiliary loss based on predicting multiple future tokens, inspired by the [DeepSeek-V3 paper](https://arxiv.org/html/2412.19437v1), to enhance training efficiency.
* [June 25, 2025] DeepSeek R1-0528 variant is now supported!
* [April 24, 2025] Llama 4 Maverick models are now supported!
* [April 14, 2025] Llama 4 Scout models are now supported. Context length is currently limited to 8k and we have many ideas for optimization but we're working on both these things.  Note that models are text-only for now, but we're working on full multi-modal support!
* **[April 7, 2025] 🚨🚨🚨 We support modular imports. This comes with an API change for `train.py`: Now you should invoke the script via `python3 -m MaxText.train MaxText/configs/base.yml run_name=...`. If you want the old behavior you can stick to an older commit `git checkout pre-module-v0.1.0` and use the older API `python MaxText/train.py MaxText/configs/base.yml run_name=...`.**
* [April 2, 2025] DeepSeek v3-0324 variant is now supported!
* [March 24, 2025] We are excited to announce support for DeepSeek v3 (671B) and v2-Lite (16B), compatible with both TPUs and GPUs. We are actively working on further optimization.
* [March 12, 2025] We are excited to announce support for Gemma 3: 4B, 12B, and 27B in text-only formats. Please see [Google Launch Blog](https://blog.google/technology/developers/gemma-3/) and [Developer Blog](https://developers.googleblog.com/en/introducing-gemma3/) for more information on Gemma 3.
* [February, 2025] (Preview): We're excited to announce the preview of building Maxtext Docker images using the JAX AI Training Images, available for both TPUs and GPUs. This provides a more reliable and consistent build environment. Learn more [Here](getting_started/Run_MaxText_via_xpk.md)

## Table of Contents

* [Getting Started](getting_started/First_run.md)
* [Runtime Performance Results](#runtime-performance-results)
* [Comparison To Alternatives](#comparison-to-alternatives)
* [Development](#development)
* [Features and Diagnostics](#features-and-diagnostics)

## Getting Started

For your first time running MaxText, we provide specific [instructions](getting_started/First_run.md).

MaxText supports training and inference of various open models. Follow user guides in the [getting started](getting_started) folder to know more.

Some extra helpful guides:
* [Gemma (generations 1-3)](https://ai.google.dev/gemma): a family of open-weights Large Language Model (LLM) by [Google DeepMind](https://deepmind.google/), based on Gemini research and technology. You can run decode and finetuning using [these instructions](end_to_end/tpu/gemma/Run_Gemma.md). For Gemma 2 and 3, use the corresponding [gemma2](end_to_end/tpu/gemma2) and [gemma3](end_to_end/tpu/gemma3) scripts for checkpoint convertion and decoding.
* [Llama2](https://llama.meta.com/llama2/): a family of open-weights Large Language Model (LLM) by Meta. You can run decode and finetuning using [these instructions](getting_started/Run_Llama2.md).
* [Mixtral](https://mistral.ai/news/mixtral-of-experts/): a family of open-weights sparse mixture-of-experts (MoE) models by Mistral AI. You can run decode and finetuning using [these instructions](end_to_end/tpu/mixtral/Run_Mixtral.md).
* [DeepSeek](https://api-docs.deepseek.com/news/news1226): a novel family of open-weights sparse MoE models by DeepSeek AI. DeepSeek-V3 features advanced techniques, including Multi-Head Latent Attention (MLA), finer-grained and shared experts, Multi-Token Prediction (MTP), and FP8 mixed precision designed for enhanced efficiency and performance. You can run pre-training, finetuning, and decoding using [these instructions](end_to_end/tpu/deepseek/Run_DeepSeek.md).

In addition to the getting started guides, there are always other MaxText capabilities that are being constantly being added! The full suite of end-to-end tests is in [end_to_end](end_to_end). We run them with a nightly cadence. They can be a good source for understanding MaxText Alternatively you can see the continuous [unit tests](.github/workflows/RunTests.yml) which are run almost continuously.

## Runtime Performance Results

More details on reproducing these results can be found in [MaxText/configs/README.md](MaxText/configs/README.md).

### TPU v5p

| No. of params | Accelerator Type | TFLOP/chip/sec | Model flops utilization (MFU) |
|---|---|---|---|
| 32B | v5p-128 | 3.28e+02 | 67.76% |
| 64B | v5p-128 | 3.23e+02 | 70.31% |
| 128B | v5p-256 | 3.15e+02 | 68.68% |
| 128B | v5p-512 | 3.15e+02 | 68.53% |
| 256B | v5p-1024 | 3.16e+02 | 68.82% |
| 512B | v5p-1024 | 2.94e+02 | 63.99% |
| 1024B | v5p-2048 | 2.49e+02 | 64.05% |
| 1024B | v5p-4096 | 2.97e+02 | 64.80% |
| 1160B | v5p-7680 | 2.95e+02 | 64.27% |
| 1160B | v5p-12288 | 3.04e+02 | 66.23% |

### TPU v5e

For 16B, 32B, 64B, and 128B models. See full run configs in [MaxText/configs/v5e/](MaxText/configs/v5e/) as `16b.sh`, `32b.sh`, `64b.sh`, `128b.sh`.

| Hardware    | 16B TFLOP/sec/chip | 16B MFU | 32B TFLOP/sec/chip | 32B MFU | 64B TFLOP/sec/chip | 64B MFU | 128B TFLOP/sec/chip | 128B MFU |
| ----------- | -----------------: | ------- | -----------------: | ------- | -----------------: | ------- | ------------------: | -------- |
| 1x v5e-256  | 120                | 61.10%  | 132                | 66.86%  | 118                | 59.90%  | 110                 | 56.06%   |
| 2x v5e-256  | 117                | 59.37%  | 128                | 64.81%  | 112                | 56.66%  | 110                 | 55.82%   |
| 4x v5e-256  | 117                | 59.14%  | 126                | 64.10%  | 110                | 55.85%  | 108                 | 54.93%   |
| 8x v5e-256  | 115                | 58.27%  | 125                | 63.67%  | 108                | 54.96%  | 104                 | 52.93%   |
| 16x v5e-256 | 111                | 56.56%  | 123                | 62.26%  | 105                | 53.29%  | 100                 | 50.86%   |
| 32x v5e-256 | 108                | 54.65%  | 119                | 60.40%  | 99                 | 50.18%  | 91                  | 46.25%   |

## Comparison to Alternatives

MaxText is heavily inspired by [MinGPT](https://github.com/karpathy/minGPT)/[NanoGPT](https://github.com/karpathy/nanoGPT), elegant standalone GPT implementations written in PyTorch and targeting Nvidia GPUs. MaxText is more complex, supporting more industry standard models and scaling to tens of thousands of chips. Ultimately MaxText has an MFU more than three times the [17%](https://twitter.com/karpathy/status/1613250489097027584?cxt=HHwWgIDUhbixteMsAAAA) reported most recently with that codebase, is massively scalable and implements a key-value cache for efficient auto-regressive decoding.

MaxText is more similar to [Nvidia/Megatron-LM](https://github.com/NVIDIA/Megatron-LM), a very well tuned LLM implementation targeting Nvidia GPUs. The two implementations achieve comparable MFUs. The difference in the codebases highlights the different programming strategies. MaxText is pure Python, relying heavily on the XLA compiler to achieve high performance. By contrast, Megatron-LM is a mix of Python and CUDA, relying on well-optimized CUDA kernels to achieve high performance.

MaxText is also comparable to [Pax](https://github.com/google/paxml). Like Pax, MaxText provides high-performance and scalable implementations of LLMs in Jax. Pax focuses on enabling powerful configuration parameters, enabling developers to change the model by editing config parameters. By contrast, MaxText is a simple, concrete implementation of various LLMs that encourages users to extend by forking and directly editing the source code.

## Features and Diagnostics
### Collect Stack Traces
When running a Single Program, Multiple Data (SPMD) job on accelerators, the overall process can hang if there is any error or any VM hangs/crashes for some reason. In this scenario, capturing stack traces will help to identify and troubleshoot the issues for the jobs running on TPU VMs.

The following configurations will help to debug a fault or when a program is stuck or hung somewhere by collecting stack traces. Change the parameter values accordingly in `MaxText/configs/base.yml`:
1. Set `collect_stack_trace: True` to enable collection of stack traces on faults or when the program is hung. This setting will periodically dump the traces for the program to help in debugging. To disable this, set `collect_stack_trace: False`.
2. Set `stack_trace_to_cloud: False` to display stack traces on console. `stack_trace_to_cloud: True` will create a temporary file in `/tmp/debugging` in the TPUs to store the stack traces. There is an agent running on TPU VMs that will periodically upload the traces from the temporary directory to cloud logging in the gcp project. You can view the traces in Logs Explorer on Cloud Logging using the following query:
```
logName="projects/<project_name>/logs/tpu.googleapis.com%2Fruntime_monitor"
jsonPayload.verb="stacktraceanalyzer"
```
3. `stack_trace_interval_seconds` signifies the duration in seconds between each stack trace collection event. Setting `stack_trace_interval_seconds: 600` will collect the stack traces every 600 seconds (10 minutes).

Here is the related PyPI package: https://pypi.org/project/cloud-tpu-diagnostics.

### Ahead of Time Compilation (AOT)
To compile your training run ahead of time, we provide a tool `train_compile.py`. This tool allows you to compile the main `train_step` in `train.py` for target hardware (e.g. a large number of v5e devices) without using the full cluster.

#### TPU Support

You may use only a CPU or a single VM from a different family to pre-compile for a TPU cluster. This compilation helps with two main goals:

* It will flag any out of memory (OOM) information, such as when the `per_device_batch_size` is set too high, with an identical OOM stack trace as if it was compiled on the target hardware.

* The ahead of time compilation can be saved and then loaded for fast startup and restart times on the target hardware.

The tool `train_compile.py` is tightly linked to `train.py` and uses the same configuration file `configs/base.yml`. Although you don't need to run on a TPU, you do need to install `jax[tpu]` in addition to other dependencies, so we recommend running `setup.sh` to install these if you have not already done so.

##### Example AOT 1: Compile ahead of time basics
After installing the dependencies listed above, you are ready to compile ahead of time:
```
# Run the below on a single machine, e.g. a CPU
python3 -m MaxText.train_compile MaxText/configs/base.yml compile_topology=v5e-256 compile_topology_num_slices=2 \
global_parameter_scale=16 per_device_batch_size=4
```

This will compile a 16B parameter MaxText model on 2 v5e pods.

##### Example AOT 2: Save compiled function, then load and run it
Here is an example that saves then loads the compiled `train_step`, starting with the save:

**Step 1: Run AOT and save compiled function**
```
# Run the below on a single machine, e.g. a CPU
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true"
python3 -m MaxText.train_compile MaxText/configs/base.yml compile_topology=v5e-256 \
compile_topology_num_slices=2 \
compiled_trainstep_file=my_compiled_train.pickle global_parameter_scale=16 \
per_device_batch_size=4 steps=10000 learning_rate=1e-3
```

**Step 2: Run train.py and load the compiled function**

To load the compiled train_step, you just need to pass `compiled_trainstep_file=my_compiled_train.pickle` into `train.py`:
```
# Run the below on each host of the target hardware, e.g. each host on 2 slices of v5e-256
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true"
python3 -m MaxText.train MaxText/configs/base.yml run_name=example_load_compile \
compiled_trainstep_file=my_compiled_train.pickle \
global_parameter_scale=16  per_device_batch_size=4 steps=10000 learning_rate=1e-3 \
base_output_directory=gs://my-output-bucket dataset_path=gs://my-dataset-bucket
```

In the save step of example 2 above we included exporting the compiler flag `LIBTPU_INIT_ARGS` and `learning_rate` because those affect the compiled object `my_compiled_train.pickle.` The sizes of the model (e.g. `global_parameter_scale`, `max_sequence_length` and `per_device_batch`) are fixed when you initially compile via `compile_train.py`, you will see a size error if you try to run the saved compiled object with different sizes than you compiled with. However a subtle note is that the **learning rate schedule** is also fixed when you run `compile_train` - which is determined by both `steps` and `learning_rate`. The optimizer parameters such as  `adam_b1` are passed only as shaped objects to the compiler - thus their real values are determined when you run `train.py`, not during the compilation. If you do pass in different shapes (e.g. `per_device_batch`), you will get a clear error message reporting that the compiled signature has different expected shapes than what was input. If you attempt to run on different hardware than the compilation targets requested via `compile_topology`, you will get an error saying there is a failure to map the devices from the compiled to your real devices. Using different XLA flags or a LIBTPU than what was compiled will probably run silently with the environment you compiled in without error. However there is no guaranteed behavior in this case; you should run in the same environment you compiled in.

#### GPU Support
Ahead-of-time compilation is also supported for GPUs with some differences from TPUs:

1. GPU does not support compilation across hardware: A GPU host is still required to run AoT compilation, but a single GPU host can compile a program for a larger cluster of the same hardware.

1. For [A3 Cloud GPUs](https://cloud.google.com/compute/docs/gpus#h100-gpus), the maximum "slice" size is a single host, and the `compile_topology_num_slices` parameter represents the number of A3 machines to precompile for.

##### Example
This example illustrates the flags to use for a multihost GPU compilation targeting a cluster of 4 A3 hosts:

**Step 1: Run AOT and save compiled function**
```
# Run the below on a single A3 machine
export XLA_FLAGS="--xla_gpu_enable_async_collectives=true"
python3 -m MaxText.train_compile MaxText/configs/base.yml compile_topology=a3 \
compile_topology_num_slices=4 \
compiled_trainstep_file=my_compiled_train.pickle global_parameter_scale=16 \
attention=dot_product per_device_batch_size=4 steps=10000 learning_rate=1e-3
```

**Step 2: Run train.py and load the compiled function**

To load the compiled train_step, you just need to pass `compiled_trainstep_file=my_compiled_train.pickle` into `train.py`:
```
# Run the below on each of the 4 target A3 hosts.
export XLA_FLAGS="--xla_gpu_enable_async_collectives=true"
python3 -m MaxText.train MaxText/configs/base.yml run_name=example_load_compile \
compiled_trainstep_file=my_compiled_train.pickle \
attention=dot_product global_parameter_scale=16  per_device_batch_size=4 steps=10000 learning_rate=1e-3 \
base_output_directory=gs://my-output-bucket dataset_path=gs://my-dataset-bucket
```

As in the TPU case, note that the compilation environment must match the execution environment, in this case by setting the same `XLA_FLAGS`.


### Automatically Upload Logs to Vertex Tensorboard
MaxText supports automatic upload of logs collected in a directory to a Tensorboard instance in Vertex AI. Follow [user guide](getting_started/Use_Vertex_AI_Tensorboard.md) to know more.

### Monitor Goodput of Your Workload
If you are interested in monitoring Goodput metrics of your workload, follow this [user guide](getting_started/Monitor_Goodput.md).
