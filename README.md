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


[![Unit Tests](https://github.com/google/maxtext/actions/workflows/UnitTests.yml/badge.svg)](https://github.com/google/maxtext/actions/workflows/UnitTests.yml)

# Overview

MaxText is a set of open source reference implementations that are **high performance**, **arbitrarily scalable**, **well-tested**, and written in pure Python/Jax to target Google Cloud TPUs and GPUs. MaxText typically achieves 55% to 60% model-flop utilization and scales from single host to very large clusters while staying simple and "optimization-free" thanks to the power of Jax and the XLA compiler.

MaxText is designed to be a starting point for ambitious LLM projects both in research and production. We encourage you to start experimenting with MaxText and then fork and modify it to meet your needs.

We have used MaxText to [demonstrate high-performance, well-converging training in int8](https://cloud.google.com/blog/products/compute/accurate-quantized-training-aqt-for-tpu-v5e) and [scale training to ~51K chips](https://cloud.google.com/blog/products/compute/the-worlds-largest-distributed-llm-training-job-on-tpu-v5e).

Key supported features:

* TPUs and GPUs (in preview)
* Training and Inference (in preview)
* Models: Llama2, Mistral and Gemma

We have used MaxText to [demonstrate high-performance, well-converging training in int8](https://cloud.google.com/blog/products/compute/accurate-quantized-training-aqt-for-tpu-v5e) and [scale training to ~51K chips](https://cloud.google.com/blog/products/compute/the-worlds-largest-distributed-llm-training-job-on-tpu-v5e).

Key supported features:
* TPUs and GPUs (in preview)
* Training and Inference (in preview)
* Models: Llama2, Mistral and Gemma

# Table of Contents

* [Getting Started](getting_started/First_run.md)
* [Runtime Performance Results](#runtime-performance-results)
* [Comparison To Alternatives](#comparison-to-alternatives)
* [Development](#development)
* [Features and Diagnostics](#features-and-diagnostics)

# Getting Started

For instructions on running MaxText the first time, see [First run](getting_started/First_run.md).

MaxText supports training and inference of various open models. For more information, see [getting started](getting_started).

See the these links for more information about the models implemented in MaxText:

* [Gemma](https://ai.google.dev/gemma): a family of open-weights Large Language Model (LLM) by [Google DeepMind](https://deepmind.google/), based on Gemini research and technology. For more information about decoding and fine tuning, see [Run Gemma](end_to_end/gemma/Run_Gemma.md).
* [Llama2](https://llama.meta.com/llama2/): a family of open-weights Large Language Model (LLM) by Meta. For more information about decoding and fine tuning, see [Run Llama2](getting_started/Run_Llama2.md).

In addition to the getting started guides, new content is added regularly! The full suite of end-to-end tests is in [end_to_end](end_to_end). We run them with a nightly cadence. They can be a good source for understanding MaxText. Alternatively, you can see the continuous [unit tests](.github/workflows/UnitTests.yml) which are run on a regular basis.

# Runtime Performance Results

This section describes the runtime performance of MaxText using different TPU versions as well as different numbers of parameters. The performance is measured by TFLOP/sec/chip and [Model Flops Utilization (MFU)](https://services.google.com/fh/files/blogs/tpu_v4_benchmarking.pdf).
You can find more details on reproducing these results can be found in [MaxText/configs/README.md](MaxText/configs/README.md).

# Runtime Performance Results

More details on reproducing these results can be found in [MaxText/configs/README.md](MaxText/configs/README.md).

## TPU v5p

| No. of params | Accelerator Type | TFLOP/chip/sec | Model flops utilization (MFU) |
|---|---|---|---|
| 32B | v5p-128 | 3.28e+02 | 71.47% |
| 64B | v5p-128 | 3.23e+02 | 70.31% |
| 128B | v5p-256 | 3.15e+02 | 68.68% |
| 128B | v5p-512 | 3.15e+02 | 68.53% |
| 256B | v5p-1024 | 3.16e+02 | 68.82% |
| 512B | v5p-1024 | 2.94e+02 | 63.99% |
| 1024B | v5p-2048 | 2.49e+02 | 64.05% |
| 1024B | v5p-4096 | 2.97e+02 | 64.80% |
| 1160B | v5p-7680 | 2.95e+02 | 64.27% |
| 1160B | v5p-12288 | 3.04e+02 | 66.23% |

## TPU v5e

For 16B, 32B, 64B, and 128B models. See full run configs in [MaxText/configs/v5e/](MaxText/configs/v5e/) as `16b.sh`, `32b.sh`, `64b.sh`, `128b.sh`.

| Hardware    | 16B TFLOP/sec/chip | 16B MFU | 32B TFLOP/sec/chip | 32B MFU | 64B TFLOP/sec/chip | 64B MFU | 128B TFLOP/sec/chip | 128B MFU |
| ----------- | -----------------: | ------- | -----------------: | ------- | -----------------: | ------- | ------------------: | -------- |
| 1x v5e-256  | 120                | 61.10%  | 132                | 66.86%  | 118                | 59.90%  | 110                 | 56.06%   |
| 2x v5e-256  | 117                | 59.37%  | 128                | 64.81%  | 112                | 56.66%  | 110                 | 55.82%   |
| 4x v5e-256  | 117                | 59.14%  | 126                | 64.10%  | 110                | 55.85%  | 108                 | 54.93%   |
| 8x v5e-256  | 115                | 58.27%  | 125                | 63.67%  | 108                | 54.96%  | 104                 | 52.93%   |
| 16x v5e-256 | 111                | 56.56%  | 123                | 62.26%  | 105                | 53.29%  | 100                 | 50.86%   |
| 32x v5e-256 | 108                | 54.65%  | 119                | 60.40%  | 99                 | 50.18%  | 91                  | 46.25%   |

# Comparison to Alternatives

MaxText is heavily inspired by [MinGPT](https://github.com/karpathy/minGPT) and [NanoGPT](https://github.com/karpathy/nanoGPT), elegant standalone GPT implementations written in PyTorch and targeting Nvidia GPUs. MaxText is more complex, supporting more industry standard models and scaling to tens of thousands of chips. Ultimately MaxText has an [MFU](https://cloud.google.com/blog/products/compute/using-cloud-tpu-multislice-to-scale-ai-workloads) more than three times the [17%](https://twitter.com/karpathy/status/1613250489097027584?cxt=HHwWgIDUhbixteMsAAAA) reported most recently with that MinGPT and NanoGPT, is massively scalable, and implements a key-value cache for efficient auto-regressive decoding.

MaxText is more similar to [Nvidia/Megatron-LM](https://github.com/NVIDIA/Megatron-LM), a well tuned LLM implementation that targets Nvidia GPUs. MaxText and Megatron-LM implementations achieve comparable MFUs. The difference in the codebases highlights different programming strategies. MaxText is written in pure Python, relying heavily on the XLA compiler to achieve high performance. By contrast, Megatron-LM is a mix of Python and CUDA, relying on well-optimized CUDA kernels to achieve high performance.

MaxText is also comparable to [Pax](https://github.com/google/paxml). Like Pax, MaxText provides high-performance and scalable implementations of LLMs in Jax. However, Pax is a framework in which developers can inject their code or configuration. By contrast, MaxText is a reference implementation designed to be forked and edited as needed.

# Features and Diagnostics

Install the [Cloud TPU diagnostics](https://pypi.org/project/cloud-tpu-diagnostics) Python package to monitor, debug and profile jobs running on Cloud TPUs.  

## Collect Stack Traces

When running a Single Program, Multiple Data (SPMD) job on accelerators, the overall process can hang if any errors occur, or if a VM hangs or crashes. Capturing stack traces will help you identify and troubleshoot the issues that are occuring.

The following configurations will help you debug a workload by collecting stack traces. Change the following parameter values accordingly in `MaxText/configs/base.yml`:

* Set `collect_stack_trace: True` to enable collection of stack traces on faults or when the program hangs. This setting periodically dumps stack traces. To disable this, set `collect_stack_trace: False`.
* Set `stack_trace_to_cloud: False` to display stack traces on the console. Or, set `stack_trace_to_cloud: True` to create a temporary file in `/tmp/debugging` in your TPU VMs to store stack traces. There is an agent running on TPU VMs that periodically uploads traces from the temporary directory to [Cloud Logging](https://cloud.google.com/logging/docs/overview). You can view the traces in [Logs Explorer](https://cloud.google.com/logging/docs/view/logs-explorer-interface) in Cloud Logging using the following query: 

   ```none
   logName="projects/<project_name>/logs/tpu.googleapis.com%2Fruntime_monitor"
   jsonPayload.verb="stacktraceanalyzer"
   ```
* `stack_trace_interval_seconds` sets the duration in seconds between each stack trace collection event. For example, setting `stack_trace_interval_seconds: 600` will collect the stack traces every 600 seconds (10 minutes).

## Ahead of Time Compilation (AOT, TPU-only)

To compile your training code ahead of time, we provide a tool called `train_compile.py`. This tool allows you to compile the main `train_step` in `train.py` for target hardware (for example, a large number of v5e devices) without using a CPU or a single GCE VM. AOT compilation helps with two main goals:

* It flags any out of memory (OOM) information. For example, when `per_device_batch_size` is set too high, the compiler will generate an OOM stack trace identical to one that generated on TPU VMs.

* The compiled code can be saved and loaded for fast startup and restart times on the target hardware.

The tool `train_compile.py` is tightly linked to `train.py` and uses the same configuration file `configs/base.yml`. Although you don't need to run on a TPU VM, you do need to install the `jax[tpu]` package. Run `setup.sh` to install this package and any dependencies.

### Example AOT 1: Compile ahead of time basics

After installing the dependencies listed above, you are ready to compile your code:

```bash
# Run the below on a single machine, for example a CPU
python3 MaxText/train_compile.py MaxText/configs/base.yml compile_topology=v5e-256 compile_topology_num_slices=2 \
global_parameter_scale=16 per_device_batch_size=4
```

This compiles a 16B parameter MaxText model on 2 v5e pods.

### Example AOT 2: Save compiled function, then load and run it

The following example saves and then loads the compiled `train_step`.

**Step 1: Run AOT and save the compiled function**

```bash
# Run these commands on a single machine (CPU). 
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true"
python3 MaxText/train_compile.py MaxText/configs/base.yml compile_topology=v5e-256 \
compile_topology_num_slices=2 \
compiled_trainstep_file=my_compiled_train.pickle global_parameter_scale=16 \
per_device_batch_size=4 steps=10000 learning_rate=1e-3
```

**Step 2: Run train.py and load the compiled function**

To load the compiled `train_step`, pass `compiled_trainstep_file=my_compiled_train.pickle` 
into `train.py`:

```bash
# Run the following command on each host of the target hardware.
# In other words, run the command on each host on 2 slices v5e-256
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true"
python3 MaxText/train.py MaxText/configs/base.yml run_name=example_load_compile \
compiled_trainstep_file=my_compiled_train.pickle \
global_parameter_scale=16  per_device_batch_size=4 steps=10000 learning_rate=1e-3 \
base_output_directory=gs://my-output-bucket dataset_path=gs://my-dataset-bucket
```

The sizes of the model (for example, `global_parameter_scale`, `max_sequence_length` and `per_device_batch`) are fixed when you use AOT compilation using `compile_train.py`. You must run a saved compiled `train_step` with a model of the same size with which it was compiled, otherwise a size error will occur. The **learning rate schedule** (which is determined by both `steps` and `learning_rate`) is also fixed when you run `compile_train`. 

Optimizer parameters such as `adam_b1` are passed only as shaped objects to the compiler, their real values are determined at runtime. If you pass in different shapes (for example, `per_device_batch`), than you compiled with, you will get a shape error message. 

If you attempt to run a compiled `train_step` on different hardware than the compilation target (using `compile_topology`), you will get an error saying there is a failure to map the devices from the compiled object to your real devices. 

While running AOT compiled code with different XLA flags or a different LIBTPU version than what you used during compilation, your code *may* run without error. However you are not guaranteed this will work. Best practice is to run your code in the same environment you compiled in.

## Automatically Upload Logs to Vertex Tensorboard

MaxText can automatically upload logs generated while your code runs to a Tensorboard instance in Vertex AI. For more information, see [user guide](getting_started/Use_Vertex_AI_Tensorboard.md).
