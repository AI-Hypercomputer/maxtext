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

# Features and diagnostics

## Collect stack traces

When running a Single Program, Multiple Data (SPMD) job on accelerators, the overall process can hang if there is any error or any VM hangs/crashes for some reason. In this scenario, capturing stack traces will help to identify and troubleshoot the issues for the jobs running on TPU VMs.

The following configurations will help to debug a fault or when a program is stuck or hung somewhere by collecting stack traces. Change the parameter values accordingly in `src/maxtext/configs/base.yml`:

1. Set `collect_stack_trace: True` to enable collection of stack traces on faults or when the program is hung. This setting will periodically dump the traces for the program to help in debugging. To disable this, set `collect_stack_trace: False`.
2. Set `stack_trace_to_cloud: False` to display stack traces on console. `stack_trace_to_cloud: True` will create a temporary file in `/tmp/debugging` in the TPUs to store the stack traces. There is an agent running on TPU VMs that will periodically upload the traces from the temporary directory to cloud logging in the gcp project. You can view the traces in Logs Explorer on Cloud Logging using the following query:

```
logName="projects/<project_name>/logs/tpu.googleapis.com%2Fruntime_monitor"
jsonPayload.verb="stacktraceanalyzer"
```

3. `stack_trace_interval_seconds` signifies the duration in seconds between each stack trace collection event. Setting `stack_trace_interval_seconds: 600` will collect the stack traces every 600 seconds (10 minutes).

Here is the related PyPI package: https://pypi.org/project/cloud-tpu-diagnostics.

(aot-compilation)=

## Ahead of Time compilation (AOT)

To compile your training run ahead of time, we provide a tool `train_compile.py`. This tool allows you to compile the main `train_step` in `train.py` for target hardware (e.g. a large number of v5e devices) without using the full cluster.

### TPU support

You may use only a CPU or a single VM from a different family to pre-compile for a TPU cluster. This compilation helps with two main goals:

- It will flag any out of memory (OOM) information, such as when the `per_device_batch_size` is set too high, with an identical OOM stack trace as if it was compiled on the target hardware.

- The ahead of time compilation can be saved and then loaded for fast startup and restart times on the target hardware.

The tool `train_compile.py` is tightly linked to `train.py` and uses the same configuration file `src/maxtext/configs/base.yml`. Although you don't need to run on a TPU, you do need to install `jax[tpu]` in addition to other dependencies, so we recommend running `setup.sh` to install these if you have not already done so.

#### Example AOT 1: Compile ahead of time basics

After installing the dependencies listed above, you are ready to compile ahead of time:

```sh
# Run the below on a single machine, e.g. a CPU
python3 -m maxtext.trainers.pre_train.train_compile src/maxtext/configs/base.yml compile_topology=v5e-256 compile_topology_num_slices=2 \
  global_parameter_scale=16 per_device_batch_size=4
```

This will compile a 16B parameter MaxText model on 2 v5e pods.

#### Example AOT 2: Save compiled function, then load and run it

Here is an example that saves then loads the compiled `train_step`, starting with the save:

**Step 1: Run AOT and save compiled function**

```sh
# Run the below on a single machine, e.g. a CPU
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true"
python3 -m maxtext.trainers.pre_train.train_compile src/maxtext/configs/base.yml compile_topology=v5e-256 \
  compile_topology_num_slices=2 \
  compiled_trainstep_file=my_compiled_train.pickle global_parameter_scale=16 \
  per_device_batch_size=4 steps=10000 learning_rate=1e-3
```

**Step 2: Run `train.py` and load the compiled function**

To load the compiled train_step, you just need to pass `compiled_trainstep_file=my_compiled_train.pickle` into `train.py`:

```sh
# Run the below on each host of the target hardware, e.g. each host on 2 slices of v5e-256
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true"
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml run_name=example_load_compile \
  compiled_trainstep_file=my_compiled_train.pickle \
  global_parameter_scale=16 per_device_batch_size=4 steps=10000 learning_rate=1e-3 \
  base_output_directory=gs://my-output-bucket dataset_path=gs://my-dataset-bucket
```

In the save step of example 2 above we included exporting the compiler flag `LIBTPU_INIT_ARGS` and `learning_rate` because those affect the compiled object `my_compiled_train.pickle.` The sizes of the model (e.g. `global_parameter_scale`, `max_sequence_length` and `per_device_batch`) are fixed when you initially compile via `compile_train.py`, you will see a size error if you try to run the saved compiled object with different sizes than you compiled with. However a subtle note is that the **learning rate schedule** is also fixed when you run `compile_train` - which is determined by both `steps` and `learning_rate`. The optimizer parameters such as `adam_b1` are passed only as shaped objects to the compiler - thus their real values are determined when you run `train.py`, not during the compilation. If you do pass in different shapes (e.g. `per_device_batch`), you will get a clear error message reporting that the compiled signature has different expected shapes than what was input. If you attempt to run on different hardware than the compilation targets requested via `compile_topology`, you will get an error saying there is a failure to map the devices from the compiled to your real devices. Using different XLA flags or a LIBTPU than what was compiled will probably run silently with the environment you compiled in without error. However there is no guaranteed behavior in this case; you should run in the same environment you compiled in.

### GPU support

Ahead-of-time compilation is also supported for GPUs with some differences from TPUs:

1. GPU does not support compilation across hardware: A GPU host is still required to run AoT compilation, but a single GPU host can compile a program for a larger cluster of the same hardware.

2. For [A3 Cloud GPUs](https://cloud.google.com/compute/docs/gpus#h100-gpus), the maximum "slice" size is a single host, and the `compile_topology_num_slices` parameter represents the number of A3 machines to precompile for.

#### Example

This example illustrates the flags to use for a multihost GPU compilation targeting a cluster of 4 A3 hosts:

**Step 1: Run AOT and save compiled function**

```sh
# Run the below on a single A3 machine
export XLA_FLAGS="--xla_gpu_enable_async_collectives=true"
python3 -m maxtext.trainers.pre_train.train_compile src/maxtext/configs/base.yml compile_topology=a3 \
  compile_topology_num_slices=4 \
  compiled_trainstep_file=my_compiled_train.pickle global_parameter_scale=16 \
  attention=dot_product per_device_batch_size=4 steps=10000 learning_rate=1e-3
```

**Step 2: Run `train.py` and load the compiled function**

To load the compiled `train_step`, you just need to pass `compiled_trainstep_file=my_compiled_train.pickle` into `train.py`:

```sh
# Run the below on each of the 4 target A3 hosts.
export XLA_FLAGS="--xla_gpu_enable_async_collectives=true"
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml run_name=example_load_compile \
  compiled_trainstep_file=my_compiled_train.pickle \
  attention=dot_product global_parameter_scale=16  per_device_batch_size=4 steps=10000 learning_rate=1e-3 \
  base_output_directory=gs://my-output-bucket dataset_path=gs://my-dataset-bucket
```

As in the TPU case, note that the compilation environment must match the execution environment, in this case by setting the same `XLA_FLAGS`.

## Automatically upload logs to Vertex AI Tensorboard

MaxText supports automatic upload of logs collected in a directory to a Tensorboard instance in Vertex AI. Follow [](use_vertex_ai_tensorboard.md) to know more.
