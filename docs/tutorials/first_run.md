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

(first-run)=

# Getting started: First run

This topic provides a basic introduction to get your MaxText workload up and running on single host and multihost environments using Cloud TPUs or NVIDIA GPUs. To help you get familiar with MaxText, we recommend starting with a single host first and then moving to multihost.

## Prerequisites: Set up storage and configure MaxText

1. To store logs and checkpoints, [Create a Cloud Storage bucket](https://cloud.google.com/storage/docs/creating-buckets) in your project. To run MaxText, the TPU or GPU VMs must have read/write permissions for the bucket. These permissions are granted by service account roles, such as the `STORAGE ADMIN` role.

1. MaxText reads a yaml file for configuration. We also recommend reviewing the configurable options in `configs/base.yml`. This file includes a decoder-only model of ~1B parameters. The configurable options can be overwritten from the command line. For instance, you can change the `steps` or `log_period` by either modifying `configs/base.yml` or by passing in `steps` and `log_period` as additional arguments to the `train.py` call. Set `base_output_directory` to a folder in the bucket you just created.

## Local development for single host

This procedure describes how to run MaxText on a single GPU or TPU host.

### Run MaxText on cloud TPUs

Local development is a convenient way to run MaxText on a single host. It doesn't scale to
multiple hosts but is a good way to learn about MaxText.

1. [Create and SSH to the single host VM of your choice](https://cloud.google.com/tpu/docs/managing-tpus-tpu-vm). You can use any available single host TPU, such as `v5litepod-8`, `v5p-8`, or `v4-8`.
1. Clone MaxText onto that TPU VM.
1. Within the root directory of the cloned repo, install dependencies and pre-commit hook by running:

```sh
python3 -m venv ~/venv-maxtext
source ~/venv-maxtext/bin/activate
bash tools/setup/setup.sh
pre-commit install
```

4. After installation completes, run training on synthetic data with the following command:

```sh
python3 -m MaxText.train src/MaxText/configs/base.yml \
  run_name=$YOUR_JOB_NAME \
  base_output_directory=gs://<my-bucket> \
  dataset_type=synthetic \
  steps=10
```

Optional: If you want to try training on a Hugging Face dataset, see [Data Input Pipeline](../guides/data_input_pipeline.md) for data input options.

5. To demonstrate model output, run the following command:

```sh
python3 -m maxtext.decode src/MaxText/configs/base.yml \
  run_name=$YOUR_JOB_NAME \
  base_output_directory=gs://<my-bucket> \
  per_device_batch_size=1
```

This command uses a model with randomly initialized weights, so the outputs are also random. To get high quality output you need pass in a checkpoint, typically via the `load_parameters_path` argument.

### Run MaxText via notebook

In the same TPU VM where you just installed all the dependencies of MaxText, You can also run training and decoding in MaxText via Notebook (for e.g., via Jupyter or Colab).

#### Decoding in MaxText via notebook

You can use [demo_decoding.ipynb](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/MaxText/examples/demo_decoding.ipynb) to try out decoding on MaxText's `Llama3.1-8b` model implementation. In this notebook, we give `"I love to"` as the prompt, and the greedily sampled first output token is `" cook"`. Please remember to provide the path to your `Llama3.1-8b` checkpoint for the `load_parameters_path` argument in the config inside the notebook. You can use [to_maxtext.py](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/MaxText/utils/ckpt_conversion/to_maxtext.py) to create a MaxText/Orbax checkpoint from a Huggingface checkpoint.

### Run MaxText on NVIDIA GPUs

1. Use `bash dependencies/scripts/docker_build_dependency_image.sh DEVICE=gpu` to build a container with the required dependencies.
1. After installation is complete, run training with the following command on synthetic data:

```sh
python3 -m MaxText.train src/MaxText/configs/base.yml \
  run_name=$YOUR_JOB_NAME \
  base_output_directory=gs://<my-bucket> \
  dataset_type=synthetic \
  steps=10
```

3. To demonstrate model output, run the following command:

```sh
python3 -m maxtext.decode src/MaxText/configs/base.yml \
  run_name=$YOUR_JOB_NAME \
  base_output_directory=gs://<my-bucket> \
  per_device_batch_size=1
```

If you see the following error when running inside a container, set a larger `--shm-size` (for example, `--shm-size=1g`):

```
Failed to execute XLA Runtime executable: run time error: custom call 'xla.gpu.all_reduce' failed: external/xla/xla/service/gpu/nccl_utils.cc:297: NCCL operation ncclCommInitRank(&comm, nranks, id, rank) failed: unhandled cuda error (run with NCCL_DEBUG=INFO for details); current tracing scope: all-reduce-start.2; current profiling annotation: XlaModule:#hlo_module=jit__unnamed_wrapped_function_,program_id=7#.
```

## Multihost development

Google Kubernetes Engine (GKE) is the recommended way to run MaxText on multiple hosts. It provides a managed environment for deploying and scaling containerized applications, including those that require TPUs or GPUs. See [Running Maxtext with XPK](../run_maxtext/run_maxtext_via_xpk.md) for details.

## Next steps: preflight optimizations

After you get workloads running, there are optimizations you can apply to improve performance. For more information, see [PREFLIGHT.md](https://github.com/google/maxtext/blob/main/PREFLIGHT.md).
