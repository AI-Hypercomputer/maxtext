# Get started with MaxText

This topic provides a basic introduction to get your MaxText workload up and running on single host and multihost environments using Cloud TPUs or NVIDIA GPUs. To help you get familiar with MaxText, we recommend starting with a single host first and then moving to multihost.

## Prerequisites: Set up storage and configure MaxText
1. To store logs and checkpoints, [Create a Cloud Storage bucket](https://cloud.google.com/storage/docs/creating-buckets) in your project. To run MaxText, the TPU or GPU VMs must have read/write permissions for the bucket. These permissions are granted by service account roles, such as the `STORAGE ADMIN` role.

2. MaxText reads a yaml file for configuration. We also recommend reviewing the configurable options in `configs/base.yml`. This file includes a decoder-only model of ~1B parameters. The configurable options can be overwritten from the command line. For instance, you can change the `steps` or `log_period` by either modifying `configs/base.yml` or by passing in `steps` and `log_period` as additional arguments to the `train.py` call. Set `base_output_directory` to a folder in the bucket you just created.

## Local development for single host
This procedure describes how to run MaxText on a single GPU or TPU host.

### Run MaxText on Cloud TPUs
Local development is a convenient way to run MaxText on a single host. It doesn't scale to
multiple hosts but is a good way to learn about MaxText.

1. [Create and SSH to the single host VM of your choice](https://cloud.google.com/tpu/docs/managing-tpus-tpu-vm). You can use any available single host TPU, such as `v5litepod-8`, `v5p-8`, or `v4-8`.
2. Clone MaxText onto that TPU VM.
3. Within the root directory of the cloned repo, install dependencies and pre-commit hook by running:
```
python3 -m venv ~/venv-maxtext
source ~/venv-maxtext/bin/activate
bash setup.sh
pre-commit install
```
4. After installation completes, run training on synthetic data with the following command:
```
python3 -m MaxText.train MaxText/configs/base.yml \
  run_name=$YOUR_JOB_NAME \
  base_output_directory=gs://<my-bucket> \
  dataset_type=synthetic \
  steps=10
```
Optional: If you want to try training on a Hugging Face dataset, see [Data Input Pipeline](https://github.com/google/maxtext/blob/main/getting_started/Data_Input_Pipeline.md) for data input options.

5. To demonstrate model output, run the following command:
```
python3 -m MaxText.decode MaxText/configs/base.yml \
  run_name=$YOUR_JOB_NAME \
  base_output_directory=gs://<my-bucket> \
  per_device_batch_size=1
```
This command uses a model with randomly initialized weights, so the outputs are also random. To get high quality output you need pass in a checkpoint, typically via the `load_parameters_path` argument.


### Run MaxText on NVIDIA GPUs
1. Use `bash docker_build_dependency_image.sh DEVICE=gpu` to build a container with the required dependencies.
2. After installation is complete, run training with the following command on synthetic data:
```
python3 -m MaxText.train MaxText/configs/base.yml \
  run_name=$YOUR_JOB_NAME \
  base_output_directory=gs://<my-bucket> \
  dataset_type=synthetic \
  steps=10  
```

3. To demonstrate model output, run the following command: 
```
python3 -m MaxText.decode MaxText/configs/base.yml \
  run_name=$YOUR_JOB_NAME \
  base_output_directory=gs://<my-bucket> \
  per_device_batch_size=1  
```

If you see the following error when running inside a container, set a larger `--shm-size` (for example, `--shm-size=1g`):
```
Failed to execute XLA Runtime executable: run time error: custom call 'xla.gpu.all_reduce' failed: external/xla/xla/service/gpu/nccl_utils.cc:297: NCCL operation ncclCommInitRank(&comm, nranks, id, rank) failed: unhandled cuda error (run with NCCL_DEBUG=INFO for details); current tracing scope: all-reduce-start.2; current profiling annotation: XlaModule:#hlo_module=jit__unnamed_wrapped_function_,program_id=7#.
```

## Multihost development
There are three patterns for running MaxText with more than one host.

1. Recommended: Google Kubernetes Engine (GKE) [Running Maxtext with XPK](Run_MaxText_via_xpk.md) - Quick experimentation and production support.
2. Google Compute Engine (GCE) [Running Maxtext with Multihost Jobs](Run_MaxText_via_multihost_job.md) - Long running production jobs with queued resources.
3. Google Compute Engine (GCE) [Running Maxtext with Multihost Runner](Run_MaxText_via_multihost_runner.md) -  Fast experiments via multiple ssh connections.

## Next steps: Preflight optimizations

After you get workloads running, there are optimizations you can apply to improve performance. For more information, see [PREFLIGHT.md](https://github.com/google/maxtext/blob/main/PREFLIGHT.md).
