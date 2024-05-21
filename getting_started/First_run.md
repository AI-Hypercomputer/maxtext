# Getting Started

We recommend starting with a single host first and then moving to multihost.

## Getting Started: Cloud Storage and Configure
1. [Create a gcs buckets](https://cloud.google.com/storage/docs/creating-buckets) in your project for storing logs and checkpoints. To run maxtext the TPU/GPU VMs must have permission to read/write the gcs bucket. These permissions are granted by service account roles, such as the `STORAGE ADMIN` role.

2. MaxText reads a yaml file for configuration. We also recommend reviewing the configurable options in `configs/base.yml`, this config includes a decoder-only model of ~1B parameters. The configurable options can be overwritten from command lines. For instance you may change the `steps` or `log_period` by either modifying `configs/base.yml` or by passing in `steps` and `log_period` as additional args to the `train.py` call. `base_output_directory` should be set to a folder in the bucket you just created.

## Getting Started: Local Development for single host

#### Running on Cloud TPUs
Local development is a convenient way to run MaxText on a single host. It doesn't scale to
multiple hosts.

1. [Create and SSH to the single-host VM of your choice.](https://cloud.google.com/tpu/docs/users-guide-tpu-vm#creating_a_cloud_tpu_vm_with_gcloud) We recommend a `v4-8`.
2. Clone MaxText onto that TPUVM.
3. Within the root directory of that `git` repo, install dependencies by running:
```
bash setup.sh
```
4. After installation completes, run training with the command on synthetic data:
```
python3 MaxText/train.py MaxText/configs/base.yml \
  run_name=$YOUR_JOB_NAME \
  base_output_directory=gs://<my-bucket> \
  dataset_type=synthetic \
  steps=10
```
Next, you can try training on a HugginFace dataset, see [Data Input Pipeline](https://github.com/google/maxtext/blob/main/getting_started/Data_Input_Pipeline.md) for data input options.

5. If you want to decode, you can decode as follows.
```
python3 MaxText/decode.py MaxText/configs/base.yml \
  run_name=$YOUR_JOB_NAME \
  base_output_directory=gs://<my-bucket> \
  per_device_batch_size=1
```
Be aware, these decodings will be random. To get high quality decodings you need pass in a checkpoint, typically via the `load_parameters_path` argument.


#### Running on NVIDIA GPUs
1. Use `bash docker_build_dependency_image.sh DEVICE=gpu` can be used to build a container with the required dependencies.
2. After installation is completed, run training with the command on synthetic data:
```
python3 MaxText/train.py MaxText/configs/base.yml \
  run_name=$YOUR_JOB_NAME \
  base_output_directory=gs://<my-bucket> \
  dataset_type=synthetic \
  steps=10  
```

3. If you want to decode, you can decode as follows.
```
python3 MaxText/decode.py MaxText/configs/base.yml \
  run_name=$YOUR_JOB_NAME \
  base_output_directory=gs://<my-bucket> \
  per_device_batch_size=1  
```

* If you see the following error when running inside a container, set a larger `--shm-size` (e.g. `--shm-size=1g`)
```
Failed to execute XLA Runtime executable: run time error: custom call 'xla.gpu.all_reduce' failed: external/xla/xla/service/gpu/nccl_utils.cc:297: NCCL operation ncclCommInitRank(&comm, nranks, id, rank) failed: unhandled cuda error (run with NCCL_DEBUG=INFO for details); current tracing scope: all-reduce-start.2; current profiling annotation: XlaModule:#hlo_module=jit__unnamed_wrapped_function_,program_id=7#.
```

## Getting Starting: Multihost development
There are three patterns for running MaxText with more than one host.

1. [GKE, recommended] [Running Maxtext with xpk](getting_started/Run_MaxText_via_xpk.md) - Quick Experimentation and Production support
2. [GCE] [Running Maxtext with Multihost Jobs](getting_started/Run_MaxText_via_multihost_job.md) - Long Running Production Jobs with Queued Resources
3. [GCE] [Running Maxtext with Multihost Runner](getting_started/Run_MaxText_via_multihost_runner.md) -  Fast experiments via multiple ssh connections.

## Getting Starting: Preflight Optimizations

Once you've gotten workloads running, there are important optimizations you might want to put on your cluster. Please check the doc [PREFLIGHT.md](https://github.com/google/maxtext/blob/main/PREFLIGHT.md)
