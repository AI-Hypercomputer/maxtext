# Getting Started

We recommend starting with a single host first and then moving to multihost.

## Getting Started: Download Dataset and Configure
You need to run these steps once per project prior to any local development or cluster experiments.

1. Create two gcs buckets in your project, one for to downloading and retrieving the dataset and the other for storing the logs.
2. Download the dataset in your gcs bucket
```
bash download_dataset.sh {GCS_PROJECT} {GCS_BUCKET_NAME}
```
3. Set config values for `base_output_directory` and `dataset_path` in `configs/base.yml`. `tokenizer_path` is full path for loading the tokenizer. MaxText assumes these GCS buckets are created in the same project and that it has permissions to read and write from them. We also recommend reviewing the configurable options in `configs/base.yml`, for instance you may change the `steps` or `log_period` by either modifying `configs/base.yml` or by passing in `steps` and `log_period` as additional args to the `train.py` call.

To run maxtext the TPUVMs must have permission to read the gcs bucket. These permissions are granted by service account roles, such as the `STORAGE ADMIN` role.

## Getting Started: Local Development for single host

Local development is a convenient way to run MaxText on a single host. It doesn't scale to
multiple hosts.

1. [Create and SSH to the single-host VM of your choice.](https://cloud.google.com/tpu/docs/users-guide-tpu-vm#creating_a_cloud_tpu_vm_with_gcloud) We recommend a `v4-8`.
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


#### Running on NVIDIA GPUs
1. Use `bash docker_build_dependency_image.sh DEVICE=gpu` can be used to build a container with the required dependencies.
2. After installation is completed, run training with the command:
```
python3 MaxText/train.py MaxText/configs/base.yml run_name=$YOUR_JOB_NAME
```
3. If you want to decode, you can decode as follows.
```
python3 MaxText/decode.py MaxText/configs/base.yml run_name=$YOUR_JOB_NAME
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
