### Getting starting with benchmark running in MaxText

Two approaches are here:

1. Run a model recipe with a single CLI command. Great to replicate performance results previously measured. See https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/trillium for examples.
2. Run several experiments pythonically across a sweep of parameters (cluster configuration, maxtext parameters) with XPK workloads.

- **xla_flags_library.py**: A grouping of xla flags organized by purpose with details on how they can be applied to a model.
- **maxtext_trillium_model_config.py**: A list of model definitions for Trillium. See optimized models here and how they apply xla flags. This config provides a pythonic way to run MaxText models.
- **benchmark_runner.py**: A cli interface to running a specific model recipe, on pathways or mcjax directly or with orchestration like xpk with one command.

```shell
# McJax with XPK
CLUSTER=my-cluster
ZONE=my-zone
PROJECT=my-project
python3 -m benchmarks.benchmark_runner xpk --project $PROJECT --zone $ZONE --cluster_name $CLUSTER --device_type v6e-256 --base_output_directory gs://maxtext-experiments-tpem/ --num_steps=5
```

```shell
# Pathways with XPK
export RUNNER=us-docker.pkg.dev/path/to/maxtext_runner
export PROXY_IMAGE=us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server
export SERVER_IMAGE=us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server

python3 -m benchmarks.benchmark_runner xpk --project $PROJECT --zone $ZONE --cluster_name $CLUSTER --device_type v6e-256 --base_output_directory gs://maxtext-experiments-tpem/ --num_steps=5 --pathways_server_image="${SERVER_IMAGE}" --pathways_proxy_server_image="${PROXY_IMAGE}" --pathways_runner_image="${RUNNER}"
```

```shell
# On-device
# Run model benchmark on current device (must run same command on all workers).
python3 -m benchmarks.benchmark_runner on-device --base_output_directory gs://maxtext-experiments-tpem/ --run_name="test-run" --num_steps=5
```

- **maxtext_xpk_runner.py**: A pythonic way to run xpk workloads! With the magic of for looping and python code, run several xpk workloads across a sweep of parameters including libtpu version, gke clusters, and maxtext parameters with one python script.

```shell
# Loop possibilities:
  # 1. Test different libtpu nightly versions.
  #  for libtpu_type in [
  #           LibTpuType.NIGHTLY
  #       ]:
  #     todays_date = time.strftime('%Y%m%d')
  #    for date in ['20241201', '20241202', todays_date]:

  # 2. Test different model configurations.
  # for remat_policy in ['qkv_proj_offloaded', 'minimal']:
  #   model.tuning_params['remat_policy'] = remat_policy

for model in list_of_models:
    # Run workloads on the below clusters
    for cluster_config in [
      v5e_cluster_config,
      v6e_cluster_config,
    ]:
      # Run workloads in the following slice configurations
      for num_slices in [1, 2 , 4]:
        # Use the libtpu dependencies from:
        for libtpu_type in [
            LibTpuType.MAXTEXT
        ]:
          wl_config = WorkloadConfig(
            model=model,
            num_slices=num_slices,
            device_type=cluster_config.device_type,
            base_output_directory=base_output_dir,
            priority="medium",
            max_restarts=0,
            libtpu_type=libtpu_type,
            libtpu_nightly_version="",
            base_docker_image=base_docker_image,
            pathways_config=None
          )
          command, name = generate_xpk_workload_cmd(
            cluster_config=cluster_config,
            wl_config=wl_config
          )

          print(f"Name of the workload is: {name} \n")
          xpk_workload_names.append(name)

          print(f"XPK command to be used is: {command} \n")
          xpk_workload_cmds.append(command)

  for xpk_workload_name, xpk_workload_cmd in zip(xpk_workload_names, xpk_workload_cmds):
    return_code = run_command_with_updates(xpk_workload_cmd, xpk_workload_name)
    if return_code != 0:
      print(f'Unable to run xpk workload: {xpk_workload_name}')

```
