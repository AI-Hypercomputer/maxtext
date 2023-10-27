#!/usr/bin/env python3

# High level idea:
# Instead of writing new config.yml, we will call xpk with maxtext commandline args
import yaml
import copy
import os
import re

def update_yaml_fields(yaml_data, update_dict, allow_new_keys=False):
    yaml_copy=copy.deepcopy(yaml_data)
    for key, value in update_dict.items():
        if not allow_new_keys:
            assert key in yaml_copy, key
        yaml_copy[key] = value
    return yaml_copy


BASE_CMD="""export LIBTPU_INIT_ARGS="--xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true" && \
python3 MaxText/train.py MaxText/configs/base.yml"""

def bname(b: bool):
    assert b == True or b == False, f'not bool: "{b}"'
    return str(b)[0]

def run_job(run_name, base_config, num_slices, **config_updates):
    maxtext_config_args = update_yaml_fields(base_config, config_updates, allow_new_keys=True)
    # TODO: Write a check that all keys are valid (match ones in base.yml)

    attempt = args.attempt
    sweep_name = args.sweep
    jobre = args.jobre
    url = f"xpk logs here"
    if not re.findall(jobre, run_name):
        print(f"SKIP: {run_name:30}", url)
        return

    print(f"RUN:  {run_name:30}", url)

    maxtext_config_args = update_yaml_fields(maxtext_config_args, {'run_name': run_name}, allow_new_keys=True)
    cmd = BASE_CMD + str(maxtext_config_args)


    xpk_cmd = ["python3", "xpk/xpk.py", "workload", "create",
    "--cluster", args.cluster,
    "--docker-image", args.docker_image,
    "--workload", run_name,
    "--tpu-type", args.tpu_type,
    "--num-slices", num_slices,
    "--command", cmd]


    if args.dryrun:
        import pprint
        pprint.pprint(xpk_cmd)
        # pprint.pprint(experiment_mhj)
        # print()
    else:
        subprocess.run(xpk_cmd, capture_output=True, check=True)


############################### FINAL runs start here

def base_test():
    return dict(
        global_parameter_scale = 2,
        base_output_directory = "gs://maxtext-experiments-multipod",
        dataset_path = "gs://max-datasets-rogue"
    )

def run_test():
    run_job("mattdavidow-b1", base_test(), 1, per_device_batch_size=1)
    run_job("mattdavidow-b1", base_test(), 1, per_device_batch_size=2)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='TPU configuration options')
    parser.add_argument('--dryrun', type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--cluster', type=str, default='', required=True)
    parser.add_argument('--docker_image', type=str, default='', required=True)
    parser.add_argument('--tpu_type', type=str, default='', required=True)
    parser.add_argument('--sweep', type=str, default='', required=True)
    parser.add_argument('--attempt', type=str, default='', required=True)
    parser.add_argument('--jobre', type=str, default='.*')
    global args
    args = parser.parse_args()
    sweep_name = args.sweep
    attempt = args.attempt

    print(args)
    sweep_fn_name = f'run_{sweep_name}'
    assert sweep_fn_name in globals(), f'{sweep_fn_name}() not defined.'
    assert attempt != ''
    sweep_fn = globals()[sweep_fn_name]
    sweep_fn()


main()