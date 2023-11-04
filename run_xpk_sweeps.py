#!/usr/bin/env python3

# High level idea:
# Instead of writing new config.yml, we will call xpk with maxtext commandline args
import yaml
import copy
import os
import re
import subprocess

def update_yaml_fields(yaml_data, update_dict, allow_new_keys=False):
    yaml_copy=copy.deepcopy(yaml_data)
    for key, value in update_dict.items():
        if not allow_new_keys:
            assert key in yaml_copy, key
        yaml_copy[key] = value
    return yaml_copy


BASE_CMD="""export LIBTPU_INIT_ARGS="--xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true" && \
python3 MaxText/train.py MaxText/configs/base.yml """

def bname(b: bool):
    assert b == True or b == False, f'not bool: "{b}"'
    return str(b)[0].lower()

def run_job(run_name, base_config, num_slices, **config_updates):
    def maxtext_arg_dict_to_str(maxtext_arg_dict):
        output_string = ""
        for key, value in maxtext_arg_dict.items():
            output_string = output_string + f"{key}={value} "
        return output_string

    if not re.findall(args.jobre, run_name):
        print(f"SKIP: {run_name:30}")
        return

    run_name = run_name + f'-a{args.attempt}'
    url = f"xpk logs here" #ex log url: https://console.cloud.google.com/kubernetes/service/us-east5/v5e-256-bodaborg/default/mattdavidow-rm-r-m-int8-t-seed1-a1/details?project=tpu-prod-env-multipod
    print(f"RUN:  {run_name:30}", url)

    # TODO: Write a check that all keys are valid (match ones in base.yml)
    maxtext_config_args = update_yaml_fields(base_config, config_updates, allow_new_keys=True)
    maxtext_config_command_line = maxtext_arg_dict_to_str(maxtext_config_args)
    cmd = BASE_CMD + maxtext_config_command_line
    #cmd = 'echo lol'
    xpk_cmd = ["python3", "xpk/xpk.py", "workload", "create",
    "--cluster", args.cluster,
    "--docker-image", args.docker_image,
    "--workload", run_name,
    "--tpu-type", args.tpu_type,
    "--num-slices", str(num_slices),
    "--command", cmd]


    #xpk_cmd = ["python3", "xpk/xpk.py", "workload", "create"]
    # TODO(mattdavidow): if dryrun run xpk in dryrun mode
    if args.dryrun:
        import pprint
        pprint.pprint(xpk_cmd)
    else:
        #subprocess.run(xpk_cmd, capture_output=True, check=True)

        try:
            completed_command = subprocess.run(xpk_cmd, capture_output=True, check=True)
            print(completed_command.stdout.decode())
        except subprocess.CalledProcessError as e:
            print(f"Error is:\n {e.stderr}")
            print(e.stdout)


############################### FINAL runs start here



def run_test():
    def base_test():
        return dict(
            global_parameter_scale = 2,
            steps=4,
            base_output_directory = "gs://maxtext-experiments-multipod",
            dataset_path = "gs://max-datasets-rogue"
        )
    run_job("mattdavidow-test-batch-1", base_test(), 1, per_device_batch_size=1)
    run_job("mattdavidow-test-batch-2", base_test(), 1, per_device_batch_size=2)

def run_remat_mystery():
    def base_remat_mystery():
        return dict(
            global_parameter_scale = 1,
            steps=3400,
            per_device_batch_size=6.0,
            learning_rate=1e-3,
            enable_checkpointing=False,
            base_output_directory = "gs://maxtext-experiments-multipod",
            dataset_path = "gs://max-datasets-rogue",   
        )

    for remat_policy in ['full', 'minimal']:
        for int8_training in [True, False]:
            for init_weights_seed in range(2):
                run_name = f"mattdavidow-rm-r-{remat_policy[0]}-int8-{bname(int8_training)}-seed{init_weights_seed}"
                run_job(run_name, base_remat_mystery(), 1, init_weights_seed=init_weights_seed, remat_policy=remat_policy, int8_training=int8_training)


def run_sweep1():
    def base_sweep1():
        return dict(
            global_parameter_scale = 1,
            steps=3400,
            per_device_batch_size=12.0,
            learning_rate=1e-3,
            enable_checkpointing=False,
            base_output_directory = "gs://maxtext-experiments-multipod",
            dataset_path = "gs://max-datasets-rogue",
            int8_training=True   
        )
    
    run_job("mattdavidow-ttt", base_sweep1(), 1, int8_ttf=False)
    run_job("mattdavidow-ttf", base_sweep1(), 1)
    run_job("mattdavidow-bfloat16", base_sweep1(), 1, int8_training=False)
    


def main():
    print("hello")
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