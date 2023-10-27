#!/usr/bin/env python3

# High level idea:
# Instead of writing new config.yml, we will call xpk with maxtext commandline args
import yaml
import copy
import os
import re

args = {
    'dryrun': True,
    'cluster': '',
    'docker_image': '',
    'tpu-type': 'v4', # 'v4' 'v5litepod-256'
    'jobre'
}


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
    maxtext_config_args = update_yaml_fields(base_config, config_updates)

    attempt = args['attempt']
    sweep_name = args['sweep']
    jobre = args['jobre']
    url = f"https://pantheon.corp.google.com/logs/query;query=timestamp%20%3E%20%222023-08-18%22%20AND%20labels.%22agent.googleapis.com%2Flog_file_path%22%3D~%22{run_name}.*%2Fmain_command_log_slice_0_worker_0%22"
    if not re.findall(jobre, run_name):
        print(f"SKIP: {run_name:30}", url)
        return

    print(f"RUN:  {run_name:30}", url)

    yml = update_yaml_fields(yml, {'run_name': run_name})
    experiment_yml_file = f"MaxText/configs/{run_name}.yml"
    with open(experiment_yml_file, 'w') as file:
        yaml.dump(yml, file)

    xpk_cmd = BASE_XPK_CMD

python3 xpk/xpk.py workload create \
--cluster bodaborgprivate5 \
--docker-image gcr.io/${PROJECT_ID}/${USER}_runner \
--workload ${USER}-first-job \
--tpu-type=v5litepod-256 \
--num-slices=1  \
--command "python3 MaxText/train.py MaxText/configs/base.yml base_output_directory=gs://maxtext-experiments-tpem/ dataset_path=gs://max-datasets-rogue/ steps=100 per_device_batch_size=1"

xpk_workload_cmd = ["python3", "xpk/xpk.py", "workload", "create", "--cluster", args.cluster, "--docker-image", args.docker-image, "--workload", run_name, "--tpu-type", args.tpu-type, "--num-slices", ]

    experiment_mhj = {
        '--RUN_NAME': run_name,
        '--BUCKET_NAME': 'mattdavidow-maxtext-br',
        '--NUM_SLICE': num_slice,
        '--TPU_TYPE': 'v5litepod-256',  # v5litepod-16
        '--VERSION': 'v2-alpha-tpuv5-lite',
        '--PROJECT': 'tpu-prod-env-multipod',
        '--ZONE': 'us-east5-b',
        '--COMMAND': mhj_cmd + experiment_yml_file,
        '--CQR_EXTRA_ARGS': ' --network=mtu9k'
        # '--COMMAND_TYPE': 'curl'  # Uncomment for Stable fleet
    }
    if args['stable']:
        experiment_mhj['--COMMAND_TYPE'] = 'curl'
        experiment_mhj['--PROJECT'] = 'tpu-prod-env-vlp-2nic'

    mhj_args = []
    for key in experiment_mhj.keys():
        mhj_args.append(key)
        mhj_args.append(str(experiment_mhj[key]))

    if args['dryrun']:
        import pprint
        # pprint.pprint(yml)
        # pprint.pprint(experiment_mhj)
        # print()
    else:
        multihost_job_main(mhj_args)

    if args['delyml']:
        os.remove(experiment_yml_file)

############################### FINAL runs start here

# Paper 16B
def baseline_s32():
    return dict(
        global_parameter_scale = 16,
        num_slice = 16,
    )

def run_s32():
    run_job("q_FFF", baseline_s32(), int8_training=False)
    run_job("q_TTF", baseline_s32())
    run_job("q_TTT", baseline_s32(), drhs_int8=True)


# Long training
def run_s38(): # 32
    # want 16000 steps on 16 slices
    run_job(f"long-FFF", ablation(gps=1), num_slice=16, fill_ratio=0.8 / 16 /1.20  , int8_training=False)
    run_job(f"long-TTF", ablation(gps=1), num_slice=16, fill_ratio=0.8 / 16 /1.20  , int8_training=True)

def base_test():
    return dict(
        global_parameter_scale = 2
        base_output_directory = "gs://maxtext-experiments-multipod"
        dataset_path = "gs://max-datasets-rogue"
    )

def run_test():
    run_job("mattdavidow-b1", base_test(), per_device_batch_size=1)
    run_job("mattdavidow-b1", base_test(), per_device_batch_size=2)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='TPU configuration options')
    parser.add_argument('--dryrun', type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--cluster', type=str, default='', required=True)
    parser.add_argument('--docker_image', type=str, default='', required=True)
    parser.add_argument('--tpu-type', type=str, default='', required=True)
    parser.add_argument('--jobre', type=str, default='.*')
    parser.add_argument('--sweep', type=str, default='', required=True)
    parser.add_argument('--attempt', type=str, default='', required=True)
    pargs = parser.parse_args()
    global args
    args = pargs.__dict__
    sweep_name = args['sweep']
    attempt = args['attempt']

    sweep_fn_name = f'run_{sweep_name}'
    assert sweep_fn_name in globals(), f'{sweep_fn_name}() not defined.'
    assert attempt != ''
    sweep_fn = globals()[sweep_fn_name]
    sweep_fn()


main()