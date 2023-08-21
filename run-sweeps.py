#!/usr/bin/env python3

from multihost_job import main as multihost_job_main
import yaml
import copy
import os

args = {
    'dryrun': True,
    'tpu': 'v4', # 'v4' 'v5'
    'pool': 'default', # 'default', 'stable-fleet'
}

###################    Common Code    ###################
def calc_chinchilla_step_count(num_params_billions, num_slices):
    base_steps = 20000 # number of Chinchilla steps for 1B model on 1 pod for per_device_batch of 4 seq length 1k
    return int(base_steps * num_params_billions / num_slices)

def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def update_yaml_fields(yaml_data, update_dict, allow_new_keys=True):
    yaml_copy=copy.deepcopy(yaml_data)
    for key, value in update_dict.items():
        if allow_new_keys or key in yaml_copy:
            yaml_copy[key] = value
    return yaml_copy

def write_yml(yml_filename, experiment_yml):
    with open(yml_filename, 'w') as file:
        yaml.dump(experiment_yml, file)

# Common multihost_job arguments (although slice is likely to change)
V5_MHJ_DICT={
    '--BUCKET_NAME': 'mattdavidow-maxtext-br',
    '--NUM_SLICE': 1,
    '--TPU_TYPE': 'v5litepod-256',  # v5litepod-16
    '--VERSION': 'v2-alpha-tpuv5-lite',
    '--PROJECT': 'tpu-prod-env-multipod',
    '--ZONE': 'us-east5-b'
}

V5_MHJ_DICT_STABLE_FLEET={
    '--BUCKET_NAME': 'mattdavidow-maxtext-br',
    '--NUM_SLICE': 1,
    '--TPU_TYPE': 'v5litepod-256',  # v5litepod-16
    '--VERSION': 'v2-alpha-tpuv5-lite',
    '--PROJECT': 'tpu-prod-env-multipod',
    '--ZONE': 'us-east5-b',
    '--COMMAND_TYPE': 'curl'
}

V4_MHJ_DICT={
    '--BUCKET_NAME': 'mattdavidow-br',  # for cloud-tpu-multipod-dev
    '--NUM_SLICE': 1,
    '--TPU_TYPE': 'v4-128',  # v4-8
    '--VERSION': 'tpu-ubuntu2204-base',
    '--PROJECT': 'cloud-tpu-multipod-dev',
    '--ZONE': 'us-central2-b',
    '--CQR_EXTRA_ARGS': ' --best-effort',
}

base_yml_file = 'MaxText/configs/base.yml'
BASE_YML_DATA=read_yaml_file(base_yml_file)

# This is a common base among all (or nearly all) sweeps
v5_base_yml_updates = {
    'per_device_batch_size':4,
    'enable_dropout':False,
    'base_output_directory':'gs://maxtext-experiments-multipod',
    'dataset_path':'gs://max-datasets-rogue',
    'remat_policy':'full',
    'learning_rate':1e-3,
}

v4_base_yml_updates = {
    'per_device_batch_size':4,
    'enable_dropout':False,
    'base_output_directory':'gs://max-experiments',
    'dataset_path':'gs://maxtext-dataset',
    'remat_policy':'full',
    'learning_rate':1e-3,
}

BASE_MHJ_CMD="""export LIBTPU_INIT_ARGS="--xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true" && \
bash setup_with_retries.sh && \
python3 MaxText/train.py """

def run_experiment(experiment_mhj):
    mhj_args = []
    for key in experiment_mhj.keys():
        mhj_args.append(key)
        mhj_args.append(str(experiment_mhj[key]))
    # print('MHJ final args: ', mhj_args)
    if args['dryrun']:
        import pprint
        pprint.pprint(mhj_args)
        print()
    else:
        multihost_job_main(mhj_args)

def get_mhj_command(base_cmd, mhj_yml_filename):
    return base_cmd + mhj_yml_filename

def run_sweep(sweep_base_yml, sweep_base_mhj_dict, experiment_list, base_run_name, attempt_number, steps=None, only_print_run_names=False):
    for experiment in experiment_list:
        experiment_mhj = update_yaml_fields(sweep_base_mhj_dict, experiment['mhj'])
        experiment_yml = update_yaml_fields(sweep_base_yml, experiment['maxtext'])

        # Calculate steps:
        if not steps:
            steps = calc_chinchilla_step_count(experiment_yml['global_parameter_scale'], experiment_mhj['--NUM_SLICE'])

        run_name= f'{base_run_name}-{experiment["name"]}-a{attempt_number}'
        if only_print_run_names:
            print(run_name)
        else:
            # Write final maxtext yml file
            experiment_yml = update_yaml_fields(experiment_yml, {'steps':steps,'run_name':run_name})
            experiment_yml_file=f"MaxText/configs/{run_name}.yml"
            write_yml(experiment_yml_file, experiment_yml)
            mhj_command=get_mhj_command(BASE_MHJ_CMD, experiment_yml_file)
            experiment_mhj = update_yaml_fields(experiment_mhj, {'--RUN_NAME':run_name,'--COMMAND':mhj_command})

            print("Running experiment: ",run_name)
            run_experiment(experiment_mhj)

            # Cleanup - delete writen yml
            os.remove(experiment_yml_file)


###################    Sweep 10 (load from 10k checkpoint, no checkpoint loading)    ###################
def run_sweep_10_load(attempt_number, only_print_run_names=False):
    # Experiment Base
    sweep_base_yml_update={
        'global_parameter_scale':8,
        'int8_training': True,
        'save_period': 2000,
        'load_from_other_directory': 'gs://maxtext-experiments-multipod/int8-sweep10-fresh-fwdT_bwdT-a2/checkpoints',
        'load_from_other_directory_step': 10000
    }

    if args['tpu'] == 'v4':
        base_yml_updates = v4_base_yml_updates
    elif args['tpu'] == 'v5':
        base_yml_updates = v5_base_yml_updates
    else:
        assert False

    BASE_YML_DATA2=update_yaml_fields(BASE_YML_DATA, base_yml_updates)
    sweep_base_yml = update_yaml_fields(BASE_YML_DATA2, sweep_base_yml_update)
    if args['tpu'] == 'v4':
        sweep_base_mhj = V4_MHJ_DICT
    elif args['tpu'] == 'v5':
        sweep_base_mhj = V5_MHJ_DICT
    else:
        assert False

    # Experiment Axes
    fwd_int8_array=[True, False]
    fwd_int8_name_array=['T', 'F']

    bwd_int8_array=[True, False]
    bwd_int8_name_array=['T', 'F']

    experiment_list = []
    for fwd_int8, fwd_int8_name in zip(fwd_int8_array, fwd_int8_name_array):
        for bwd_int8, bwd_int8_name in zip(bwd_int8_array, bwd_int8_name_array):
            experiment_name = f"fwd{fwd_int8_name}_bwd{bwd_int8_name}"
            maxtext_config={'fwd_int8':fwd_int8, 'bwd_int8':bwd_int8}
            experiment_list.append({'name':experiment_name, 'maxtext':maxtext_config,'mhj':{}})

    base_run_name='int8-sweep10-load-10k'
    run_sweep(sweep_base_yml, sweep_base_mhj, experiment_list, base_run_name, attempt_number, only_print_run_names=only_print_run_names)


###################    Sweep 10 (fresh, no checkpoint loading)    ###################
def run_sweep_10_fresh(attempt_number, only_print_run_names=False):
    # Experiment Base
    sweep_base_yml_update={
        'global_parameter_scale':8,
        'int8_training': True,
        'save_period': 2000
    }

    if args['tpu'] == 'v4':
        base_yml_updates = v4_base_yml_updates
    elif args['tpu'] == 'v5':
        base_yml_updates = v5_base_yml_updates
    else:
        assert False

    BASE_YML_DATA2=update_yaml_fields(BASE_YML_DATA, base_yml_updates)
    sweep_base_yml = update_yaml_fields(BASE_YML_DATA2, sweep_base_yml_update)
    if args['tpu'] == 'v4':
        sweep_base_mhj = V4_MHJ_DICT
    elif args['tpu'] == 'v5':
        sweep_base_mhj = V5_MHJ_DICT
    else:
        assert False

    # Experiment Axes
    fwd_int8_array=[True, False]
    fwd_int8_name_array=['T', 'F']

    bwd_int8_array=[True, False]
    bwd_int8_name_array=['T', 'F']

    experiment_list = []
    for fwd_int8, fwd_int8_name in zip(fwd_int8_array, fwd_int8_name_array):
        for bwd_int8, bwd_int8_name in zip(bwd_int8_array, bwd_int8_name_array):
            experiment_name = f"fwd{fwd_int8_name}_bwd{bwd_int8_name}"
            maxtext_config={'fwd_int8':fwd_int8, 'bwd_int8':bwd_int8}
            experiment_list.append({'name':experiment_name, 'maxtext':maxtext_config,'mhj':{}})

    base_run_name='int8-sweep10-fresh'
    run_sweep(sweep_base_yml, sweep_base_mhj, experiment_list, base_run_name, attempt_number, only_print_run_names=only_print_run_names)

###################    Sweep 9    ###################
def run_sweep_9(attempt_number, only_print_run_names=False):
    # Experiment Base
    sweep_base_yml_update={
        'global_parameter_scale':8,
        'int8_training': True,
        'save_period': 2000,
        'learning_rate': 5e-3,
    }

    if args['tpu'] == 'v4':
        base_yml_updates = v4_base_yml_updates
    elif args['tpu'] == 'v5':
        base_yml_updates = v5_base_yml_updates
    else:
        assert False

    BASE_YML_DATA2=update_yaml_fields(BASE_YML_DATA, base_yml_updates)
    sweep_base_yml = update_yaml_fields(BASE_YML_DATA2, sweep_base_yml_update)
    if args['tpu'] == 'v4':
        sweep_base_mhj = V4_MHJ_DICT
    elif args['tpu'] == 'v5':
        sweep_base_mhj = V5_MHJ_DICT
    else:
        assert False

    # Experiment Axes
    experiment_list = [
        {'name':"baseline",'maxtext':{},'mhj':{}},
        {'name':"adam_eps_1e-6",'maxtext':{'adam_eps':1e-6}, 'mhj':{}},
        {'name':"adam_eps_1e-7",'maxtext':{'adam_eps':1e-7}, 'mhj':{}},
        {'name':"adam_b1_95", 'maxtext':{'adam_b1':0.85}, 'mhj':{}},
        {'name':"adam_b1_85", 'maxtext':{'adam_b1':0.85}, 'mhj':{}},
        {'name':"adam_b1_80", 'maxtext':{'adam_b1':0.80}, 'mhj':{}},
        {'name':"adam_b2_93", 'maxtext':{'adam_b2':0.98}, 'mhj':{}},
        {'name':"adam_b2_93", 'maxtext':{'adam_b2':0.93}, 'mhj':{}},
        {'name':"adam_b2_90", 'maxtext':{'adam_b2':0.90}, 'mhj':{}},
        {'name':"adam_b2_85", 'maxtext':{'adam_b2':0.85}, 'mhj':{}},
        ]

    base_run_name='int8-sweep9'
    run_sweep(sweep_base_yml, sweep_base_mhj, experiment_list, base_run_name, attempt_number, only_print_run_names=only_print_run_names)

###################    Sweep 8    ###################
def run_sweep_8(base_run_name, attempt_number, only_print_run_names=False):
    # Experiment Base
    sweep8_base_yml_update={
        'global_parameter_scale':8,
        'load_from_other_directory':'gs://maxtext-experiments-multipod/mattdavidow-sweep-clipping-a1_int8T_size8_pods4_clippingOff_key1/checkpoints',
        'load_from_other_directory_step':10000,
        'adam_eps':1e-3,
        'int8_training': True
    }

    if args['tpu'] == 'v4':
        base_yml_updates = v4_base_yml_updates
    elif args['tpu'] == 'v5':
        base_yml_updates = v5_base_yml_updates
    else:
        assert False

    BASE_YML_DATA2=update_yaml_fields(BASE_YML_DATA, base_yml_updates)
    sweep8_base_yml = update_yaml_fields(BASE_YML_DATA2, sweep8_base_yml_update)
    if args['tpu'] == 'v4':
        sweep8_base_mhj = V4_MHJ_DICT
    elif args['tpu'] == 'v5':
        sweep8_base_mhj = V5_MHJ_DICT
    else:
        assert False
    sweep8_steps = 15000

    # Experiment Axes
    fwd_int8_array=['True','False']
    fwd_int8_name_array=['T', 'F']

    bwd_int8_array=['True','False']
    bwd_int8_name_array=['T', 'F']

    experiment_list = []
    for fwd_int8, fwd_int8_name in zip(fwd_int8_array, fwd_int8_name_array):
        for bwd_int8, bwd_int8_name in zip(bwd_int8_array, bwd_int8_name_array):
            experiment_name = f"fwd{fwd_int8_name}_bwd{bwd_int8_name}"
            maxtext_config={'fwd_int8':fwd_int8, 'bwd_int8':bwd_int8}
            experiment_list.append({'name':experiment_name, 'maxtext':maxtext_config,'mhj':{}})

    run_sweep(sweep8_base_yml, sweep8_base_mhj, experiment_list, base_run_name, attempt_number,steps=sweep8_steps, only_print_run_names=only_print_run_names)

###################    Sweep 7    ###################
def run_sweep_7(base_run_name, attempt_number, only_print_run_names=False):

    # All experiments inherit:
    sweep7_base_yml_update={
        'global_parameter_scale':8,
        'load_from_other_directory':'gs://maxtext-experiments-multipod/mattdavidow-sweep-clipping-a1_int8T_size8_pods4_clippingOff_key1/checkpoints',
        'load_from_other_directory_step':2000,
        'learning_rate':5e-3,
        'adam_eps':1e-3
    }

    # Experiment Axis:
    experiment_list = [
        {'name':"baseline",'maxtext':{},'mhj':{}},
        {'name':"adam_eps_1e-2",'maxtext':{'adam_eps':1e-2}, 'mhj':{}},
        {'name':"adam_eps_1e-8",'maxtext':{'adam_eps':1e-8}, 'mhj':{}},
        {'name':"adam_b1_85", 'maxtext':{'adam_b1':0.85}, 'mhj':{}},
        {'name':"adam_b1_80", 'maxtext':{'adam_b1':0.80}, 'mhj':{}},
        {'name':"adam_b2_93", 'maxtext':{'adam_b2':0.93}, 'mhj':{}},
        {'name':"adam_b2_90", 'maxtext':{'adam_b2':0.90}, 'mhj':{}},
        {'name':"adam_b2_85", 'maxtext':{'adam_b2':0.85}, 'mhj':{}},
        ]

    if args['tpu'] == 'v4':
        base_yml_updates = v4_base_yml_updates
    elif args['tpu'] == 'v5':
        base_yml_updates = v5_base_yml_updates
    else:
        assert False

    BASE_YML_DATA2=update_yaml_fields(BASE_YML_DATA, base_yml_updates)
    sweep7_base_yml = update_yaml_fields(BASE_YML_DATA2, sweep7_base_yml_update)

    if args['tpu'] == 'v4':
        sweep7_base_mhj = V4_MHJ_DICT
    elif args['tpu'] == 'v5':
        sweep7_base_mhj = V5_MHJ_DICT
    else:
        assert False

    for experiment in experiment_list:
        # print("Starting experiment: ", experiment['name']) # DEBUG
        # Define "independent mhj config: All except --COMMAND and --RUN_NAME"
        experiment_mhj = update_yaml_fields(sweep7_base_mhj, experiment['mhj'])

        # Define "indepedent" maxtext configs (all except step count and run_name)
        experiment_yml = update_yaml_fields(sweep7_base_yml, experiment['maxtext'])

        # Calculate steps:
        # We don't have compute/time to run to Chinchilla steps on 1 pod = 160k
        #steps = calc_chinchilla_step_count(experiment_yml['global_parameter_scale'], experiment_mhj['--NUM_SLICE'])
        steps = 15000

        # Define run_name:
        run_name= f'{base_run_name}-{experiment["name"]}-a{attempt_number}'
        experiment_yml = update_yaml_fields(experiment_yml, {'steps':steps,'run_name':run_name})
        experiment_yml_file=f"MaxText/configs/{run_name}.yml"
        write_yml(experiment_yml_file, experiment_yml)

        # Define mhj command:
        # print("sweep7_base_mhj: ", sweep7_base_mhj)
        # print("experiment_yml: ", experiment_yml)

        mhj_command=get_mhj_command(BASE_MHJ_CMD, experiment_yml_file)

        # Update mhj and maxtext dicts with these dependent values (steps, run_name, command)
        experiment_mhj = update_yaml_fields(experiment_mhj, {'--RUN_NAME':run_name,'--COMMAND':mhj_command})

        # print('experiment_yml: ', experiment_yml) # DEBUG
        # print('experiment_mhj: ', experiment_mhj) # DEBUG

        print(run_name)
        if not only_print_run_names:
            run_experiment(experiment_mhj)

        # Cleanup - delete writen yml
        os.remove(experiment_yml_file)

###################    Sweep t1 (test 1)    ###################
def sweep_test1(attempt: str):
    # Experiment Base
    sweep8_base_yml_update={
        'global_parameter_scale':1,
        # 'global_parameter_scale':8,
        # 'load_from_other_directory':'gs://maxtext-experiments-multipod/mattdavidow-sweep-clipping-a1_int8T_size8_pods4_clippingOff_key1/checkpoints',
        # 'load_from_other_directory_step':10000,
        # 'adam_eps':1e-3,
        # 'int8_training': True
        'int8_training': False
    }
    steps = 2000

    if args['tpu'] == 'v4':
        base_yml_updates = v4_base_yml_updates
    elif args['tpu'] == 'v5':
        base_yml_updates = v5_base_yml_updates
    else:
        assert False

    BASE_YML_DATA2=update_yaml_fields(BASE_YML_DATA, base_yml_updates)
    sweep8_base_yml = update_yaml_fields(BASE_YML_DATA2, sweep8_base_yml_update)
    if args['tpu'] == 'v4':
        sweep8_base_mhj = V4_MHJ_DICT
    elif args['tpu'] == 'v5':
        sweep8_base_mhj = V5_MHJ_DICT
    else:
        assert False

    experiment_list = []
    experiment_name = f"lew_test1"
    maxtext_config={}
    experiment_list.append({'name':experiment_name, 'maxtext':maxtext_config,'mhj':{}})

    run_sweep(sweep8_base_yml, sweep8_base_mhj, experiment_list, 'test1', attempt, steps=steps, only_print_run_names=False)


def bname(b: bool):
    assert b == True or b == False
    return str(b)[0]


def base_yml(model_size, load_from=None, load_step=None):
    return base_yml


def run_job(
        run_name,
        maxtext_config,
        *,
        model_size=8,
):

    yml = read_yaml_file('MaxText/configs/base.yml')
    yml = update_yaml_fields(yml, maxtext_config)
    num_slice = yml['num_slice']

    yml = update_yaml_fields(yml, {
        'global_parameter_scale': model_size,
        'steps': calc_chinchilla_step_count(model_size, num_slice)
    })

    attempt = args['attempt']
    sweep_name = args['sweep']
    run_name = f'int8-{sweep_name}-a{attempt}-{run_name}'
    yml = update_yaml_fields(yml, {'run_name': run_name})
    experiment_yml_file = f"MaxText/configs/{run_name}.yml"
    write_yml(experiment_yml_file, yml)
    mhj_command = get_mhj_command(BASE_MHJ_CMD, experiment_yml_file)
    experiment_mhj = update_yaml_fields(V5_MHJ_DICT, {'--RUN_NAME': run_name,'--COMMAND': mhj_command, '--NUM_SLICE': num_slice})

    print("Running experiment: ", run_name)
    mhj_args = []
    for key in experiment_mhj.keys():
        mhj_args.append(key)
        mhj_args.append(str(experiment_mhj[key]))

    if args['dryrun']:
        import pprint
        pprint.pprint(mhj_args)
        print()
    else:
        multihost_job_main(mhj_args)

    if args['delyml']:
        os.remove(experiment_yml_file)


def run_s11():
    def run(fwd_int8: bool, bwd_int8: bool, clip:str):
        clip_letter=clip[0]
        config={'fwd_int8':fwd_int8, 'bwd_int8':bwd_int8}
        if clip == 'global':
            config['clip_by_global_norm'] = 1.0
        elif clip == 'rms':
            config['clip_by_block_rms'] = 5e-4
        else:
            assert clip == 'none'
        run_name= f'fwd{bname(fwd_int8)}_bwd{bname(bwd_int8)}_clip-{clip_letter}'
        # load_from='int8-sweep10-fresh-fwdT_bwdT-a2', load_step=1000
        run_job(run_name, config)

    run(True, True, 'none')
    run(True, True, 'global')
    run(True, True, 'rms')   # no idea if this is a good value, but charts on 1B model and large LR suggest yes.
    run(True, False, 'none')
    run(True, False, 'global')
    run(True, False, 'rms')   # no idea if this is a good value, but charts on 1B model and large LR suggest yes.


def run_s12():
    def run(
            run_name,
            *,
            fwd = True,
            bwd = True,
            load_from='int8-sweep10-fresh-fwdT_bwdT-a2',
            quantize_logits=True,
            use_dq=False,
    ):
        config = {
            'fwd_int8': fwd,
            'bwd_int8': bwd,
            'use_dqdg': use_dq,
            'quantize_logits': quantize_logits,
        }
        if load_from is not None:
            config.update({
                'load_from_other_directory': f'gs://maxtext-experiments-multipod/{load_from}/checkpoints',
                'load_from_other_directory_step': 2000
            })
        # run_name= f'{bname(fwd)}{bname(bwd)}-load{bname(load_from is not None)}'
        run_job(run_name, config)


    run('baseline')
    run('bwdF-logitsF', bwd=False, quantize_logits=False)
    run('bwdT-logitsF', bwd=True,  quantize_logits=False)
    run('dqT', use_dq=True)

    # Logits sweep
    # take S10 and rerun without logits quantization  TF vs TT
    # dqdg sweep
    # take S10 and rerun with dqdg, just TT


def run_s15():
    def run(
            *,
            clip_by_ucb = True,
            fwd_int8 = True,
            dlhs_int8 = True,
            drhs_int8 = True,
            lr_mul = 1.,
    ):
        config = {
            'load_from_other_directory': f'gs://maxtext-experiments-multipod/int8-sweep10-fresh-fwdT_bwdT-a2/checkpoints',
            'load_from_other_directory_step': 10000,

            'fwd_int8': fwd_int8,
            'dlhs_int8': dlhs_int8,
            'drhs_int8': drhs_int8,
            'clip_by_ucb': clip_by_ucb,
            'learning_rate': 1.e-3 * lr_mul,
        }
        run_name = f'{bname(fwd_int8)}{bname(dlhs_int8)}{bname(drhs_int8)}_ucb{bname(clip_by_ucb)}-LR{int(lr_mul)}'

        run_job(run_name, config)

    # Q: Which one is the sourse of the loss loss?
    run(dlhs_int8=True, drhs_int8=False)
    run(dlhs_int8=False, drhs_int8=True)
    # A: It seems that the loss loss of  drhs_int8=True is twice bigger than dlhs_int8=True, but spikes are terrible.

    # Q: Does clip_by_ucb help in general?  Does it help with quantization?
    run(dlhs_int8=True, drhs_int8=True)
    run(dlhs_int8=True, drhs_int8=True, lr_mul=100.0)

    # TODO: standard grad clip?

def run_s16():
    config = {
        'fwd_int8': True,
        'dlhs_int8': True,
        'drhs_int8': True,
        'learning_rate': 1.e-3,
        'num_slice': 4,
        'save_period': 1000,
    }
    run_job('TTT-checkpoint_baseline-4s', config)

def run_s16b():
    def run(
            *,
            fwd = True,
            dlhs = True,
            drhs = True,
            lr_mul = 1.0,
    ):
        config = {
            'fwd_int8': fwd,
            'dlhs_int8': dlhs,
            'drhs_int8': drhs,
            'learning_rate': 1.e-3 * lr_mul,
            'num_slice': 4,
            'save_period': 1000,
            'load_from_other_directory': f'gs://maxtext-experiments-multipod/int8-s16-a1-TTT-checkpoint_baseline-4s/checkpoints',
            'load_from_other_directory_step': 4000, # end of warmup
        }
        run_job('TTT-checkpoint_baseline-4s', config)
    run()
    # run(dlhs = False, drhs=False)
    # run(fwd = False, dlhs = False, drhs=False)
    # run(lr_mul=10.0)
    # run(lr_mul=100.0)

def run_s17():
    def run(
            *,
            clip_by_global_norm = 1.0,
            fwd_int8 = True,
            dlhs_int8 = True,
            drhs_int8 = True,
            lr_mul = 1.,
    ):
        config = {
            'load_from_other_directory': f'gs://maxtext-experiments-multipod/int8-sweep10-fresh-fwdT_bwdT-a2/checkpoints',
            'load_from_other_directory_step': 10000,

            'fwd_int8': fwd_int8,
            'dlhs_int8': dlhs_int8,
            'drhs_int8': drhs_int8,
            'clip_by_global_norm': clip_by_global_norm,
            'learning_rate': 1.e-3 * lr_mul,
        }
        run_name = f'{bname(fwd_int8)}{bname(dlhs_int8)}{bname(drhs_int8)}_global{bname(clip_by_global_norm)}-LR{int(lr_mul)}'

        run_job(run_name, config)

    run()
    run(lr_mul=10.)
    run(lr_mul=10.)
    run(drhs_int8=False)
    run(dlhs_int8=False)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='TPU configuration options')
    parser.add_argument('--dryrun', type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--delyml', type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--tpu', type=str, default='v5')
    parser.add_argument('--pool', type=str, default='default') # 'default' or 'stable-fleet'
    parser.add_argument('--sweep', type=str, default='')
    parser.add_argument('--attempt', type=str, default='')
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
