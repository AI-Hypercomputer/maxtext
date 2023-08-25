#!/usr/bin/env python3

from multihost_job import main as multihost_job_main
import yaml
import copy
import os

args = {
    'dryrun': True,
    'tpu': 'v4', # 'v4' 'v5'
    'stable': False,
}


def update_yaml_fields(yaml_data, update_dict, allow_new_keys=True):
    yaml_copy=copy.deepcopy(yaml_data)
    for key, value in update_dict.items():
        if allow_new_keys or key in yaml_copy:
            yaml_copy[key] = value
    return yaml_copy


BASE_MHJ_CMD="""export LIBTPU_INIT_ARGS="--xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true" && \
bash setup_with_retries.sh && \
python3 MaxText/train.py """


def bname(b: bool):
    assert b == True or b == False, f'not bool: "{b}"'
    return str(b)[0]


def run_job(run_name, maxtext_config):
    model_size = maxtext_config['global_parameter_scale']
    with open('MaxText/configs/base.yml', 'r') as file:
        yml = yaml.safe_load(file)

    yml = update_yaml_fields(yml, maxtext_config)
    num_slice = yml['num_slice']
    tokens_per_seq = yml['max_target_length']
    seqs_per_chip = yml['per_device_batch_size']

    def calc_chinchilla_step_count(num_params_billions, num_slice, seqs_per_chip, tokens_per_seq):
        billion = 2 ** 30
        chips_per_slice = 256
        needed_tokens = num_params_billions * billion * 20
        tokens_per_step = tokens_per_seq * seqs_per_chip * chips_per_slice * num_slice
        needed_steps = int(needed_tokens / tokens_per_step)
        return needed_steps
    lr_steps = calc_chinchilla_step_count(num_params_billions=model_size, num_slice=num_slice, seqs_per_chip=seqs_per_chip, tokens_per_seq=tokens_per_seq)

    yml = update_yaml_fields(yml, {
        'steps': lr_steps,
        'learning_rate_schedule_steps': lr_steps,
    })

    attempt = args['attempt']
    sweep_name = args['sweep']
    run_name = f'int8-{sweep_name}-a{attempt}-{run_name}'
    yml = update_yaml_fields(yml, {'run_name': run_name})
    experiment_yml_file = f"MaxText/configs/{run_name}.yml"
    with open(experiment_yml_file, 'w') as file:
        yaml.dump(yml, file)

    experiment_mhj = {
        '--RUN_NAME': run_name,
        '--BUCKET_NAME': 'mattdavidow-maxtext-br',
        '--NUM_SLICE': num_slice,
        '--TPU_TYPE': 'v5litepod-256',  # v5litepod-16
        '--VERSION': 'v2-alpha-tpuv5-lite',
        '--PROJECT': 'tpu-prod-env-multipod',
        '--ZONE': 'us-east5-b',
        '--COMMAND': BASE_MHJ_CMD + experiment_yml_file,
        # '--COMMAND_TYPE': 'curl'  # Uncomment for Stable fleet
    }
    if args['stable']:
        experiment_mhj['--COMMAND_TYPE'] = 'curl'
        experiment_mhj['--PROJECT'] = 'tpu-prod-env-vlp-2nic'


    # V4_MHJ_DICT={
    #     '--BUCKET_NAME': 'mattdavidow-br',  # for cloud-tpu-multipod-dev
    #     '--NUM_SLICE': 1,
    #     '--TPU_TYPE': 'v4-128',  # v4-8
    #     '--VERSION': 'tpu-ubuntu2204-base',
    #     '--PROJECT': 'cloud-tpu-multipod-dev',
    #     '--ZONE': 'us-central2-b',
    #     '--CQR_EXTRA_ARGS': ' --best-effort',
    # }
    # And this.
    # 'base_output_directory':'gs://max-experiments',
    # 'dataset_path':'gs://maxtext-dataset',

    print("Running experiment: ", run_name)
    mhj_args = []
    for key in experiment_mhj.keys():
        mhj_args.append(key)
        mhj_args.append(str(experiment_mhj[key]))

    if args['dryrun']:
        import pprint
        pprint.pprint(maxtext_config)
        pprint.pprint(experiment_mhj)

        print()
    else:
        multihost_job_main(mhj_args)

    if args['delyml']:
        os.remove(experiment_yml_file)



def run_s16():
    config = {
        'fwd_int8': True,
        'dlhs_int8': True,
        'drhs_int8': True,
        'learning_rate': 1.e-3,
        'num_slice': 4,
        'save_period': 1000,
        'global_parameter_scale': 8,
    }
    run_job('TTT-checkpoint_baseline-4s', config)

def run_s16_load():
    def run(
            *,
            fwd = True,
            dlhs = True,
            drhs = True,
            lr_mul = 1.0,
            clip_global = 0.0,
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
            'clip_by_global_norm': clip_global,
            'global_parameter_scale': 8,
        }
        run_name = f'4s-L-{bname(fwd)}{bname(dlhs)}{bname(drhs)}_global{int(clip_global*10)}-LR{int(lr_mul)}'
        run_job(run_name, config)
    run()
    run(dlhs=False, drhs=False)
    run(fwd=False, dlhs=False, drhs=False)
    run(lr_mul=10.0)
    run(lr_mul=10.0, clip_global=0.5)

# This is a warmup checkpoint generation for S19
def run_s18_8B_16seq_warmup():
    config = {
        'fwd_int8': True,
        'dlhs_int8': True,
        'drhs_int8': True,
        'learning_rate': 1.e-3,
        'num_slice': 4,
        'per_device_batch_size': 16,
        'save_period': 1000,
        'global_parameter_scale': 8,
    }
    run_job('yep', config)

# This is a sweep on: FFF,TFF,TTF,TTT
# For pseudo-final eval on 8B model. 4pods, 16seq
def run_s19():
    def run(
            *,
            fwd = True,
            dlhs = True,
            drhs = True,
    ):
        config = {
            'load_from_other_directory': f'gs://maxtext-experiments-multipod/int8-s18_8B_16seq_warmup-a1-yep/checkpoints',
            'load_from_other_directory_step': 1000,
            'num_slice': 4,
            'per_device_batch_size': 16,
            'fwd_int8': fwd,
            'dlhs_int8': dlhs,
            'drhs_int8': drhs,
            'global_parameter_scale': 8,
            # 'learning_rate': 1.e-3 * lr_mul,
        }
        run_name = f'{bname(fwd)}{bname(dlhs)}{bname(drhs)}'
        run_job(run_name, config)

    run(fwd=False, dlhs=False, drhs=False)
    run(fwd=True, dlhs=False, drhs=False)
    run(fwd=True, dlhs=True, drhs=False)
    run(fwd=True, dlhs=True, drhs=True)


# Same as S19 but back to 4seq and added gradient clipping.
def run_s20_base(
        *,
        fwd = True,
        dlhs = True,
        drhs = True,
        clip_global = 0.3,
        clip_by_ucb = 0, # 0 or 1
        # lrs = 0,  # This is a small delta to LR, meant as a 'seed' replacement
        lr_mul = 1.0,  # This is a small delta to LR, meant as a 'seed' replacement
):
    config = {
        # For seq16
        # 'load_from_other_directory': f'gs://maxtext-experiments-multipod/int8-s18_8B_16seq_warmup-a1-yep/checkpoints',
        # 'load_from_other_directory_step': 1000,
        'save_period': 1000,
        'load_from_other_directory': 'gs://maxtext-experiments-multipod/int8-s16-a1-TTT-checkpoint_baseline-4s/checkpoints',
        'load_from_other_directory_step': 4000, # end of warmup
        'num_slice': 4,
        'per_device_batch_size': 4,
        'fwd_int8': fwd,
        'dlhs_int8': dlhs,
        'drhs_int8': drhs,
        'clip_by_global_norm': clip_global,
        'clip_by_ucb': clip_by_ucb,
        # 'learning_rate': 1.e-3 * (1.0 + lrs / 10000.0),
        'learning_rate': 1.e-3 * lr_mul,
        'global_parameter_scale': 8,
    }
    run_name = f'{bname(fwd)}{bname(dlhs)}{bname(drhs)}-cg{int(clip_global*10):02}-cucb{clip_by_ucb}-lr{int(lr_mul*10):03}'
    run_job(run_name, config)


# S19 was spikey, back to 4seq. Add clip_global.
def run_s20():
    run_s20_base(fwd=False, dlhs=False, drhs=False)
    run_s20_base(dlhs=False, drhs=False)
    run_s20_base(drhs=False)
    run_s20_base(clip_global=0.2)
    run_s20_base(clip_global=0.3)
    run_s20_base(clip_global=0.5)


# S20 we were unlucky with spikes.
# Add more similar runs.
# Add UCB clipping.
def run_s21():
    run_s20_base(drhs=False, clip_global=0.2)
    run_s20_base(drhs=False, clip_global=0.3)
    run_s20_base(drhs=False, clip_global=0.5)
    run_s20_base(drhs=False, clip_global=0.0, clip_by_ucb=1)
    run_s20_base(drhs=False, clip_global=0.0, clip_by_ucb=1, lr_mul=2.0)
    run_s20_base(drhs=False, clip_global=0.0, clip_by_ucb=1, lr_mul=5.0)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='TPU configuration options')
    parser.add_argument('--dryrun', type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--delyml', type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--stable', type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--tpu', type=str, default='v5')
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
