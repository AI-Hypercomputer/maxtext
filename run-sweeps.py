from multihost_job import main as multihost_job_main
import yaml
import copy
import os

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
BASE_MHJ_DICT={
    '--BUCKET_NAME':'mattdavidow-maxtext-br',
    '--NUM_SLICE':1,
    '--TPU_TYPE':'v5litepod-256',
    '--VERSION':'v2-alpha-tpuv5-lite',
    '--PROJECT':'tpu-prod-env-multipod',
    '--ZONE':'us-east5-b'
}

base_yml_file = 'MaxText/configs/base.yml'
BASE_YML_DATA=read_yaml_file(base_yml_file)
# This is a common base among all (or nearly all) sweeps
base_yml_updates = {
    'per_device_batch_size':4,
    'enable_dropout':False,
    'base_output_directory':'gs://maxtext-experiments-multipod',
    'dataset_path':'gs://max-datasets-rogue',
    'remat_policy':'full'
}
BASE_YML_DATA=update_yaml_fields(BASE_YML_DATA, base_yml_updates)

BASE_MHJ_CMD="""export LIBTPU_INIT_ARGS="--xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true" && \
bash setup_with_retries.sh && \
python3 MaxText/train.py """

def run_experiment(experiment_mhj):
    mhj_args = []
    for key in experiment_mhj.keys():
        mhj_args.append(key)
        mhj_args.append(str(experiment_mhj[key]))
    # print('MHJ final args: ', mhj_args)
    multihost_job_main(mhj_args)

def get_mhj_command(base_cmd, mhj_yml):
    return base_cmd + mhj_yml

def run_sweep(sweep_base_yml, sweep_base_mhj_dict, experiment_list, base_run_name, attempt_number, steps=None):
    for experiment in experiment_list:
        experiment_mhj = update_yaml_fields(sweep_base_mhj_dict, experiment['mhj'])
        experiment_yml = update_yaml_fields(sweep_base_yml, experiment['maxtext'])
        
        # Calculate steps:
        if not steps:
            steps = calc_chinchilla_step_count(experiment_yml['global_parameter_scale'], experiment_mhj['--NUM_SLICE'])

        run_name= f'{base_run_name}-{experiment["name"]}-a{attempt_number}'

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



###################    Sweep 8    ###################
def run_sweep_8(base_run_name, attempt_number):
    # Experiment Base
    sweep8_base_yml_update={
        'global_parameter_scale':8,
        'load_from_other_directory':'gs://maxtext-experiments-multipod/mattdavidow-sweep-clipping-a1_int8T_size8_pods4_clippingOff_key1/checkpoints',
        'load_from_other_directory_step':10000,
        'adam_eps':1e-3,
        'int8_training': True
    }
    sweep8_base_yml = update_yaml_fields(BASE_YML_DATA, sweep8_base_yml_update)
    sweep8_base_mhj = BASE_MHJ_DICT
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

    run_sweep(sweep8_base_yml, sweep8_base_mhj, experiment_list, base_run_name, attempt_number,steps=sweep8_steps)

###################    Sweep 7    ###################
def run_sweep_7(base_run_name, attempt_number):
 
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

    sweep7_base_yml = update_yaml_fields(BASE_YML_DATA, sweep7_base_yml_update)
    sweep7_base_mhj = BASE_MHJ_DICT

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

        print("Running experiment: ",run_name)
        run_experiment(experiment_mhj)
        
        # Cleanup - delete writen yml
        os.remove(experiment_yml_file)


run_sweep_8('mattdavidow-sweep8', 2)
#run_sweep_7('mattdavidow-sweep7',2)