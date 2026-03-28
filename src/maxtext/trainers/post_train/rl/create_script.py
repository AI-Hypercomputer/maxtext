import os
from jinja2 import Template
import argparse

def generate_rl_config(
    metadata_name, 
    batch_size, 
    rollout_data_parallelism, 
    rollout_tensor_parallelism, 
    rollout_expert_parallelism, 
    trainer_devices_fraction, 
    subslice_shape, 
    enable_single_controller, 
    sampler_devices_fraction, 
    base_output_directory, 
    run_name,
    hf_token,
    extra_config
):
    script_template = """#!/bin/bash
CLUSTER_NAME=next-devx-1
DEVICE_TYPE=tpu7x-4x4x4
PROJECT=tpu-prod-env-automated
ZONE=us-central1
IMAGE_DIR=gcr.io/cloud-tpu-multipod-dev/sanbao/maxtext_reshard_image:latest

command="pip install --no-deps git+https://github.com/AI-Hypercomputer/pathways-utils.git@v0.1.4 && \\
pip install src/maxtext/integration/vllm && \\
HF_TOKEN={{ hf_token }} JAX_RANDOM_WEIGHTS=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 NEW_MODEL_DESIGN=1 TPU_MIN_LOG_LEVEL=0 TF_CPP_MIN_LOG_LEVEL=0 TPU_STDERR_LOG_LEVEL=0 JAX_PLATFORMS=proxy,cpu JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 ENABLE_PATHWAYS_PERSISTENCE=1 \\
python3 -m src.maxtext.trainers.post_train.rl.reshard_debug src/maxtext/configs/post_train/rl.yml \\
model_name=qwen3-30b-a3b \\
tokenizer_path=Qwen/Qwen3-30B-A3B \\
run_name={{ run_name }} \\
base_output_directory={{ base_output_directory }} \\
hf_access_token={{ hf_token }} \\
batch_size={{ batch_size }} \\
rl.num_generations=8 \\
num_batches=10 \\
rollout_data_parallelism={{ rollout_data_parallelism }} \\
rollout_tensor_parallelism={{ rollout_tensor_parallelism }} \\
rollout_expert_parallelism={{ rollout_expert_parallelism }} \\
hbm_utilization_vllm=0.6 \\
scan_layers=True \\
allow_split_physical_axes=True \\
vllm_hf_overrides='{architectures: [\\"MaxTextForCausalLM\\"]}' \\
vllm_additional_config='{maxtext_config: {model_name: qwen3-30b-a3b, allow_split_physical_axes: true, log_config: false, weight_dtype: bfloat16}}' \\
trainer_devices_fraction={{ trainer_devices_fraction }} \\
subslice_shape='{{ subslice_shape }}' \\
enable_single_controller={{ enable_single_controller }} \\
sampler_devices_fraction={{ sampler_devices_fraction }} {{extra_config}}"

python3 ~/Documents/xpk/run.py workload create-pathways  --workload {{ metadata_name }} \\
--docker-image ${IMAGE_DIR} \\
--cluster ${CLUSTER_NAME} \\
--tpu-type=${DEVICE_TYPE} \\
--project=$PROJECT \\
--zone=$ZONE \\
--num-slices=1  \\
--priority=high \\
--custom-pathways-worker-args="--xprof_max_trace_buffers=16384" \\
--command "${command}"
"""

    t = Template(script_template)
    rendered_script = t.render(
        metadata_name=metadata_name,
        batch_size=batch_size,
        rollout_data_parallelism=rollout_data_parallelism,
        rollout_tensor_parallelism=rollout_tensor_parallelism,
        rollout_expert_parallelism=rollout_expert_parallelism,
        trainer_devices_fraction=trainer_devices_fraction,
        subslice_shape=subslice_shape,
        enable_single_controller=enable_single_controller,
        sampler_devices_fraction=sampler_devices_fraction,
        base_output_directory=base_output_directory,
        run_name=run_name,
        hf_token=hf_token,
        extra_config=extra_config
    )
    
    return rendered_script

# Example Usage:
"""
python ./maxtext/src/maxtext/trainers/post_train/rl/create_script_235b.py \
      --metadata_name "${workload_name}" \
      --trainer_chips "${trainer_chips}" \
      --number_of_sampler_chips_per_replica "${sampler_chips}" \
      --sampler_replicas 1 \
      --base_output_directory "${base_output_directory}" \
      --hf_token "${hf_token}" \
      --store_directory "${store_path}" \
      --enable_tp "${enable_tp}"
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_name", type=str, required=True)
    parser.add_argument("--trainer_chips", type=int, required=True)
    parser.add_argument("--number_of_sampler_chips_per_replica", type=int, required=True)
    parser.add_argument("--sampler_replicas", type=int, required=True)
    parser.add_argument("--base_output_directory", type=str, required=True)
    parser.add_argument("--hf_token", type=str, required=True)
    parser.add_argument("--store_directory", type=str, required=True)
    parser.add_argument("--enable_tp", action="store_true", default=False, help="Enable tensor parallelism")
    parser.add_argument("--enable_ep", action="store_true", default=False, help="Enable expert parallelism")
    args = parser.parse_args()
    print(vars(args))


    # for v7x-128
    extra_config = ""
    number_of_chips = 64
    batch_size = args.trainer_chips * 2
    trainer_devices_fraction = args.trainer_chips / number_of_chips
    rollout_data_parallelism = args.sampler_replicas
    sampler_chips = args.number_of_sampler_chips_per_replica * args.sampler_replicas
    assert sampler_chips + args.trainer_chips <= number_of_chips, "Total number of chips used by trainer and sampler must be less than or equal to available chips"
    if args.enable_tp and args.enable_ep:
        rollout_tensor_parallelism = args.number_of_sampler_chips_per_replica * 2
        rollout_expert_parallelism = rollout_tensor_parallelism // 4 if rollout_tensor_parallelism >= 4 else 1
        assert rollout_tensor_parallelism % rollout_expert_parallelism == 0, "rollout_tensor_parallelism must be divisible by rollout_expert_parallelism"
        rollout_tensor_parallelism = 4 if rollout_tensor_parallelism >= 4 else rollout_tensor_parallelism
    elif args.enable_ep:
        rollout_tensor_parallelism = 1
        rollout_expert_parallelism = args.number_of_sampler_chips_per_replica * 2
    elif args.enable_tp:
        rollout_tensor_parallelism = args.number_of_sampler_chips_per_replica * 2
        rollout_expert_parallelism = 1
        extra_config += " enable_dp_attention=True" if rollout_tensor_parallelism >= 4 else ""
    else:
        assert False, "At least one of tensor parallelism or expert parallelism must be enabled"

    sampler_devices_fraction = sampler_chips / number_of_chips
    if args.trainer_chips == 4:
      enable_single_controller = "true"
    else:
        enable_single_controller = "false"

    subslice_shape_status = {
        1: "1,1,1",
        2: "2,1,1",
        4: "2,2,1",
        8: "2,2,2",
        16: "2,2,4",
        32: "2,4,4",
        64: "4,4,4",
        128: "4,4,8"}
    subslice_shape = subslice_shape_status.get(args.trainer_chips, "")
    
    output_directory = os.path.join(args.base_output_directory, args.metadata_name)

    result = generate_rl_config(
        metadata_name=args.metadata_name,
        batch_size=batch_size,
        rollout_data_parallelism=rollout_data_parallelism,
        rollout_tensor_parallelism=rollout_tensor_parallelism,
        rollout_expert_parallelism=rollout_expert_parallelism,
        trainer_devices_fraction=trainer_devices_fraction,
        subslice_shape=subslice_shape,
        enable_single_controller=enable_single_controller,
        sampler_devices_fraction=sampler_devices_fraction,
        base_output_directory=output_directory,
        run_name=args.metadata_name,
        hf_token=args.hf_token,
        extra_config=extra_config
    )
    # if the script directory does not exist, create it
    if not os.path.exists(args.store_directory):
        os.makedirs(args.store_directory)
    output_script_path = os.path.join(args.store_directory, f"{args.metadata_name}.sh")
    
    with open(output_script_path, "w") as f:
        f.write(result)