RUN_NAME=${1}
SIZE=${2}
LOAD_FROM_OTHER=${3}

ckpt_dir=gs://maxtext-experiments-multipod/${LOAD_FROM_OTHER}/checkpoints
bd=gs://maxtext-experiments-multipod
dataset=gs://max-datasets-rogue
python3 MaxText/train.py MaxText/configs/base.yml run_name=${RUN_NAME} steps=17 global_parameter_scale=${SIZE} base_output_directory=$bd dataset_path=$dataset learning_rate=1e-2 global_parameter_scale=1 per_device_batch_size=1 enable_checkpointing=True async_checkpointing=False load_from_other_directory=$ckpt_dir save_period=15 enable_profiler=False