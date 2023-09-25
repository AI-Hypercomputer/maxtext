RUN_NAME=${1}
SIZE=${2}
LOAD_FROM_OTHER=${3}

bd=gs://maxtext-experiments-multipod
dataset=gs://max-datasets-rogue
python3 MaxText/train.py MaxText/configs/base.yml run_name=${RUN_NAME} load_from_other_directory=${LOAD_FROM_OTHER} load_from_other_directory_step=5 steps=22 global_parameter_scale=${SIZE} base_output_directory=$bd dataset_path=$dataset learning_rate=1e-2 per_device_batch_size=1 enable_checkpointing=True async_checkpointing=False save_period=5 enable_profiler=False