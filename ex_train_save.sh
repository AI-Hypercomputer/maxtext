RUN_NAME=${1}
SIZE=${2}

bd=gs://maxtext-experiments-multipod
dataset=gs://max-datasets-rogue
python3 MaxText/train.py MaxText/configs/base.yml run_name=${RUN_NAME} steps=12 global_parameter_scale=${SIZE} base_output_directory=$bd dataset_path=$dataset learning_rate=1e-2 per_device_batch_size=1 enable_checkpointing=True async_checkpointing=False save_period=5 enable_profiler=False 