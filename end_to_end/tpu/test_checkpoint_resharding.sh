#!/bin/bash
set -ex

RUN_NAME=${1}_$(date +%Y-%m-%d-%H)
OUTPUT_PATH=${2}
DATASET_PATH=${3}

# Train and save checkpoint - sharded with DCN Data Parallelism + ICI FSDP Parallelism
python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME steps=101\
    metrics_file='saved_metrics.txt' checkpoint_period=20 base_output_directory=$OUTPUT_PATH dataset_path=$DATASET_PATH\
    dcn_data_parallelism=2 dcn_fsdp_parallelism=1 ici_fsdp_parallelism=4 ici_tensor_parallelism=1 collect_stack_trace=False

# Retrieve checkpoint - sharded with DCN Data Parallelism + ICI FSDP + Tensor Parallelism
python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME steps=102\
    metrics_file='restored_metrics.txt' base_output_directory=$OUTPUT_PATH dataset_path=$DATASET_PATH\
    dcn_data_parallelism=2 dcn_fsdp_parallelism=1 ici_fsdp_parallelism=2 ici_tensor_parallelism=2 collect_stack_trace=False

python3 end_to_end/tpu/eval_assert.py checkpoint_save_restore metrics.txt learning/loss
