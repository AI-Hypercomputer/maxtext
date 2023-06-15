#!/bin/bash
set -e

RUN_NAME=${1}_$(date +%Y-%m-%d-%H)
OUTPUT_PATH=${2}
DATASET_PATH=${3}


#Train
python3 MaxText/train.py MaxText/configs/base.yml run_name=${RUN_NAME}_1 steps=5 metrics_file='run_1_metrics.txt'\
    enable_checkpointing=False enable_data_shuffling=False enable_dropout=False base_output_directory=$OUTPUT_PATH dataset_path=$DATASET_PATH

python3 MaxText/train.py MaxText/configs/base.yml run_name=${RUN_NAME}_2 steps=5 metrics_file='run_2_metrics.txt'\
    enable_checkpointing=False enable_data_shuffling=False enable_dropout=False base_output_directory=$OUTPUT_PATH dataset_path=$DATASET_PATH

python3 end_to_end/eval_assert.py determinism metrics.txt learning/loss 
