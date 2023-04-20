#!/bin/bash
set -e

RUN_NAME=${1}
OUTPUT_PATH=${2}
DATASET_PATH=${3}


#Setup and Train
bash setup.sh

python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME steps=500 dcn_data_parallelism=2 ici_fsdp_parallelism=8\
    metrics_file='saved_metrics.txt' save_period=20 base_output_directory=$OUTPUT_PATH dataset_path=$DATASET_PATH

python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME steps=501 dcn_data_parallelism=2 ici_fsdp_parallelism=8\
    metrics_file='restored_metrics.txt' base_output_directory=$OUTPUT_PATH dataset_path=$DATASET_PATH

python3 end_to_end/eval_assert.py metrics.txt 0 learning/loss
