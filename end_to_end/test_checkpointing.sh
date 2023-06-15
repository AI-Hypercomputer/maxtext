#!/bin/bash
set -e

RUN_NAME=${1}_$(date +%Y-%m-%d-%H)
OUTPUT_PATH=${2}
DATASET_PATH=${3}


#Train
python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME steps=501\
    metrics_file='saved_metrics.txt' save_period=20 base_output_directory=$OUTPUT_PATH dataset_path=$DATASET_PATH

python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME steps=502\
    metrics_file='restored_metrics.txt' base_output_directory=$OUTPUT_PATH dataset_path=$DATASET_PATH

python3 end_to_end/eval_assert.py checkpoint_save_restore metrics.txt learning/loss
