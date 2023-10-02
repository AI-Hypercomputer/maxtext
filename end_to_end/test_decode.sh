#!/bin/bash
set -e

NUM_TOKEN_THRESHOLD=${1}
OUTPUT_PATH=${2}
DATASET_PATH=${3}
# Run name is optional 4th input - our daily XLML tests will use one.


if [ -z ${4} ]
then 
    RUN_NAME=${USER}_$(date +%Y-%m-%d-%H-%M-%S)
else
    RUN_NAME=${4}_$(date +%Y-%m-%d-%H)
fi

#Train
python3 MaxText/decode.py MaxText/configs/base.yml run_name=$RUN_NAME\
    steps=50 enable_checkpointing=False metrics_file='metrics.txt'\
    base_output_directory=$OUTPUT_PATH dataset_path=$DATASET_PATH

python3 end_to_end/eval_assert.py metrics_average metrics.txt $NUM_TOKEN_THRESHOLD num_tokens
