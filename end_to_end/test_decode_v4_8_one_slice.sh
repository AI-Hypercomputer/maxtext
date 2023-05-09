#!/bin/bash
set -e

USER=${1}
NUM_TOKEN_THRESHOLD=${2}
OUTPUT_PATH=${3}
DATASET_PATH=${4}
# Run name is optional 5th input - our daily XLML tests will use one.


if [ -z ${5} ]
then 
    RUN_NAME=${USER}_$(date +%Y-%m-%d-%H-%M-%S)
else
    RUN_NAME=${5}_$(date +%Y-%m-%d-%H)
fi

#Setup and Train
bash setup.sh
python3 MaxText/decode.py MaxText/configs/base.yml run_name=$RUN_NAME\
    steps=100 enable_checkpointing=False metrics_file='metrics.txt'\
    base_output_directory=$OUTPUT_PATH

python3 end_to_end/eval_assert.py metrics.txt $NUM_TOKEN_THRESHOLD num_tokens metrics_average
