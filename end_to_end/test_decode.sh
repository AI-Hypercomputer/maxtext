#!/bin/bash
set -ex

NUM_TOKEN_THRESHOLD=${1}
OUTPUT_PATH=${2}
DATASET_PATH=${3}
# Run name is optional 4th input - our daily XLML tests will use one.


if [ -z ${4} ]
then
    RUN_NAME=${USER}_$(date +%Y-%m-%d-%H-%M-%S)-${RANDOM}
else
    RUN_NAME=${4}_$(date +%Y-%m-%d-%H)-${RANDOM}
fi

if [ -z ${5} ]
then
    ICI_TENSOR_PARALLELISM=4
else
    ICI_TENSOR_PARALLELISM=${5}
fi

# Decode without checkpoint
python3 MaxText/decode.py MaxText/configs/base.yml run_name=$RUN_NAME\
    steps=50 enable_checkpointing=False metrics_file=/tmp/${RUN_NAME}_metrics.txt \
    base_output_directory=$OUTPUT_PATH dataset_path=$DATASET_PATH \
    attention=dot_product ici_tensor_parallelism=${ICI_TENSOR_PARALLELISM}
