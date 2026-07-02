#!/bin/bash
set -e

OUTPUT_PATH=${2}
DATASET_PATH=${3}
# Run name is optional 4th input - our daily XLML tests will use one.

if [ -z ${4} ]
then
    RUN_NAME=${USER}_$(date +%Y-%m-%d-%H-%M-%S)
else
    RUN_NAME=${4}_$(date +%Y-%m-%d-%H)
fi

# Decode with CPU - 8B
python3 end_to_end/cpu/16b.sh RUN_NAME=$RUN_NAME OUTPUT_PATH=$OUTPUT_PATH DATASET_PATH=$DATASET_PATH