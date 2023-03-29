#!/bin/bash
set -e

USER=${1}
TFLOP_THRESHOLD=${2}


if [ -z ${3} ]
then 
    RUN_NAME=${USER}_$(date +%Y-%m-%d-%H-%M-%S)
else
    RUN_NAME=${3}_$(date +%Y-%m-%d-%H)
fi

#Setup and Train
bash setup.sh
python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME\
    steps=150 reuse_example_batch=1 remat_policy='full' metrics_file='metrics.txt'

python3 end_to_end/eval_assert.py metrics.txt $TFLOP_THRESHOLD perf/per_device_tflops_per_sec
