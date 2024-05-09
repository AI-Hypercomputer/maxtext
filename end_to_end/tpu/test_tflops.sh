#!/bin/bash
set -ex

USER=${1}
TFLOP_THRESHOLD=${2}
OUTPUT_PATH=${3}
DATASET_PATH=${4}


if [ -z ${5} ]
then 
    RUN_NAME=${USER}_$(date +%Y-%m-%d-%H-%M-%S)
else
    RUN_NAME=${5}_$(date +%Y-%m-%d-%H)
fi

#Train
python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME\
    steps=150 reuse_example_batch=1 remat_policy='full' enable_profiler=True enable_checkpointing=False metrics_file='metrics.txt'\
    base_output_directory=$OUTPUT_PATH dataset_path=$DATASET_PATH log_period=150

python3 end_to_end/tpu/eval_assert.py metrics_average metrics.txt $TFLOP_THRESHOLD perf/per_device_tflops_per_sec
