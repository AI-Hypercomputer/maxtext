#!/bin/bash
set -ex

RUN_NAME=${1}_$(date +%Y-%m-%d-%H)
OUTPUT_PATH=${2}
DATASET_PATH=${3}
VOCAB_PATH=$OUTPUT_PATH/vocab_test_creation_$RUN_NAME


#Train
python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME steps=5 enable_checkpointing=False\
    base_output_directory=$OUTPUT_PATH dataset_path=$DATASET_PATH tokenizer_path=$VOCAB_PATH

python3 end_to_end/tpu/eval_assert.py vocab_creation $VOCAB_PATH
