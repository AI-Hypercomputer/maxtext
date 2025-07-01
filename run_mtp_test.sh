#!/bin/bash

# A simple script to launch a MaxText training run.
# It takes the RUN_NAME and OUTPUT_PATH from the command line.

RUN_NAME=$1
OUTPUT_PATH=$2

# This points to the public C4 dataset, which is standard for this type of validation.
DATASET_PATH="gs://maxtext-dataset/"

# The core command that starts training.
# It passes all configuration details to train.py.
# The `${@:3}` part cleverly passes any additional arguments from our command
# line directly to train.py, which is how we'll control MTP.
python3 -m MaxText.train \
  MaxText/configs/base.yml \
  run_name=$RUN_NAME \
  base_output_directory=$OUTPUT_PATH \
  dataset_path=$DATASET_PATH \
  steps=25000 \
  enable_checkpointing=False \
  eval_interval=1000 \
  ${@:3}
