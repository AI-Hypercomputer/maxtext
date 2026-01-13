#!/bin/bash
set -ex

# Set output directory
export BASE_OUTPUT_DIRECTORY="gs://chfu-eu-maxtext/distillation/outputs"

/home/chfu_google_com/maxtext/maxtext_venv/bin/python3 \
  src/MaxText/distillation/train_distill.py \
  src/MaxText/configs/distillation_c4.yml \
  run_name=distillation-test-llama3-1-8b  \
  base_output_directory=${BASE_OUTPUT_DIRECTORY} \
  checkpoint_period=2000 \
  log_period=10 \
  save_checkpoint_on_completion=False \
  use_vertex_tensorboard=False \
  vertex_tensorboard_region=us-central1 \
  steps=20 
