#!/bin/bash
set -ex

# Set output directory
export BASE_OUTPUT_DIRECTORY="/home/chfu_google_com/maxtext/my-maxtext-outputs"

/home/chfu_google_com/maxtext/maxtext_venv/bin/python3 \
  src/MaxText/distillation/train_distill.py \
  src/MaxText/configs/distillation.yml \
  run_name=distillation-test-llama3-1-8b  \
  base_output_directory=${BASE_OUTPUT_DIRECTORY} \
  checkpoint_period=2000 \
  log_period=10 \
  save_checkpoint_on_completion=True \
  use_vertex_tensorboard=True \
  vertex_tensorboard_region=us-central1 \
  steps=20 
