#!/bin/bash
# Script to run a simple MaxText training job on a single host.

# Set output directory
OUTPUT_DIR="/home/chfu_google_com/maxtext/my-maxtext-outputs"
mkdir -p "${OUTPUT_DIR}"

# Run training
/home/chfu_google_com/maxtext/maxtext_venv/bin/python3 -m MaxText.train src/MaxText/configs/base.yml \
  run_name=test_run \
  base_output_directory="${OUTPUT_DIR}" \
  dataset_type=synthetic \
  steps=10
