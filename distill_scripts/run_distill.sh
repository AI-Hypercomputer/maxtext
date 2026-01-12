#!/bin/bash
set -ex

# Set output directory
export BASE_OUTPUT_DIRECTORY="/home/chfu_google_com/maxtext/my-maxtext-outputs"

echo "Starting Distillation Run..."
echo "Output Directory: ${BASE_OUTPUT_DIRECTORY}"

# Run distillation with random weights for both Student and Teacher
# This is just to verify the distillation pipeline works.
/home/chfu_google_com/maxtext/maxtext_venv/bin/python3 src/MaxText/distillation/train_distill.py src/MaxText/configs/base.yml \
  run_name=distill_test_$(date +%Y%m%d_%H%M%S) \
  base_output_directory=${BASE_OUTPUT_DIRECTORY} \
  model_name=llama2-7b \
  dataset_type=tfds \
  dataset_path=gs://maxtext-dataset \
  dataset_name='c4/en:3.0.1' \
  steps=10 \
  per_device_batch_size=1 \
  teacher_overrides='{"model_name": "llama2-7b", "load_parameters_path": "gs://maxtext-llama/llama-2-7b/maxtext-ckpt/2024-03-27-04-24/0/items"}'

echo "Distillation Run Complete."
