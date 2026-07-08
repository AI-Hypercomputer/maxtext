#!/bin/bash
cd /home/snehalv_google_com/maxtext
export PYTHONPATH=$PWD/src
echo "Starting conversion..."
/home/snehalv_google_com/venv-maxtext/bin/python src/maxtext/checkpoint_conversion/to_maxtext.py \
    src/maxtext/configs/base.yml \
    model_name=deepseek4-tiny \
    base_output_directory=gs://snehalv-data/deepseek4-conversion-pr/scanned/ \
    scan_layers=true \
    skip_jax_distributed_system_initialization=true \
    --hf_model_path=tests/end_to_end/tpu/deepseek/v4-284b/hf_tiny_model > conversion.log 2>&1
echo "Finished with exit code: $?"
