#!/bin/bash

# Script to convert the distilled Llama 3.1 8B MaxText (Tunix/NNX) checkpoint to Hugging Face format.
# This uses the custom to_huggingface_tunix module to handle the NNX parameter structure.

export PYTHONPATH=src

/home/chfu_google_com/maxtext/maxtext_venv/bin/python3 -m MaxText.utils.ckpt_conversion.to_huggingface_tunix \
    src/MaxText/configs/base.yml \
    model_name=llama3.1-8b \
    load_parameters_path=gs://chfu-eu-maxtext/distillation/outputs/distillation-test-llama3-1-8b/checkpoints/20/model_params \
    base_output_directory=/home/chfu_google_com/maxtext/tmp_conversion/converted_hf \
    scan_layers=true \
    use_multimodal=false \
    weight_dtype=bfloat16 \
    base_num_query_heads=16 \
    head_dim=256 \
    base_num_kv_heads=4
