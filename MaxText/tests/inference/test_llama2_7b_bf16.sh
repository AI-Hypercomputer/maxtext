#!/bin/bash

# Define the arguments in an array
args=(
  "MaxText/decode.py"
  "MaxText/configs/base.yml"
  "tokenizer_path=assets/tokenizer.llama2"
  "model_name=llama2-7b"
  "load_parameters_path=gs://runner-maxtext-logs/direct_generate_param_only_checkpoint_2024-06-11-04-13/checkpoints/0/items/"
  "checkpoint_is_quantized=false"
  "weight_dtype=bfloat16"
  "max_prefill_predict_length=16"
  "max_target_length=32"
  "ici_fsdp_parallelism=1"
  "ici_autoregressive_parallelism=1"
  "ici_tensor_parallelism=-1"
  "scan_layers=false"
  "per_device_batch_size=1"
  "attention=paged"
  "pagedattn_num_pages=64"
  "pagedattn_tokens_per_page=8"
  "pagedattn_pages_per_compute_block=4"
)

# Execute the Python script with the arguments
python "${args[@]}"

