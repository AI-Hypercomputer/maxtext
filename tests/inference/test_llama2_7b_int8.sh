#!/bin/bash

# Define the arguments in an array
args=(
  "-m"
  "maxtext.decode"
  "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}/configs/base.yml"
  "tokenizer_path="${MAXTEXT_ASSETS_ROOT:-${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/assets/tokenizers}}"/tokenizer.llama2"
  "model_name=llama2-7b"
  "load_parameters_path=gs://msingh-bkt/checkpoints/quant_llama2-7b-chat/20241120034012/int8_"
  "checkpoint_is_quantized=true"
  "quantization=int8"
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
python3 "${args[@]}"
