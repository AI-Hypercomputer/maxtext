#!/bin/bash

# This file is documentation for how to get started with DeepSeek v4.

# This file runs Step 1 on CPU.
# 1. Dequantize the HuggingFace checkpoint (FP8) to bf16, then convert it to MaxText-compatible checkpoint (bf16):
#    Scanned format is better for training; unscanned format is better for decoding.

set -ex

export MODEL_NAME='deepseek4-1600b'
export TOKENIZER_PATH='deepseek-ai/DeepSeek-V4-Pro'

# Installing torch for checkpoint conversion and forward_pass_logit_checker.py
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

: "${BASE_OUTPUT_PATH:?BASE_OUTPUT_PATH is required and must be set}"
BASE_OUTPUT_PATH="${BASE_OUTPUT_PATH%/}"
echo "using BASE_OUTPUT_PATH = ${BASE_OUTPUT_PATH}"

# Step 1: Checkpoint conversion
# DeepSeek-V4-Pro-0813 is released only in FP8 (routed experts FP4) at
# https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813; dequantize it to bf16 (about 3.1 TiB).
# Set CKPT_DISK_LOCATION to reuse a bf16 copy, or FP8_DISK_LOCATION to reuse a local FP8 copy.
LOCAL_DIR="${LOCAL_DIR:-/tmp}"
if [ -z "${CKPT_DISK_LOCATION}" ]; then
  FP8_LOCAL_PATH="${FP8_DISK_LOCATION:-${LOCAL_DIR}/hf-fp8}"
  if [ -z "${FP8_DISK_LOCATION}" ]; then
    hf download deepseek-ai/DeepSeek-V4-Pro-0813 --revision 72e1d3230f6c080a530b0a1d46f8eb4602340597 \
      --local-dir "${FP8_LOCAL_PATH}"
  fi
  export BF16_LOCAL_PATH="${LOCAL_DIR}/hf-bf16"
  python3 -m maxtext.checkpoint_conversion.standalone_scripts.deepseek_dequantize \
    --input-path="${FP8_LOCAL_PATH}" \
    --output-path="${BF16_LOCAL_PATH}"
else
  export BF16_LOCAL_PATH="${CKPT_DISK_LOCATION}"
fi

# Step 2: Extract tid2eid routing tables to tid2eid.safetensors
echo "Extracting tid2eid routing tables..."
python3 - << 'EOPY'
import glob
import os
import sys
from safetensors import safe_open
from safetensors.numpy import save_file

hf_dir = os.environ.get("BF16_LOCAL_PATH", "/tmp/hf-bf16")
out_dir = "/tmp"
st_files = sorted(glob.glob(os.path.join(hf_dir, "*.safetensors")))
tables = {}
for sf_path in st_files:
  with safe_open(sf_path, framework="numpy") as f:
    for k in f.keys():
      for layer_idx in range(3):
        if k in (f"layers.{layer_idx}.ffn.gate.tid2eid", f"model.layers.{layer_idx}.ffn.gate.tid2eid"):
          tables[f"layers_{layer_idx}"] = f.get_tensor(k)
if len(tables) == 3:
  out_path = os.path.join(out_dir, "tid2eid.safetensors")
  save_file(tables, out_path)
  print(f"Successfully extracted {len(tables)} tid2eid tables to {out_path}")
else:
  sys.exit(f"Error: expected 3 tid2eid tables, found {len(tables)}")
EOPY

if [ -f /tmp/tid2eid.safetensors ]; then
  gcloud storage cp /tmp/tid2eid.safetensors "${BASE_OUTPUT_PATH}/tid2eid.safetensors"
fi

# scanned
python3 -m maxtext.checkpoint_conversion.to_maxtext src/maxtext/configs/base.yml \
  model_name="${MODEL_NAME}" \
  scan_layers=true \
  attention=dot_product \
  base_output_directory="${BASE_OUTPUT_PATH}/scanned" \
  hf_access_token="${HF_TOKEN:-}" \
  hardware=cpu \
  skip_jax_distributed_system=True \
  async_checkpointing=False \
  --hf_model_path="${BF16_LOCAL_PATH}" \
  --lazy_load_tensors=True \
  --simulated_cpu_devices_count=1 \
  --save_dtype=bfloat16 \
  checkpoint_storage_concurrent_gb=256

# unscanned
python3 -m maxtext.checkpoint_conversion.to_maxtext src/maxtext/configs/base.yml \
  model_name="${MODEL_NAME}" \
  scan_layers=false \
  attention=dot_product \
  base_output_directory="${BASE_OUTPUT_PATH}/unscanned" \
  hf_access_token="${HF_TOKEN:-}" \
  hardware=cpu \
  skip_jax_distributed_system=True \
  async_checkpointing=False \
  --hf_model_path="${BF16_LOCAL_PATH}" \
  --lazy_load_tensors=True \
  --simulated_cpu_devices_count=1 \
  --save_dtype=bfloat16 \
  checkpoint_storage_concurrent_gb=256
