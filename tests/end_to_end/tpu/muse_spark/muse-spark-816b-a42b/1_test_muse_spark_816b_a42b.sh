#!/bin/bash

# This file is documentation for how to get started with Muse-Spark-Open-816B-A42B.

# This file runs Step 1 once on a high-memory CPU VM (it is not part of the daily test).
# 1. Convert the HuggingFace checkpoint (bf16) to MaxText-compatible checkpoints (bf16):
#    the scanned format is better for training; the unscanned format is better for decoding.
# Step 2 (2_test_muse_spark_816b_a42b.sh) then runs the logit check, pre-training, fine-tuning and decoding.

set -ex
idx=$(date +%Y-%m-%d-%H-%M)
export MODEL_NAME='muse-spark-816b-a42b'

# Installing torch for CPU checkpoint conversion.
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# `meta-models/Muse-Spark-Open-816B-A42B` is gated: either export HF_TOKEN, or point HF_MODEL_PATH to a local copy
# of the safetensors (the conversion reads them lazily, one tensor at a time).
HF_MODEL_ARGS=()
if [ -n "${HF_MODEL_PATH}" ]; then
  HF_MODEL_ARGS=(--hf_model_path="${HF_MODEL_PATH}")
fi

# Non-Googlers please remember to point `MODEL_BUCKET` to a GCS bucket that you own.
export MODEL_BUCKET=${MODEL_BUCKET:-gs://maxtext-model-checkpoints}

# Scanned checkpoint (scan_layers=true): 2 unrolled prefix layers + 15 scanned 4-layer blocks.
python3 -m maxtext.checkpoint_conversion.to_maxtext \
  "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}/base.yml" \
  model_name=${MODEL_NAME} \
  base_output_directory=${MODEL_BUCKET}/${MODEL_NAME}/scanned/${idx} \
  scan_layers=true \
  hardware=cpu \
  skip_jax_distributed_system=true \
  "${HF_MODEL_ARGS[@]}" \
  --save_dtype=bfloat16

# Unscanned checkpoint (scan_layers=false).
python3 -m maxtext.checkpoint_conversion.to_maxtext \
  "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}/base.yml" \
  model_name=${MODEL_NAME} \
  base_output_directory=${MODEL_BUCKET}/${MODEL_NAME}/unscanned/${idx} \
  scan_layers=false \
  hardware=cpu \
  skip_jax_distributed_system=true \
  "${HF_MODEL_ARGS[@]}" \
  --save_dtype=bfloat16

echo "Wrote MaxText compatible checkpoints to ${MODEL_BUCKET}/${MODEL_NAME}/[scanned|unscanned]/${idx}"
