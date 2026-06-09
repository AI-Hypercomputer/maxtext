#!/bin/bash
# Script to convert a scanned MaxText checkpoint to an unscanned MaxText checkpoint.
#
# It does this in two steps:
# 1. Converts scanned MaxText checkpoint -> Hugging Face format
# 2. Converts Hugging Face format -> unscanned MaxText checkpoint (scan_layers=False)
#
# NOTE: This conversion process requires a significant amount of RAM (~70 GB for 35B model).
# If running this locally on a machine with less than 96 GB RAM, it may fail with OOM.
# You can launch this as a single-host TPU VM workload via xpk:
#
#   xpk workload create \
#     --cluster mlperf-v5p \
#     --zone europe-west4-b \
#     --project cloud-tpu-multipod-dev \
#     --workload qwen3-5-unscan-job \
#     --tpu-type v5p-8 \
#     --num-slices 1 \
#     --docker-image gcr.io/tpu-prod-env-multipod/maxtext_post_training_nightly:latest \
#     --command "export HF_TOKEN=\$HF_TOKEN && cd /maxtext && bash unscan_checkpoint.sh"
#

set -ex

# Ensure only worker 0 runs the conversion when running in a multi-host GKE JobSet
if [ -n "${JOB_COMPLETION_INDEX}" ] && [ "${JOB_COMPLETION_INDEX}" -ne 0 ]; then
  echo "This is worker ${JOB_COMPLETION_INDEX}. Exiting to let worker 0 perform conversion."
  exit 0
fi

MODEL_NAME="qwen3.5-35b-a3b"
CHECKPOINT_DIR="gs://snehalv-data/qwen3-5/unscanned/qwen3.5-35b-a3b_2026-06-09-01-41/checkpoints/10"
SCANNED_CKPT_PATH="${CHECKPOINT_DIR}/model_params"
HF_CKPT_PATH="temp_hf_checkpoint"
UNSCANNED_CKPT_PATH="${CHECKPOINT_DIR}/unscanned_checkpoint"

# Disable fsspec directory caching to avoid multi-host / eventual consistency sync delays
export FSSPEC_USE_LISTINGS_CACHE=no
export GCSFS_LISTING_CACHE=False

# If HF_TOKEN environment variable is not set, print warning
if [ -z "${HF_TOKEN}" ]; then
  echo "WARNING: HF_TOKEN is not set. Hugging Face downloads/uploads might fail."
fi

echo "=========================================================="
echo "Step 1: Converting Scanned MaxText -> Hugging Face format"
echo "=========================================================="
python3 src/maxtext/checkpoint_conversion/to_huggingface.py \
  src/maxtext/configs/base.yml \
  model_name="${MODEL_NAME}" \
  load_parameters_path="${SCANNED_CKPT_PATH}" \
  base_output_directory="${HF_CKPT_PATH}" \
  scan_layers=True \
  weight_dtype=bfloat16 \
  skip_jax_distributed_system=True

echo "=========================================================="
echo "Step 2: Converting Hugging Face -> Unscanned MaxText format"
echo "=========================================================="
python3 -m maxtext.checkpoint_conversion.to_maxtext \
  --lazy_load_tensors=True \
  --hf_model_path="${HF_CKPT_PATH}" \
  src/maxtext/configs/base.yml \
  model_name="${MODEL_NAME}" \
  base_output_directory="${UNSCANNED_CKPT_PATH}" \
  scan_layers=False \
  skip_jax_distributed_system=True \
  hardware=cpu

# Clean up intermediate local HF checkpoint folder to free space
rm -rf "${HF_CKPT_PATH}"

echo "=========================================================="
echo "Unscanning checkpoint conversion completed successfully!"
echo "Unscanned checkpoint is stored at: ${UNSCANNED_CKPT_PATH}"
echo "=========================================================="
