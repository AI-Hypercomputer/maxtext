#!/bin/bash

# Nemotron-3-Nano-Omni-30B-A3B text backbone (nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16): Step 1, on CPU.
# Converts the HuggingFace checkpoint to an unscanned MaxText checkpoint. The 52-layer hybrid pattern does not repeat,
# so this model supports only scan_layers=false: there is no scanned checkpoint.
# ---
# Example usage:
#   export HF_TOKEN=hf_...   # or HF_MODEL_PATH=/path/to/local/hf/checkpoint
#   BASE_OUTPUT_PATH=gs://<bucket>/nemotron3 bash tests/end_to_end/tpu/nemotron3/30b-a3b/1_test_nemotron3.sh
# ---

set -ex

export MODEL_NAME="${MODEL_NAME:-nemotron3-nano-omni-30b-a3b}"
export HF_MODEL_PATH="${HF_MODEL_PATH:-}"
export BASE_OUTPUT_PATH="${BASE_OUTPUT_PATH:-gs://your-gcs-bucket/${MODEL_NAME}_maxtext_ckpt}"

if [ -z "${HF_TOKEN}" ] && [ -z "${HF_MODEL_PATH}" ]; then
  echo "Error: set HF_TOKEN or HF_MODEL_PATH."
  exit 1
fi
BASE_OUTPUT_PATH=${BASE_OUTPUT_PATH%/}
echo "Using BASE_OUTPUT_PATH = ${BASE_OUTPUT_PATH}"

HF_LOCAL_ARG=""
if [ -n "${HF_MODEL_PATH}" ]; then
  HF_LOCAL_ARG="hf_model_path=${HF_MODEL_PATH}"
fi

# Router biases (e_score_correction_bias) are float32 in the HF checkpoint and stay float32 in the output: the 128
# biases of a layer sit within about 0.1 of each other, so bfloat16 rounding would change expert selection.
# shellcheck disable=SC2086
JAX_PLATFORMS=cpu python3 -m maxtext.checkpoint_conversion.to_maxtext \
  "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}/base.yml" \
  model_name="${MODEL_NAME}" \
  base_output_directory="${BASE_OUTPUT_PATH}/unscanned" \
  hf_access_token="${HF_TOKEN:-}" \
  scan_layers=false \
  use_multimodal=false \
  lazy_load_tensors=true \
  ${HF_LOCAL_ARG}
