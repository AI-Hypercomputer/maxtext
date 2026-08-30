#!/bin/bash

# This file is documentation for how to get started with Qwen3.8-2.4T-A95B.
#
# This file runs Step 2 on TPU7x:
#
# 1. Convert the HuggingFace checkpoint (bf16) to MaxText-compatible
#    checkpoint (bf16):
#      - scanned format is better for training / fine-tuning
#      - unscanned format is better for decoding
#
# 2. Run:
#      - forward-logit correctness check
#      - pre-training smoke test
#      - fine-tuning smoke test
#      - decoding smoke test
#
#
# ---------------------------------------------------------------------------
# HF golden logits
# ---------------------------------------------------------------------------
#
# Generate the golden logits independently on a large-memory CPU machine:
#
# python3 -m tests.assets.logits_generation.generate_hf_golden_logits \
#   --model-id=Qwen/Qwen3.8-2.4T-A95B \
#   --output-path=golden_data_qwen3.8-2.4t-a95b.jsonl \
#   --prompts='I love to' \
#   --hf-model-path=$LOCAL_BF16_PATH \
#   --trust-remote-code=False \
#   --hf-load-dtype=bfloat16
#
# Then upload:
#
# gcloud storage cp \
#   golden_data_qwen3.8-2.4t-a95b.jsonl \
#   gs://YOUR_BUCKET/qwen3.8-2.4t-a95b/golden_data_qwen3.8-2.4t-a95b.jsonl
#
#
# Expected initial TPU topology:
#
#   256 physical TPU7x chips
#   512 JAX TPU devices
#
#   FSDP = 64
#   EP   = 8
#
#   64 * 8 = 512
#

set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"

export MAXTEXT_REPO_ROOT="${REPO_ROOT}"
export MAXTEXT_CONFIGS_DIR="${REPO_ROOT}/src/maxtext/configs"

# Local XPK checkout must win over the /deps copy from the base image.
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"

cd "${REPO_ROOT}"

echo "===== SOURCE CHECK ====="
echo "REPO_ROOT=${REPO_ROOT}"
echo "MAXTEXT_REPO_ROOT=${MAXTEXT_REPO_ROOT}"
echo "MAXTEXT_CONFIGS_DIR=${MAXTEXT_CONFIGS_DIR}"
echo "PYTHONPATH=${PYTHONPATH}"

python3 - <<'PY'
import maxtext.configs.pyconfig as pyconfig
import maxtext.configs.types as types

print("pyconfig =", pyconfig.__file__)
print("types    =", types.__file__)

assert pyconfig.__file__.startswith("/app/"), pyconfig.__file__
assert types.__file__.startswith("/app/"), types.__file__

print("PASS: using layered MaxText source")
PY

export MODEL_NAME='qwen3.8-2.4t-a95b'
export TOKENIZER_PATH='Qwen/Qwen3.8-2.4T-A95B'


# ---------------------------------------------------------------------------
# Torch is needed by forward_pass_logit_checker.py
# ---------------------------------------------------------------------------

python3 -m pip install torch \
  --index-url https://download.pytorch.org/whl/cpu


# ---------------------------------------------------------------------------
# Base output location
# ---------------------------------------------------------------------------

if [ -z "${BASE_OUTPUT_PATH}" ]; then
  # FILL THIS IN.
  export BASE_OUTPUT_PATH=gs://yujiedeng-maxtext-dev/model_bringup/test/qwen3.8-2.4t-a95b/e2e/$(date +%Y-%m-%d-%H-%M)
  echo "BASE_OUTPUT_PATH is not set"
fi

BASE_OUTPUT_PATH=${BASE_OUTPUT_PATH%/}

echo "using BASE_OUTPUT_PATH = ${BASE_OUTPUT_PATH}"


# ---------------------------------------------------------------------------
# Converted MaxText checkpoints
#
# FILL THESE IN with your actual stable GCS checkpoint paths.
# ---------------------------------------------------------------------------

SCANNED_CKPT_PATH=gs://yujiedeng-maxtext-dev/model_bringup/model_zoo/qwem3_8_mt_scanned/scanned/0/items

UNSCANNED_CKPT_PATH=gs://yujiedeng-maxtext-dev/model_bringup/model_zoo/qwem3_8_mt_unscanned/unscanned-v2/0/items


# ---------------------------------------------------------------------------
# Fine-tuning dataset
#
# Same convention as the Qwen3.5 E2E test.
# ---------------------------------------------------------------------------

export DATASET_PATH=gs://maxtext-dataset


# ---------------------------------------------------------------------------
# Golden logits
#
# Same protocol as qwen3.5:
#
#   1. use packaged golden if it exists
#   2. otherwise fetch known golden from GCS
# ---------------------------------------------------------------------------

GOLDEN_LOGITS_DISK_LOCATION="/deps/tests/assets/golden_logits/golden_data_${MODEL_NAME}.jsonl"

if [ ! -f "${GOLDEN_LOGITS_DISK_LOCATION}" ]; then

  # FILL THIS IN.
  GOLDEN_LOGITS_PATH=gs://yujiedeng-maxtext-dev/model_bringup/test/qwen3.8-2.4t-a95b/golden_data_qwen3.8-2.4t-a95b.jsonl

  GOLDEN_LOGITS_DISK_LOCATION=/tmp/golden_data.jsonl

  gcloud storage cp \
    ${GOLDEN_LOGITS_PATH} \
    ${GOLDEN_LOGITS_DISK_LOCATION}
fi


# ---------------------------------------------------------------------------
# TPU topology sanity check
#
# TPU7x exposes two JAX devices per physical chip.
#
# 256 physical chips should therefore expose 512 JAX devices.
# ---------------------------------------------------------------------------
 
python3 - <<'PY'
import os

from maxtext.configs import pyconfig
import maxtext.configs.pyconfig as pyconfig_module

print("pyconfig module:", pyconfig_module.__file__)
print("MAXTEXT_CONFIGS_DIR:", os.environ["MAXTEXT_CONFIGS_DIR"])

assert pyconfig_module.__file__.startswith("/app/")
assert os.environ["MAXTEXT_CONFIGS_DIR"].startswith("/app/")

cfg = pyconfig.initialize(
    [
        os.path.join("/app/src/maxtext", "train.py"),
        os.path.join(
            os.environ["MAXTEXT_CONFIGS_DIR"],
            "base.yml",
        ),
    ],
    skip_jax_distributed_system=True,
    model_name="qwen3.8-2.4t-a95b",
    run_name="q38-pre-logit-check",
    base_output_directory="/tmp/q38-pre-logit-check",
)

print("model_name    =", cfg.model_name)
print("decoder_block =", cfg.decoder_block)
print("layers        =", cfg.base_num_decoder_layers)
print("experts       =", cfg.num_experts)

assert cfg.model_name == "qwen3.8-2.4t-a95b"
assert cfg.base_num_decoder_layers == 92
assert cfg.num_experts == 512

print("PASS: Qwen3.8 config before logit verification")
PY

# ===========================================================================
# 2.1 Forward-logit correctness
# ===========================================================================
#
# Same philosophy as qwen3.5:
#
# HF BF16 golden
#       vs
# MaxText scanned checkpoint
#
# We deliberately use the high-precision MaxText path here.
#

python3 -m tests.utils.forward_pass_logit_checker \
  ${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}/base.yml \
  base_output_directory=${BASE_OUTPUT_PATH} \
  run_name=forward_logits_check \
  load_parameters_path=${SCANNED_CKPT_PATH} \
  scan_layers=true \
  use_multimodal=false \
  attention=dot_product \
  per_device_batch_size=1 \
  model_name=${MODEL_NAME} \
  tokenizer_type=huggingface \
  tokenizer_path=${TOKENIZER_PATH} \
  max_prefill_predict_length=4 \
  max_target_length=4 \
  async_checkpointing=false \
  sparse_matmul=True \
  megablox=True \
  ici_data_parallelism=1 \
  ici_tensor_parallelism=1 \
  ici_fsdp_parallelism=32 \
  ici_expert_parallelism=8 \
  weight_dtype=float32 \
  dtype=float32 \
  activations_in_float32=true \
  matmul_precision=highest \
  float32_logits=true \
  float32_qk_product=true \
  --golden_logits_path=${GOLDEN_LOGITS_DISK_LOCATION} \
  --atol=1.5 \
  --rtol=1.5 \
  --max_kl_div=0.2


# ===========================================================================
# 2.2 Pre-training smoke test
# ===========================================================================
#
# Purpose:
#
#   - initialize the entire Qwen3.8 MaxText model
#   - run forward
#   - run backward
#   - exercise MoE routing
#   - exercise optimizer update
#
# This does NOT load the converted checkpoint.
# It verifies that the architecture is trainable from scratch.
#
# Same 5-step protocol as qwen3.5.
#

python3 -m maxtext.trainers.pre_train.train \
  ${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}/base.yml \
  base_output_directory=${BASE_OUTPUT_PATH} \
  run_name=q38_pre_training \
  model_name=${MODEL_NAME} \
  tokenizer_type=huggingface \
  tokenizer_path=${TOKENIZER_PATH} \
  dataset_type=synthetic \
  enable_checkpointing=false \
  attention=flash \
  sparse_matmul=True \
  megablox=True \
  dtype=bfloat16 \
  weight_dtype=bfloat16 \
  per_device_batch_size=1 \
  steps=5 \
  max_target_length=1024 \
  ici_data_parallelism=1 \
  ici_tensor_parallelism=1 \
  ici_fsdp_parallelism=32 \
  ici_expert_parallelism=8


# ===========================================================================
# 2.3 Fine-tuning smoke test
# ===========================================================================
#
# This is more important for checkpoint bring-up than pre-training:
#
#   scanned converted checkpoint
#             ↓
#          restore
#             ↓
#          forward
#             ↓
#          backward
#             ↓
#       optimizer update
#
# So this verifies that the checkpoint is usable for actual training.
#

python3 -m maxtext.trainers.pre_train.train \
  ${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}/base.yml \
  base_output_directory=${BASE_OUTPUT_PATH} \
  run_name=q38_fine_tuning \
  model_name=${MODEL_NAME} \
  tokenizer_type=huggingface \
  tokenizer_path=${TOKENIZER_PATH} \
  dataset_path=${DATASET_PATH} \
  enable_checkpointing=true \
  async_checkpointing=false \
  load_parameters_path=${SCANNED_CKPT_PATH} \
  scan_layers=true \
  attention=flash \
  sparse_matmul=True \
  megablox=True \
  dtype=bfloat16 \
  weight_dtype=bfloat16 \
  per_device_batch_size=1 \
  steps=5 \
  max_target_length=1024 \
  ici_data_parallelism=1 \
  ici_tensor_parallelism=1 \
  ici_fsdp_parallelism=32 \
  ici_expert_parallelism=8


# ===========================================================================
# 2.4 Decoding smoke test
# ===========================================================================
#
# Decode uses the unscanned checkpoint.
#
# This tests:
#
#   - unscanned checkpoint restore
#   - prefill
#   - full-attention KV cache
#   - GatedDeltaNet recurrent state
#   - autoregressive cache updates
#
# Note:
# decode currently uses the HuggingFace tokenizer, so HF_TOKEN is supplied
# following the Qwen3.5 E2E convention.
#

python3 -m maxtext.inference.decode \
  ${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}/base.yml \
  base_output_directory=${BASE_OUTPUT_PATH} \
  run_name=decode \
  model_name=${MODEL_NAME} \
  tokenizer_type=huggingface \
  tokenizer_path=${TOKENIZER_PATH} \
  hf_access_token=${HF_TOKEN} \
  load_parameters_path=${UNSCANNED_CKPT_PATH} \
  scan_layers=false \
  attention=dot_product \
  sparse_matmul=True \
  megablox=True \
  dtype=bfloat16 \
  weight_dtype=bfloat16 \
  per_device_batch_size=1 \
  max_prefill_predict_length=64 \
  max_target_length=128 \
  ici_data_parallelism=1 \
  ici_tensor_parallelism=1 \
  ici_fsdp_parallelism=32 \
  ici_expert_parallelism=8 \
  decode_sampling_strategy=greedy \
  prompt="An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and outputs are all vectors. The output is "


echo "============================================================"
echo "PASS: Qwen3.8-2.4T-A95B Step-2 E2E"
echo "============================================================"