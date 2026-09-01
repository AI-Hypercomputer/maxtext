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
#!/bin/bash
#
# Qwen3.8-2.4T-A95B Step 2 E2E test
#
# Target hardware:
#   v5p-1024
#   512 physical chips
#   1024 JAX devices
#
# Test stages:
#   1. HF golden <-> MaxText forward-logit verification
#   2. 5-step synthetic pretraining
#   3. 5-step fine-tuning from scanned checkpoint
#   4. Greedy decoding from unscanned checkpoint
#
# MMLU / JetStream is intentionally kept as a separate test.
#

set -euxo pipefail

###############################################################################
# Environment
###############################################################################

export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=65536"

export MAXTEXT_REPO_ROOT=/app
export PYTHONPATH=/app/src

export MODEL_NAME="qwen3.8-2.4t-a95b"
export TOKENIZER_PATH="Qwen/Qwen3.8-2.4T-A95B"
export BASE_OUTPUT_PATH=gs://yujiedeng-maxtext-dev/model_bringup/test/qwen3.8-2.4t-a95b/e2e/$(date +%Y-%m-%d-%H-%M)



# ---------------------------------------------------------------------------
# v5p-1024 topology:
#
#   512 physical chips
#   1024 JAX devices
#
#   FSDP 128 x EP 8 = 1024
# ---------------------------------------------------------------------------
export ICI_FSDP=32
export ICI_EP=8
export DCN_FSDP=1

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
  "${MAXTEXT_REPO_ROOT}/src/maxtext/configs/base.yml" \
  base_output_directory="${BASE_OUTPUT_PATH}" \
  run_name=forward_logits_check_unscanned \
  load_parameters_path="${UNSCANNED_CKPT_PATH}" \
  scan_layers=false \
  use_multimodal=false \
  model_name="${MODEL_NAME}" \
  tokenizer_type=huggingface \
  tokenizer_path="${TOKENIZER_PATH}" \
  attention=dot_product \
  sparse_matmul=true \
  megablox=true \
  per_device_batch_size=1 \
  max_prefill_predict_length=4 \
  max_target_length=4 \
  async_checkpointing=false \
  ici_data_parallelism=1 \
  ici_tensor_parallelism=1 \
  ici_fsdp_parallelism="${ICI_FSDP}" \
  ici_expert_parallelism="${ICI_EP}" \
  dcn_fsdp_parallelism="${DCN_FSDP}" \
  dcn_data_parallelism=1 \
  weight_dtype=float16 \
  dtype=float16 \
  activations_in_float32=false \
  matmul_precision=highest \
  float32_logits=true \
  float32_qk_product=true \
  --golden_logits_path="${GOLDEN_LOGITS_DISK_LOCATION}" \
  --atol=1.5 \
  --rtol=1.5 \
  --max_kl_div=0.2

python3 -m tests.utils.forward_pass_logit_checker \
  "${MAXTEXT_REPO_ROOT}/src/maxtext/configs/base.yml" \
  base_output_directory="${BASE_OUTPUT_PATH}" \
  run_name=forward_logits_check \
  load_parameters_path="${SCANNED_CKPT_PATH}" \
  scan_layers=true \
  use_multimodal=false \
  model_name="${MODEL_NAME}" \
  tokenizer_type=huggingface \
  tokenizer_path="${TOKENIZER_PATH}" \
  attention=dot_product \
  sparse_matmul=true \
  megablox=true \
  per_device_batch_size=1 \
  max_prefill_predict_length=4 \
  max_target_length=4 \
  async_checkpointing=false \
  ici_data_parallelism=1 \
  ici_tensor_parallelism=1 \
  ici_fsdp_parallelism="${ICI_FSDP}" \
  ici_expert_parallelism="${ICI_EP}" \
  dcn_fsdp_parallelism="${DCN_FSDP}" \
  dcn_data_parallelism=1 \
  weight_dtype=float16 \
  dtype=float16 \
  activations_in_float32=false \
  matmul_precision=highest \
  float32_logits=true \
  float32_qk_product=true \
  --golden_logits_path="${GOLDEN_LOGITS_DISK_LOCATION}" \
  --atol=1.5 \
  --rtol=1.5 \
  --max_kl_div=0.2

echo "PASS: forward logit verification"

###############################################################################
# 2. Pretraining smoke test
#
# We deliberately use BF16 for:
#   - parameters
#   - activations
#   - gradients
#   - Adam first moment
#
# Adam second moment inherits weight_dtype in current MaxText.
#
# Full remat minimizes activation HBM.
#
# num_vocab_tiling=4 reduces memory associated with the 248k vocabulary loss.
###############################################################################

echo "============================================================"
echo "2/4: Pretraining smoke test"
echo "============================================================"

python3 -m maxtext.trainers.pre_train.train \
  "${MAXTEXT_REPO_ROOT}/src/maxtext/configs/base.yml" \
  base_output_directory="${BASE_OUTPUT_PATH}" \
  run_name=q38_pre_training \
  model_name="${MODEL_NAME}" \
  tokenizer_type=huggingface \
  tokenizer_path="${TOKENIZER_PATH}" \
  dataset_type=synthetic \
  enable_checkpointing=false \
  scan_layers=true \
  use_multimodal=false \
  attention=flash \
  sparse_matmul=true \
  megablox=true \
  dtype=bfloat16 \
  weight_dtype=bfloat16 \
  grad_dtype=bfloat16 \
  mu_dtype=bfloat16 \
  opt_type=adamw \
  remat_policy=full \
  num_vocab_tiling=4 \
  per_device_batch_size=1 \
  steps=5 \
  max_target_length=1024 \
  max_inflight_computations=1 \
  ici_data_parallelism=1 \
  ici_tensor_parallelism=1 \
  ici_fsdp_parallelism="${ICI_FSDP}" \
  ici_expert_parallelism="${ICI_EP}" \
  dcn_fsdp_parallelism="${DCN_FSDP}" \
dcn_data_parallelism=1 \

echo "PASS: pretraining smoke test"

###############################################################################
# 3. Fine-tuning / checkpoint-load training smoke test
#
# Important:
#   We want to test checkpoint restore + optimizer + backward/update.
#   We DO NOT want this 2.4T smoke test writing enormous training checkpoints.
###############################################################################

echo "============================================================"
echo "3/4: Fine-tuning smoke test"
echo "============================================================"

python3 -m maxtext.trainers.pre_train.train \
  "${MAXTEXT_REPO_ROOT}/src/maxtext/configs/base.yml" \
  base_output_directory="${BASE_OUTPUT_PATH}" \
  run_name=q38_fine_tuning \
  model_name="${MODEL_NAME}" \
  tokenizer_type=huggingface \
  tokenizer_path="${TOKENIZER_PATH}" \
  dataset_path="${DATASET_PATH}" \
  load_parameters_path="${SCANNED_CKPT_PATH}" \
  scan_layers=true \
  use_multimodal=false \
  enable_checkpointing=true \
  async_checkpointing=false \
  save_checkpoint_on_start=false \
  save_checkpoint_on_completion=false \
  checkpoint_period=1000000 \
  attention=flash \
  sparse_matmul=true \
  megablox=true \
  dtype=bfloat16 \
  weight_dtype=bfloat16 \
  grad_dtype=bfloat16 \
  mu_dtype=bfloat16 \
  opt_type=adamw \
  remat_policy=full \
  num_vocab_tiling=4 \
  per_device_batch_size=1 \
  steps=5 \
  max_target_length=1024 \
  max_inflight_computations=1 \
  ici_data_parallelism=1 \
  ici_tensor_parallelism=1 \
  ici_fsdp_parallelism="${ICI_FSDP}" \
  ici_expert_parallelism="${ICI_EP}" \
  dcn_data_parallelism=1

echo "PASS: fine-tuning smoke test"

###############################################################################
# 4. Native MaxText decoding
#
# Decode uses the UNSCANNED checkpoint.
###############################################################################

echo "============================================================"
echo "4/4: Decode smoke test"
echo "============================================================"

python3 -m maxtext.inference.decode \
  "${MAXTEXT_REPO_ROOT}/src/maxtext/configs/base.yml" \
  base_output_directory="${BASE_OUTPUT_PATH}" \
  run_name=decode \
  model_name="${MODEL_NAME}" \
  tokenizer_type=huggingface \
  tokenizer_path="${TOKENIZER_PATH}" \
  hf_access_token="${HF_TOKEN:-}" \
  load_parameters_path="${UNSCANNED_CKPT_PATH}" \
  scan_layers=false \
  use_multimodal=false \
  attention=dot_product \
  sparse_matmul=true \
  megablox=true \
  dtype=bfloat16 \
  weight_dtype=bfloat16 \
  per_device_batch_size=1 \
  max_prefill_predict_length=64 \
  max_target_length=128 \
  ici_data_parallelism=1 \
  ici_tensor_parallelism=1 \
  ici_fsdp_parallelism="${ICI_FSDP}" \
  ici_expert_parallelism="${ICI_EP}" \
  dcn_fsdp_parallelism="${DCN_FSDP}" \
  dcn_data_parallelism=1 \
  decode_sampling_strategy=greedy \
  prompt="An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and outputs are all vectors. The output is "

echo "PASS: decode smoke test"

###############################################################################
# Done
###############################################################################

echo
echo "============================================================"
echo "PASS: Qwen3.8-2.4T-A95B E2E"
echo "============================================================"
echo "  [PASS] BF16 forward-logit verification"
echo "  [PASS] BF16 AdamW pretraining"
echo "  [PASS] BF16 checkpoint-loaded fine-tuning"
echo "  [PASS] BF16 greedy decoding"
echo "============================================================"