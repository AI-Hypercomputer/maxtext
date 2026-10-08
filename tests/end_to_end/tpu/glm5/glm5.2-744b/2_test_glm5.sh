#!/bin/bash

# This file runs Step 2 on a TPU cluster for GLM-5.2 (Cross-Layer IndexShare).
# 1. Forward-pass logit verification against HuggingFace golden logits (or on-the-fly HF model).
# 2. Distributed pre-training with GLM-5.2's 78-layer Cross-Layer IndexShare topology.
# 3. Autoregressive decoding using unscanned weights and MLA KV cache (`mla_naive_kvcache=false`).
#
# Step 1 overrides `indexer_topk=2` (`LOGIT_INDEXER_TOPK`, matching `2_test_deepseek.sh`) so
# `kv_len > indexer_topk` exercises sparse top-k selection and Cross-Layer IndexShare across all
# 78 layers instead of taking the `kv_len <= indexer_topk` dense causal shortcut.
# Pre-generated golden logits must therefore be produced from a HuggingFace checkpoint config with
# matching `"index_topk": 2` (e.g. `GENERATE_GOLDEN_LOGITS=true bash 1_test_glm5.sh`, or by staging
# a symlink directory whose `config.json` sets `"index_topk": 2` and running):
# python3 -m tests.assets.logits_generation.generate_hf_golden_logits \
#   --model-id=zai-org/GLM-5.2 \
#   --output-path=golden_data_glm5.2-744b.jsonl \
#   --prompts='I love to;Today is a;What is the' \
#   --hf-model-path="${STAGED_HF_DIR_TOPK2}" \
#   --trust-remote-code=True \
#   --hf-load-dtype=bfloat16

set -ex
set -o pipefail

export MODEL_NAME="${MODEL_NAME:-glm5.2-744b}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-zai-org/GLM-5.2}"

if [ -z "${BASE_OUTPUT_PATH:-}" ]; then
  # Non-Googlers please remember to point `BASE_OUTPUT_PATH` to GCS buckets that you own;
  # this script uses internal buckets for testing.
  export BASE_OUTPUT_PATH="gs://runner-maxtext-logs/$(date +%Y-%m-%d-%H-%M)"
  echo "BASE_OUTPUT_PATH is not set; using ${BASE_OUTPUT_PATH}"
fi
BASE_OUTPUT_PATH="${BASE_OUTPUT_PATH%/}"
echo "Using BASE_OUTPUT_PATH = ${BASE_OUTPUT_PATH}"

# Standard MaxText GCS checkpoint conventions (produced by `1_test_glm5.sh`):
#   SCANNED_CKPT_PATH=gs://maxtext-model-checkpoints/glm5.2-744b/scanned/0/items
#   UNSCANNED_CKPT_PATH=gs://maxtext-model-checkpoints/glm5.2-744b/unscanned/0/items
#   GOLDEN_LOGITS_PATH=gs://maxtext-test-assets/golden_data_${MODEL_NAME}.jsonl
# Fail fast before installing dependencies when required checkpoint paths are unset.
if [ "${RUN_LOGIT:-true}" = "true" ]; then
  : "${SCANNED_CKPT_PATH:?Please set SCANNED_CKPT_PATH (e.g. gs://maxtext-model-checkpoints/glm5.2-744b/scanned/0/items)}"
  DEFAULT_GOLDEN_DISK="/deps/tests/assets/golden_logits/golden_data_${MODEL_NAME}.jsonl"
  if [ -z "${GOLDEN_LOGITS_PATH:-}" ] && [ -z "${HF_MODEL_PATH:-}" ] && [ ! -f "${DEFAULT_GOLDEN_DISK}" ]; then
    : "${GOLDEN_LOGITS_PATH:?Please set GOLDEN_LOGITS_PATH (e.g. gs://maxtext-test-assets/golden_data_glm5.2-744b.jsonl) or HF_MODEL_PATH}"
  fi
fi
if [ "${RUN_DECODE:-true}" = "true" ]; then
  : "${UNSCANNED_CKPT_PATH:?Please set UNSCANNED_CKPT_PATH (e.g. gs://maxtext-model-checkpoints/glm5.2-744b/unscanned/0/items)}"
fi

# Ensure JetStream and dependencies are installed when running inference.decode
if [ "${RUN_DECODE:-true}" = "true" ]; then
  if ! python3 -c "import jetstream" &>/dev/null; then
    python3 -m src.dependencies.scripts.install_pre_train_extra_deps --with-tf
  fi
fi

# Install CPU torch for forward_pass_logit_checker.py / tokenizer helpers
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

CONFIG_FILE="${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}/base.yml"

# 1. Forward-Pass Golden Logit Verification (sparse `indexer_topk` + Cross-Layer IndexShare)
if [ "${RUN_LOGIT:-true}" = "true" ]; then
  LOGIT_INDEXER_TOPK="${LOGIT_INDEXER_TOPK:-2}"
  LOGIT_COMPARE_ARGS=()
  if [ -n "${GOLDEN_LOGITS_PATH:-}" ]; then
    if [[ "${GOLDEN_LOGITS_PATH}" =~ ^gs:// ]]; then
      GOLDEN_LOGITS_DISK_LOCATION="/tmp/golden_data_${MODEL_NAME}.jsonl"
      gcloud storage cp "${GOLDEN_LOGITS_PATH}" "${GOLDEN_LOGITS_DISK_LOCATION}"
    else
      GOLDEN_LOGITS_DISK_LOCATION="${GOLDEN_LOGITS_PATH}"
    fi
    LOGIT_COMPARE_ARGS=(
      "--golden_logits_path=${GOLDEN_LOGITS_DISK_LOCATION}"
      "--atol=${ATOL:-1.5}"
      "--rtol=${RTOL:-1.5}"
      "--max_kl_div=${MAX_KL_DIV:-0.1}"
    )
  elif [ -n "${HF_MODEL_PATH:-}" ]; then
    EFFECTIVE_HF_MODEL_PATH="${HF_MODEL_PATH}"
    if [ -d "${HF_MODEL_PATH}" ] && [ -f "${HF_MODEL_PATH}/config.json" ]; then
      EFFECTIVE_HF_MODEL_PATH="/tmp/hf_model_${MODEL_NAME}_topk_${LOGIT_INDEXER_TOPK}"
      rm -rf "${EFFECTIVE_HF_MODEL_PATH}"
      mkdir -p "${EFFECTIVE_HF_MODEL_PATH}"
      python3 -c '
import json, os, sys
src_dir, dst_dir, topk = os.path.abspath(sys.argv[1]), sys.argv[2], int(sys.argv[3])
for entry in os.listdir(src_dir):
  if entry == "config.json":
    continue
  os.symlink(os.path.join(src_dir, entry), os.path.join(dst_dir, entry))
with open(os.path.join(src_dir, "config.json"), "r", encoding="utf-8") as f:
  cfg = json.load(f)
cfg["index_topk"] = topk
with open(os.path.join(dst_dir, "config.json"), "w", encoding="utf-8") as f:
  json.dump(cfg, f, indent=2)
' "${HF_MODEL_PATH}" "${EFFECTIVE_HF_MODEL_PATH}" "${LOGIT_INDEXER_TOPK}"
    fi
    LOGIT_COMPARE_ARGS=(
      "--run_hf_model=True"
      "--hf_model_path=${EFFECTIVE_HF_MODEL_PATH}"
      "--max_kl_div=${MAX_KL_DIV:-0.1}"
    )
  else
    LOGIT_COMPARE_ARGS=(
      "--golden_logits_path=${DEFAULT_GOLDEN_DISK}"
      "--atol=${ATOL:-1.5}"
      "--rtol=${RTOL:-1.5}"
      "--max_kl_div=${MAX_KL_DIV:-0.1}"
    )
  fi

  # shellcheck disable=SC2086
  python3 -m tests.utils.forward_pass_logit_checker "${CONFIG_FILE}" \
    base_output_directory="${BASE_OUTPUT_PATH}" \
    run_name=forward_logits_check \
    load_parameters_path="${SCANNED_CKPT_PATH}" \
    scan_layers=true \
    attention=dot_product \
    per_device_batch_size="${LOGIT_PER_DEVICE_BATCH_SIZE:-1}" \
    model_name="${MODEL_NAME}" \
    tokenizer_path="${TOKENIZER_PATH}" \
    override_model_config=true \
    indexer_topk="${LOGIT_INDEXER_TOPK}" \
    max_prefill_predict_length="${LOGIT_MAX_PREFILL_LENGTH:-512}" \
    max_target_length="${LOGIT_MAX_TARGET_LENGTH:-512}" \
    async_checkpointing=false \
    sparse_matmul="${LOGIT_SPARSE_MATMUL:-false}" \
    ici_fsdp_parallelism="${LOGIT_ICI_FSDP:--1}" \
    ici_expert_parallelism="${LOGIT_ICI_EP:-4}" \
    checkpoint_storage_concurrent_gb=1024 \
    weight_dtype="${LOGIT_WEIGHT_DTYPE:-bfloat16}" \
    dtype="${LOGIT_DTYPE:-bfloat16}" \
    activations_in_float32="${LOGIT_ACTIVATIONS_IN_FLOAT32:-false}" \
    matmul_precision="${LOGIT_MATMUL_PRECISION:-highest}" \
    float32_logits="${LOGIT_FLOAT32_LOGITS:-true}" \
    float32_qk_product="${LOGIT_FLOAT32_QK_PRODUCT:-true}" \
    ${LOGIT_EXTRA_FLAGS:-} \
    "${LOGIT_COMPARE_ARGS[@]}"
fi

# 2. Distributed Pre-Training with GLM-5.2 Cross-Layer IndexShare (e.g. 64 TPU cores: FSDP=16, EP=4)
if [ "${RUN_PRETRAIN:-true}" = "true" ]; then
  # shellcheck disable=SC2086
  python3 -m maxtext.trainers.pre_train.train "${CONFIG_FILE}" \
    base_output_directory="${BASE_OUTPUT_PATH}" \
    run_name=pretrain_glm52 \
    model_name="${MODEL_NAME}" \
    override_model_config=true \
    dataset_type=synthetic \
    tokenizer_type=huggingface \
    tokenizer_path="${TOKENIZER_PATH}" \
    per_device_batch_size="${PRETRAIN_PER_DEVICE_BATCH_SIZE:-1}" \
    max_target_length="${PRETRAIN_MAX_TARGET_LENGTH:-2048}" \
    indexer_topk="${PRETRAIN_INDEXER_TOPK:-1024}" \
    scan_layers=true \
    dcn_pipeline_parallelism=1 \
    dcn_data_parallelism=-1 \
    ici_pipeline_parallelism=1 \
    ici_fsdp_transpose_parallelism=1 \
    ici_fsdp_parallelism="${PRETRAIN_ICI_FSDP:--1}" \
    ici_expert_parallelism="${PRETRAIN_ICI_EP:-4}" \
    allow_split_physical_axes=true \
    use_iota_embed=true \
    remat_policy=custom \
    decoder_layer_input=offload \
    opt_type=sgd \
    enable_checkpointing=false \
    steps="${PRETRAIN_STEPS:-10}" \
    ${PRETRAIN_EXTRA_FLAGS:-}
fi

# 3. Autoregressive Decoding with GLM-5.2 Cross-Layer IndexShare (unscanned + MLA KV cache)
if [ "${RUN_DECODE:-true}" = "true" ]; then
  # shellcheck disable=SC2086
  python3 -m maxtext.inference.decode "${CONFIG_FILE}" \
    base_output_directory="${BASE_OUTPUT_PATH}" \
    run_name=decode_glm52 \
    model_name="${MODEL_NAME}" \
    override_model_config=true \
    tokenizer_type=huggingface \
    tokenizer_path="${TOKENIZER_PATH}" \
    load_parameters_path="${UNSCANNED_CKPT_PATH}" \
    scan_layers=false \
    attention=dot_product \
    sparse_matmul="${DECODE_SPARSE_MATMUL:-false}" \
    dtype=bfloat16 \
    weight_dtype=bfloat16 \
    per_device_batch_size=1 \
    indexer_topk="${DECODE_INDEXER_TOPK:-4}" \
    max_prefill_predict_length="${DECODE_MAX_PREFILL:-64}" \
    max_target_length="${DECODE_MAX_TARGET:-128}" \
    ici_fsdp_parallelism="${DECODE_ICI_FSDP:--1}" \
    ici_tensor_parallelism="${DECODE_ICI_TP:-1}" \
    ici_expert_parallelism="${DECODE_ICI_EP:-4}" \
    checkpoint_storage_concurrent_gb=1024 \
    mla_naive_kvcache=false \
    prompt="${DECODE_PROMPT:-The capital of France is}" \
    ${DECODE_EXTRA_FLAGS:-}
fi
