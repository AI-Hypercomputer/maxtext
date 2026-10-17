#!/bin/bash

# Nemotron-3-Nano-Omni-30B-A3B text backbone (nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16): Step 2, on TPU.
# Stages (each toggled by RUN_<STAGE>=true|false):
#   LOGIT      fp32 logit check (go/maxtext-model-bringup §6.1 flags) against the golden in tests/assets/golden_logits.
#   LOGIT_LONG fp32 and bf16 logit checks on long prompts (129 / 1025 / 4096 tokens) from GOLDEN_LONG_DIR.
#   DECODE     greedy decode; output must start with the HF reference continuation.
#   PRETRAIN   pre-training from scratch on C4.
#   FINETUNE   SFT from the converted checkpoint on HuggingFaceH4/ultrachat_200k.
#
# Logit checks use tests/utils/nemotron3_logit_check.py, not forward_pass_logit_checker alone: Nemotron's router biases
# sit within about 0.1 of each other, so exact fp32 top-6 ties occur on real text, and free-routing KL measures those
# ties rather than implementation error. The check replays the HF top-k, records MaxText's own top-k, and fails on any
# disagreement that is not a near-tie (see its docstring for the gates). The short golden is
# tests/assets/golden_logits/golden_data_nemotron3-nano-omni-30b-a3b.jsonl, falling back to
# gs://maxtext-test-assets/golden_data_nemotron3-nano-omni-30b-a3b.jsonl.
# Goldens were generated with a patched HF reference (the generator is kept with the bringup harness);
# tests/assets/logits_generation/generate_hf_golden_logits.py runs the unpatched HF code, which is wrong past
# token 128 for this model.
#
# The model supports only scan_layers=false. Every stage ends with `E2E_STAGE_<name>_RC=<rc>`.

set -ex

export MODEL_NAME="${MODEL_NAME:-nemotron3-nano-omni-30b-a3b}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16}"
CONFIGS_DIR="${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"

if [ -z "${BASE_OUTPUT_PATH}" ]; then
  # Non-Googlers please point BASE_OUTPUT_PATH to a GCS bucket that you own.
  export BASE_OUTPUT_PATH="gs://runner-maxtext-logs/$(date +%Y-%m-%d-%H-%M)"
  echo "BASE_OUTPUT_PATH is not set"
fi
BASE_OUTPUT_PATH="${BASE_OUTPUT_PATH%/}"
echo "using BASE_OUTPUT_PATH = ${BASE_OUTPUT_PATH}"

# Converted by 1_test_nemotron3.sh.
UNSCANNED_CKPT_PATH="${UNSCANNED_CKPT_PATH:-gs://maxtext-model-checkpoints/${MODEL_NAME}/unscanned/0/items}"
export DATASET_PATH="${DATASET_PATH:-gs://maxtext-dataset}"

COMMON_FLAGS="model_name=${MODEL_NAME} base_output_directory=${BASE_OUTPUT_PATH} scan_layers=false \
tokenizer_type=huggingface tokenizer_path=${TOKENIZER_PATH}"
# fp32 flags from go/maxtext-model-bringup §6.1.
FP32_FLAGS="dtype=float32 weight_dtype=float32 activations_in_float32=true matmul_precision=${LOGIT_MATMUL_PRECISION:-high} \
float32_logits=true"
BF16_FLAGS="dtype=bfloat16 weight_dtype=bfloat16 activations_in_float32=false float32_logits=true"
LOGIT_FLAGS="load_parameters_path=${UNSCANNED_CKPT_PATH} attention=dot_product per_device_batch_size=1 \
sparse_matmul=${LOGIT_SPARSE_MATMUL:-true} megablox=${LOGIT_MEGABLOX:-false} return_routed_experts=true \
ici_fsdp_parallelism=${LOGIT_ICI_FSDP:--1} ici_expert_parallelism=${LOGIT_ICI_EP:-1} ${LOGIT_EXTRA_FLAGS:-}"

fetch() {  # fetch <gcs-or-local path> <local path>
  if [ "$1" = "$2" ]; then
    return 0
  fi
  if [[ "$1" == gs://* ]]; then gcloud storage cp "$1" "$2"; else cp "$1" "$2"; fi
}

if [ "${RUN_LOGIT:-true}" = "true" ]; then
  echo "=== STAGE: logit check (fp32) ==="
  GOLDEN="${MAXTEXT_REPO_ROOT:-$PWD}/tests/assets/golden_logits/golden_data_${MODEL_NAME}.jsonl"
  if [ ! -f "${GOLDEN}" ]; then
    fetch "${GOLDEN_LOGITS_PATH:-gs://maxtext-test-assets/golden_data_${MODEL_NAME}.jsonl}" /tmp/golden_data.jsonl
    GOLDEN=/tmp/golden_data.jsonl
  fi
  rc=0
  stage_rc=0
  # shellcheck disable=SC2086
  python3 -m tests.utils.nemotron3_logit_check "${CONFIGS_DIR}/base.yml" ${COMMON_FLAGS} ${FP32_FLAGS} ${LOGIT_FLAGS} \
    run_name=logit_fp32 max_target_length=8 \
    --golden_logits_path="${GOLDEN}" --golden_dtype=fp32 --gate=d_p3_9 --modes=replay,free \
    --check_id_prefix=e2e.short --output_jsonl="${LOGIT_OUTPUT_JSONL:-/tmp/nemotron3_logits.jsonl}" || rc=$?
  echo "E2E_STAGE_LOGIT_RC=${rc}"
  [ "${rc}" = "0" ] || stage_rc=1
  [ "${stage_rc}" = "0" ]
fi

if [ "${RUN_LOGIT_LONG:-false}" = "true" ]; then
  echo "=== STAGE: logit check, long prompts (fp32 and bf16) ==="
  GOLDEN_LONG_DIR="${GOLDEN_LONG_DIR:?set GOLDEN_LONG_DIR to the directory holding the golden_s<len>.pkl files}"
  stage_rc=0
  for tag in ${LOGIT_LONG_TAGS:-s129 s1025 s4096}; do
    fetch "${GOLDEN_LONG_DIR%/}/golden_${tag}.pkl" "/tmp/golden_${tag}.pkl"
    len="${tag#s}"
    if [[ " ${LOGIT_LONG_DTYPES:-fp32 bf16} " == *" fp32 "* ]]; then
      rc=0
      # shellcheck disable=SC2086
      python3 -m tests.utils.nemotron3_logit_check "${CONFIGS_DIR}/base.yml" ${COMMON_FLAGS} ${FP32_FLAGS} ${LOGIT_FLAGS} \
        run_name="logit_fp32_${tag}" max_target_length="${len}" \
        --golden_logits_path="/tmp/golden_${tag}.pkl" --golden_dtype=fp32 --gate="${LOGIT_FP32_GATE:-d_p3_9}" \
        --modes=replay,free --check_id_prefix="e2e.${tag}.${LOGIT_MATMUL_PRECISION:-high}" \
        --output_jsonl="${LOGIT_OUTPUT_JSONL:-/tmp/nemotron3_logits.jsonl}" || rc=$?
      echo "E2E_STAGE_LOGIT_FP32_${tag}_RC=${rc}"
      [ "${rc}" = "0" ] || stage_rc=1
    fi
    if [[ " ${LOGIT_LONG_DTYPES:-fp32 bf16} " == *" bf16 "* ]]; then
      rc=0
      # shellcheck disable=SC2086
      python3 -m tests.utils.nemotron3_logit_check "${CONFIGS_DIR}/base.yml" ${COMMON_FLAGS} ${BF16_FLAGS} ${LOGIT_FLAGS} \
        run_name="logit_bf16_${tag}" max_target_length="${len}" \
        --golden_logits_path="/tmp/golden_${tag}.pkl" --golden_dtype=bf16 --gate=d_p3_10 --modes=replay,free \
        --check_id_prefix="e2e.${tag}" --output_jsonl="${LOGIT_OUTPUT_JSONL:-/tmp/nemotron3_logits.jsonl}" || rc=$?
      echo "E2E_STAGE_LOGIT_BF16_${tag}_RC=${rc}"
      [ "${rc}" = "0" ] || stage_rc=1
    fi
  done
  [ "${stage_rc}" = "0" ]
fi

if [ "${RUN_DECODE:-true}" = "true" ]; then
  echo "=== STAGE: decode ==="
  # Each case is "prompt<TAB>expected": expected is the patched HF oracle's fp32 greedy continuation of the prompt
  # (16 tokens, truncated after EOS as decode.py does), from the golden's greedy_tokens_fp32.
  DECODE_CASES="${DECODE_CASES:-}"
  if [[ "${DECODE_CASES}" == gs://* ]]; then
    fetch "${DECODE_CASES}" /tmp/decode_cases.tsv
    DECODE_CASES=/tmp/decode_cases.tsv
  elif [ -z "${DECODE_CASES}" ]; then
    DECODE_CASES=/tmp/decode_cases.tsv
    printf 'I love to\t travel and explore new places. I also enjoy cooking and trying out new recipes.\n' > "${DECODE_CASES}"
  fi
  rc=0
  while IFS=$'\t' read -r prompt expected; do
    [ -z "${prompt}" ] && continue
    if [ -z "${expected}" ]; then
      echo "ERROR: empty expected continuation for prompt \"${prompt}\""
      rc=1
      continue
    fi
    # shellcheck disable=SC2086
    python3 -m maxtext.inference.decode "${CONFIGS_DIR}/base.yml" ${COMMON_FLAGS} \
      ${DECODE_PRECISION_FLAGS:-${FP32_FLAGS}} \
      run_name=decode hf_access_token="${HF_TOKEN:-}" load_parameters_path="${UNSCANNED_CKPT_PATH}" \
      attention=dot_product sparse_matmul=true megablox=false per_device_batch_size=1 \
      max_prefill_predict_length="${DECODE_MAX_PREFILL:-64}" max_target_length="${DECODE_MAX_TARGET:-80}" \
      ici_fsdp_parallelism="${DECODE_ICI_FSDP:-1}" ici_tensor_parallelism=1 ici_expert_parallelism="${DECODE_ICI_EP:--1}" \
      prompt="${prompt}" autoregressive_decode_assert="${expected}" ${DECODE_EXTRA_FLAGS:-} < /dev/null || rc=1
    echo "E2E_DECODE_CASE prompt=\"${prompt}\" rc=${rc}"
  done < "${DECODE_CASES}"
  echo "E2E_STAGE_DECODE_RC=${rc}"
  [ "${rc}" = "0" ]
fi

check_loss() {  # check_loss <log>: the last reported loss must be finite and positive.
  local loss
  loss=$(awk -F"loss: " '/completed step:/ {print $2}' "$1" | awk -F"," '{print $1}' | tail -n 1 | tr -d ' ')
  if [ -z "${loss}" ] || [ "$(awk -v v="${loss}" 'BEGIN { print (v + 0 > 0 && v + 0 < 1e30) ? 1 : 0 }')" != "1" ]; then
    echo "ERROR: invalid final loss: ${loss}"
    return 1
  fi
}

if [ "${RUN_PRETRAIN:-true}" = "true" ]; then
  echo "=== STAGE: pre-train ==="
  # shellcheck disable=SC2086
  python3 -m maxtext.trainers.pre_train.train "${CONFIGS_DIR}/base.yml" ${COMMON_FLAGS} \
    run_name=pre_training dataset_type=grain grain_file_type=tfrecord dataset_path="${DATASET_PATH}" \
    enable_checkpointing=false attention=flash sparse_matmul=true megablox=true dtype=bfloat16 weight_dtype=float32 \
    per_device_batch_size="${PRETRAIN_PER_DEVICE_BATCH_SIZE:-1}" steps="${PRETRAIN_STEPS:-20}" \
    max_target_length="${PRETRAIN_MAX_TARGET:-4096}" remat_policy="${PRETRAIN_REMAT:-full}" \
    ici_fsdp_parallelism="${PRETRAIN_ICI_FSDP:--1}" ici_expert_parallelism="${PRETRAIN_ICI_EP:-8}" \
    ${PRETRAIN_EXTRA_FLAGS:-} 2>&1 | tee /tmp/pretrain_output.log
  rc=${PIPESTATUS[0]}
  [ "${rc}" = "0" ] && { check_loss /tmp/pretrain_output.log || rc=1; }
  echo "E2E_STAGE_PRETRAIN_RC=${rc}"
  [ "${rc}" = "0" ]
fi

if [ "${RUN_FINETUNE:-true}" = "true" ]; then
  echo "=== STAGE: SFT ==="
  # No checkpoint saves: with optimizer_memory_host_offload, MaxText's save of the host-resident step counter
  # fails (XLA E1200). Checkpointing stays enabled so load_parameters_path restores on multi-host slices too.
  # shellcheck disable=SC2086
  python3 -m maxtext.trainers.post_train.sft.train_sft_native "${CONFIGS_DIR}/post_train/sft.yml" ${COMMON_FLAGS} \
    run_name=sft load_parameters_path="${UNSCANNED_CKPT_PATH}" hf_access_token="${HF_TOKEN:-}" \
    enable_checkpointing=true save_checkpoint_on_start=false save_checkpoint_on_completion=false checkpoint_period=100000 \
    attention="${SFT_ATTENTION:-flash}" sparse_matmul=true megablox=true dtype=bfloat16 weight_dtype=float32 \
    per_device_batch_size="${SFT_PER_DEVICE_BATCH_SIZE:-1}" steps="${SFT_STEPS:-10}" \
    max_target_length="${SFT_MAX_TARGET:-1024}" remat_policy="${SFT_REMAT:-full}" \
    ici_fsdp_parallelism="${SFT_ICI_FSDP:--1}" ici_expert_parallelism="${SFT_ICI_EP:-4}" \
    ${SFT_EXTRA_FLAGS:-} 2>&1 | tee /tmp/sft_output.log
  rc=${PIPESTATUS[0]}
  [ "${rc}" = "0" ] && { check_loss /tmp/sft_output.log || rc=1; }
  echo "E2E_STAGE_FINETUNE_RC=${rc}"
  [ "${rc}" = "0" ]
fi

if [ -n "${E2E_RESULTS_GCS:-}" ] && [ -f "${LOGIT_OUTPUT_JSONL:-/tmp/nemotron3_logits.jsonl}" ]; then
  gcloud storage cp "${LOGIT_OUTPUT_JSONL:-/tmp/nemotron3_logits.jsonl}" "${E2E_RESULTS_GCS%/}/nemotron3_logits.jsonl"
fi
echo "Nemotron-3-Nano-Omni end-to-end tests completed successfully!"
