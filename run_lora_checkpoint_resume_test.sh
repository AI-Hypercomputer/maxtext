#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -o pipefail

# Detect directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_DIR="${SCRIPT_DIR}"

VENV_PYTHON="/home/jackyf_google_com/maxtext/.venv/bin/python"
if [ ! -f "${VENV_PYTHON}" ]; then
  VENV_PYTHON="${WORKSPACE_DIR}/.venv/bin/python"
fi
if [ ! -f "${VENV_PYTHON}" ]; then
  VENV_PYTHON="python3"
fi

export PYTHONPATH="${WORKSPACE_DIR}/src"

# Parse CLI arguments and options
SPECIFIED_MODEL=""
SPECIFIED_TRAINER=""
SPECIFIED_STEP="all"
SCAN_LAYERS_VAL="True"
LORA_QTYPE="nf4"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model)
      SPECIFIED_MODEL="$2"
      shift 2
      ;;
    -t|--trainer)
      SPECIFIED_TRAINER="$2"
      shift 2
      ;;
    -s|--step)
      SPECIFIED_STEP="$2"
      shift 2
      ;;
    --scan|--scan-layers)
      SCAN_LAYERS_VAL="$2"
      shift 2
      ;;
    -q|--qtype)
      LORA_QTYPE="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [-m|--model MODEL_NAME] [-t|--trainer TRAINER_TYPE] [-s|--step STEP_NUM] [--scan True|False] [-q|--qtype nf4|int8]"
      exit 0
      ;;
    *)
      if [ -z "${SPECIFIED_MODEL}" ]; then
        SPECIFIED_MODEL="$1"
      elif [ -z "${SPECIFIED_TRAINER}" ]; then
        SPECIFIED_TRAINER="$1"
      fi
      shift
      ;;
  esac
done

# Fallback to environment variables if set
SPECIFIED_MODEL="${SPECIFIED_MODEL:-${TEST_MODEL_NAME}}"
SPECIFIED_TRAINER="${SPECIFIED_TRAINER:-${TEST_TRAINER}}"

if [ -n "${SPECIFIED_MODEL}" ]; then
  MODELS=("${SPECIFIED_MODEL}")
else
  MODELS=("qwen3-4b" "gemma3-4b" "llama3.1-8b" "gpt-oss-20b" "gemma4-26b")
fi

if [ -n "${SPECIFIED_TRAINER}" ]; then
  TRAINERS=("${SPECIFIED_TRAINER}")
else
  TRAINERS=("pre_train" "sft_native" "sft_custom")
fi

echo "=========================================================="
echo "Starting Flax NNX LoRA Comprehensive E2E Test Suite"
echo "Models under test:  ${MODELS[*]}"
echo "Trainers under test:${TRAINERS[*]}"
echo "Scan Layers:        ${SCAN_LAYERS_VAL}"
echo "Workspace:          ${WORKSPACE_DIR}"
echo "Python:             ${VENV_PYTHON}"
echo "PYTHONPATH:         ${PYTHONPATH}"
echo "=========================================================="

LOG_DIR="${WORKSPACE_DIR}/maxtext_output/lora_resume_test_logs"
mkdir -p "${LOG_DIR}"

for TEST_MODEL_NAME in "${MODELS[@]}"; do
for TRAINER in "${TRAINERS[@]}"; do
  echo -e "\n=========================================================="
  echo "TESTING MODEL: ${TEST_MODEL_NAME} | TRAINER: ${TRAINER}"
  echo "=========================================================="

  # Setup module and config paths
  if [ "${TRAINER}" == "pre_train" ]; then
    MODULE_NAME="maxtext.trainers.pre_train.train"
    CONFIG_PATH="src/maxtext/configs/base.yml"
  elif [ "${TRAINER}" == "sft_native" ]; then
    MODULE_NAME="maxtext.trainers.post_train.sft.train_sft_native"
    CONFIG_PATH="src/maxtext/configs/base.yml"
  elif [ "${TRAINER}" == "sft_custom" ]; then
    MODULE_NAME="maxtext.trainers.post_train.sft.train_sft"
    CONFIG_PATH="src/maxtext/configs/post_train/sft.yml"
  fi

  # Directories
  BASE_RUN="lora_resume_test_${TEST_MODEL_NAME}_${TRAINER}_base"
  WORKLOAD_RUN="lora_resume_test_${TEST_MODEL_NAME}_${TRAINER}_workload"

  if [ "${SPECIFIED_STEP}" = "all" ] || [ "${SPECIFIED_STEP}" = "1" ]; then
    rm -rf "${WORKSPACE_DIR}/maxtext_output/${BASE_RUN}"
    rm -rf "${WORKSPACE_DIR}/maxtext_output/${WORKLOAD_RUN}"
  elif [ "${SPECIFIED_STEP}" = "2" ]; then
    rm -rf "${WORKSPACE_DIR}/maxtext_output/${WORKLOAD_RUN}"
  fi

  LOG_PREFIX="${LOG_DIR}/${TEST_MODEL_NAME}_${TRAINER}"

  # 1. Generate base-only checkpoint
  if [ "${SPECIFIED_STEP}" = "all" ] || [ "${SPECIFIED_STEP}" = "1" ]; then
    echo "[1/4] Generating base-only checkpoint..."
    "${VENV_PYTHON}" -m "${MODULE_NAME}" "${CONFIG_PATH}" \
      run_name="${BASE_RUN}" \
      model_name="${TEST_MODEL_NAME}" scan_layers=${SCAN_LAYERS_VAL} pure_nnx=True dataset_type=synthetic steps=10 \
      enable_checkpointing=True checkpoint_period=10 \
      enable_goodput_recording=False enable_checkpoint_cloud_logger=False monitor_goodput=False log_period=1 \
      override_model_config=True base_num_decoder_layers=2 base_emb_dim=128 base_mlp_dim=256 base_num_query_heads=4 base_num_kv_heads=4 head_dim=128 \
      max_target_length=128 vocab_size=256 per_device_batch_size=8 \
      lora.enable_lora=False \
      > "${LOG_PREFIX}_step1_base.log" 2>&1
    STEP1_STATUS=$?
    echo "Base Checkpoint Exit Status: ${STEP1_STATUS}"
    if [ ${STEP1_STATUS} -ne 0 ]; then
      echo "Error: ${TEST_MODEL_NAME} ${TRAINER} base checkpoint generation failed! See logs in ${LOG_PREFIX}_step1_base.log"
      exit 1
    fi
  fi

  # Find items or model_params based on trainer layout
  BASE_CHECKPOINT_PATH=$(find "${WORKSPACE_DIR}/maxtext_output/${BASE_RUN}/checkpoints" -name "items" -o -name "model_params" | head -n 1)
  if [ -z "${BASE_CHECKPOINT_PATH}" ] || [ ! -d "${BASE_CHECKPOINT_PATH}" ]; then
    echo "Error: Could not find generated base checkpoint directory under maxtext_output/${BASE_RUN}/checkpoints"
    exit 1
  fi
  echo "Found Base Checkpoint: ${BASE_CHECKPOINT_PATH}"

  # 2. Train with LoRA (saves checkpoint at step 10)
  if [ "${SPECIFIED_STEP}" = "all" ] || [ "${SPECIFIED_STEP}" = "2" ]; then
    sleep 2
    echo "[2/4] Training with LoRA starting from base checkpoint..."
    "${VENV_PYTHON}" -m "${MODULE_NAME}" "${CONFIG_PATH}" \
      run_name="${WORKLOAD_RUN}" \
      model_name="${TEST_MODEL_NAME}" scan_layers=${SCAN_LAYERS_VAL} pure_nnx=True dataset_type=synthetic steps=15 \
      load_parameters_path="${BASE_CHECKPOINT_PATH}" \
      enable_checkpointing=True checkpoint_period=10 \
      enable_goodput_recording=False enable_checkpoint_cloud_logger=False monitor_goodput=False log_period=1 \
      override_model_config=True base_num_decoder_layers=2 base_emb_dim=128 base_mlp_dim=256 base_num_query_heads=4 base_num_kv_heads=4 head_dim=128 \
      max_target_length=128 vocab_size=256 per_device_batch_size=8 \
      lora.enable_lora=True lora.lora_rank=4 lora.lora_alpha=8.0 lora.lora_weight_qtype=${LORA_QTYPE} lora.lora_tile_size=16 sharding_tolerance=1.0 \
      > "${LOG_PREFIX}_step2_lora_train.log" 2>&1
    STEP2_STATUS=$?
    echo "LoRA Train Exit Status: ${STEP2_STATUS}"
    if [ ${STEP2_STATUS} -ne 0 ]; then
      echo "Error: ${TEST_MODEL_NAME} ${TRAINER} LoRA initial training failed! See logs in ${LOG_PREFIX}_step2_lora_train.log"
      exit 1
    fi
  fi

  # Find lora checkpoint folder (any items or model_params folder under checkpoints)
  LORA_CHECKPOINT_PATH=$(find "${WORKSPACE_DIR}/maxtext_output/${WORKLOAD_RUN}/checkpoints" -type d \( -name "items" -o -name "model_params" \) | grep -v "/0/" | head -n 1)
  if [ -z "${LORA_CHECKPOINT_PATH}" ]; then
    LORA_CHECKPOINT_PATH=$(find "${WORKSPACE_DIR}/maxtext_output/${WORKLOAD_RUN}/checkpoints" -type d \( -name "items" -o -name "model_params" \) | head -n 1)
  fi
  echo "Found Saved LoRA Checkpoint: ${LORA_CHECKPOINT_PATH}"

  # 3. Resume training from step 10 under same run_name
  if [ "${SPECIFIED_STEP}" = "all" ] || [ "${SPECIFIED_STEP}" = "3" ]; then
    sleep 2
    echo "[3/4] Resuming training under same run name (workload name)..."
    "${VENV_PYTHON}" -m "${MODULE_NAME}" "${CONFIG_PATH}" \
      run_name="${WORKLOAD_RUN}" \
      model_name="${TEST_MODEL_NAME}" scan_layers=${SCAN_LAYERS_VAL} pure_nnx=True dataset_type=synthetic steps=20 \
      enable_checkpointing=True checkpoint_period=10 \
      enable_goodput_recording=False enable_checkpoint_cloud_logger=False monitor_goodput=False log_period=1 \
      override_model_config=True base_num_decoder_layers=2 base_emb_dim=128 base_mlp_dim=256 base_num_query_heads=4 base_num_kv_heads=4 head_dim=128 \
      max_target_length=128 vocab_size=256 per_device_batch_size=8 \
      lora.enable_lora=True lora.lora_rank=4 lora.lora_alpha=8.0 lora.lora_weight_qtype=${LORA_QTYPE} lora.lora_tile_size=16 sharding_tolerance=1.0 \
      > "${LOG_PREFIX}_step3_lora_resume.log" 2>&1
    STEP3_STATUS=$?
    echo "LoRA Resume Exit Status: ${STEP3_STATUS}"
    if [ ${STEP3_STATUS} -ne 0 ]; then
      echo "Error: ${TEST_MODEL_NAME} ${TRAINER} LoRA resume training failed! See logs in ${LOG_PREFIX}_step3_lora_resume.log"
      exit 1
    fi
  fi

  # 4. Verify standalone restore of saved LoRA checkpoint
  if [ "${SPECIFIED_STEP}" = "all" ] || [ "${SPECIFIED_STEP}" = "4" ]; then
    sleep 2
    echo "[4/4] Verifying standalone restore of LoRA checkpoint..."
    "${VENV_PYTHON}" -m "${MODULE_NAME}" "${CONFIG_PATH}" \
      run_name="lora_restore_verify_${TEST_MODEL_NAME}_${TRAINER}_$(date +%s)" \
      model_name="${TEST_MODEL_NAME}" scan_layers=${SCAN_LAYERS_VAL} pure_nnx=True dataset_type=synthetic steps=11 \
      load_parameters_path="${BASE_CHECKPOINT_PATH}" \
      lora.lora_restore_path="${LORA_CHECKPOINT_PATH}" \
      enable_checkpointing=True checkpoint_period=1000 \
      enable_goodput_recording=False enable_checkpoint_cloud_logger=False monitor_goodput=False log_period=1 \
      override_model_config=True base_num_decoder_layers=2 base_emb_dim=128 base_mlp_dim=256 base_num_query_heads=4 base_num_kv_heads=4 head_dim=128 \
      max_target_length=128 vocab_size=256 per_device_batch_size=8 \
      lora.enable_lora=True lora.lora_rank=4 lora.lora_alpha=8.0 lora.lora_weight_qtype=${LORA_QTYPE} lora.lora_tile_size=16 sharding_tolerance=1.0 \
      > "${LOG_PREFIX}_step4_lora_restore.log" 2>&1
    STEP4_STATUS=$?
    echo "LoRA Standalone Restore Exit Status: ${STEP4_STATUS}"
    if [ ${STEP4_STATUS} -ne 0 ]; then
      echo "Error: ${TEST_MODEL_NAME} ${TRAINER} LoRA standalone restore failed! See logs in ${LOG_PREFIX}_step4_lora_restore.log"
      exit 1
    fi
  fi

done
done

echo ""
echo "=========================================================="
echo "Asserting Accuracy and Parsing Performance Metrics"
echo "=========================================================="

"${VENV_PYTHON}" -c "
import glob
import re
import os

log_dir = '${LOG_DIR}'
models = [$(printf '"%s", ' "${MODELS[@]}")]
trainers = [$(printf '"%s", ' "${TRAINERS[@]}")]

# Find all log files
log_files = glob.glob(os.path.join(log_dir, '*.log'))

results = []
all_passed = True

def parse_loss(filepath, target_step):
    if not os.path.exists(filepath):
        return None, f'File not found: {os.path.basename(filepath)}'
    with open(filepath, 'r') as f:
        content = f.read()
    pattern = rf'completed step: {target_step},.*loss: ([\d\.]+)'
    match = re.search(pattern, content)
    if match:
        return float(match.group(1)), None
    return None, f'Step {target_step} not found in {os.path.basename(filepath)}'

print('\n### E2E LoRA Core Verification Results\n')
print('| Model | Trainer | Final Train Loss | Initial Resume Loss | Loss Continuity | Standalone Restore | Status |')
print('|---|---|---|---|---|---|---|')

for model in models:
    for trainer in trainers:
        step2_log = os.path.join(log_dir, f'{model}_{trainer}_step2_lora_train.log')
        step3_log = os.path.join(log_dir, f'{model}_{trainer}_step3_lora_resume.log')
        step4_log = os.path.join(log_dir, f'{model}_{trainer}_step4_lora_restore.log')
        
        # SFT trainers save/resume at step 10/11 vs pre_train step 10/11
        train_step = 15 if trainer == 'sft_custom' else 14
        resume_step = 16 if trainer == 'sft_custom' else 15
        
        train_loss, err2 = parse_loss(step2_log, train_step)
        resume_loss, err3 = parse_loss(step3_log, resume_step)
        
        restore_passed = os.path.exists(step4_log)
        if restore_passed:
            with open(step4_log, 'r') as f:
                restore_content = f.read()
            restore_passed = 'completed step: 10' in restore_content or 'completed step: 0' in restore_content
        
        if train_loss is not None and resume_loss is not None:
            diff = abs(train_loss - resume_loss)
            rel_diff = diff / max(abs(train_loss), 1e-5)
            loss_passed = diff < 0.05 or rel_diff < 0.05
            continuity_str = f'PASSED' if loss_passed else f'FAILED (Diff: {diff:.6f}, RelDiff: {rel_diff:.4f})'
        else:
            loss_passed = False
            continuity_str = 'FAILED (Logs empty)'
            
        restore_str = 'PASSED' if restore_passed else 'FAILED'
        row_passed = loss_passed and restore_passed
        if not row_passed:
            all_passed = False
            
        status_str = 'PASSED' if row_passed else 'FAILED'
        train_str = f'{train_loss:.6f} (Step {train_step})' if train_loss else 'N/A'
        resume_str = f'{resume_loss:.6f} (Step {resume_step})' if resume_loss else 'N/A'
        
        print(f'| {model} | {trainer} | {train_str} | {resume_str} | {continuity_str} | {restore_str} | {status_str} |')

if not all_passed:
    print('\nFAILURE: One or more correctness assertions failed across the models/trainers.')
    exit(1)

print('\nSUCCESS: All Flax NNX LoRA E2E correctness assertions passed successfully!')
"

echo "COMPREHENSIVE TEST COMPLETE."