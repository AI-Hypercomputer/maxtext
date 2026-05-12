#!/bin/bash
#
# Copyright 2023-2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# OLMo 3 7B pre-training launcher for MaxText. Hyperparameters mirror
# OLMo-core's pretrain-1.py (global batch ≈4M tokens, peak LR 3e-4 cosine to
# 0.1×, 2k warmup, β=(0.9, 0.95), ε=1e-8, WD 0.1, grad-clip 1.0, z-loss 1e-5,
# logits in fp32, skip-step-on-spike). Reference run:
# https://wandb.ai/ai2-llm/Olmo-3-1025-7B
#
# Required env:
#   INDEX_PATH            JSON index from tools/data_generation/build_olmo_npy_index.py
#   GCS_BASE              gs:// prefix recorded in the index (e.g. gs://my-bucket/)
#   LOCAL_MOUNT           gcsfuse mount of GCS_BASE on this host
#   OUTPUT_DIR            base_output_directory (checkpoints + logs)
#   HF_TOKEN              tokenizer download (or HF_SECRETS=<file>)
#
# Optional (one of):
#   LOAD_PARAMETERS_PATH  cold-start init weights (Orbax dir; e.g. converted
#                         AI2 stage1-stepN snapshot). Ignored on resume. If
#                         unset, MaxText uses fresh random init from the
#                         olmo3-7b-pt model config.
#
# Optional (with defaults):
#   RUN_NAME, TARGET_GLOBAL_BATCH (=512), PER_DEVICE_BATCH, TOTAL_DEVICES,
#   GRAD_ACCUM_STEPS (=1), STEPS, WARMUP_STEPS (=2000), LR_SCHEDULE_STEPS
#   (=STEPS), LEARNING_RATE (=3e-4), COSINE_FINAL_FRAC (=0.1), CHECKPOINT_PERIOD,
#   DATA_SEED (=42), VENV_PATH, MOUNT_GCSFUSE (=0).
#
# Usage:
#   INDEX_PATH=... GCS_BASE=gs://... LOCAL_MOUNT=... OUTPUT_DIR=... \
#   LOAD_PARAMETERS_PATH=gs://.../stage1-step0/0/items HF_SECRETS=~/.hf_token.sh \
#   bash src/maxtext/trainers/pre_train/scripts/olmo/run_olmo3_7b_stage1.sh
#
# Resume: keep RUN_NAME and OUTPUT_DIR stable. Orbax picks up the latest
# checkpoint; the OLMo grain sampler resumes its data position via stateless
# initial_step = step × per-host-batch.

set -euo pipefail

# Lives at src/maxtext/trainers/pre_train/scripts/olmo/; walk up five levels to repo root.
MAXTEXT_ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"

VENV_PATH="${VENV_PATH:-${MAXTEXT_ROOT}/maxtext_venv}"
HF_SECRETS="${HF_SECRETS:-}"
INDEX_PATH="${INDEX_PATH:?INDEX_PATH is required (path to olmo index JSON)}"
GCS_BASE="${GCS_BASE:?GCS_BASE is required (e.g. gs://my-bucket/)}"
LOCAL_MOUNT="${LOCAL_MOUNT:?LOCAL_MOUNT is required (gcsfuse mount path of GCS_BASE)}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required (base_output_directory)}"
# Optional: if unset, MaxText random-inits from the olmo3-7b-pt config.
LOAD_PARAMETERS_PATH="${LOAD_PARAMETERS_PATH:-}"

RUN_NAME="${RUN_NAME:-olmo3_7b_stage1}"

# Force exactly one trailing slash on both halves of the path-remap pair —
# the OLMo data source does a literal `replacement + path[len(prefix):]`,
# so trailing slashes must match (otherwise gs://bucket/dataset/foo →
# /mnt/local/dataset/foo silently becomes /mnt/localdataset/foo).
GCS_BASE="${GCS_BASE%/}/"
LOCAL_MOUNT="${LOCAL_MOUNT%/}/"

# Hyperparameters (defaults match OLMo-core pretrain-1.py).
TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-512}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
SEQ_LEN="${SEQ_LEN:-8192}"
STEPS="${STEPS:-1414078}"  # AI2 stage-1 horizon: 5.928T tokens at 4.19M tokens/step
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
# Cosine schedule horizon. Defaults to STEPS for full-horizon runs; for
# partial runs that overlay against a reference curve, set this to the
# reference horizon (e.g. 1193317 for OLMo-3 pretrain-1's 5T-token schedule
# at 4.19M tokens/step) so warmup stays absolute and cosine shape matches.
LR_SCHEDULE_STEPS="${LR_SCHEDULE_STEPS:-${STEPS}}"
LEARNING_RATE="${LEARNING_RATE:-3.0e-4}"
COSINE_FINAL_FRAC="${COSINE_FINAL_FRAC:-0.1}"
ADAM_B1="${ADAM_B1:-0.9}"
ADAM_B2="${ADAM_B2:-0.95}"
ADAM_EPS="${ADAM_EPS:-1.0e-8}"
ADAM_WD="${ADAM_WD:-0.1}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
Z_LOSS="${Z_LOSS:-1.0e-5}"
CHECKPOINT_PERIOD="${CHECKPOINT_PERIOD:-1000}"
DATA_SEED="${DATA_SEED:-42}"
TOKENIZER_PATH="${TOKENIZER_PATH:-allenai/Olmo-3-7B-Instruct}"

# Activate venv if present; otherwise rely on system Python (e.g. inside an
# XPK runner image where MaxText deps are already installed).
# shellcheck disable=SC1090,SC1091
if [[ -d "${VENV_PATH}" ]]; then
  source "${VENV_PATH}/bin/activate"
fi
if [[ -n "${HF_SECRETS:-}" && -f "${HF_SECRETS}" ]]; then
  # shellcheck disable=SC1090
  source "${HF_SECRETS}"
fi
: "${HF_TOKEN:?HF_TOKEN must be set (or HF_SECRETS pointing at a file that exports it)}"
export PYTHONPATH="${MAXTEXT_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Ironwood-tuned XLA flags (loss-neutral; MFU 27% → 41% → 44.5%).
# Initial set from AI-Hypercomputer's tpu-recipes/ironwood/llama3.1-70b/8k-bf16-tpu7x-4x4x4
# (gets us to 41%); extended set sourced from a Borg-launched llama3.1-8b run that
# hit 47% on gf_4x4x4 — extends sparse-core collective offload to all-gather/2d-all-
# gather/reduce-scatter (FSDP path), bumps scoped VMEM 60→64 MB, sets DVFS p-state=7,
# disables async collective fusion (interferes with SC offload). Append rather than
# replace so a caller can still inject extra flags.
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-} \
  --xla_tpu_scoped_vmem_limit_kib=65536 \
  --xla_tpu_bf16_emission_mode=NATIVE_EMISSION \
  --xla_tpu_dvfs_p_state=7 \
  --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true \
  --xla_tpu_enable_sparse_core_collective_offload_all_gather=true \
  --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true \
  --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true \
  --xla_tpu_use_tc_device_shape_on_sc=True \
  --xla_sc_disable_megacore_partitioning=True \
  --xla_tpu_enable_async_collective_fusion_fuse_all_gather=false"

# Optional gcsfuse mount; the XPK wrapper sets MOUNT_GCSFUSE=1 so each pod
# mounts its own copy at startup.
if [[ "${MOUNT_GCSFUSE:-0}" = "1" ]]; then
  bucket="${GCS_BASE#gs://}"; bucket="${bucket%/}"
  echo "Mounting gs://${bucket} at ${LOCAL_MOUNT} via gcsfuse..."
  bash "${MAXTEXT_ROOT}/src/dependencies/scripts/setup_gcsfuse.sh" \
    DATASET_GCS_BUCKET="${bucket}" MOUNT_PATH="${LOCAL_MOUNT}"
fi

# Decide per-device batch from target global batch and visible devices.
# bash-level jax.device_count() returns LOCAL devices only in multi-host XPK
# jobs (e.g. 32 instead of 128 on tpu7x-4x4x4); pass TOTAL_DEVICES to override.
if [[ -n "${TOTAL_DEVICES:-}" ]]; then
  NUM_DEVICES="${TOTAL_DEVICES}"
else
  NUM_DEVICES=$(python3 -c 'import jax; print(jax.device_count())')
fi
if [[ -z "${PER_DEVICE_BATCH:-}" ]]; then
  total_slots=$(( NUM_DEVICES * GRAD_ACCUM_STEPS ))
  if (( TARGET_GLOBAL_BATCH % total_slots != 0 )); then
    echo "ERROR: TARGET_GLOBAL_BATCH=${TARGET_GLOBAL_BATCH} is not divisible by " \
         "num_devices×grad_accum=${NUM_DEVICES}×${GRAD_ACCUM_STEPS}=${total_slots}." >&2
    exit 1
  fi
  PER_DEVICE_BATCH=$(( TARGET_GLOBAL_BATCH / total_slots ))
fi

EFFECTIVE_GLOBAL=$(( PER_DEVICE_BATCH * NUM_DEVICES * GRAD_ACCUM_STEPS ))
if (( EFFECTIVE_GLOBAL != TARGET_GLOBAL_BATCH )); then
  echo "ERROR: per_device(${PER_DEVICE_BATCH}) × devices(${NUM_DEVICES}) × " \
       "grad_accum(${GRAD_ACCUM_STEPS}) = ${EFFECTIVE_GLOBAL}, expected " \
       "${TARGET_GLOBAL_BATCH}." >&2
  exit 1
fi

# warmup_steps_fraction is scaled by the schedule horizon (LR_SCHEDULE_STEPS),
# not the run length (STEPS), so the LR curve matches the reference for any
# prefix of training.
WARMUP_FRAC=$(python3 -c "print(${WARMUP_STEPS}/${LR_SCHEDULE_STEPS})")

echo "=== OLMo 3 7B Stage 1 ==="
echo "  run_name           : ${RUN_NAME}"
echo "  output_dir         : ${OUTPUT_DIR}/${RUN_NAME}"
echo "  init weights       : ${LOAD_PARAMETERS_PATH:-<random init>}"
echo "  index              : ${INDEX_PATH}"
echo "  data path remap    : ${GCS_BASE} → ${LOCAL_MOUNT}"
echo "  num_devices        : ${NUM_DEVICES}"
echo "  per_device_batch   : ${PER_DEVICE_BATCH}"
echo "  grad_accum_steps   : ${GRAD_ACCUM_STEPS}"
echo "  global_batch       : ${EFFECTIVE_GLOBAL} instances ($(( EFFECTIVE_GLOBAL * SEQ_LEN )) tokens)"
echo "  seq_len            : ${SEQ_LEN}"
echo "  steps              : ${STEPS}"
echo "  peak LR            : ${LEARNING_RATE}"
echo "  cosine final frac  : ${COSINE_FINAL_FRAC}"
echo "  warmup steps/frac  : ${WARMUP_STEPS} / ${WARMUP_FRAC}  (over schedule_steps=${LR_SCHEDULE_STEPS})"
echo "  adam β1,β2,ε       : ${ADAM_B1}, ${ADAM_B2}, ${ADAM_EPS}"
echo "  weight decay       : ${ADAM_WD} (excluded for token_embedder/embedding)"
echo "  grad clip          : ${GRAD_CLIP}"
echo "  z-loss multiplier  : ${Z_LOSS}"
echo "  ckpt period        : ${CHECKPOINT_PERIOD}"
echo

# Note on dtypes: relies on base.yml defaults (dtype=bfloat16,
# weight_dtype=float32) which match AI2's OLMo-core mixed-precision setup —
# bf16 forward/backward, fp32 master weights, fp32 Adam state. Overriding
# weight_dtype=bfloat16 silently demotes Adam's m,v to bf16 and loses a
# fraction of every tiny early-warmup update.
python -m maxtext.trainers.pre_train.train \
  "${MAXTEXT_ROOT}/src/maxtext/configs/base.yml" \
  model_name=olmo3-7b-pt \
  run_name="${RUN_NAME}" \
  base_output_directory="${OUTPUT_DIR}" \
  load_parameters_path="${LOAD_PARAMETERS_PATH}" \
  dataset_type=olmo_grain \
  olmo_index_path="${INDEX_PATH}" \
  olmo_path_remap_from="${GCS_BASE}" \
  olmo_path_remap_to="${LOCAL_MOUNT}" \
  olmo_apply_ngram_filter=True \
  data_shuffle_seed="${DATA_SEED}" \
  per_device_batch_size="${PER_DEVICE_BATCH}" \
  gradient_accumulation_steps="${GRAD_ACCUM_STEPS}" \
  max_target_length="${SEQ_LEN}" \
  steps="${STEPS}" \
  learning_rate="${LEARNING_RATE}" \
  learning_rate_schedule_steps="${LR_SCHEDULE_STEPS}" \
  learning_rate_final_fraction="${COSINE_FINAL_FRAC}" \
  warmup_steps_fraction="${WARMUP_FRAC}" \
  adam_b1="${ADAM_B1}" \
  adam_b2="${ADAM_B2}" \
  adam_eps="${ADAM_EPS}" \
  adam_weight_decay="${ADAM_WD}" \
  adamw_mask='["token_embedder/embedding"]' \
  gradient_clipping_threshold="${GRAD_CLIP}" \
  z_loss_multiplier="${Z_LOSS}" \
  cast_logits_to_fp32=True \
  logits_dot_in_fp32=True \
  skip_step_on_spikes=True \
  enable_checkpointing=True \
  async_checkpointing=True \
  checkpoint_period="${CHECKPOINT_PERIOD}" \
  save_checkpoint_on_completion=True \
  tokenizer_type=huggingface \
  tokenizer_path="${TOKENIZER_PATH}" \
  remat_policy=custom \
  decoder_layer_input=device \
  context=device \
  query_proj=device \
  key_proj=device \
  value_proj=device \
  qkv_proj=device \
  out_proj=device \
  mlpwi_0=device \
  mlpwo=device \
  use_iota_embed=True \
  use_tokamax_splash=True \
  attention=flash \
  sa_block_q=2048 \
  sa_block_kv=2048 \
  sa_block_kv_compute=2048 \
  sa_block_q_dkv=2048 \
  sa_block_kv_dkv=2048 \
  sa_block_kv_dkv_compute=2048 \
  sa_q_layout=SEQ_MINOR \
  sa_k_layout=SEQ_MINOR \
  sa_v_layout=HEAD_DIM_MINOR \
  sa_use_fused_bwd_kernel=True
