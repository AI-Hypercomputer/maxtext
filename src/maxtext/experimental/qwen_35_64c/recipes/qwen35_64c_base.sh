#!/bin/bash
# Copyright 2026 Google LLC
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

set -euo pipefail

# ==============================================================================
# Baseline Recipe: Qwen3.5-397B-A17B on 64-chip TPU v7x (tpu7x:4x4x4, 128 TCs)
# - Sequence Length: 64K (S = 65,536)
# - Per-Device Batch Size: PDB = 0.25 (Global Batch Size = 32, GA = 1)
# - Mesh: FSDP = 16, EP = 8 (ep_d2d = 2, expert = 4), CP = EP = 8
# - MoE Capacity Factor: 1.25 (Ragged Splash MoE)
# All derived recipes source this baseline and only override what differs.
# ==============================================================================

export TOPOLOGY="${TOPOLOGY:-tpu7x:4x4x4}"
export JOB_NAME="${JOB_NAME:-qwen35-397b-64k-cf125-pdb025}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/qwen35_64c/${JOB_NAME}}"
export C4_DATA_DIR="${C4_DATA_DIR:-}"

# Benchmark & Optimizer runtime flags
export DATASET_TYPE="${DATASET_TYPE:-synthetic}"
export STEPS="${STEPS:-18}"
export WARMUP_STEPS="${WARMUP_STEPS:-4}"
export PROFILE_STEPS="${PROFILE_STEPS:-3}"
export LEARNING_RATE="${LEARNING_RATE:-1e-9}"
export SEED="${SEED:-42}"
export DUMP_XLA="${DUMP_XLA:-false}"
export EXTRA_LIBTPU_ARGS="${EXTRA_LIBTPU_ARGS:-}"

# Model, Sharding & Remat baseline configuration (matches Qwen35Config defaults)
export SEQ_LEN="${SEQ_LEN:-65536}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
export FSDP_SIZE="${FSDP_SIZE:-16}"
export EP_SIZE="${EP_SIZE:-8}"
export CAPACITY_FACTOR="${CAPACITY_FACTOR:-1.25}"
export SCHEDULE_MODE="${SCHEDULE_MODE:-batchsplit}"
export DECOUPLE_EP_D2D="${DECOUPLE_EP_D2D:-true}"
export SA_UNROLL_BATCH="${SA_UNROLL_BATCH:-false}"
export MOE_BWD_STASH_OVERFLOW="${MOE_BWD_STASH_OVERFLOW:-true}"
export SAVE_DECODER_INPUT_HBM="${SAVE_DECODER_INPUT_HBM:-true}"
export OFFLOAD_DECODER_INPUT="${OFFLOAD_DECODER_INPUT:-false}"
export SAVE_MOE_GATE_UP_SUBLAYERS="${SAVE_MOE_GATE_UP_SUBLAYERS:-[1, 3]}"
export OFFLOAD_MOE_GATE_UP_SUBLAYERS="${OFFLOAD_MOE_GATE_UP_SUBLAYERS:-[0, 2]}"
export SAVE_MOE_DISP_EP_SUBLAYERS="${SAVE_MOE_DISP_EP_SUBLAYERS:-[]}"
export SAVE_KEY_VALUE_PROJ="${SAVE_KEY_VALUE_PROJ:-true}"
export SAVE_GDN_RESIDUALS="${SAVE_GDN_RESIDUALS:-false}"
export OFFLOAD_GDN_QKV="${OFFLOAD_GDN_QKV:-false}"
export SAVE_OUT_PROJ_HBM="${SAVE_OUT_PROJ_HBM:-true}"
export OFFLOAD_OUT_PROJ="${OFFLOAD_OUT_PROJ:-true}"
export PEEL_TAIL_HBM_SUPERBLOCKS="${PEEL_TAIL_HBM_SUPERBLOCKS:-1}"
export SA_BLOCK_KV_DKV_COMPUTE="${SA_BLOCK_KV_DKV_COMPUTE:-512}"
if [[ -z "${EXTRA_CONFIG_OVERRIDES_JSON:-}" ]]; then
  export EXTRA_CONFIG_OVERRIDES_JSON="{}"
fi

if [[ -z "${CONFIG_OVERRIDES_JSON:-}" ]]; then
  CONFIG_OVERRIDES_JSON="$(python3 - << 'EOF'
import json
import os

def _bool(name: str) -> bool:
  return os.environ[name].strip().lower() in ("1", "true", "yes")

def _int_list(name: str) -> list[int]:
  return [int(x) for x in json.loads(os.environ[name])]

baseline = {
    "seq_len": 65536,
    "global_batch_size": 32,
    "grad_accum_steps": 1,
    "fsdp_size": 16,
    "ep_size": 8,
    "capacity_factor": 1.25,
    "schedule_mode": "batchsplit",
    "decouple_ep_d2d": True,
    "sa_unroll_batch": False,
    "moe_bwd_stash_overflow": True,
    "save_decoder_input_hbm": True,
    "offload_decoder_input": False,
    "save_moe_gate_up_sublayers": [1, 3],
    "offload_moe_gate_up_sublayers": [0, 2],
    "save_moe_disp_ep_sublayers": [],
    "save_key_value_proj": True,
    "save_gdn_residuals": False,
    "offload_gdn_qkv": False,
    "save_out_proj_hbm": True,
    "offload_out_proj": True,
    "peel_tail_hbm_superblocks": 1,
    "sa_block_kv_dkv_compute": 512,
}

current = {
    "seq_len": int(os.environ["SEQ_LEN"]),
    "global_batch_size": int(os.environ["GLOBAL_BATCH_SIZE"]),
    "grad_accum_steps": int(os.environ["GRAD_ACCUM_STEPS"]),
    "fsdp_size": int(os.environ["FSDP_SIZE"]),
    "ep_size": int(os.environ["EP_SIZE"]),
    "capacity_factor": float(os.environ["CAPACITY_FACTOR"]),
    "schedule_mode": os.environ["SCHEDULE_MODE"].strip(),
    "decouple_ep_d2d": _bool("DECOUPLE_EP_D2D"),
    "sa_unroll_batch": _bool("SA_UNROLL_BATCH"),
    "moe_bwd_stash_overflow": _bool("MOE_BWD_STASH_OVERFLOW"),
    "save_decoder_input_hbm": _bool("SAVE_DECODER_INPUT_HBM"),
    "offload_decoder_input": _bool("OFFLOAD_DECODER_INPUT"),
    "save_moe_gate_up_sublayers": _int_list("SAVE_MOE_GATE_UP_SUBLAYERS"),
    "offload_moe_gate_up_sublayers": _int_list("OFFLOAD_MOE_GATE_UP_SUBLAYERS"),
    "save_moe_disp_ep_sublayers": _int_list("SAVE_MOE_DISP_EP_SUBLAYERS"),
    "save_key_value_proj": _bool("SAVE_KEY_VALUE_PROJ"),
    "save_gdn_residuals": _bool("SAVE_GDN_RESIDUALS"),
    "offload_gdn_qkv": _bool("OFFLOAD_GDN_QKV"),
    "save_out_proj_hbm": _bool("SAVE_OUT_PROJ_HBM"),
    "offload_out_proj": _bool("OFFLOAD_OUT_PROJ"),
    "peel_tail_hbm_superblocks": int(os.environ["PEEL_TAIL_HBM_SUPERBLOCKS"]),
    "sa_block_kv_dkv_compute": int(os.environ["SA_BLOCK_KV_DKV_COMPUTE"]),
}

overrides = {k: v for k, v in current.items() if v != baseline[k]}
extra = json.loads(os.environ.get("EXTRA_CONFIG_OVERRIDES_JSON", "{}"))
overrides.update(extra)
print(json.dumps(overrides, separators=(",", ": ")))
EOF
)"
  export CONFIG_OVERRIDES_JSON
fi

MODE="${MODE:-train}"
if [[ "${1:-}" == "--aot" ]]; then
  MODE="aot"
  shift
elif [[ "${1:-}" == "--dry-run" ]]; then
  MODE="dry_run"
  shift
fi

if [[ "${MODE}" == "dry_run" ]]; then
  echo "JOB_NAME=${JOB_NAME}"
  echo "CONFIG_OVERRIDES_JSON=${CONFIG_OVERRIDES_JSON}"
  exit 0
elif [[ "${MODE}" == "aot" ]]; then
  AOT_OUT_DIR="${AOT_OUT_DIR:-/tmp/${JOB_NAME}_aot}"
  exec python3 -m maxtext.experimental.qwen_35_64c.aot_compile \
    --exp_name="${JOB_NAME}" \
    --out_dir="${AOT_OUT_DIR}" \
    --topology="${TOPOLOGY}" \
    --config_overrides_json="${CONFIG_OVERRIDES_JSON}" \
    --extra_libtpu_args="${EXTRA_LIBTPU_ARGS}" \
    "$@"
else
  exec python3 -m maxtext.experimental.qwen_35_64c.benchmark_train \
    --dataset_type="${DATASET_TYPE}" \
    --c4_data_dir="${C4_DATA_DIR}" \
    --steps="${STEPS}" \
    --warmup_steps="${WARMUP_STEPS}" \
    --profile_steps="${PROFILE_STEPS}" \
    --learning_rate="${LEARNING_RATE}" \
    --seed="${SEED}" \
    --dump_xla="${DUMP_XLA}" \
    --config_overrides_json="${CONFIG_OVERRIDES_JSON}" \
    --extra_libtpu_args="${EXTRA_LIBTPU_ARGS}" \
    --output_dir="${OUTPUT_DIR}" \
    "$@"
fi
