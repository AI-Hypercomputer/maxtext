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
# Baseline recipe runner: Flint-40B (RecursiveSI/flint) on 64-chip TPU v7x
# (tpu7x:4x4x4, 128 TensorCores). Defaults to the 8K setup (S=8192, GBS=1024,
# PDB=8, FSDP=128, CP=1, EP=1); derived recipes override only what differs.
#
# Run the same command on every TPU VM host of the slice:
#   bash flint_64c_<recipe>.sh             # train + benchmark (benchmark_train)
#   bash flint_64c_<recipe>.sh --aot       # offline AOT compile + HBM/HLO audit
#   bash flint_64c_<recipe>.sh --dry-run   # print the resolved settings
#   bash flint_64c_<recipe>.sh --data=c4   # pre-tokenized C4 (set C4_DATA_DIR)
# ==============================================================================

MODE="${MODE:-train}"
PASSTHROUGH_ARGS=()
for arg in "$@"; do
  case "${arg}" in
    --aot) MODE="aot" ;;
    --dry-run) MODE="dry_run" ;;
    --data=*) export DATASET_TYPE="${arg#--data=}" ;;
    *) PASSTHROUGH_ARGS+=("${arg}") ;;
  esac
done

export JOB_NAME="${JOB_NAME:-flint-40b-64c-8k}"
export TOPOLOGY="${TOPOLOGY:-tpu7x:4x4x4}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/flint40b/${JOB_NAME}}"
export C4_DATA_DIR="${C4_DATA_DIR:-}"

export DATASET_TYPE="${DATASET_TYPE:-synthetic}"
if [[ "${DATASET_TYPE}" == "c4" ]]; then
  export STEPS="${STEPS:-60}" LEARNING_RATE="${LEARNING_RATE:-3e-5}" C4_SHUFFLE_SEED="${C4_SHUFFLE_SEED:-42}"
else
  export STEPS="${STEPS:-18}" LEARNING_RATE="${LEARNING_RATE:-1e-9}" C4_SHUFFLE_SEED="${C4_SHUFFLE_SEED:--1}"
fi
export BENCH_WARMUP_STEPS="${BENCH_WARMUP_STEPS:-4}" PROFILE_STEPS="${PROFILE_STEPS:-3}"
export PROFILE_ALL_PROCESSES="${PROFILE_ALL_PROCESSES:-false}"
export SEED="${SEED:-42}" DUMP_XLA="${DUMP_XLA:-false}" ENABLE_OPTIMIZED_STACK="${ENABLE_OPTIMIZED_STACK:-true}"

export SEQ_LEN="${SEQ_LEN:-8192}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1024}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
export FSDP_SIZE="${FSDP_SIZE:-128}"
export CP_SIZE="${CP_SIZE:-1}"
export EP_SIZE="${EP_SIZE:-1}"
# Extra user compiler flags, appended after the tuned TPU v7x flags that the
# Python entrypoints add when OPTIMIZED_LIBTPU_ARGS=true.
export EXTRA_LIBTPU_ARGS="${EXTRA_LIBTPU_ARGS:-}"
export PARAM_DTYPE="${PARAM_DTYPE:-float32}"
export COMPUTE_DTYPE="${COMPUTE_DTYPE:-bfloat16}"
export REMAT_POLICY="${REMAT_POLICY:-auto}"
export SAVE_EXTRA_SELECTIVE_RESIDUALS="${SAVE_EXTRA_SELECTIVE_RESIDUALS:-auto}"
export MOE_IMPL="${MOE_IMPL:-pallas_gmm_v2}"
export SC_COLLECTIVE_MODE="${SC_COLLECTIVE_MODE:-dual_sc}"
export ROUTED_W_TILED="${ROUTED_W_TILED:-auto}"
export MOE_BIJECTIVE_PERMUTE="${MOE_BIJECTIVE_PERMUTE:-true}"
export MOE_SLOT_MAJOR_PACK="${MOE_SLOT_MAJOR_PACK:-false}"
export MOE_ZERO_SCATTER_ROUTER="${MOE_ZERO_SCATTER_ROUTER:-true}"
export USE_SC_MOE_GATHER_REDUCE="${USE_SC_MOE_GATHER_REDUCE:-true}"
export USE_SC_COUNTING_SORT="${USE_SC_COUNTING_SORT:-true}"
export USE_PALLAS_ROUTER_TOPK="${USE_PALLAS_ROUTER_TOPK:-true}"
export COMPACT_ROUTER_MAPS="${COMPACT_ROUTER_MAPS:-true}"
export SKIP_ROUTER_AUX_STATS_WHEN_ZERO_COEF="${SKIP_ROUTER_AUX_STATS_WHEN_ZERO_COEF:-true}"
export REPLICATE_ROUTER_GATE="${REPLICATE_ROUTER_GATE:-true}"
export FUSE_LAYER_PROJECTIONS="${FUSE_LAYER_PROJECTIONS:-true}"
export PRECOMPUTE_ROPE_TABLES="${PRECOMPUTE_ROPE_TABLES:-true}"
export PREFETCH_ROUTED_WEIGHTS="${PREFETCH_ROUTED_WEIGHTS:-true}"
export TRANSPOSE_ROUTED_DOWN_AG="${TRANSPOSE_ROUTED_DOWN_AG:-true}"
export TRANSPOSE_DENSE_RHS_AG="${TRANSPOSE_DENSE_RHS_AG:-true}"
export NUM_VOCAB_CHUNKS="${NUM_VOCAB_CHUNKS:-1}"
export VOCAB_TOKEN_CHUNK_SIZE="${VOCAB_TOKEN_CHUNK_SIZE:-2048}"
export USE_FUSED_VOCAB_VJP_SCAN="${USE_FUSED_VOCAB_VJP_SCAN:-auto}"
export SPLASH_GLOBAL_BLOCK="${SPLASH_GLOBAL_BLOCK:-auto}"
export SPLASH_DIAG_SKIP="${SPLASH_DIAG_SKIP:-false}"
export SPLASH_FWD_STRIP_SCHEDULE="${SPLASH_FWD_STRIP_SCHEDULE:-auto}"
export SPLASH_COMPACT_LSE_RESIDUAL="${SPLASH_COMPACT_LSE_RESIDUAL:-true}"
export SPLASH_DKV_HEAD_STITCH="${SPLASH_DKV_HEAD_STITCH:-false}"
export SPLASH_GLOBAL_DKV_HEAD_STITCH="${SPLASH_GLOBAL_DKV_HEAD_STITCH:-auto}"
export SPLASH_USE_LOCAL_HALO_CP="${SPLASH_USE_LOCAL_HALO_CP:-true}"
export SPLASH_IGNORE_SEGMENT_IDS="${SPLASH_IGNORE_SEGMENT_IDS:-auto}"
export SPLASH_USE_BASE2_EXP="${SPLASH_USE_BASE2_EXP:-true}"
export USE_FP8="${USE_FP8:-auto}"
export USE_LB_CP_LAYOUT="${USE_LB_CP_LAYOUT:-true}"
export EARLY_GLOBAL_KV_PROJ="${EARLY_GLOBAL_KV_PROJ:-true}"
export USE_PALLAS_LB_KV_REORDER="${USE_PALLAS_LB_KV_REORDER:-true}"
export USE_FUSED_QK_PROLOGUE="${USE_FUSED_QK_PROLOGUE:-true}"
if [[ -z "${EXTRA_CONFIG_OVERRIDES_JSON:-}" ]]; then
  export EXTRA_CONFIG_OVERRIDES_JSON="{}"
fi

flint_bool() {
  local raw="${1//[[:space:]]/}"
  case "${raw,,}" in
    1|true|yes) printf 'true' ;;
    *) printf 'false' ;;
  esac
}

OPTIMIZED_LIBTPU_ARGS="$(flint_bool "${OPTIMIZED_LIBTPU_ARGS:-${ENABLE_OPTIMIZED_STACK}}")"
export OPTIMIZED_LIBTPU_ARGS

flint_auto_bool() {
  local raw="${1//[[:space:]]/}"
  if [[ "${raw,,}" == "auto" ]]; then
    printf '%s' "${2}"
  else
    flint_bool "${raw}"
  fi
}

build_config_overrides_json() {
  local s=$(( SEQ_LEN )) gbs=$(( GLOBAL_BATCH_SIZE )) ga=$(( GRAD_ACCUM_STEPS ))
  local fsdp=$(( FSDP_SIZE )) cp=$(( CP_SIZE )) ep=$(( EP_SIZE ))
  local raw_remat="${REMAT_POLICY//[[:space:]]/}" raw_moe="${MOE_IMPL//[[:space:]]/}"
  local raw_sc="${SC_COLLECTIVE_MODE//[[:space:]]/}" raw_p="${PARAM_DTYPE//[[:space:]]/}"
  local raw_c="${COMPUTE_DTYPE//[[:space:]]/}" raw_g="${SPLASH_GLOBAL_BLOCK//[[:space:]]/}"
  local opt_enabled resolved_remat sel_dflt fp8_dflt tiled_dflt diag_dflt d_skip joined
  opt_enabled="$(flint_bool "${ENABLE_OPTIMIZED_STACK}")"
  local tok_per_tc=$(( ((gbs / ga) * s) / (fsdp * cp * ep) ))
  if [[ "${raw_remat}" == "auto" ]]; then
    if [[ "${opt_enabled}" == "true" ]]; then
      if (( tok_per_tc <= 16384 && fsdp >= 64 && ep == 1 )); then
        resolved_remat="selective"
      else
        resolved_remat="global_selective"
      fi
    else
      resolved_remat="layer"
    fi
  else
    resolved_remat="${raw_remat}"
  fi
  local -a fields=(
    "\"seq_len\": ${s}" "\"global_batch_size\": ${gbs}" "\"grad_accum_steps\": ${ga}"
    "\"fsdp_size\": ${fsdp}" "\"cp_size\": ${cp}" "\"ep_size\": ${ep}"
  )
  if [[ "${opt_enabled}" == "true" ]]; then
    local local_q gbq gbkv gbkv_c gbq_dkv gbkv_dkv gbkv_dkv_c st_q st_dkv
    local_q=$(( cp > 1 ? s / (2 * cp) : s ))
    if [[ "${raw_g,,}" == "auto" ]]; then
      gbq=$(( local_q < 512 ? local_q : 512 ))
      gbkv=$(( s < 2048 ? s : 2048 ))
      gbkv_c=$(( gbkv < 512 ? gbkv : 512 ))
      gbq_dkv=$(( local_q < 256 ? local_q : 256 ))
      gbkv_dkv=$(( s < 4096 ? s : 4096 ))
      gbkv_dkv_c=$(( gbkv_dkv < 512 ? gbkv_dkv : 512 ))
    else
      gbq=$(( raw_g ))
      gbkv=$(( raw_g ))
      gbkv_c=$(( raw_g ))
      gbq_dkv=$(( raw_g ))
      gbkv_dkv=$(( raw_g ))
      gbkv_dkv_c=$(( raw_g ))
    fi
    if [[ "${resolved_remat}" == "selective" ]]; then
      sel_dflt="true"
    else
      sel_dflt="false"
    fi
    if [[ "${resolved_remat}" == "selective" || "${resolved_remat}" == "global_selective" ]]; then
      fp8_dflt="true"
    else
      fp8_dflt="false"
    fi
    if (( (2048 / (fsdp * cp)) % 8 == 0 )); then
      tiled_dflt="true"
    else
      tiled_dflt="false"
    fi
    if (( gbq >= 256 )); then
      diag_dflt="true"
    else
      diag_dflt="false"
    fi
    d_skip="$(flint_auto_bool "${SPLASH_DIAG_SKIP}" "${diag_dflt}")"
    local g_stitch_dflt fused_vocab_dflt
    if [[ "${d_skip}" == "true" ]]; then
      st_q=1
      st_dkv=1
      g_stitch_dflt="false"
    else
      st_q=8
      st_dkv=4
      g_stitch_dflt="true"
    fi
    if (( NUM_VOCAB_CHUNKS == 1 )); then
      fused_vocab_dflt="true"
    else
      fused_vocab_dflt="false"
    fi
    fields+=(
      "\"remat_policy\": \"${resolved_remat}\""
      "\"save_extra_selective_residuals\": $(flint_auto_bool "${SAVE_EXTRA_SELECTIVE_RESIDUALS}" "${sel_dflt}")"
      "\"moe_impl\": \"${raw_moe}\""
      "\"sc_collective_mode\": \"${raw_sc}\""
      "\"routed_w_tiled\": $(flint_auto_bool "${ROUTED_W_TILED}" "${tiled_dflt}")"
      "\"moe_bijective_permute\": $(flint_bool "${MOE_BIJECTIVE_PERMUTE}")"
      "\"fuse_layer_projections\": $(flint_bool "${FUSE_LAYER_PROJECTIONS}")"
      "\"num_vocab_chunks\": $(( NUM_VOCAB_CHUNKS ))"
      "\"vocab_token_chunk_size\": $(( VOCAB_TOKEN_CHUNK_SIZE ))"
      "\"use_fused_vocab_vjp_scan\": $(flint_auto_bool "${USE_FUSED_VOCAB_VJP_SCAN}" "${fused_vocab_dflt}")"
      "\"precompute_rope_tables\": $(flint_bool "${PRECOMPUTE_ROPE_TABLES}")"
      "\"splash_block_q\": 512"
      "\"splash_block_kv\": 512"
      "\"splash_block_kv_compute\": 256"
      "\"splash_block_q_dkv\": 512"
      "\"splash_block_kv_dkv\": 512"
      "\"splash_block_kv_dkv_compute\": 512"
      "\"splash_local_dq_reduction_steps\": 1"
      "\"splash_num_stacked_q_heads\": 8"
      "\"splash_num_stacked_q_heads_dkv\": 8"
      "\"splash_use_scheduler\": true"
      "\"splash_use_scheduler_dkv\": false"
      "\"splash_dkv_reuse_scaled_q\": true"
      "\"splash_use_local_halo_cp\": $(flint_bool "${SPLASH_USE_LOCAL_HALO_CP}")"
      "\"splash_ignore_segment_ids\": $(flint_auto_bool "${SPLASH_IGNORE_SEGMENT_IDS}" "true")"
      "\"splash_fwd_strip_schedule\": \"${SPLASH_FWD_STRIP_SCHEDULE//[[:space:]]/}\""
      "\"splash_compact_lse_residual\": $(flint_bool "${SPLASH_COMPACT_LSE_RESIDUAL}")"
      "\"splash_dkv_head_stitch\": $(flint_bool "${SPLASH_DKV_HEAD_STITCH}")"
      "\"splash_global_dkv_head_stitch\": $(flint_auto_bool "${SPLASH_GLOBAL_DKV_HEAD_STITCH}" "${g_stitch_dflt}")"
      "\"splash_global_block_q\": ${gbq}"
      "\"splash_global_block_kv\": ${gbkv}"
      "\"splash_global_block_kv_compute\": ${gbkv_c}"
      "\"splash_global_block_q_dkv\": ${gbq_dkv}"
      "\"splash_global_block_kv_dkv\": ${gbkv_dkv}"
      "\"splash_global_block_kv_dkv_compute\": ${gbkv_dkv_c}"
      "\"splash_global_num_stacked_q_heads\": ${st_q}"
      "\"splash_global_num_stacked_q_heads_dkv\": ${st_dkv}"
      "\"splash_global_use_scheduler\": true"
      "\"splash_global_use_scheduler_dkv\": false"
      "\"splash_global_dkv_reuse_scaled_q\": true"
      "\"splash_qk_diag_skip\": ${d_skip}"
      "\"splash_sv_diag_skip\": ${d_skip}"
      "\"splash_use_base2_exp\": $(flint_bool "${SPLASH_USE_BASE2_EXP}")"
      "\"replicate_router_gate\": $(flint_bool "${REPLICATE_ROUTER_GATE}")"
      "\"moe_slot_major_pack\": $(flint_bool "${MOE_SLOT_MAJOR_PACK}")"
      "\"prefetch_routed_weights\": $(flint_bool "${PREFETCH_ROUTED_WEIGHTS}")"
      "\"transpose_routed_down_ag\": $(flint_bool "${TRANSPOSE_ROUTED_DOWN_AG}")"
      "\"transpose_dense_rhs_ag\": $(flint_bool "${TRANSPOSE_DENSE_RHS_AG}")"
      "\"moe_zero_scatter_router\": $(flint_bool "${MOE_ZERO_SCATTER_ROUTER}")"
      "\"use_sc_moe_gather_reduce\": $(flint_bool "${USE_SC_MOE_GATHER_REDUCE}")"
      "\"use_sc_counting_sort\": $(flint_bool "${USE_SC_COUNTING_SORT}")"
      "\"use_pallas_router_topk\": $(flint_bool "${USE_PALLAS_ROUTER_TOPK}")"
      "\"compact_router_maps\": $(flint_bool "${COMPACT_ROUTER_MAPS}")"
      "\"skip_router_aux_stats_when_zero_coef\": $(flint_bool "${SKIP_ROUTER_AUX_STATS_WHEN_ZERO_COEF}")"
      "\"use_lb_cp_layout\": $(flint_bool "${USE_LB_CP_LAYOUT}")"
      "\"early_global_kv_proj\": $(flint_bool "${EARLY_GLOBAL_KV_PROJ}")"
      "\"use_pallas_lb_kv_reorder\": $(flint_bool "${USE_PALLAS_LB_KV_REORDER}")"
      "\"use_fused_qk_prologue\": $(flint_bool "${USE_FUSED_QK_PROLOGUE}")"
      "\"use_fp8\": $(flint_auto_bool "${USE_FP8}" "${fp8_dflt}")"
    )
  elif [[ "${resolved_remat}" != "layer" ]]; then
    fields+=("\"remat_policy\": \"${resolved_remat}\"")
  fi
  if [[ "${opt_enabled}" != "true" && "$(flint_auto_bool "${USE_FP8}" "false")" == "true" ]]; then
    fields+=("\"use_fp8\": true")
  fi
  if [[ "${raw_p}" != "float32" ]]; then fields+=("\"param_dtype\": \"${raw_p}\""); fi
  if [[ "${raw_c}" != "bfloat16" ]]; then fields+=("\"compute_dtype\": \"${raw_c}\""); fi
  local extra_trim="${EXTRA_CONFIG_OVERRIDES_JSON#"${EXTRA_CONFIG_OVERRIDES_JSON%%[![:space:]]*}"}"
  extra_trim="${extra_trim%"${extra_trim##*[![:space:]]}"}"
  if [[ "${extra_trim}" != "{}" ]]; then
    if [[ "${extra_trim}" != \{*\} ]]; then
      echo "EXTRA_CONFIG_OVERRIDES_JSON must be a JSON object, got: ${EXTRA_CONFIG_OVERRIDES_JSON}" >&2
      return 1
    fi
    local extra_body="${extra_trim:1:-1}"
    if [[ -n "${extra_body//[[:space:]]/}" ]]; then fields+=("${extra_body}"); fi
  fi
  joined="$(printf ',%s' "${fields[@]}")"
  printf '{%s}' "${joined:1}"
}

if [[ -z "${CONFIG_OVERRIDES_JSON:-}" ]]; then
  CONFIG_OVERRIDES_JSON="$(build_config_overrides_json)"
  export CONFIG_OVERRIDES_JSON
fi

if [[ "${MODE}" == "dry_run" ]]; then
  printf "JOB_NAME=%s\nDATASET_TYPE=%s\nC4_SHUFFLE_SEED=%s\nSTEPS=%s\nLEARNING_RATE=%s\nCONFIG_OVERRIDES_JSON=%s\nEXTRA_LIBTPU_ARGS=%s\nOPTIMIZED_LIBTPU_ARGS=%s\n" \
    "${JOB_NAME}" "${DATASET_TYPE}" "${C4_SHUFFLE_SEED}" "${STEPS}" "${LEARNING_RATE}" "${CONFIG_OVERRIDES_JSON}" "${EXTRA_LIBTPU_ARGS}" "${OPTIMIZED_LIBTPU_ARGS}"
  exit 0
fi

if [[ "${MODE}" == "aot" ]]; then
  exec python3 -m maxtext.experimental.flint40b.aot_compile \
    --exp_name="${JOB_NAME}" --out_dir="${AOT_OUT_DIR:-/tmp/flint40b/${JOB_NAME}_aot}" --topology="${TOPOLOGY}" \
    --config_overrides_json="${CONFIG_OVERRIDES_JSON}" \
    --optimized_libtpu_args="${OPTIMIZED_LIBTPU_ARGS}" --extra_libtpu_args="${EXTRA_LIBTPU_ARGS}" \
    ${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"}
fi

if [[ "${MODE}" != "train" ]]; then
  echo "Unknown MODE=${MODE} (expected train, aot, or dry_run)" >&2
  exit 1
fi

exec python3 -m maxtext.experimental.flint40b.benchmark_train \
  --output_dir="${OUTPUT_DIR}" --dataset_type="${DATASET_TYPE}" \
  --c4_data_dir="${C4_DATA_DIR}" --c4_shuffle_seed="${C4_SHUFFLE_SEED}" \
  --steps="${STEPS}" --bench_warmup_steps="${BENCH_WARMUP_STEPS}" \
  --profile_steps="${PROFILE_STEPS}" --profile_all_processes="${PROFILE_ALL_PROCESSES}" \
  --learning_rate="${LEARNING_RATE}" --seed="${SEED}" --dump_xla="${DUMP_XLA}" \
  --config_overrides_json="${CONFIG_OVERRIDES_JSON}" \
  --optimized_libtpu_args="${OPTIMIZED_LIBTPU_ARGS}" --extra_libtpu_args="${EXTRA_LIBTPU_ARGS}" \
  ${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"}
