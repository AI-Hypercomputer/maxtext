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
# Shared recipe runner: Lineage DeepSeek-V3 671B (+ 1 MTP layer) on a 64-chip
# TPU v7x slice (tpu7x:4x4x4, 128 TensorCores, 16 hosts).
#
# Defaults to the 8K recipe (SEQ_LEN=8192, PDB=1.0, GBS=128); derived scripts
# (dsv3_64c_8k.sh, dsv3_64c_16k.sh) override only the sequence-specific knobs.
#
# Run the same command on every TPU VM host of the slice:
#   bash dsv3_64c_8k.sh              # train + profile (synthetic data default)
#   bash dsv3_64c_16k.sh             # 16K context train + profile
#   bash dsv3_64c_16k.sh --aot       # offline AOT compile + HBM/cost audit
#   bash dsv3_64c_16k.sh --dry-run   # print resolved flags and command
#   DATASET_PATH=gs://YOUR_BUCKET/c4 bash dsv3_64c_16k.sh --data=c4
# Any extra `key=value` arguments are forwarded to MaxText last.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAXTEXT_ROOT="${MAXTEXT_ROOT:-$(cd "${SCRIPT_DIR}/../../../../.." && pwd)}"

MODE="${MODE:-train}"
DRY_RUN="${DRY_RUN:-false}"
PASSTHROUGH_ARGS=()
for arg in "$@"; do
  case "${arg}" in
    --aot) MODE="aot" ;;
    --dry-run) DRY_RUN="true" ;;
    --data=*) DATASET_TYPE="${arg#--data=}" ;;
    *) PASSTHROUGH_ARGS+=("${arg}") ;;
  esac
done

PYTHON_BIN="${PYTHON_BIN:-python3}"
BASE_CONFIG="${BASE_CONFIG:-${MAXTEXT_ROOT}/src/maxtext/configs/base.yml}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MAXTEXT_ROOT}/src/maxtext/assets/tokenizers/tokenizer_llama3.tiktoken}"

SEQ_LEN="${SEQ_LEN:-8192}"
RUN_NAME="${RUN_NAME:-dsv3-671b-64c-${SEQ_LEN}}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/maxtext_lineage/${RUN_NAME}}"

# Normalize dataset mode: `c4` aliases MaxText's pre-tokenized `c4_mlperf`.
RAW_DATASET_TYPE="${DATASET_TYPE:-synthetic}"
case "${RAW_DATASET_TYPE}" in
  synthetic) RESOLVED_DATASET_TYPE="synthetic" ;;
  c4|c4_mlperf) RESOLVED_DATASET_TYPE="c4_mlperf" ;;
  *)
    echo "ERROR: Unsupported DATASET_TYPE='${RAW_DATASET_TYPE}' (expected 'synthetic', 'c4', or 'c4_mlperf')." >&2
    exit 1
    ;;
esac

DATASET_PATH="${DATASET_PATH:-}"
DATASET_NAME="${DATASET_NAME:-c4/en:3.0.5}"
EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-c4/en:3.0.9}"
LOAD_PARAMETERS_PATH="${LOAD_PARAMETERS_PATH:-}"

if [[ "${DRY_RUN}" != "true" && "${MODE}" == "train" && "${RESOLVED_DATASET_TYPE}" == "c4_mlperf" && -z "${DATASET_PATH}" ]]; then
  echo "ERROR: Set DATASET_PATH (e.g. gs://YOUR_BUCKET/c4_tfds) when running with --data=c4." >&2
  exit 1
fi

# Training and profiling defaults.
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1.0}"
EVAL_PER_DEVICE_BATCH_SIZE="${EVAL_PER_DEVICE_BATCH_SIZE:-0.5}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
STEPS="${STEPS:-12}"
if [[ "${RESOLVED_DATASET_TYPE}" == "c4_mlperf" ]]; then
  LEARNING_RATE="${LEARNING_RATE:-2.4e-5}"
else
  LEARNING_RATE="${LEARNING_RATE:-1e-9}"
fi
EVAL_INTERVAL="${EVAL_INTERVAL:--1}"
EVAL_STEPS="${EVAL_STEPS:-4}"
PROFILER="${PROFILER:-xplane}"
SKIP_FIRST_N_STEPS_FOR_PROFILER="${SKIP_FIRST_N_STEPS_FOR_PROFILER:-5}"
PROFILER_STEPS="${PROFILER_STEPS:-4}"
SEED="${SEED:-1729}"

# Offline AOT compile settings (`--aot`).
COMPILE_TOPOLOGY="${COMPILE_TOPOLOGY:-tpu7x-128}"
NUM_SLICES="${NUM_SLICES:-1}"
COMPILED_TRAINSTEP_FILE="${COMPILED_TRAINSTEP_FILE:-}"

# Lineage 64-chip recipe knobs.
LINEAGE_LAYOUT_4X4X4="${LINEAGE_LAYOUT_4X4X4:-ep8_fsdp64}"
LINEAGE_QUANTIZATION="${LINEAGE_QUANTIZATION:-fp8_full_all_splash_qk_e3_combine_per_hop}"
LINEAGE_FP8_SPLASH_OUT="${LINEAGE_FP8_SPLASH_OUT:-true}"
LINEAGE_OUT_PROJ_2D="${LINEAGE_OUT_PROJ_2D:-true}"
LINEAGE_BWD_FP8_COMBINE="${LINEAGE_BWD_FP8_COMBINE:-false}"
LINEAGE_NORM_REDUCE_TC="${LINEAGE_NORM_REDUCE_TC:-false}"
LINEAGE_ROUTER_METADATA_SC0="${LINEAGE_ROUTER_METADATA_SC0:-false}"
LINEAGE_CP_EXPERT_AXES="${LINEAGE_CP_EXPERT_AXES:-false}"
LINEAGE_STREAM_PARAM_CAST="${LINEAGE_STREAM_PARAM_CAST:-true}"
LINEAGE_REMAT_DENSE_LAYERS="${LINEAGE_REMAT_DENSE_LAYERS:-true}"
LINEAGE_REMAT_MTP="${LINEAGE_REMAT_MTP:-true}"
LINEAGE_OFFLOAD_MLA_FIELDS="${LINEAGE_OFFLOAD_MLA_FIELDS:-orig_x,kv_down,splash_out}"
LINEAGE_OFFLOAD_MOE_FIELDS="${LINEAGE_OFFLOAD_MOE_FIELDS:-out_proj_out}"
LINEAGE_OFFLOAD_BANK="${LINEAGE_OFFLOAD_BANK:-false}"
LINEAGE_OFFLOAD_W_GRAD="${LINEAGE_OFFLOAD_W_GRAD:-false}"
LINEAGE_MAX_ASYNC_OVERLAP_ENTRY="${LINEAGE_MAX_ASYNC_OVERLAP_ENTRY:-false}"
LINEAGE_LOSS_CHUNK_TOKENS="${LINEAGE_LOSS_CHUNK_TOKENS:-2048}"
CAPACITY_FACTOR="${CAPACITY_FACTOR:-1.25}"
RAGGED_BUFFER_FACTOR="${RAGGED_BUFFER_FACTOR:-0.0}"

# Splash MLA block sizes and causal diagonal-skip knobs.
SA_BLOCK_Q="${SA_BLOCK_Q:-2048}"
SA_BLOCK_KV="${SA_BLOCK_KV:-2048}"
SA_BLOCK_KV_COMPUTE="${SA_BLOCK_KV_COMPUTE:-1024}"
SA_BLOCK_Q_DKV="${SA_BLOCK_Q_DKV:-2048}"
SA_BLOCK_KV_DKV="${SA_BLOCK_KV_DKV:-2048}"
SA_BLOCK_KV_DKV_COMPUTE="${SA_BLOCK_KV_DKV_COMPUTE:-2048}"
LINEAGE_DIAG_BAND_SKIP="${LINEAGE_DIAG_BAND_SKIP:-true}"
LINEAGE_DQ_REDUCTION_STEPS="${LINEAGE_DQ_REDUCTION_STEPS:-3}"
LINEAGE_QK_DIAG_GRID="${LINEAGE_QK_DIAG_GRID:-8}"

# Tuned TPU v7x compiler flags.
# At 8K (`auto`), `pipeline_memory_host_transfers=false` is fastest on real C4
# data, while `true` avoids extra synthetic-data host-staging wait.
BF16_PROPAGATION="${BF16_PROPAGATION:-true}"
RAW_PIPE_HOST="${PIPELINE_MEMORY_HOST_TRANSFERS:-auto}"
if [[ "${RAW_PIPE_HOST}" == "auto" ]]; then
  if [[ "${RESOLVED_DATASET_TYPE}" == "synthetic" ]]; then
    RESOLVED_PIPE_HOST="true"
  else
    RESOLVED_PIPE_HOST="false"
  fi
else
  RESOLVED_PIPE_HOST="${RAW_PIPE_HOST}"
fi
EXTRA_LIBTPU_ARGS="${EXTRA_LIBTPU_ARGS:-}"

XLA_FLAG_LIST=(
  "--xla_tpu_scoped_vmem_limit_kib=65536"
  "--xla_tpu_enable_offloading_copy_to_sparsecore=false"
  "--xla_tpu_enable_sparse_core_collective_offload_dense_all_to_all=false"
  "--xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false"
  "--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=500000"
  "--xla_all_gather_latency_bound_threshold_in_bytes=1000000"
  "--xla_tpu_pipeline_memory_host_transfers=${RESOLVED_PIPE_HOST}"
  "--xla_tpu_scheduler_percent_shared_memory_limit=100"
  "--xla_tpu_enable_data_parallel_all_reduce_opt=false"
  "--xla_tpu_dcn_all_reduce_combiner_threshold_bytes=134217728"
  "--xla_tpu_dvfs_p_state=7"
  "--xla_tpu_enable_sparse_core_collective_offload_ragged_all_to_all=false"
  "--xla_tpu_enable_async_ragged_all_to_all=true"
  "--xla_tpu_force_async_all_to_all=true"
)
if [[ "${BF16_PROPAGATION}" == "false" ]]; then
  XLA_FLAG_LIST+=("--xla_jf_bf16_propagation=false")
fi

COMPILE_XLA_FLAGS="${XLA_FLAG_LIST[*]}"
if [[ -n "${EXTRA_LIBTPU_ARGS}" ]]; then
  COMPILE_XLA_FLAGS="${COMPILE_XLA_FLAGS} ${EXTRA_LIBTPU_ARGS}"
fi
export LIBTPU_INIT_ARGS="${COMPILE_XLA_FLAGS} --xla_tpu_spmd_rng_bit_generator_unsafe=true"

MAXTEXT_ARGS=(
  "${BASE_CONFIG}"
  "run_name=${RUN_NAME}"
  "base_output_directory=${OUTPUT_DIR}"
  "model_name=deepseek3-671b-lineage"
  "override_model_config=true"
  "use_lineage=true"
  "use_stream_chunking=true"
  "mesh_axes=['dcn','x','y','z','core']"
  "dcn_parallelism=[1,1,1,1,1]"
  "ici_parallelism=[1,4,4,4,2]"
  "max_target_length=${SEQ_LEN}"
  "per_device_batch_size=${PER_DEVICE_BATCH_SIZE}"
  "eval_per_device_batch_size=${EVAL_PER_DEVICE_BATCH_SIZE}"
  "gradient_accumulation_steps=${GRAD_ACCUM_STEPS}"
  "first_num_dense_layers=3"
  "base_num_decoder_layers=61"
  "mtp_num_layers=1"
  "mtp_reuse_input_embedding=true"
  "n_routing_groups=8"
  "topk_routing_group=4"
  "dtype=bfloat16"
  "mu_dtype=bfloat16"
  "grad_dtype=bfloat16"
  "opt_type=adamw"
  "lr_schedule_type=cosine"
  "learning_rate=${LEARNING_RATE}"
  "learning_rate_schedule_steps=12000"
  "warmup_steps_fraction=0.00034"
  "learning_rate_final_fraction=0.000417"
  "attention=flash"
  "decoder_block=deepseek"
  "attention_type=mla"
  "use_iota_embed=false"
  "tokenizer_type=tiktoken"
  "tokenizer_path=${TOKENIZER_PATH}"
  "data_shuffle_seed=${SEED}"
  "init_weights_seed=${SEED}"
  "enable_checkpointing=false"
  "async_checkpointing=false"
  "save_checkpoint_on_completion=false"
  "save_checkpoint_on_start=false"
  "warmup_programs_in_init=false"
  "jax_distributed_initialization_timeout=900"
  "lineage_layout_4x4x4=${LINEAGE_LAYOUT_4X4X4}"
  "lineage_quantization=${LINEAGE_QUANTIZATION}"
  "lineage_fp8_splash_out=${LINEAGE_FP8_SPLASH_OUT}"
  "lineage_out_proj_2d=${LINEAGE_OUT_PROJ_2D}"
  "lineage_bwd_fp8_combine=${LINEAGE_BWD_FP8_COMBINE}"
  "lineage_norm_reduce_tc=${LINEAGE_NORM_REDUCE_TC}"
  "lineage_router_metadata_sc0=${LINEAGE_ROUTER_METADATA_SC0}"
  "lineage_cp_expert_axes=${LINEAGE_CP_EXPERT_AXES}"
  "lineage_stream_param_cast=${LINEAGE_STREAM_PARAM_CAST}"
  "lineage_remat_dense_layers=${LINEAGE_REMAT_DENSE_LAYERS}"
  "lineage_remat_mtp=${LINEAGE_REMAT_MTP}"
  "lineage_offload_mla_fields=${LINEAGE_OFFLOAD_MLA_FIELDS}"
  "lineage_offload_moe_fields=${LINEAGE_OFFLOAD_MOE_FIELDS}"
  "lineage_offload_bank=${LINEAGE_OFFLOAD_BANK}"
  "lineage_offload_w_grad=${LINEAGE_OFFLOAD_W_GRAD}"
  "lineage_max_async_overlap_entry=${LINEAGE_MAX_ASYNC_OVERLAP_ENTRY}"
  "lineage_loss_chunk_tokens=${LINEAGE_LOSS_CHUNK_TOKENS}"
  "capacity_factor=${CAPACITY_FACTOR}"
  "ragged_buffer_factor=${RAGGED_BUFFER_FACTOR}"
  "sa_block_q=${SA_BLOCK_Q}"
  "sa_block_kv=${SA_BLOCK_KV}"
  "sa_block_kv_compute=${SA_BLOCK_KV_COMPUTE}"
  "sa_block_q_dkv=${SA_BLOCK_Q_DKV}"
  "sa_block_kv_dkv=${SA_BLOCK_KV_DKV}"
  "sa_block_kv_dkv_compute=${SA_BLOCK_KV_DKV_COMPUTE}"
  "lineage_diag_band_skip=${LINEAGE_DIAG_BAND_SKIP}"
  "lineage_dq_reduction_steps=${LINEAGE_DQ_REDUCTION_STEPS}"
  "lineage_qk_diag_grid=${LINEAGE_QK_DIAG_GRID}"
  "compile_xla_flags=${COMPILE_XLA_FLAGS}"
  "dataset_type=${RESOLVED_DATASET_TYPE}"
  "steps=${STEPS}"
  "eval_interval=${EVAL_INTERVAL}"
  "profiler=${PROFILER}"
  "skip_first_n_steps_for_profiler=${SKIP_FIRST_N_STEPS_FOR_PROFILER}"
  "profiler_steps=${PROFILER_STEPS}"
  "enable_tpu_profiling_options=true"
  "tpu_num_sparse_core_tiles_to_trace=0"
  "upload_all_profiler_results=false"
)

if [[ "${RESOLVED_DATASET_TYPE}" == "c4_mlperf" ]]; then
  MAXTEXT_ARGS+=(
    "dataset_path=${DATASET_PATH}"
    "dataset_name=${DATASET_NAME}"
    "eval_dataset_name=${EVAL_DATASET_NAME}"
    "train_data_columns=['ids']"
    "eval_data_columns=['ids']"
    "tokenize_train_data=false"
    "tokenize_eval_data=false"
    "train_shard_in_read=true"
    "train_interleave_cycle_length=4"
  )
fi

if [[ -n "${LOAD_PARAMETERS_PATH}" ]]; then
  MAXTEXT_ARGS+=("load_parameters_path=${LOAD_PARAMETERS_PATH}")
fi

if (( EVAL_INTERVAL > 0 )); then
  MAXTEXT_ARGS+=(
    "eval_steps=${EVAL_STEPS}"
    "target_eval_loss=0.0"
  )
fi

if [[ "${MODE}" == "aot" ]]; then
  ENTRYPOINT="maxtext.trainers.pre_train.train_compile"
  MAXTEXT_ARGS+=(
    "compile_topology=${COMPILE_TOPOLOGY}"
    "compile_topology_num_slices=${NUM_SLICES}"
  )
  if [[ -n "${COMPILED_TRAINSTEP_FILE}" ]]; then
    MAXTEXT_ARGS+=("compiled_trainstep_file=${COMPILED_TRAINSTEP_FILE}")
  fi
else
  ENTRYPOINT="maxtext.trainers.pre_train.train"
fi

if (( ${#PASSTHROUGH_ARGS[@]} > 0 )); then
  MAXTEXT_ARGS+=("${PASSTHROUGH_ARGS[@]}")
fi

CMD=("${PYTHON_BIN}" "-m" "${ENTRYPOINT}" "${MAXTEXT_ARGS[@]}")

if [[ "${DRY_RUN}" == "true" ]]; then
  printf 'LIBTPU_INIT_ARGS=%q\n' "${LIBTPU_INIT_ARGS}"
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

exec "${CMD[@]}"
