#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Qwen3.5-397B-A17B through MaxTextTrainingEngine, at the v7x trainer cells benchmarked with train.py.
#
# The MaxText flags are those of the 128-chip train.py benchmark job, so an engine number here and a
# train.py number there differ only in the driver and the loss.
#
# usage: qwen35_397b_engine_benchmark.sh <cell> <mode> <loss> [extra...]
#   cell   128        128 chips (tpu7x-256): FSDP32 x CP4 x EP2, micro-batch 64 x 16 = GBS 1024
#          128-mbs32  128 chips: FSDP16 x CP8 x EP2, micro-batch 32 x 32 = GBS 1024. Not FSDP32: under
#                     cp-as-ep the MoE shards the micro-batch over fsdp x expert, so FSDP32 x EP2 needs
#                     a micro-batch that is a multiple of 64.
#          256        256 chips (tpu7x-512): FSDP64 x CP4 x EP2, micro-batch 128 x 8 = GBS 1024
#   mode   aot        compile for the cell's topology on this host (CPU is enough) and report memory
#          run        execute on the TPU slice this runs on and report throughput
#   loss   sft | grpo GRPO runs with num_vocab_tiling=1: the GRPO loss needs the logits, which vocab
#                     tiling never materializes. SFT keeps the benchmark's 16.
#   extra  engine_benchmark flags (--router_replay, --steps=5, --eval_batches=2, --profile_steps=1,
#          --compute_logps_chunk_size=2048, --report_path=gs://...) and MaxText overrides (key=value), in
#          any order; an override given here replaces the preset of the same key.
#
# env:
#   PYTHON        interpreter, default python3.
#   DRY_RUN=1     print the command instead of running it.
set -euo pipefail

usage() {
  sed -n '/^# usage:/,/^# env:/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d' >&2
  exit 2
}
[[ $# -ge 3 ]] || usage
CELL=$1; MODE=$2; LOSS=$3; shift 3

case "$CELL" in
  128)       TOPOLOGY=tpu7x-256; MESH=(ici_fsdp_parallelism=32 ici_context_parallelism=4 ici_expert_parallelism=2); BATCH=(per_device_batch_size=0.25 gradient_accumulation_steps=16) ;;
  128-mbs32) TOPOLOGY=tpu7x-256; MESH=(ici_fsdp_parallelism=16 ici_context_parallelism=8 ici_expert_parallelism=2); BATCH=(per_device_batch_size=0.125 gradient_accumulation_steps=32) ;;
  256)       TOPOLOGY=tpu7x-512; MESH=(ici_fsdp_parallelism=64 ici_context_parallelism=4 ici_expert_parallelism=2); BATCH=(per_device_batch_size=0.25 gradient_accumulation_steps=8) ;;
  *) echo "unknown cell: $CELL" >&2; usage ;;
esac
case "$MODE" in aot|run) ;; *) echo "unknown mode: $MODE" >&2; usage ;; esac
case "$LOSS" in
  sft)  VOCAB_TILING=16 ;;
  grpo) VOCAB_TILING=1 ;;
  *) echo "unknown loss: $LOSS" >&2; usage ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

XLA_FLAGS_V7X=(
  --xla_tpu_use_tc_device_shape_on_sc=true
  --xla_sc_disable_megacore_partitioning=true
  --xla_tpu_enable_offloading_gather_to_sparsecore=true
  --xla_tpu_enable_sparse_core_collective_offload_all_gather=true
  --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true
  --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true
  --xla_tpu_enable_sparse_core_reduce_scatter_v2=true
  --xla_tpu_use_single_sparse_core_for_all_gather_offload=true
  --xla_tpu_enable_concurrent_sparse_core_offloading=true
  --xla_tpu_aggressive_opt_barrier_removal=true
  --xla_tpu_scoped_vmem_limit_kib=65536
  --xla_tpu_enable_sublane_major_scaling_bitcast_fusion=false
)

MODEL=(
  model_name=qwen3.5-397b-a17b base_num_decoder_layers=60 override_model_config=true scan_layers=True
  use_multimodal=false dataset_type=synthetic max_target_length=65536
  opt_type=adamw adam_weight_decay=0.0 adam_b1=0.9 adam_b2=0.999 adam_eps=1e-8 dtype=bfloat16 mu_dtype=bfloat16
  megablox=true sparse_matmul=true use_tokamax_gmm=true use_gmm_v2=true use_gmm_v2_heuristic_tiling=true
  merge_gating_gmm=false use_ring_of_experts=true use_ragged_sort=true use_custom_sort_vjp=false
  ragged_buffer_factor=2.0 use_random_routing=True num_moe_token_chunks=4 moe_chunk_barrier=false
  attention=flash use_tokamax_splash=true use_splash_scheduler=true sa_block_q=512 sa_block_kv=1024
  sa_block_kv_compute=512 sa_block_q_dkv=1024 sa_block_kv_dkv=2048 sa_block_kv_dkv_compute=1024
  sa_fuse_reciprocal=false sa_use_base2_exp=true dq_reduction_steps=3 gdn_chunk_size=64 use_gdn_kernel=true
  custom_mesh_and_rule=cp-as-ep ici_tensor_parallelism=1 context_parallel_strategy=ring
  context_parallel_load_balance=False allow_split_physical_axes=False remat_policy=custom decoder_layer_input=device
  gdn=device gdn_conv=device
  use_iota_embed=false tokenizer_type=huggingface tokenizer_path=assets/tokenizers/qwen3-tokenizer packing=false
  enable_checkpointing=false learning_rate=1e-5 "num_vocab_tiling=$VOCAB_TILING"
)

HARNESS=(); OVERRIDES=()
for arg in "$@"; do
  if [[ $arg == --* ]]; then HARNESS+=("$arg"); else OVERRIDES+=("$arg"); fi
done

if [[ $MODE == aot ]]; then
  PLATFORM=("compile_topology=$TOPOLOGY" compile_topology_num_slices=1 "compile_xla_flags=${XLA_FLAGS_V7X[*]}")
  HARNESS=(--hbm_gib_per_device=94.74 ${HARNESS[@]+"${HARNESS[@]}"})
else
  # A live run takes the XLA flags from the environment, as the train.py job did. Added, not replaced.
  export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-} ${XLA_FLAGS_V7X[*]}"
  PLATFORM=()
fi

# `${a[@]+"${a[@]}"}` rather than `"${a[@]}"` for arrays that may be empty: bash before 4.4 (macOS
# ships 3.2) treats an empty array as unset, which `set -u` turns into an error.
COMMAND=(
  "${PYTHON:-python3}" -m maxtext.training_engine.engine_benchmark "--mode=$MODE" "--loss_type=$LOSS"
  ${HARNESS[@]+"${HARNESS[@]}"}
  "$REPO_ROOT/src/maxtext/configs/base.yml" "run_name=qwen35_397b_engine_${CELL}_${LOSS}"
  "${MODEL[@]}" "${MESH[@]}" "${BATCH[@]}" ${PLATFORM[@]+"${PLATFORM[@]}"} ${OVERRIDES[@]+"${OVERRIDES[@]}"}
)
if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf '%q ' "${COMMAND[@]}"; echo
  exit 0
fi
cd "$REPO_ROOT"
exec "${COMMAND[@]}"
