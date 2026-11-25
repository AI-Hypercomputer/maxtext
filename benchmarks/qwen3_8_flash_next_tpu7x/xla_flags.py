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

"""Grouped XLA / libtpu flags for Qwen3.8-Flash-Next 64-chip TPU7x recipes.

Derived from the TPU7x compiler flag set used on 64-chip Ironwood (`tpu7x:4x4x4`),
split into A/B-testable groups with Qwen3.8-specific empirical defaults:

  * `BASE_FLAGS` enables `WORKAROUNDS + SPARSECORE_COLLECTIVES_CONCURRENT` by
    default. Isolated 64-chip TPU7x 8K A/B profiling showed that enabling
    concurrent SparseCore collective offload without the 2 MiB min-size cutoff
    saves `-36.3 ms/step` (`+2.1%` tok/s/chip; `-29.5 ms` on MoE
    `ragged_all_to_all` and `-14.5 ms` on `all-reduce-scatter`) while keeping
    Qwen3.8's narrower `emb_dim=2560` FSDP collectives offloaded to SparseCore.
  * `SPARSECORE_COLLECTIVES_2MIB_MIN_SIZE` and `SCHEDULER_AND_MEMORY` remain
    explicit opt-in groups: on 8K, the 2 MiB threshold pushes `emb_dim=2560` FSDP
    `all-gather` and `reduce-scatter` collectives back onto TensorCore, while
    `SCHEDULER_AND_MEMORY` regressed MoE custom fusions and data formatting
    (`+22 ms/step` combined).
  * Excludes `--xla_tpu_enable_offloading_{gather,scatter,copy}_to_sparsecore=false`
    from defaults; only evaluate after index-routed PLE and TC ragged MoE eliminate
    SparseCore gather/scatter contention.
  * Excludes `--xla_tpu_dvfs_p_state=7` from defaults (opt-in only where approved).
  * Keeps `--xla_tpu_enable_gather_subgraph_pipelining=false` in `WORKAROUNDS` for
    `libtpu` builds prior to the MoE gather-subgraph pipelining compiler fix;
    drop `WORKAROUNDS` on newer `libtpu` builds to pick up pipelined MoE gather
    speedups.
"""

# Workarounds for `libtpu` builds prior to the gather-subgraph pipelining fix.
WORKAROUNDS = " --xla_tpu_enable_gather_subgraph_pipelining=false"

# Group 1a (default in BASE_FLAGS): Concurrent SparseCore collective offload
# (AG / RS / AR) without a 2 MiB min-size cutoff, keeping Qwen3.8's narrower
# 2560-dim FSDP collectives eligible for SparseCore offload.
SPARSECORE_COLLECTIVES_CONCURRENT = (
    " --xla_tpu_use_tc_device_shape_on_sc=true"
    " --xla_sc_disable_megacore_partitioning=true"
    " --xla_tpu_enable_concurrent_sparse_core_offloading=true"
    " --xla_tpu_enable_sparse_core_collective_offload_all_gather=true"
    " --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true"
    " --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true"
    " --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true"
    " --xla_tpu_enable_sparse_core_reduce_scatter_v2=true"
    " --xla_tpu_use_single_sparse_core_for_all_gather_offload=false"
    " --xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false"
    " --xla_tpu_disable_sparse_core_collective_offload_remover=true"
    " --xla_tpu_enable_sparse_core_hierarchical_reduce_scatter=true"
    " --xla_tpu_enable_sparse_core_hierarchical_all_reduce=true"
    " --xla_tpu_enable_sparse_core_reduce_scatter_padding=true"
    " --xla_tpu_enable_reduce_scatter_span_buffer_utilization_optimization=true"
    " --xla_tpu_sparse_core_reduce_scatter_latency_multiplier=6"
)

# Group 1b (opt-in): 2 MiB SparseCore offload thresholds (useful for large >=4096-dim
# models, but un-offloads Qwen3.8's 2560-dim FSDP collectives on 8K).
SPARSECORE_COLLECTIVES_2MIB_MIN_SIZE = (
    " --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=2000000"
    " --xla_tpu_sparse_core_reduce_scatter_offload_min_size_in_bytes=2000000"
    " --xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=2000000"
)

# Combined Group 1 bundle matching long-context TPU7x recipes.
SPARSECORE_COLLECTIVES = SPARSECORE_COLLECTIVES_CONCURRENT + SPARSECORE_COLLECTIVES_2MIB_MIN_SIZE

# Group 2 (opt-in): Scheduler, VMEM budget, PCIe host-offload pacing, and SC prefetching.
SCHEDULER_AND_MEMORY = (
    " --xla_tpu_enable_all_experimental_scheduler_features=true"
    " --xla_tpu_nd_short_transfer_max_chunks=2048"
    " --xla_tpu_aggressive_opt_barrier_removal=false"
    " --xla_tpu_scoped_vmem_limit_kib=65472"
    " --xla_tpu_enable_sublane_major_scaling_bitcast_fusion=false"
    " --xla_tpu_pcie_bandwidth_multiplier=0.05"
    " --xla_tpu_scheduler_percent_shared_memory_limit=112"
    " --xla_jf_rematerialization_percent_shared_memory_limit=90"
    " --xla_tpu_autofdo=false"
    " --xla_sc_tile_tasks_prefetch_ahead=4"
)

# Opt-in flag keeping contiguous 2D MoE activation row gathers on TensorCore HBM DMA
# (while leaving 1D s32 scatter offload enabled on SparseCore).
DISABLE_SC_GATHER_OFFLOAD = " --xla_tpu_enable_offloading_gather_to_sparsecore=false"

# Keeps layout / transpose copies on TensorCore VMEM/HBM instead of saturating SparseCore.
DISABLE_SC_COPY_OFFLOAD = " --xla_tpu_enable_offloading_copy_to_sparsecore=false"

# Keeps synchronous <6 MB metadata / scale / small-projection all-gathers on TensorCore ICI
# while offloading >=6.55 MB FSDP weights and 335 MB EP token payloads to SparseCore.
SPARSECORE_AG_6MIB_MIN_SIZE = " --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=6000000"

# Opt-in 64 KiB SparseCore RS/AR min-size cutoff: keeps micro-collectives (<64 KiB, e.g.
# shared_expert_gate all-reduce) on TensorCore ICI while keeping >=204 KiB FSDP weight shards
# on SparseCore.
SPARSECORE_RS_AR_64KIB_MIN_SIZE = (
    " --xla_tpu_sparse_core_reduce_scatter_offload_min_size_in_bytes=65536"
    " --xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=65536"
)

# Opt-in only AFTER index-routed PLE and TC ragged MoE eliminate SC gather/scatter.
DISABLE_SC_DATA_MOVEMENT_OFFLOAD = (
    " --xla_tpu_enable_offloading_gather_to_sparsecore=false"
    " --xla_tpu_enable_offloading_scatter_to_sparsecore=false"
    " --xla_tpu_enable_offloading_copy_to_sparsecore=false"
)

# SparseCore LHS offload queueing + layer-scheduler flags (without global VMEM / HBM
# limit overrides) to overlap async SparseCore collectives (`_ep_combine_tokens`, QSA KV AG).
LHS_QUEUEING_FLAGS = (
    " --xla_tpu_enable_sparse_core_collective_aggregator=true"
    " --xla_tpu_enable_sparse_core_offload_queuing_in_lhs=true"
    " --xla_tpu_enable_latency_hiding_layer_scheduler=true"
    " --xla_tpu_enable_layer_scheduler_for_dependent_collectives=true"
    " --xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=false"
)

# Opt-in global VMEM / scheduler shared-memory overrides (excluded from default recipes).
OPT_IN_LHS_MEMORY_FLAGS = " --xla_tpu_scoped_vmem_limit_kib=65472" " --xla_tpu_scheduler_percent_shared_memory_limit=140"

# Opt-in PCIe pacing flags for `tensors_to_offload` (`pinned_host`) activation offload arms.
HOST_OFFLOAD_PACING_FLAGS = " --xla_tpu_pcie_bandwidth_multiplier=0.03" " --xla_tpu_host_transfer_overlap_limit=1"

# Empirically validated default bundle for 64-chip TPU7x recipes; opt-in groups
# above can be composed explicitly for per-sequence-length A/B sweeps.
BASE_FLAGS = WORKAROUNDS + SPARSECORE_COLLECTIVES_CONCURRENT
HIGH_TOKEN_FLAGS = BASE_FLAGS + DISABLE_SC_COPY_OFFLOAD + SPARSECORE_AG_6MIB_MIN_SIZE
