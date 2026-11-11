# Copyright 2023–2026 Google LLC
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

"""XLA / libtpu flags for the DeepSeek-V3 671B FP8 TPU7x recipes.

Flags shared by every scale are grouped by purpose; the scale-dependent ones
are separate constants. Recipes in `config.py` concatenate the groups they use.
"""

# Async collectives: all-gather / all-reduce / reduce-scatter scheduling.
ASYNC_COLLECTIVES = (
    " --xla_msa_enable=false"
    " --xla_enable_async_all_gather=true"
    " --xla_enable_async_all_reduce=true"
    " --xla_jf_single_phase_ring_max_kib=256"
    " --xla_max_concurrent_async_all_gathers=2"
    " --xla_tpu_prefer_async_allgather_to_allreduce=true"
    " --xla_tpu_enable_ag_backward_pipelining=true"
    " --xla_tpu_enable_async_collective_fusion_fuse_all_gather=false"
    " --xla_tpu_enable_cross_module_binomial_all_reduce=true"
    " --xla_tpu_small_all_reduce_binomial_threshold_bytes=262144"
    " --xla_tpu_enable_3d_reduce_scatter_decomposer=false"
    " --xla_tpu_aggressive_opt_barrier_removal=true"
)

# Layer scheduler instead of the latency-hiding layer scheduler.
SCHEDULER = (
    " --xla_tpu_enable_latency_hiding_layer_scheduler=false"
    " --xla_tpu_enable_layer_scheduler_for_dependent_collectives=true"
    " --xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=true"
    " --xla_tpu_rerun_latency_hiding_scheduler_post_sc_assignment=true"
    " --xla_tpu_scheduler_percent_shared_memory_limit=150"
)

# SparseCore collective offload (all-gather, all-reduce, reduce-scatter).
SPARSECORE_OFFLOAD = (
    " --xla_sc_disable_megacore_partitioning=True"
    " --xla_tpu_use_tc_device_shape_on_sc=True"
    " --xla_tpu_enable_concurrent_sparse_core_offloading=true"
    " --xla_tpu_enable_offloading_copy_to_sparsecore=false"
    " --xla_tpu_enable_offloading_gather_to_sparsecore=false"
    " --xla_tpu_enable_sparse_core_collective_aggregator=true"
    " --xla_tpu_enable_sparse_core_collective_offload_all_gather=true"
    " --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true"
    " --xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true"
    " --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true"
    " --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true"
    " --xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true"
    " --xla_tpu_enable_sparse_core_hierarchical_all_reduce=true"
    " --xla_tpu_enable_sparse_core_offload_queuing_in_lhs=true"
    " --xla_tpu_enable_sparse_core_reduce_scatter_v2=true"
    " --xla_tpu_sparse_core_all_gather_latency_multiplier=1"
    " --xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3"
    " --xla_tpu_use_single_sparse_core_for_all_gather_offload=true"
    " --xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false"
)

# Chip, memory and debug settings.
CHIP = (
    " --xla_tpu_bf16_emission_mode=NATIVE_EMISSION"
    " --xla_tpu_dvfs_p_state=7"
    " --xla_tpu_pcie_bandwidth_multiplier=0.03"
    " --xla_tpu_scoped_vmem_limit_kib=65472"
    " --xla_xprof_register_llo_debug_info=false"
)

# Flags common to all scales (2k, 4k, 8k).
BASE_FLAGS = ASYNC_COLLECTIVES + SCHEDULER + SPARSECORE_OFFLOAD + CHIP

# --- Scale-dependent flags ---

# All-reduce offload on one SparseCore (2k, 4k) or split across both (8k).
SINGLE_SPARSE_CORE_ALL_REDUCE = " --xla_tpu_use_single_sparse_core_for_all_reduce_offload=true"
SPLIT_SPARSE_CORE_ALL_REDUCE = " --xla_tpu_use_single_sparse_core_for_all_reduce_offload=false"

# 40 MiB all-reduce combiner thresholds (8k).
ALL_REDUCE_COMBINER_40MIB = (
    " --xla_tpu_arf_combiner_threshold_in_bytes=41943040" " --xla_jf_crs_combiner_threshold_in_bytes=41943040"
)
