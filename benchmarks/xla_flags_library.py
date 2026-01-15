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
"""This file contains commonly-used XLA Flags."""

### NOTICE ###
# These are potential optimization flags that are used within MaxText models.
# They may or may not provide performance improvements in extended use cases
# but provide a recommendation for potential grouping of xla flags and their
# usage. Please refer to their usage for examples.

REMOVE = "remove"
ADD_SERVER = "add_server"
ADD_PROXY = "add_proxy"
ADD_WORKER = "add_worker"

# xla tpu scoped vmem defines the amount of vmem used for the current hlo op.
# The remaining vmem is used for prefetching latter op needs.
# These limits are experimentally recommended values for compute bound models.
_DENSE_VMEM_LIMIT = 98304
_MOE_VMEM_LIMIT = 81920

DENSE_VMEM_LIMIT_FLAG = f" --xla_tpu_scoped_vmem_limit_kib={_DENSE_VMEM_LIMIT}"
MOE_VMEM_LIMIT_FLAG = f" --xla_tpu_scoped_vmem_limit_kib={_MOE_VMEM_LIMIT}"
CUSTOM_VMEM_LIMIT_FLAG = "--xla_tpu_scoped_vmem_limit_kib={vmem_limit}".format

# Continuation Fusion (CF) for All Gather Collectives
# Continuation Fusion is a form of parallelizing compute work with collectives.
CF_FOR_ALL_GATHER = (
    " --xla_tpu_enable_async_collective_fusion=true"
    " --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true"
    " --xla_tpu_enable_async_collective_fusion_multiple_steps=true"
    " --xla_tpu_overlap_compute_collective_tc=true"
    " --xla_enable_async_all_gather=true"
)

# Continuation Fusion (CF) for All Reduce Collectives
# Continuation Fusion is a form of parallelizing compute work with collectives.
CF_FOR_ALL_REDUCE = (
    " --xla_tpu_enable_async_collective_fusion=true"
    " --xla_tpu_enable_async_collective_fusion_fuse_all_reduce=true"
    " --xla_tpu_enable_async_collective_fusion_multiple_steps=true"
    " --xla_tpu_overlap_compute_collective_tc=true"
    " --xla_enable_async_all_reduce=true"
)

# Continuation Fusion (CF) for All Gather and All Reduce Collectives.
# Continuation Fusion is a form of parallelizing compute work with collectives.
CF_FOR_ALL_REDUCE_AND_ALL_GATHER = (
    " --xla_enable_async_all_reduce=true"
    " --xla_enable_async_all_gather=true"
    " --xla_tpu_overlap_compute_collective_tc=true"
    " --xla_tpu_enable_async_collective_fusion=true"
    " --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true"
    " --xla_tpu_enable_async_collective_fusion_multiple_steps=true"
)


# Base Flags needed when enabling sparsecore offloading
ENABLE_SPARSECORE_OFFLOADING_BASE_FLAGS = (
    " --xla_tpu_use_tc_device_shape_on_sc=true"
    " --xla_sc_enable_instruction_fusion=false"
    " --xla_sc_disjoint_spmem=false"
    " --xla_sc_disable_megacore_partitioning=true"
    " --2a886c8_chip_config_name=megachip_tccontrol"
)


# Enable SparseCore All Gather (1D), Reduce Scatter (1D) and All Reduce (ND)
ENABLE_SPARSECORE_OFFLOADING_FOR_RS_AG_AR = (
    " --xla_tpu_enable_async_collective_fusion_fuse_all_gather=false"
    " --xla_tpu_enable_async_collective_fusion_fuse_all_reduce=false"
    " --xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=false"
    " --xla_tpu_enable_sparse_core_collective_offload_all_gather=true"
    " --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true"
    " --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true"
    " --xla_tpu_enable_all_gather_offload_tracing=true"
    " --xla_tpu_enable_reduce_scatter_offload_tracing=true"
    " --xla_tpu_enable_all_reduce_offload_tracing=true"
) + ENABLE_SPARSECORE_OFFLOADING_BASE_FLAGS

# Enable SparseCore Reduce Scatter (SC RS)
# Either one of CF or SC can be enabled at a time.
ENABLE_SPARSECORE_OFFLOADING_FOR_REDUCE_SCATTER = (
    " --xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=false"
    " --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true"
    " --xla_tpu_enable_reduce_scatter_offload_tracing=true"
) + ENABLE_SPARSECORE_OFFLOADING_BASE_FLAGS

# Enable SparseCore All Gather (SC AG).
# Either one of CF or SC can be enabled at a time.
ENABLE_SPARSECORE_OFFLOADING_FOR_ALL_GATHER = (
    " --xla_tpu_enable_async_collective_fusion_fuse_all_gather=false"
    " --xla_tpu_enable_sparse_core_collective_offload_all_gather=true"
    " --xla_tpu_enable_all_gather_offload_tracing=true"
) + ENABLE_SPARSECORE_OFFLOADING_BASE_FLAGS

# Enable SparseCore All Reduce (SC AR)
# Either one of CF or SC can be enabled at a time.
# This is useful for reducing the gradient reduction all-reduce time with
# overlapping with compute during that time.
ENABLE_SPARSECORE_OFFLOADING_FOR_ALL_REDUCE = (
    " --xla_tpu_enable_async_collective_fusion_fuse_all_reduce=false"
    " --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true"
    " --xla_tpu_enable_all_reduce_offload_tracing=true"
) + ENABLE_SPARSECORE_OFFLOADING_BASE_FLAGS

# Better memory layout for all-reduce (AR).
LAYOUT_FOR_ALL_REDUCE_SCATTER = (
    " --xla_tpu_use_minor_sharding_for_major_trivial_input=true"
    " --xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1"
    " --xla_tpu_assign_all_reduce_scatter_layout=true"
)

# Enable Reduce Scatter (RS) Fusion in certain cases by merging
# All-reduce (AR) and Dynamic-Slice (DS).
REDUCE_SCATTER_FUSION = (
    " --xla_tpu_use_minor_sharding_for_major_trivial_input=true"
    " --xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1"
)

# xla_tpu_data_parallel_opt_different_sized_ops:
#   enable pipelining of data parallel ops across multiple iterations
# xla_tpu_enable_data_parallel_all_reduce_opt:
#   optimize DCN all-reduces used for data parallel sharding
DATA_PARALLEL_OVERLAP = (
    " --xla_tpu_enable_data_parallel_all_reduce_opt=true" " --xla_tpu_data_parallel_opt_different_sized_ops=true"
)

# Enable Enhanced Launch Barrier.
# Gracefully handle dispatch failures on TPU workers. For Pathways is required
# for error propagation and out-of-order execution detection.
ENHANCED_LAUNCH_BARRIER = " --xla_tpu_use_enhanced_launch_barrier=true"

# Host offloading Flags. These are optimizations recommended when using host
# offloading.
HOST_OFFLOAD_FLAGS = (
    " --xla_tpu_enable_all_experimental_scheduler_features=true"
    " --xla_tpu_enable_scheduler_memory_pressure_tracking=true"
    " --xla_tpu_host_transfer_overlap_limit=24"
    " --xla_tpu_aggressive_opt_barrier_removal=ENABLED"
    " --xla_lhs_prioritize_async_depth_over_stall=ENABLED"
    " --xla_tpu_enable_ag_backward_pipelining=true"
    " --xla_should_allow_loop_variant_parameter_in_chain=ENABLED"
    " --xla_should_add_loop_invariant_op_in_chain=ENABLED"
    " --xla_max_concurrent_host_send_recv=100"
    " --xla_tpu_scheduler_percent_shared_memory_limit=100"
    " --xla_latency_hiding_scheduler_rerun=2"
)

# Flags to optimize pipeline parallelism over DCN with large host offloading.
PIPELINING_FLAGS = " --xla_tpu_iova_dma_chunk_size_bytes=16777216"  # breaks DMA to/from host into 16M chunks

# Disable bundle-aware CostModel which was causing worse perf b/357103386.
# Some fusions in the backward pass of the model were 3x slower without this.
DISABLE_BUNDLE_AWARE_COST_MODEL = " --xla_tpu_use_bundle_aware_cost_model_for_fusions=false"

# Enable Silent Data Corruption (SDC) Checker
# SDC Checker will check for chip / ici / hardware corruption events.
# Below is a configuration of finding non-deterministic failures by
# rerunning the HLO and LLO and verifying the results match each time.
# Enabling this will reduce performance by a factor of 1/(repeat_count + 1).
ENABLE_SDC_CHECKER = False
SDC_CHECKER = (
    " --xla_tpu_enable_sdc_checker=true"
    " --xla_tpu_sdc_check_halt_on_detection=true"
    " --xla_tpu_sdc_replicate_llo=true --xla_tpu_sdc_check_repeat_count=5"
)
if ENABLE_SDC_CHECKER:
  ENABLE_DEBUG_LOGS = True

# Enable Debug Logs
# These are a set of debug logs recommended to be part of a workload.
ENABLE_DEBUG_LOGS = False
DEBUG_LOGS = {
    "TPU_STDERR_LOG_LEVEL": "0",
    "TF_CPP_MIN_LOG_LEVEL": "0",
    "TPU_MIN_LOG_LEVEL": "0",
    "TPU_VMODULE": "tpu_configuration_ops_impl=3",  # Enable TPU logging
}

# Disables collective matmul operations.
DISABLE_COLLECTIVE_MATMUL = " --xla_jf_spmd_threshold_for_windowed_einsum_mib=1000000"
# Disable "megacore fusion allow ags".
DISABLE_MEGACORE_FUSION_ALLOW_AGS = " --xla_tpu_megacore_fusion_allow_ags=false"
# Enable async collective permute.
ENABLE_ASYNC_COLLECTIVE_PERMUTE = " --xla_enable_async_collective_permute=true"
