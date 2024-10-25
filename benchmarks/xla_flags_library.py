"""XLA Flags."""

_DENSE_VMEM_LIMIT = 98304
_MOE_VMEM_LIMIT = 81920
DENSE_VMEM_LIMIT_FLAG = f" --xla_tpu_scoped_vmem_limit_kib={_DENSE_VMEM_LIMIT}"
MOE_VMEM_LIMIT_FLAG = f" --xla_tpu_scoped_vmem_limit_kib={_MOE_VMEM_LIMIT}"
CUSTOM_VMEM_LIMIT_FLAG = (
    lambda vmem_limit: f"--xla_tpu_scoped_vmem_limit_kib={vmem_limit}"
)

# Enabled by default but lets make sure.
CF_FOR_ALL_GATHER = (
    " --xla_tpu_enable_async_collective_fusion=true"
    " --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true"
    " --xla_tpu_enable_async_collective_fusion_multiple_steps=true"
    " --xla_tpu_overlap_compute_collective_tc=true"
    " --xla_enable_async_all_gather=true"
)

CF_FOR_ALL_REDUCE = (
    " --xla_tpu_enable_async_collective_fusion=true"
    " --xla_tpu_enable_async_collective_fusion_fuse_all_reduce=true"
    " --xla_tpu_enable_async_collective_fusion_multiple_steps=true"
    " --xla_tpu_overlap_compute_collective_tc=true"
    " --xla_enable_async_all_reduce=true"
)

CF_FOR_ALL_REDUCE_AND_ALL_GATHER = (
    " --xla_enable_async_all_reduce=true"
    " --xla_enable_async_all_gather=true"
    " --xla_tpu_overlap_compute_collective_tc=true"
    " --xla_tpu_enable_async_collective_fusion=true"
    " --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true"
    " --xla_tpu_enable_async_collective_fusion_multiple_steps=true"
)

#Only ready for 1D All-Gather but should support 2D soon, and
# hopefully All-Reduce soon.
ENABLE_SPARECORE_OFFLOADING_FOR_1D_ALL_GATHER = (
    " --xla_sc_disable_megacore_partitioning=true"
    " --xla_tpu_enable_async_collective_fusion_fuse_all_gather=false"
    " --xla_tpu_enable_all_gather_offload_tracing=true"
    " --xla_tpu_use_tc_device_shape_on_sc=true"
    " --xla_tpu_enable_sparse_core_collective_offload_all_gather=true"
    " --xla_sc_enable_instruction_fusion=false"
    " --xla_sc_disjoint_spmem=false"
    # Interesting flags to try:
    # " --xla_tpu_enable_offloading_gather_to_sparsecore=true"
    # " --xla_tpu_enable_offloading_reduce_to_sparsecore=true"
    # " --xla_tpu_enable_offloading_scatter_to_sparsecore=true"
)

# Better memory layout for all-reduce
LAYOUT_FOR_ALL_REDUCE_SCATTER = (
    " --xla_tpu_use_minor_sharding_for_major_trivial_input=true"
    " --xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1"
    " --xla_tpu_assign_all_reduce_scatter_layout=true"
)

# THESE FLAGS need to be verified that they do something
IDK_FLAGS_P1 = (
    " --xla_tpu_use_minor_sharding_for_major_trivial_input=true"
    " --xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1"
)

# THESE FLAGS need to be verified that they do something
IDK_FLAGS_P2 = (
    " --xla_tpu_enable_data_parallel_all_reduce_opt=true"
    " --xla_tpu_data_parallel_opt_different_sized_ops=true"
)

# Enable SDC Checker
ENABLE_SDC_CHECKER = False
SDC_CHECKER = (
    " --xla_tpu_enable_sdc_checker=true"
    " --xla_tpu_sdc_check_halt_on_detection=true"
    " --xla_tpu_sdc_replicate_llo=true --xla_tpu_sdc_check_repeat_count=5"
)
if ENABLE_SDC_CHECKER:
  ENABLE_DEBUG_LOGS = True

# Enable Debug Logs
ENABLE_DEBUG_LOGS = False
DEBUG_LOGS = {
    "TPU_STDERR_LOG_LEVEL": "0",
    "TF_CPP_MIN_LOG_LEVEL": "0",
    "TPU_MIN_LOG_LEVEL": "0",
    "TPU_VMODULE": "tpu_configuration_ops_impl=3",  # Enable TPU logging
}