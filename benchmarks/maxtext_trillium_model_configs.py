"""
 Copyright 2024 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """
"""Shared Benchmark config for v6e orchestrations."""

import dataclasses
import os.path
import typing
from tempfile import gettempdir
#from benchmarks import xla_flags_library

# TODO(vbarr@) Abstract software features like checkpointing,
# real data / synthetic data out of this config
# TODO(vbarr@) Make slice dependent configurations to allow for a model's tuning
# to adjust at scales.

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
    " --xla_tpu_enable_data_parallel_all_reduce_opt=true"
    " --xla_tpu_data_parallel_opt_different_sized_ops=true"
)

# Enable Enhanced Launch Barrier.
# Gracefully handle dispatch failures on TPU workers. For Pathways is required
# for error propagation and out-of-order execution detection.
ENHANCED_LAUNCH_BARRIER = (
    " --xla_tpu_use_enhanced_launch_barrier=true"
)

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
PIPELINING_FLAGS = (
    " --xla_tpu_iova_dma_chunk_size_bytes=16777216" # breaks DMA to/from host into 16M chunks
)

ASYNC_CP = (
" --xla_enable_async_collective_permute=true"
)


# Disable bundle-aware CostModel which was causing worse perf b/357103386.
# Some fusions in the backward pass of the model were 3x slower without this.
DISABLE_BUNDLE_AWARE_COST_MODEL = (
    " --xla_tpu_use_bundle_aware_cost_model_for_fusions=false"
)

ASYNC_A2A = (
    " --xla_tpu_enable_async_all_to_all=true"
)



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
DISABLE_COLLECTIVE_MATMUL = (
    " --xla_jf_spmd_threshold_for_windowed_einsum_mib=1000000"
)






# The minimum set of tuning params required for pathways. Does not include
# checkpointing params.
BASE_PATHWAYS_TUNING_PARAMS = {
    "checkpoint_storage_use_ocdbt": False,
    "checkpoint_storage_use_zarr3": False,
    "enable_pathways_goodput": True,
    "enable_single_controller": True,
    "metrics_file": "metrics.txt",
    "goodput_upload_interval_seconds": 30,
}

# The set of tuning params required for long-running pathways jobs.
PATHWAYS_LONG_RUN_CHECKPOINTING_TUNING_PARAMS = {
    "enable_checkpointing": True,
    "async_checkpointing": True,
    "checkpoint_period": 100,
    "enable_checkpoint_cloud_logger": True,
}

# The set of tuning params required for short-running pathways jobs.
PATHWAYS_SHORT_RUN_CHECKPOINTING_TUNING_PARAMS = {
    "enable_checkpointing": True,
    "async_checkpointing": True,
    "checkpoint_period": 20,
    "enable_checkpoint_cloud_logger": True,
}









@dataclasses.dataclass
class MaxTextModel:
  model_name: str
  model_type: str
  tuning_params: dict[str, typing.Any]
  xla_flags: str

  # Additional pathways tuning params as necessary. Adding
  # enable_single_controller=True to pathways_tuning_params is not necessary.
  pathways_tuning_params: dict[str, typing.Any] = None

  # XLA flags for pathways, if different from the default. Some flags may not
  # be supported by pathways e.g. "--2a886c8_chip_config_name".
  pathways_xla_flag_options: dict[str, typing.Any] = None


trillium_model_dict = {}


# Run this for new definitions that should be part of the library.
def _add_to_model_dictionary(
    model_dictionary: dict[str, MaxTextModel], maxtext_model: MaxTextModel
) -> MaxTextModel:
  model_dictionary[maxtext_model.model_name.replace("-", "_")] = maxtext_model
  return maxtext_model


# Ran with a docker built from and XPK runner ran from:
# docker_image_flag = '--docker-image="gcr.io/tpu-prod-env-multipod/mattdavidow_ep_first"'
# 
# commit 1fb44401c22c5267924513909781435536942e26 (HEAD -> mattdavidow-dream-ep-first, origin/mattdavidow-dream-ep-first)
# Author: gobbleturk <mattdavidow@google.com>
# Date:   Mon May 19 00:10:49 2025 +0000
#      add async CP

matt_dream_v1 = _add_to_model_dictionary(
  trillium_model_dict,
  MaxTextModel(
    model_name="matt_dream_v1",
    model_type="default",
    tuning_params={
        "steps": 50,
        "per_device_batch_size": 1.0,
        "remat_policy": "full",
        "max_target_length": 4096,
        "enable_checkpointing": False,
        "dataset_type": "synthetic",
        "base_output_directory": "gs://maxtext-experiments-multipod",
        "decoder_block": "mixtral",
        "ici_expert_parallelism": 64,
        "ici_tensor_parallelism": 4,
        "allow_split_physical_axes": True,
        "custom_mesh": "hybrid_ring_64x4",
        "num_experts": 64, # 256
        "num_experts_per_tok": 2,
        "base_emb_dim": 10240, #7168
        "base_mlp_dim": 40960,
        "base_num_query_heads": 64,
        "base_num_kv_heads": 16,
        "head_dim": 256,
        "skip_first_n_steps_for_profiler": 40,
        "sparse_matmul": False, # False
        "megablox": False, # True
        "capacity_factor": 1,
        "profiler": "xplane",
        "opt_type": "sgd",
        "dump_hlo": True,
        "weight_dtype": "bfloat16",
        
        # PP
        "base_num_decoder_layers": 16, # PP * 8
        "dcn_pipeline_parallelism": 2,
        "num_pipeline_microbatches": 4, # PP * 2 or since we are sad PP * 1
        "num_layers_per_pipeline_stage": 2,
        # "scan_layers": False,
    },
    xla_flags=(
        MOE_VMEM_LIMIT_FLAG
        + REDUCE_SCATTER_FUSION
        #CF_FOR_ALL_GATHER
        + LAYOUT_FOR_ALL_REDUCE_SCATTER
        + PIPELINING_FLAGS
        + ASYNC_A2A
        #+ PP_MORE_FLAGS
        + ASYNC_CP
    ),
  )
)



# Ran with a docker built from and XPK runner ran from:
# docker_image_flag = '--docker-image="gcr.io/tpu-prod-env-multipod/mattdavidow_ep_first"'
# 
# commit 1c213eb20026eb9877ebb14768295c4b0e2e1b97 (HEAD -> mattdavidow-dream-ep-first, origin/mattdavidow-dream-ep-first)
# Author: gobbleturk <mattdavidow@google.com>
# Date:   Sun May 18 20:33:59 2025 +0000

#     Add async a2a flag

matt_dream_pure_ep_v1 = _add_to_model_dictionary(
  trillium_model_dict,
  MaxTextModel(
    model_name="matt_dream_v1",
    model_type="default",
    tuning_params={
        "steps": 20,
        "per_device_batch_size": 1,
        "remat_policy": "full",
        "max_target_length": 4096,
        "enable_checkpointing": False,
        "dataset_type": "synthetic",
        "base_output_directory": "gs://maxtext-experiments-multipod",
        "decoder_block": "mixtral",
        "ici_expert_parallelism": 256,
        "num_experts": 256, # 256
        "num_experts_per_tok": 8,
        "base_emb_dim": 8192, #7168
        "base_mlp_dim": 32768,
        "base_num_query_heads": 64,
        "base_num_kv_heads": 16,
        "head_dim": 256,
        "skip_first_n_steps_for_profiler": 12,
        "sparse_matmul": False, # False
        "megablox": False, # True
        "capacity_factor": 1,
        "profiler": "xplane",
        "opt_type": "sgd",
        "weight_dtype": "bfloat16",
        "dump_hlo": True,
        
        
        # PP
        "base_num_decoder_layers": 4, # PP * 8
        # "dcn_pipeline_parallelism": 2,
        # "num_pipeline_microbatches": 2, # PP * 2 or since we are sad PP * 1
        # "num_layers_per_pipeline_stage": 2,
        # "scan_layers": False,
    },
    xla_flags=(
        MOE_VMEM_LIMIT_FLAG
        + REDUCE_SCATTER_FUSION
        #CF_FOR_ALL_GATHER
        + LAYOUT_FOR_ALL_REDUCE_SCATTER
        + PIPELINING_FLAGS
        #+ PP_MORE_FLAGS
    ),
  )
)