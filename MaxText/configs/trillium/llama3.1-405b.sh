# This config will work out of the box for any number of trillium-256 slices.
#
# Command Flags:
# OUTPUT_PATH (Required, unless base_output_directory is already set in base.yml)
# DATASET_PATH (Required, unless dataset_path is already set in base.yml)
# RUN_NAME (Required, unless run_name is already set in base.yml or running with XPK/GKE)
#
# Example to invoke this script:
# bash MaxText/configs/trillium/llama3.1-405b.sh OUTPUT_PATH="gs://<your_output_path>"
#

set -e

export EXECUTABLE="train.py" # or train_compile.py
export RUN_PREFLIGHT="true"

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

# The setup accommodates two cases:
# 1) Passing the 'RUN_NAME' variable at runtime
# 2) Propagating the 'M_RUN_NAME' variable within an Airflow sweeping workflow
if [ -n "$RUN_NAME" ];
then
    export M_RUN_NAME=$RUN_NAME
fi

# Set up network optimizations
if [ "$RUN_PREFLIGHT" = "true" ]; then
    bash preflight.sh
fi


# gcloud storage cp gs://libtpu_internal/raymondzou/ghostlite/2024-11-07-00:15:07-libtpu.so ./libtpu.so

# export TPU_LIBRARY_PATH=$PWD/libtpu.so

# Questionable flags: --xla_tpu_enable_scheduler_memory_pressure_tracking=enabled 

# Train
export LIBTPU_INIT_ARGS="--megascale_use_insecure_grpc=true --xla_tpu_enable_all_experimental_scheduler_features=true --xla_tpu_host_transfer_overlap_limit=24 \
--xla_tpu_aggressive_opt_barrier_removal=ENABLED --xla_lhs_prioritize_async_depth_over_stall=ENABLED --xla_tpu_enable_ag_backward_pipelining=true --xla_should_allow_loop_variant_parameter_in_chain=ENABLED \
--xla_should_add_loop_invariant_op_in_chain=ENABLED --xla_max_concurrent_host_send_recv=100 --xla_tpu_scheduler_percent_shared_memory_limit=98 --xla_latency_hiding_scheduler_rerun=1 \
--xla_tpu_scoped_vmem_limit_kib=98304 \
--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true \
--xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true --xla_tpu_use_bundle_aware_cost_model_for_fusions=false "

export TPU_PREMAPPED_BUFFER_SIZE=4294967296

# This is for testing optimizer compute offloading

python3 MaxText/$EXECUTABLE MaxText/configs/base.yml model_name=llama3.1-405b \
  steps=30 per_device_batch_size=1 enable_checkpointing=false \
  ici_tensor_parallelism=1 dcn_fsdp_parallelism=1 optimizer_host_offload=true \
  remat_policy=custom decoder_layer_input=offload key_proj=offload value_proj=offload \
  max_target_length=8192 \
  base_output_directory=gs://runner-maxtext-logs \
  dataset_type=synthetic allow_split_physical_axes=true profiler=xplane \
  sa_block_q=1024 sa_block_q_dkv=2048 sa_block_q_dq=2048 skip_first_n_steps_for_profiler=10 profiler_steps=2

# python3 MaxText/$EXECUTABLE MaxText/configs/base.yml model_name=llama3.1-405b \
#   steps=15 per_device_batch_size=0.25 enable_checkpointing=false \
#   ici_tensor_parallelism=4 dcn_fsdp_parallelism=2 \
#   max_target_length=8192 \
#   base_output_directory=gs://runner-maxtext-logs \
#   dataset_type=synthetic allow_split_physical_axes=true profiler=xplane \
#   sa_block_q=1024 sa_block_q_dkv=2048 sa_block_q_dq=2048






# optimizer_host_offload=true
# query_proj=offload key_proj=offload value_proj=offload out_proj=offload \
# custom_mesh=hybrid_ring_64x4
# key_proj=remat value_proj=remat \
# tensors_to_save='key_proj,value_proj' tensors_to_offload='decoder_layer_input,query_proj,out_proj'

# dcn_fsdp_parallelism=2 ici_tensor_parallelism=8 ici_fsdp_transpose_parallelism=32\
# remat_policy=qkv_proj_offloaded

# --xla_tpu_decompose_all_gather_einsum=true --xla_tpu_decompose_einsum_reduce_scatter=true

# host ops flags
# --xla_tpu_enable_all_experimental_scheduler_features=true --xla_tpu_enable_scheduler_memory_pressure_tracking=ENABLED --xla_tpu_host_transfer_overlap_limit=24 --xla_tpu_aggressive_opt_barrier_removal=ENABLED --xla_lhs_prioritize_async_depth_over_stall=ENABLED --xla_tpu_enable_ag_backward_pipelining=True --xla_should_allow_loop_variant_parameter_in_chain=ENABLED --xla_should_add_loop_invariant_op_in_chain=ENABLED --xla_max_concurrent_host_send_recv=100 --xla_tpu_scheduler_percent_shared_memory_limit=98 xla_latency_hiding_scheduler_rerun=2 


# This is for testing 4 slices

# export LIBTPU_INIT_ARGS='--xla_tpu_scoped_vmem_limit_kib=98304 --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true --xla_tpu_enable_all_experimental_scheduler_features=true --xla_tpu_enable_scheduler_memory_pressure_tracking=true --xla_tpu_host_transfer_overlap_limit=24 --xla_tpu_aggressive_opt_barrier_removal=ENABLED --xla_lhs_prioritize_async_depth_over_stall=ENABLED --xla_tpu_enable_ag_backward_pipelining=true --xla_should_allow_loop_variant_parameter_in_chain=ENABLED --xla_should_add_loop_invariant_op_in_chain=ENABLED --xla_max_concurrent_host_send_recv=100 --xla_tpu_scheduler_percent_shared_memory_limit=100 --xla_latency_hiding_scheduler_rerun=2' && export JAX_PLATFORMS=tpu,cpu && export TPU_PREMAPPED_BUFFER_SIZE=4294967296 && echo 4294967296 && export ENABLE_PJRT_COMPATIBILITY=true
 
 
# #  --xla_sc_disable_megacore_partitioning=true  --xla_tpu_enable_async_collective_fusion_fuse_all_gather=false  --xla_tpu_enable_all_gather_offload_tracing=true  --xla_tpu_use_tc_device_shape_on_sc=true  --xla_tpu_enable_sparse_core_collective_offload_all_gather=true  --xla_sc_enable_instruction_fusion=false  --xla_sc_disjoint_spmem=false  --2a886c8_chip_config_name=megachip_tccontrol  --xla_tpu_enable_sparse_core_collective_offload_all_reduce=false' && export JAX_PLATFORMS=tpu,cpu && export TPU_PREMAPPED_BUFFER_SIZE=4294967296 && echo 4294967296 && export ENABLE_PJRT_COMPATIBILITY=true

# python3 MaxText/train.py MaxText/configs/base.yml per_device_batch_size=1 ici_fsdp_parallelism=64 \
# ici_tensor_parallelism=4 dcn_fsdp_parallelism=4 allow_split_physical_axes=True custom_mesh=hybrid_ring_64x4 \
# remat_policy=custom decoder_layer_input=offload query_proj=offload key_proj=offload value_proj=offload out_proj=offload \
# max_target_length=8192 attention=flash gcs_metrics=True use_iota_embed=True \
# dataset_path=gs://max-datasets-rogue dataset_type=synthetic reuse_example_batch=1 enable_checkpointing=False \
# profiler=xplane sa_block_q=1024 sa_block_q_dkv=2048 sa_block_q_dq=2048  steps=20 enable_checkpointing=false \
# model_name=llama3.1-405b base_output_directory=gs://runner-maxtext-logs 