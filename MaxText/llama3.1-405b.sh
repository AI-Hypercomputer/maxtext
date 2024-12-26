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

# pip installing libtpu from a certain date
pip install libtpu-nightly==0.1.dev20241206 -f https://storage.googleapis.com/libtpu-releases/index.html 

# VMEM flag
LIBTPU_FLAGS="--xla_tpu_scoped_vmem_limit_kib=98304"
# GRPC flags
# LIBTPU_FLAGS+=" --megascale_grpc_premap_memory_bytes=17179869184 --xla_tpu_enable_sunk_dcn_allreduce_done_with_host_reduction=true"
# All reduce scatter
# LIBTPU_FLAGS+=" xla_tpu_use_minor_sharding_for_major_trivial_input=true \
#     --xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1 --xla_tpu_assign_all_reduce_scatter_layout=true"
# CF for AG
LIBTPU_FLAGS+=" --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true \
    --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true \
    --xla_enable_async_all_gather=true"
# Host offloading flags
LIBTPU_FLAGS+=" --xla_tpu_enable_all_experimental_scheduler_features=true --xla_tpu_host_transfer_overlap_limit=24 \
    --xla_tpu_aggressive_opt_barrier_removal=true --xla_lhs_prioritize_async_depth_over_stall=ENABLED --xla_tpu_enable_ag_backward_pipelining=true --xla_should_allow_loop_variant_parameter_in_chain=ENABLED \
    --xla_should_add_loop_invariant_op_in_chain=ENABLED --xla_max_concurrent_host_send_recv=100 --xla_tpu_scheduler_percent_shared_memory_limit=90 --xla_latency_hiding_scheduler_rerun=2"
# SC offloading only for AG
# LIBTPU_FLAGS+=" --xla_sc_disable_megacore_partitioning=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=false \
#     --xla_tpu_enable_all_gather_offload_tracing=true --xla_tpu_use_tc_device_shape_on_sc=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true \
#     --xla_sc_enable_instruction_fusion=false --xla_sc_disjoint_spmem=false --2a886c8_chip_config_name=megachip_tccontrol"
# SC offloading only for AR
LIBTPU_FLAGS+=" --xla_sc_disable_megacore_partitioning=true --xla_tpu_enable_all_reduce_offload_tracing=false --xla_tpu_enable_async_collective_fusion_fuse_all_reduce=false \
    --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true --xla_tpu_use_tc_device_shape_on_sc=true \
    --xla_sc_enable_instruction_fusion=false --xla_sc_disjoint_spmem=false --2a886c8_chip_config_name=megachip_tccontrol"
# SC offloading for AG and AR
# LIBTPU_FLAGS+=" --xla_sc_disable_megacore_partitioning=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=false --xla_tpu_enable_all_gather_offload_tracing=false \
#     --xla_tpu_enable_all_reduce_offload_tracing=false --xla_tpu_use_tc_device_shape_on_sc=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true \
#     --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true --xla_sc_enable_instruction_fusion=false \
#     --xla_sc_disjoint_spmem=false --2a886c8_chip_config_name=megachip_tccontrol"
# CM
# LIBTPU_FLAGS+=" --xla_tpu_enable_windowed_einsum_for_all_gather=false --xla_tpu_enable_windowed_einsum_for_reduce_scatter=false \
#     --xla_tpu_decompose_all_gather_einsum=true --xla_tpu_decompose_einsum_reduce_scatter=true"

RUN_NAME=mohitkhatwani-60-slices-run4

# LIBTPU_FLAGS+=" --xla_dump_to=gs://runner-maxtext-logs/$RUN_NAME/HLO_dumps/"


export TPU_PREMAPPED_BUFFER_SIZE=17179869184
export LIBTPU_INIT_ARGS=$LIBTPU_FLAGS

# networking fix
echo "4096 41943040 314572800" > /proc/sys/net/ipv4/tcp_rmem



python3 MaxText/$EXECUTABLE MaxText/configs/base.yml model_name=llama3.1-405b run_name=$RUN_NAME \
  steps=20 per_device_batch_size=0.25 max_target_length=8192 enable_checkpointing=false \
  ici_tensor_parallelism=4 dcn_fsdp_parallelism=2 custom_mesh=hybrid_ring_64x4 \
  remat_policy=custom query_proj=offload out_proj=offload key_proj=device value_proj=device mlpwo=device mlpwi_0=offload mlpwi_1=offload decoder_layer_input=device \
  base_output_directory=gs://runner-maxtext-logs \
  dataset_type=synthetic allow_split_physical_axes=true profiler=xplane skip_first_n_steps_for_profiler=17 profiler_steps=3 \
  sa_block_q=1024 sa_block_q_dkv=2048 sa_block_q_dq=2048 use_iota_embed=True



# python3 MaxText/train.py MaxText/configs/base.yml per_device_batch_size=1 ici_fsdp_parallelism=64 ici_tensor_parallelism=4 dcn_fsdp_parallelism=2 allow_split_physical_axes=True custom_mesh=hybrid_ring_64x4 remat_policy=custom decoder_layer_input=offload query_proj=offload key_proj=offload value_proj=offload out_proj=offload max_target_length=8192 attention=flash gcs_metrics=True use_iota_embed=True dataset_path=gs://max-datasets-rogue dataset_type=synthetic reuse_example_batch=1 enable_checkpointing=False profiler=xplane skip_first_n_steps_for_profiler=17 profiler_steps=3 sa_block_q=1024 sa_block_q_dkv=2048 sa_block_q_dq=2048  steps=20 enable_checkpointing=false model_name=llama3.1-405b base_output_directory=gs://runner-maxtext-logs use_vertex_tensorboard=false vertex_tensorboard_project="" vertex_tensorboard_region="" run_name="llama3-1-405b-8192-fsdp-48-20241206"

# custom remat policy for pdb=0.5
# remat_policy=custom decoder_layer_input=offload query_proj=offload out_proj=offload key_proj=device value_proj=offload mlpwo=offload mlpwi_0=offload mlpwi_1=remat

# for pdb=0.25
# remat_policy=custom query_proj=offload out_proj=offload key_proj=device value_proj=device mlpwo=device mlpwi_0=offload mlpwi_1=offload decoder_layer_input=device