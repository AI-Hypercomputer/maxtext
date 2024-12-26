
set -e
pip install libtpu-nightly==0.1.dev20241206 -f https://storage.googleapis.com/libtpu-releases/index.html 

export TPU_PREMAPPED_BUFFER_SIZE=17179869184

# networking fix
echo "4096 41943040 314572800" > /proc/sys/net/ipv4/tcp_rmem

# CF AG flags
# --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true \
    # --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true

export LIBTPU_INIT_ARGS=' --xla_tpu_scoped_vmem_limit_kib=98304 --xla_tpu_enable_all_experimental_scheduler_features=true \
    --xla_tpu_enable_scheduler_memory_pressure_tracking=true --xla_tpu_host_transfer_overlap_limit=24 --xla_tpu_aggressive_opt_barrier_removal=ENABLED --xla_lhs_prioritize_async_depth_over_stall=ENABLED \
    --xla_tpu_enable_ag_backward_pipelining=true --xla_should_allow_loop_variant_parameter_in_chain=ENABLED --xla_should_add_loop_invariant_op_in_chain=ENABLED --xla_max_concurrent_host_send_recv=100 \
    --xla_tpu_scheduler_percent_shared_memory_limit=100 --xla_latency_hiding_scheduler_rerun=2'

python3 MaxText/train.py MaxText/configs/base.yml per_device_batch_size=1 ici_fsdp_parallelism=64 ici_tensor_parallelism=4 \
    allow_split_physical_axes=True custom_mesh=hybrid_ring_64x4 \
    remat_policy=custom decoder_layer_input=offload query_proj=device key_proj=device value_proj=device out_proj=offload optimizer_memory_host_offload=True \
    max_target_length=8192 attention=flash gcs_metrics=True use_iota_embed=True dataset_path=gs://max-datasets-rogue dataset_type=synthetic reuse_example_batch=1 \
    enable_checkpointing=False profiler=xplane sa_block_q=1024 sa_block_q_dkv=2048 sa_block_q_dq=2048  steps=20 enable_checkpointing=false model_name=llama3.1-405b \
    base_output_directory=gs://runner-maxtext-logs run_name="llama3-1-405b-8192-opt-off-1-20241216"