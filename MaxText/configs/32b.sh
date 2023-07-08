echo "Running 32b.sh"
RUN_NAME=${1}

export TPU_LIBRARY_PATH='/lib/libtpu.so'
bash rto_setup.sh && export LIBTPU_INIT_ARGS='--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true'
echo "train.py"
python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME\
    steps=6 per_device_batch_size=4 enable_checkpointing=false\
    enable_profiler=true remat_policy=full base_emb_dim=8192 base_mlp_dim=32768\
    base_num_heads=32 base_num_decoder_layers=40 head_dim=256\
    max_target_length=2048 metrics_file='metrics.txt'






