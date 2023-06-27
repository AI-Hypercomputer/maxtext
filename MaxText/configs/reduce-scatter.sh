
RUN_NAME=${1}

bash setup.sh && bash network_setup.sh && export LIBTPU_INIT_ARGS='--xla_tpu_enable_data_parallel_all_reduce_opt=true'
python3 MaxText/train.py MaxText/configs/base.yml run_name=${RUN_NAME}\
    steps=10 per_device_batch_size=2 enable_checkpointing=false\
    enable_profiler=true remat_policy=proj base_emb_dim=8192 base_mlp_dim=16384\
    base_num_heads=24 base_num_decoder_layers=36 head_dim=256\
    max_target_length=2048 metrics_file='metrics.txt' log_period=8