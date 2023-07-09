echo "Running repro-raymond.sh"
RUN_NAME=${1}
PROJ_LIST=${2}
BATCH_SIZE=${3}

#export TPU_LIBRARY_PATH='/lib/libtpu.so'
python3 MaxText/train.py MaxText/configs/base.yml run_name=${RUN_NAME}\
    steps=5 per_device_batch_size=${BATCH_SIZE} enable_checkpointing=false\
    enable_profiler=true remat_policy=proj base_emb_dim=256 base_mlp_dim=256\
    base_num_heads=24 base_num_decoder_layers=36 head_dim=256\
    max_target_length=512 metrics_file='metrics.txt' log_period=8 proj_list=${PROJ_LIST}