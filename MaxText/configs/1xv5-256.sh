export TPU_PREMAPPED_BUFFER_SIZE=4294967296
#export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true TPU_MEGACORE=MEGACORE_DENSE"
python3 MaxText/train.py MaxText/configs/base.yml run_name=1xv5-256 dcn_data_parallelism=1 ici_fsdp_parallelism=256 steps=10 per_device_batch_size=2  enable_profiler=true remat_policy=full scale=3
# TFLOP/s 165, 22B
