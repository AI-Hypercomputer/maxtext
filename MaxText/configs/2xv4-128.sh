export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true TPU_MEGACORE=MEGACORE_DENSE"
python3 MaxText/train.py MaxText/configs/base.yml run_name=2xv4-64 dcn_data_parallelism=2 ici_fsdp_parallelism=64 steps=10 per_device_batch_size=16  enable_profiler=true remat_policy=full scale=3
# 158TFLOP/s, 22B params
