export LIBTPU_INIT_ARGS="--xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
USE_INT8_TRAINING=${1}
BATCH_SIZE=${2}
SCALE=${3}
RUN_NAME=${4}

STEPS=5

python3 MaxText/train.py MaxText/configs/base.yml \
    steps=${STEPS} learning_rate=0.001 warmup_steps=2000 enable_profiler=true enable_checkpointing=false \
    enable_dropout=false enable_data_shuffling=false run_name=${RUN_NAME}\
    use_int8_training=${USE_INT8_TRAINING} metrics_file=metrics.txt\
    remat_policy=full\
    base_output_directory=gs://maxtext-experiments-multipod dataset_path=gs://max-datasets-rogue/\
    per_device_batch_size=${BATCH_SIZE} global_parameter_scale=${SCALE}