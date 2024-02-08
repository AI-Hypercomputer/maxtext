echo "Running 128b.sh forked"
# Stop execution if any command exits with error
set -e
export XLA_FLAGS="--xla_dump_to=20240208_scantrue_offload"
export TPU_STDERR_LOG_LEVEL=0
export TPU_MIN_LOG_LEVEL=0
export JAX_USE_PJRT_C_API_ON_TPU=1
export TF_CPP_MIN_LOG_LEVEL=0
# Train
export LIBTPU_INIT_ARGS="--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
python3 MaxText/train_compile.py MaxText/configs/base.yml run_name=$RUN_NAME\
    steps=10 per_device_batch_size=.5 enable_checkpointing=false\
    enable_profiler=true global_parameter_scale=128\
    ici_fsdp_parallelism=16 ici_tensor_parallelism=16\
    max_target_length=2048 base_output_directory=gs://runner-maxtext-logs\
    dataset_path=$DATASET_PATH use_iota_embed=true reuse_example_batch=1\
    dataset_type=synthetic attention='flash' gcs_metrics=true\
    compile_topology=v5e-256 compile_topology_num_slices=1\
    remat_policy=minimal
