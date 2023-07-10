#!/bin/bash
set -e

USER=${1}
TFLOP_THRESHOLD=${2}
OUTPUT_PATH=${3}
DATASET_PATH=${4}


if [ -z ${5} ]
then 
    RUN_NAME=${USER}
else
    RUN_NAME=${5}
fi

# Train
export LIBTPU_INIT_ARGS="--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME\
    steps=150 per_device_batch_size=6 enable_checkpointing=false\
    enable_profiler=false remat_policy=full base_emb_dim=6144 base_mlp_dim=24576\
    base_num_heads=24 base_num_decoder_layers=36 head_dim=256\
    max_target_length=2048 metrics_file='metrics.txt' base_output_directory=$OUTPUT_PATH\
    dataset_path=$DATASET_PATH log_period=150

# Assert TFLOP/s
python3 end_to_end/eval_assert.py metrics_average metrics.txt $TFLOP_THRESHOLD perf/per_device_tflops_per_sec
