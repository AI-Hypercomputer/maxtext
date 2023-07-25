#!/bin/bash
set -e

RUN_NAME=$1

export LIBTPU_INIT_ARGS="--xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"

base_command="python3 MaxText/train.py MaxText/configs/base.yml \
    base_emb_dim=1024 base_mlp_dim=1024 base_num_heads=16 \
    per_device_batch_size=3 enable_checkpointing=false enable_profiler=true \
    steps=10 enable_profiler=true"

bfloat16_command=$base_command" run_name=$RUN_NAME-bfloat16 metrics_file=bfloat16_metrics.txt"
aqt_command=$base_command" run_name=$RUN_NAME-aqt metrics_file=aqt_metrics.txt use_int8_training=true"

echo "Starting bfloat16 run"
eval ${bfloat16_command}
echo "Finished bfloat16 run"
last_3_lines=$(tail -n 3 bfloat16_metrics.txt)
echo "Printing last 3 lines of metrics:"
echo "${last_3_lines}"
gsutil cp bfloat16_metrics.txt gs://mattdavidow-maxtext-br/${RUN_NAME}_metrics_bfloat16.txt

sleep 10

echo "Starting AQT run"
eval ${aqt_command}
echo "Finished aqt run"
last_3_lines=$(tail -n 3 aqt_metrics.txt)
echo "Printing last 3 lines of metrics:"
echo "${last_3_lines}"
gsutil cp aqt_metrics.txt gs://mattdavidow-maxtext-br/${RUN_NAME}_metrics_aqt.txt