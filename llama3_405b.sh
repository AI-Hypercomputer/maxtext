#!/bin/bash


XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true \
--xla_gpu_enable_triton_gemm=false \
--xla_gpu_enable_highest_priority_async_stream=true \
--xla_gpu_all_reduce_combine_threshold_bytes=134217728 \
--xla_gpu_all_gather_combine_threshold_bytes=1073741824 \
--xla_gpu_reduce_scatter_combine_threshold_bytes=33554432 \
--xla_gpu_enable_pipelined_all_gather=true \
--xla_gpu_enable_pipelined_reduce_scatter=true \
--xla_gpu_enable_pipelined_all_reduce=true \
--xla_gpu_enable_while_loop_double_buffering=true \
--xla_gpu_enable_triton_softmax_fusion=false \
--xla_gpu_enable_all_gather_combine_by_dim=false \
--xla_gpu_enable_reduce_scatter_combine_by_dim=false \
--xla_disable_hlo_passes=rematerialization \
--xla_gpu_shard_autotuning=false";

# Launch llama2 7b
export MODEL_NAME=llama2-7b
export NUM_NODES=2

# Launch llama2 70b
export MODEL_NAME=llama2-70b
export NUM_NODES=2

# Launch llama3.1 405b
export MODEL_NAME=llama3.1-405b
export NUM_NODES=1

# Common parameters
export CLUSTER_NAME=a3plus-benchmark
export ZONE=australia-southeast1
export WORKLOAD_NAME=lance-$(echo $MODEL_NAME | sed 's/\.//g')
export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/llama2-xprof_0920_lance
export DEVICE_TYPE=h100-mega-80gb-8
export OUTPUT_BUCKET=gs://lancewang-dev-supercomputer-testing/maxtext_gpu
export RUN_NAME=maxtext-$MODEL_NAME
CONFIG_NAME=$(echo $MODEL_NAME | sed 's/-/_/g')


