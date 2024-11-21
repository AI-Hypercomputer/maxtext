#!/bin/bash



# Launch llama2 7b
# export MODEL_NAME=llama2-7b

# Launch llama2 70b
# export MODEL_NAME=llama2-70b

# Launch llama3.1 405b
export MODEL_NAME=llama3.1-405b

# Common parameters
export CLUSTER_NAME=a3plus-benchmark
export ZONE=australia-southeast1

#export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/llama2-xprof_1001_lance # 128 layers
# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/llama2-xprof_1002_126layers_lance
# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/llama2-xprof_1003_nolayers_pinned_lance
# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/llama2-xprof_1004_nolayers_nightly_lance
# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/llama2-xprof_1008-base-1004_nolayers_nightly_lance
# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/llama2-xprof_1010_nolayers_nightly_lance
# For PGLE
# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/llama2-1022_lance
# For 2% tolerance issue
export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/llama2-1022_405b_lance
# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/llama2-1104_405b_lance


export DEVICE_TYPE=h100-mega-80gb-8
export OUTPUT_PATH=lancewang-dev-supercomputer-testing/maxtext_gpu
export OUTPUT_BUCKET=gs://$OUTPUT_PATH
export RUN_NAME=maxtext-$MODEL_NAME
CONFIG_NAME=$(echo $MODEL_NAME | sed 's/-/_/g')

# export JAX_ENABLE_PGLE=false
export JAX_ENABLE_PGLE=true
export JAX_PGLE_AGGREGATION_PERCENTILE=50
export JAX_SHARE_AUTOTUNE_CONFIG_BETWEEN_HOSTS=true
export JAX_PGLE_PROFILING_RUNS=3
export STRICT_CHECKER=false
# JAX_PGLE_AGGREGATION_PERCENTILE=$JAX_PGLE_AGGREGATION_PERCENTILE
# JAX_SHARE_AUTOTUNE_CONFIG_BETWEEN_HOSTS=$JAX_SHARE_AUTOTUNE_CONFIG_BETWEEN_HOSTS
# JAX_PGLE_PROFILING_RUNS=$JAX_PGLE_PROFILING_RUNS
# TF_CPP_VMODULE=profile_guided_latency_estimator=10
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
# CUDA_DEVICE_MAX_CONNECTIONS=1

cat <<EOF > env.txt
TF_CPP_VMODULE=latency_hiding_scheduler=2,profile_guided_latency_estimator=2
NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE=/usr/local/nvidia/lib64/a3plus_guest_config.textproto
NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS=600000
JAX_ENABLE_PGLE=$JAX_ENABLE_PGLE
JAX_REMOVE_CUSTOM_PARTITIONING_PTR_FROM_CACHE_KEY=$JAX_ENABLE_PGLE
JAX_DEBUG_LOG_MODULES=jax._src.compiler,jax._src.cache_key,jax._src.interpreters.xla,jax._src.pjit
XLA_FLAGS=--xla_gpu_enable_latency_hiding_scheduler=true \
--xla_gpu_graph_level=0 \
--xla_gpu_enable_triton_gemm=false \
--xla_gpu_enable_highest_priority_async_stream=true \
--xla_gpu_all_reduce_combine_threshold_bytes=536870912 \
--xla_gpu_all_gather_combine_threshold_bytes=536870912 \
--xla_gpu_reduce_scatter_combine_threshold_bytes=536870912 \
--xla_gpu_enable_pipelined_all_gather=true \
--xla_gpu_enable_pipelined_reduce_scatter=true \
--xla_gpu_enable_pipelined_all_reduce=true \
--xla_gpu_enable_while_loop_double_buffering=true \
--xla_disable_hlo_passes=rematerialization \
--xla_gpu_enable_triton_softmax_fusion=false \
--xla_gpu_enable_all_gather_combine_by_dim=false \
--xla_gpu_enable_reduce_scatter_combine_by_dim=false
EOF

# --xla_gpu_enable_pgle_accuracy_checker=$STRICT_CHECKER \

# Jon doesn't have this one
# --xla_gpu_shard_autotuning=false


# --xla_dump_hlo_pass_re=.* \
# --xla_dump_to=/tmp/xla_dump/ \

# Removed in order to run old images
# --xla_gpu_enable_pgle_accuracy_checker=true \

# There are the diffs between mine and Olech's
# --xla_gpu_all_reduce_combine_threshold_bytes=134217728 \
# --xla_gpu_all_gather_combine_threshold_bytes=1073741824 \
# --xla_gpu_reduce_scatter_combine_threshold_bytes=33554432 \
# --xla_gpu_enable_triton_softmax_fusion=false \
# --xla_gpu_enable_all_gather_combine_by_dim=false \
# --xla_gpu_enable_reduce_scatter_combine_by_dim=false \
# --xla_gpu_shard_autotuning=false \

