#!/bin/bash

export MODEL_NAME=llama2-7b

# Launch llama2 70b
# export MODEL_NAME=llama2-70b

# Launch llama3.1 405b
# export MODEL_NAME=llama3.1-405b

MODEL_SIZE=$(echo $MODEL_NAME | grep -o '[0-9]\+b')

# Common parameters
export CLUSTER_NAME=a3plus-benchmark

# export ZONE=us-central1-b
export ZONE=australia-southeast1

export DEVICE_TYPE=h100-mega-80gb-8
export OUTPUT_PATH=lancewang-dev-supercomputer-testing/maxtext_gpu
export OUTPUT_BUCKET=gs://$OUTPUT_PATH
export XLA_DUMP=gs://$OUTPUT_PATH/xla/1107
export RUN_NAME=maxtext-$MODEL_NAME
CONFIG_NAME=$(echo $MODEL_NAME | sed 's/-/_/g')

export JAX_ENABLE_PGLE=false
# export JAX_ENABLE_PGLE=true
export STRICT_CHECKER=true

export JAX_PGLE_AGGREGATION_PERCENTILE=50
export JAX_SHARE_AUTOTUNE_CONFIG_BETWEEN_HOSTS=true
export JAX_PGLE_PROFILING_RUNS=3

# JAX_PGLE_AGGREGATION_PERCENTILE=$JAX_PGLE_AGGREGATION_PERCENTILE
# JAX_SHARE_AUTOTUNE_CONFIG_BETWEEN_HOSTS=$JAX_SHARE_AUTOTUNE_CONFIG_BETWEEN_HOSTS
# JAX_PGLE_PROFILING_RUNS=$JAX_PGLE_PROFILING_RUNS
# TF_CPP_VMODULE=profile_guided_latency_estimator=10
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
# CUDA_DEVICE_MAX_CONNECTIONS=1

cat <<EOF > env.txt
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,NET,ENV,TUNING,COLL
NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE=/usr/local/nvidia/lib64/a3plus_guest_config.textproto
NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS=600000
JAX_ENABLE_PGLE=$JAX_ENABLE_PGLE
JAX_REMOVE_CUSTOM_PARTITIONING_PTR_FROM_CACHE_KEY=$JAX_ENABLE_PGLE
JAX_DEBUG_LOG_MODULES=compiler
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NCCL_ALGO=Ring,Tree
NCCL_BUFFSIZE=8388608
NCCL_FASTRAK_CTRL_DEV=eth0
NCCL_FASTRAK_ENABLE_CONTROL_CHANNEL=0
NCCL_FASTRAK_ENABLE_HOTPATH_LOGGING=0
NCCL_FASTRAK_IFNAME=eth1,eth2,eth3,eth4,eth5,eth6,eth7,eth8
NCCL_FASTRAK_NUM_FLOWS=2
NCCL_FASTRAK_USE_LLCM=1
NCCL_FASTRAK_USE_SNAP=1
NCCL_MIN_NCHANNELS=4
NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE=/usr/local/nvidia/lib64/a3plus_guest_config.textproto
NCCL_TUNER_CONFIG_PATH=/usr/local/nvidia/lib64/a3plus_tuner_config.textproto
NCCL_TUNER_PLUGIN=libnccl-tuner.so
XLA_FLAGS=--xla_gpu_enable_latency_hiding_scheduler=true \
--xla_gpu_threshold_for_windowed_einsum_mib=0 \
--xla_gpu_multi_streamed_windowed_einsum \
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



# export LOCAL_IMAGE_NAME=gcr.io/tpu-prod-env-multipod/jonbolin-maxtext-gpu:20241008-1
# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/llama2-xprof_1010_nolayers_nightly_lance
# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/llama2-1104_405b_lance
export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/maxtext_collective_benchmarking
export WORKLOAD_NAME=lancewang-collective-benchmarking-${RANDOM:0:2}
export NUM_NODES=64

COMMAND="python3 mt_all_gather_benchmark.py --num_nodes=$NUM_NODES"; 

COMMAND='export LD_LIBRARY_PATH=/usr/local/cuda-12.6/compat:$LD_LIBRARY_PATH;'"${COMMAND};";

echo 'COMMAND' ${COMMAND}
python ../xpk/xpk.py workload delete --cluster $CLUSTER_NAME --workload $WORKLOAD_NAME; 
python ../xpk/xpk.py workload create --cluster $CLUSTER_NAME --workload $WORKLOAD_NAME --command "${COMMAND}" --docker-image=$LOCAL_IMAGE_NAME --device-type=$DEVICE_TYPE --num-nodes=$NUM_NODES --scheduler=gke.io/topology-aware-auto --env-file=env.txt ;
