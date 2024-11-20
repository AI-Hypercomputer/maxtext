#!/bin/bash

export MODEL_NAME=llama2-7b

# Launch llama2 70b
# export MODEL_NAME=llama2-70b

# Launch llama3.1 405b
# export MODEL_NAME=llama3.1-405b

# Common parameters
export CLUSTER_NAME=a3plus-benchmark
export ZONE=australia-southeast1

export DEVICE_TYPE=h100-mega-80gb-8
export OUTPUT_PATH=lancewang-dev-supercomputer-testing/maxtext_gpu
export OUTPUT_BUCKET=gs://$OUTPUT_PATH
export RUN_NAME=maxtext-$MODEL_NAME
CONFIG_NAME=$(echo $MODEL_NAME | sed 's/-/_/g')

# Enable PGLE
# export JAX_ENABLE_PGLE=false
export JAX_ENABLE_PGLE=true
export JAX_PGLE_AGGREGATION_PERCENTILE=50
export JAX_SHARE_AUTOTUNE_CONFIG_BETWEEN_HOSTS=true
export JAX_PGLE_PROFILING_RUNS=3

# Turn off checker
export STRICT_CHECKER=true

# JAX_PGLE_AGGREGATION_PERCENTILE=$JAX_PGLE_AGGREGATION_PERCENTILE
# JAX_SHARE_AUTOTUNE_CONFIG_BETWEEN_HOSTS=$JAX_SHARE_AUTOTUNE_CONFIG_BETWEEN_HOSTS
# JAX_PGLE_PROFILING_RUNS=$JAX_PGLE_PROFILING_RUNS
# TF_CPP_VMODULE=profile_guided_latency_estimator=10
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
# CUDA_DEVICE_MAX_CONNECTIONS=1

cat <<EOF > env.txt
NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE=/usr/local/nvidia/lib64/a3plus_guest_config.textproto
NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS=600000
JAX_ENABLE_COMPILATION_CACHE=True
JAX_ENABLE_PGLE=$JAX_ENABLE_PGLE
JAX_REMOVE_CUSTOM_PARTITIONING_PTR_FROM_CACHE_KEY=$JAX_ENABLE_PGLE
JAX_DEBUG_LOG_MODULES=jax._src.compiler
XLA_FLAGS=--xla_gpu_enable_latency_hiding_scheduler=true \
--xla_gpu_enable_triton_gemm=false \
--xla_gpu_graph_level=0 \
--xla_gpu_enable_highest_priority_async_stream=true \
--xla_gpu_all_reduce_combine_threshold_bytes=536870912 \
--xla_gpu_all_gather_combine_threshold_bytes=536870912 \
--xla_gpu_reduce_scatter_combine_threshold_bytes=536870912 \
--xla_gpu_enable_pipelined_all_gather=true \
--xla_gpu_enable_pipelined_reduce_scatter=true \
--xla_gpu_enable_pipelined_all_reduce=true \
--xla_gpu_enable_while_loop_double_buffering=true \
--xla_disable_hlo_passes=rematerialization \
--xla_gpu_pgle_accuracy_checker='PGLE_STRICTNESS_LEVEL_ERROR' \
--xla_gpu_enable_triton_softmax_fusion=false \
--xla_gpu_enable_all_gather_combine_by_dim=false \
--xla_gpu_enable_reduce_scatter_combine_by_dim=false \
--xla_gpu_shard_autotuning=false
EOF

# --xla_dump_to=/tmp/xla_dump/ \
# --xla_gpu_experimental_dump_fdo_profiles \
#--xla_gpu_enable_pgle_accuracy_checker=$STRICT_CHECKER \
# The current image doesn't support this one yet
# --xla_gpu_pgle_accuracy_checker='PGLE_STRICTNESS_LEVEL_ERROR' \

call_train() {
    export WORKLOAD_NAME=$USER-pgle-7b-$1n-levon-lance-${RANDOM:0:2}

    export NUM_NODES=$1

    export PER_DEVICE_BATCH_SIZE=$2
    export ICI_TP=$3
    # export ICI_TP=1

    export DCN_FSDP=$NUM_NODES
    #export DCN_FSDP=32

    export DCN_PP=1
    export NUM_LAYERS_PER_PP_STAGE=$(expr 128 / $DCN_PP) # Layers are modified to 128 for short term solution
    # export NUM_LAYERS_PER_PP_STAGE=$(expr 126 / $DCN_PP)

    export REMAT_POLICY=$4
    # export REMAT_POLICY=full
    # export REMAT_POLICY=minimal

    # export ATTENTION=cudnn_flash_te
    export ATTENTION=dot_product
}

# input 1: number of nodes
# input 2: per device batch size
# input 3: TP
# input 4: remat policy

# Config 1, TP
# call_non_pp 2 1 8 save_qkv_proj

# Congig 2, FSDP
# call_non_pp 768 1 1 save_qkv_proj
# call_train 2 1 1 full
call_train 2 1 8 minimal

# export LOCAL_IMAGE_NAME=gcr.io/tpu-prod-env-multipod/jonbolin-maxtext-gpu:20241008-1
# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/llama2-xprof_1010_nolayers_nightly_lance
# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/llama2-1022_lance

export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/lance-1119-dev-rebased

#  --xprof_gpu_cupti_collector_max_callback_api_events=20971520 --xprof_gpu_cupti_collector_max_activity_api_events=20971520

# COMMAND="python3 MaxText/train.py MaxText/configs/models/gpu/$CONFIG_NAME.yml hardware=gpu run_name=$RUN_NAME steps=10 max_target_length=4096 model_name=$MODEL_NAME enable_checkpointing=false attention=dot_product dataset_type=synthetic async_checkpointing=false base_output_directory=$OUTPUT_BUCKET logits_dot_in_fp32=false use_iota_embed=true dcn_pipeline_parallelism=1 dcn_data_parallelism=2 per_device_batch_size=1 ici_tensor_parallelism=8 weight_dtype=bfloat16 remat_policy=minimal && gsutil -m cp -r /tmp/xla_dump/ $OUTPUT_BUCKET";

COMMAND="python3 MaxText/train.py MaxText/configs/models/gpu/$CONFIG_NAME.yml hardware=gpu run_name=$RUN_NAME steps=10 max_target_length=4096 model_name=$MODEL_NAME enable_checkpointing=false attention=$ATTENTION dataset_type=synthetic async_checkpointing=false base_output_directory=$OUTPUT_BUCKET logits_dot_in_fp32=false use_iota_embed=true dcn_pipeline_parallelism=$DCN_PP dcn_fsdp_parallelism=$DCN_FSDP per_device_batch_size=$PER_DEVICE_BATCH_SIZE ici_tensor_parallelism=$ICI_TP weight_dtype=bfloat16 remat_policy=$REMAT_POLICY && gsutil -m cp -r /tmp/xla_dump/ $OUTPUT_BUCKET";

# Disable profiler
# profiler=xplane skip_first_n_steps_for_profiler=5" 

COMMAND='export LD_LIBRARY_PATH=/usr/local/cuda-12.6/compat:$LD_LIBRARY_PATH;'"${COMMAND}"; 

# COMMAND='ls -d -1 MaxText/**/*'

python ../xpk/xpk.py  workload delete --cluster $CLUSTER_NAME --workload $WORKLOAD_NAME;

python ../xpk/xpk.py  workload create --cluster $CLUSTER_NAME --workload $WORKLOAD_NAME --command "${COMMAND}" --docker-image=$LOCAL_IMAGE_NAME --device-type=$DEVICE_TYPE --num-nodes=$NUM_NODES --scheduler=gke.io/topology-aware-auto --env-file=env.txt ;