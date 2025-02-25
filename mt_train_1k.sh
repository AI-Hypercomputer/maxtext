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

# export JAX_ENABLE_PGLE=true
export JAX_ENABLE_PGLE=false
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

# YY: temp add JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none to bypass the PGLE issue

cat <<EOF > env.txt

# NCCL_DEBUG=INFO
# NCCL_DEBUG_SUBSYS=INIT,NET,ENV,TUNING,COLL
# NCCL_DEBUG_FILE=/var/log/yy/nccl_log.%h.%p
XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE=/usr/local/nvidia/lib64/a3plus_guest_config.textproto
NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS=600000
JAX_ENABLE_PGLE=$JAX_ENABLE_PGLE
JAX_REMOVE_CUSTOM_PARTITIONING_PTR_FROM_CACHE_KEY=true
JAX_DEBUG_LOG_MODULES=compiler
JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none
XLA_FLAGS=--xla_gpu_enable_latency_hiding_scheduler=true \
--xla_gpu_graph_level=0 \
--xla_gpu_enable_triton_gemm=false \
--xla_gpu_enable_highest_priority_async_stream=true \
--xla_gpu_all_reduce_combine_threshold_bytes=134217728 \
--xla_gpu_all_gather_combine_threshold_bytes=1073741824 \
--xla_gpu_reduce_scatter_combine_threshold_bytes=33554432 \
--xla_gpu_enable_pipelined_all_gather=true \
--xla_gpu_enable_pipelined_reduce_scatter=true \
--xla_gpu_enable_pipelined_all_reduce=true \
--xla_gpu_enable_while_loop_double_buffering=true \
--xla_disable_hlo_passes=rematerialization \
--xla_gpu_enable_all_gather_combine_by_dim=false \
--xla_gpu_enable_reduce_scatter_combine_by_dim=false \
--xla_gpu_threshold_for_windowed_einsum_mib=0 \
--xla_gpu_multi_streamed_windowed_einsum
EOF




# YY
# What's wrong with this flag: --xla_gpu_enable_triton_softmax_fusion=false \

#
#

# Cause the program to crash
# --xla_gpu_use_memcpy_local_p2p=true

# MFU drops to 9%, should not enable collective matmul in some cases
# --xla_gpu_threshold_for_windowed_einsum_mib=0 \
# --xla_gpu_multi_streamed_windowed_einsum


# --xla_gpu_enable_pgle_accuracy_checker=$STRICT_CHECKER \
# --xla_dump_to=/tmp/xla_dump/
# NCCL_DEBUG=INFO
# NCCL_DEBUG_SUBSYS=INIT,NET,ENV,TUNING,COLL
# --xla_gpu_threshold_for_windowed_einsum_mib=0 \
# --xla_gpu_multi_streamed_windowed_einsum \
# --xla_gpu_graph_level=0 \
# --xla_gpu_pgle_accuracy_checker=PGLE_STRICTNESS_LEVEL_ERROR \

# --xla_gpu_enable_pgle_accuracy_checker=$STRICT_CHECKER \

# export LOCAL_IMAGE_NAME=gcr.io/tpu-prod-env-multipod/jonbolin-maxtext-gpu:20241008-1


# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/llama2-xprof_1010_nolayers_nightly_lance # cuda working, but 128 node not working.

# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/llama2-1104_405b_lance #cuda attention kernel has issue

# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/lance-1119-dev-rebased # pgle has issue

# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/lance-1124 # still flash attention not working
# export LOCAL_IMAGE_NAME=gcr.io/supercomputer-testing/lance-0106-pinned
# export LOCAL_IMAGE_NAME=gcr.io/supercomputer-testing/lance-0206-nightly
# export LOCAL_IMAGE_NAME=gcr.io/tpu-prod-env-multipod/maxtext_gpu_jax_pinned
# export LOCAL_IMAGE_NAME=gcr.io/supercomputer-testing/lance-0218-tp-fix
# export LOCAL_IMAGE_NAME=gcr.io/supercomputer-testing/lance-0219-tp-fix2-pp-fix
# export LOCAL_IMAGE_NAME=gcr.io/supercomputer-testing/lance-0220-tp-fix2-pp-fix # Rebase back to Feb 12 with PP fix

# Cannot locate the ip address, so the jax stable stack cannot be used directly on xpk GPU
# export LOCAL_IMAGE_NAME=gcr.io/supercomputer-testing/lance-0220-stable-stack-tp-fix2-pp-fix


# export LOCAL_IMAGE_NAME=gcr.io/supercomputer-testing/lance-0220-stable-stack-additial_deps_keep_data_sharding

# export LOCAL_IMAGE_NAME=gcr.io/supercomputer-testing/lance-0220-stable-stack-additional-deps # Now this version contains the first fix, and all the deps

# export LOCAL_IMAGE_NAME=gcr.io/supercomputer-testing/lance-nv-0212:latest # This version has fix 1 and 2
# export LOCAL_IMAGE_NAME=gcr.io/supercomputer-testing/lance-nv-0212-no-fix # This version reverts all the fix
# export LOCAL_IMAGE_NAME=gcr.io/supercomputer-testing/lance-0201-stable-stack
export LOCAL_IMAGE_NAME=gcr.io/supercomputer-testing/lance-0215-stable-stack
# export LOCAL_IMAGE_NAME=gcr.io/supercomputer-testing/lance-nv-0212-no-local-maxtext
# export LOCAL_IMAGE_NAME=gcr.io/supercomputer-testing/lance-nv-1204-no-local-maxtext # No PTX compiliation provider is available
# export LOCAL_IMAGE_NAME=gcr.io/supercomputer-testing/lance-nv-0110-no-local-maxtext # No PTX
gcr.io/supercomputer-testing/lance-nv-0110-jsts

call_config() {

    declare -A args

    # Parse named arguments
    while [[ "$#" -gt 0 ]]; do
        case $1 in
            --*) key="${1#--}"; args["$key"]="$2"; shift ;; # Remove "--" and set key-value
            *) echo "Unknown parameter passed: $1"; return 1 ;;
        esac
        shift # Move to the next argument
    done

    local NUM_LAYERS=${args[NUM_LAYERS]}
    local NUM_NODES=${args[NUM_NODES]}
    local PER_DEVICE_BATCH_SIZE=${args[PER_DEVICE_BATCH_SIZE]}
    local ICI_TP=${args[ICI_TP]}
    local T_SEQ=${args[T_SEQ]}

    local DCN_FSDP=${args[DCN_FSDP]}
    local DCN_PP=${args[DCN_PP]}
    local NUM_LAYERS_PER_PP_STAGE=${args[NUM_LAYERS_PER_PP_STAGE]} # Layers are modified to 128 for short term solution
    # export NUM_LAYERS_PER_PP_STAGE=$(expr 126 / $DCN_PP)

    local REMAT_POLICY=${args[REMAT_POLICY]}
    # export REMAT_POLICY=full
    # export REMAT_POLICY=minimal

    local ATTENTION=${args[ATTENTION]}
    # local ATTENTION=cudnn_flash_te
    # export ATTENTION=dot_product

    local WORKLOAD_NAME=$USER-$MODEL_SIZE-${NUM_NODES}n${ICI_TP}tp${DCN_FSDP}fsdp${DCN_PP}pp${NUM_LAYERS_PER_PP_STAGE}l-${RANDOM:0:2}

    echo 'NUM_NODES' ${NUM_NODES} 'PER_DEVICE_BATCH_SIZE' ${PER_DEVICE_BATCH_SIZE} 'ICI_TP' ${ICI_TP} 'DCN_FSDP' ${DCN_FSDP} 'DCN_PP' ${DCN_PP} 'NUM_LAYERS_PER_PP_STAGE' ${NUM_LAYERS_PER_PP_STAGE} 'REMAT_POLICY' ${REMAT_POLICY} 'ATTENTION' ${ATTENTION} WORKLOAD_NAME ${WORKLOAD_NAME}

    COMMAND="python3 MaxText/train.py MaxText/configs/models/gpu/$CONFIG_NAME.yml hardware=gpu run_name=$RUN_NAME steps=10 max_target_length=4096 model_name=$MODEL_NAME enable_checkpointing=false attention=$ATTENTION dataset_type=synthetic async_checkpointing=false base_output_directory=$OUTPUT_BUCKET logits_dot_in_fp32=false use_iota_embed=true scan_layers=true ici_tensor_parallelism=$ICI_TP dcn_fsdp_parallelism=$DCN_FSDP dcn_pipeline_parallelism=$DCN_PP per_device_batch_size=$PER_DEVICE_BATCH_SIZE num_layers_per_pipeline_stage=$NUM_LAYERS_PER_PP_STAGE weight_dtype=bfloat16 remat_policy=$REMAT_POLICY profiler=xplane skip_first_n_steps_for_profiler=5";
    # ici_tensor_sequence_parallelism=$T_SEQ
    # quantization=fp8

    COMMAND="${COMMAND};""gsutil -m cp -r /var/log/yy $OUTPUT_BUCKET";

    echo 'COMMAND is:' ${COMMAND}
    python ../xpk/xpk.py workload delete --cluster $CLUSTER_NAME --workload $WORKLOAD_NAME;
    python ../xpk/xpk.py workload create --cluster $CLUSTER_NAME --workload $WORKLOAD_NAME --command "${COMMAND}" --docker-image=$LOCAL_IMAGE_NAME --device-type=$DEVICE_TYPE --num-nodes=$NUM_NODES --scheduler=gke.io/topology-aware-auto --env-file=env.txt ;
}

# input 1: 768 nodes 6k cluster
# input 2: per device batch size
# input 3: TP
# input 4: FSDP
# input 5: PP
# input 6: number of layers per PP stage
# input 7:remat policy

# Testing the xpk and commands
# call_config 2 1 8 1 0 save_qkv_proj
# submit 126

# Test run with 2 nodes, cuda kernel fails
export NODES=16
# call_config --NUM_LAYERS 126  --NUM_NODES $NODES --PER_DEVICE_BATCH_SIZE 1 --T_SEQ 1 --ICI_TP 8 --ICI_FSDP 1 --DCN_FSDP $NODES --DCN_PP 1 --REMAT_POLICY full --ATTENTION cudnn_flash_te --NUM_LAYERS_PER_PP_STAGE 0
call_config --NUM_LAYERS 126  --NUM_NODES $NODES --PER_DEVICE_BATCH_SIZE 1 --T_SEQ 1 --ICI_TP 1 --ICI_FSDP 8 --DCN_FSDP $NODES --DCN_PP 1 --REMAT_POLICY save_qkv_proj --ATTENTION cudnn_flash_te --NUM_LAYERS_PER_PP_STAGE 0


# call_config --NUM_LAYERS 126  --NUM_NODES 2 --PER_DEVICE_BATCH_SIZE 1 --T_SEQ 8 --ICI_TP 1 --ICI_FSDP 1 --DCN_FSDP 2 --DCN_PP 1 --REMAT_POLICY save_qkv_proj --ATTENTION cudnn_flash_te --NUM_LAYERS_PER_PP_STAGE 0

# call_config --NUM_LAYERS 126  --NUM_NODES 2 --PER_DEVICE_BATCH_SIZE 1 --ICI_TP 8 --DCN_FSDP 2 --DCN_PP 1 --REMAT_POLICY save_qkv_proj --ATTENTION dot_product --NUM_LAYERS_PER_PP_STAGE 0
# call_config --NUM_LAYERS 126  --NUM_NODES 2 --PER_DEVICE_BATCH_SIZE 1 --ICI_TP 8 --DCN_FSDP 2 --DCN_PP 1 --REMAT_POLICY save_qkv_proj --ATTENTION cudnn_flash_te --NUM_LAYERS_PER_PP_STAGE 0

#
# call_config --NUM_LAYERS 126  --NUM_NODES 64 --PER_DEVICE_BATCH_SIZE 1 --T_SEQ 8 --ICI_TP 1 --DCN_FSDP 64 --DCN_PP 1 --REMAT_POLICY full --ATTENTION cudnn_flash_te --NUM_LAYERS_PER_PP_STAGE 0


# call_config --NUM_LAYERS 126  --NUM_NODES 128 --PER_DEVICE_BATCH_SIZE 1 --ICI_TP 8 --DCN_FSDP 128 --DCN_PP 1 --REMAT_POLICY save_qkv_proj --ATTENTION cudnn_flash_te --NUM_LAYERS_PER_PP_STAGE 0

#OOM
# call_config --NUM_LAYERS 126  --NUM_NODES 128 --PER_DEVICE_BATCH_SIZE 1 --ICI_TP 8 --DCN_FSDP 64 --DCN_PP 1 --REMAT_POLICY save_qkv_proj --ATTENTION cudnn_flash_te --NUM_LAYERS_PER_PP_STAGE 0


# call_config --NUM_LAYERS 126 --NUM_NODES 128 --PER_DEVICE_BATCH_SIZE 1 --ICI_TP 8 --DCN_FSDP 16 --DCN_PP 1 --REMAT_POLICY save_qkv_proj --ATTENTION cudnn_flash_te --NUM_LAYERS_PER_PP_STAGE 0
# call_config --NUM_LAYERS 126 --NUM_NODES 128 --PER_DEVICE_BATCH_SIZE 1 --ICI_TP 8 --DCN_FSDP 8 --DCN_PP 1 --REMAT_POLICY save_qkv_proj --ATTENTION cudnn_flash_te --NUM_LAYERS_PER_PP_STAGE 0

# OOM
# call_config --NUM_LAYERS 126 --NUM_NODES 128 --PER_DEVICE_BATCH_SIZE 1 --ICI_TP 8 --DCN_FSDP 1 --DCN_PP 1 --REMAT_POLICY save_qkv_proj --ATTENTION cudnn_flash_te --NUM_LAYERS_PER_PP_STAGE 0

# Does DCN FSDP 32 work, 33% MFU
# call_config --NUM_LAYERS 126 --NUM_NODES 128 --PER_DEVICE_BATCH_SIZE 1 --ICI_TP 8 --DCN_FSDP 32 --DCN_PP 1 --REMAT_POLICY save_qkv_proj --ATTENTION cudnn_flash_te --NUM_LAYERS_PER_PP_STAGE 0

# Using TP of 8 for best perf
# call_config --NUM_LAYERS 126 --NUM_NODES 128 --PER_DEVICE_BATCH_SIZE 1 --ICI_TP 8 --DCN_FSDP 64 --DCN_PP 1 --REMAT_POLICY save_qkv_proj --ATTENTION cudnn_flash_te --NUM_LAYERS_PER_PP_STAGE 0

# Only using TP of 4 to see perf
# call_config --NUM_LAYERS 126 --NUM_NODES 128 --PER_DEVICE_BATCH_SIZE 1 --ICI_TP 4 --DCN_FSDP 64 --DCN_PP 1 --REMAT_POLICY save_qkv_proj --ATTENTION cudnn_flash_te --NUM_LAYERS_PER_PP_STAGE 0




# Config 0, 8 TP, 64 FSDP, save_qkv_proj should be good
# call_config --NUM_NODES 128 --PER_DEVICE_BATCH_SIZE 1 --ICI_TP 8 --DCN_FSDP 64 --DCN_PP 1 --REMAT_POLICY save_qkv_proj --ATTENTION dot_product --NUM_LAYERS_PER_PP_STAGE 0

# Config 0, 8 TP, 32 FSDP, full should be good
# call_config --NUM_NODES 128 --PER_DEVICE_BATCH_SIZE 1 --ICI_TP 8 --DCN_FSDP 32 --DCN_PP 1 --REMAT_POLICY full --ATTENTION dot_product --NUM_LAYERS_PER_PP_STAGE 0

# call_config 128 1 8 1 0 save_qkv_proj

# # Config 1, PP, remat full
# call_pp 512 3 8 2 16 full
# submit 128

# # Congig 2, PP,
# call_pp 512 1 8 2 16 save_qkv_proj
# submit 128

# Pure FSDP, OOM
# call_config 128 1 1 1 0 save_qkv_proj
# submit 126

# Pure FSDP
# call_config 128 1 1 1 0 full
# submit 126

# Pure FSDP on 2 nodes, 7b
# call_config 128 1 1 1 0 full
# submit 126

# call_config 16 1 1 1 16 save_qkv_proj

# submit 126

# call_config 8 1 1 1 16 save_qkv_proj
# submit 126

# call_config 4 1 1 1 16 save_qkv_proj
# submit 126

# call_config 2 1 1 1 16 save_qkv_proj
# submit 126
