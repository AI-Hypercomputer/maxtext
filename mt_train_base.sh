#!/bin/bash

# export MODEL_NAME=llama2-7b

# Launch llama2 70b
# export MODEL_NAME=llama2-70b

# Launch llama3.1 405b
export MODEL_NAME=llama3.1-405b

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

export JAX_ENABLE_PGLE=true
# export JAX_ENABLE_PGLE=false
export STRICT_CHECKER=true

export JAX_PGLE_AGGREGATION_PERCENTILE=50
export JAX_SHARE_AUTOTUNE_CONFIG_BETWEEN_HOSTS=true
export JAX_PGLE_PROFILING_RUNS=3

cat <<EOF > env.txt

NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE=/usr/local/nvidia/lib64/a3plus_guest_config.textproto
NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS=600000
JAX_ENABLE_PGLE=$JAX_ENABLE_PGLE
JAX_REMOVE_CUSTOM_PARTITIONING_PTR_FROM_CACHE_KEY=true
NVTE_FUSED_ATTN=1
XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
CUDA_DEVICE_MAX_CONNECTIONS=1
XLA_FLAGS=--xla_gpu_enable_latency_hiding_scheduler=true \
--xla_gpu_enable_command_buffer= \
--xla_gpu_enable_triton_gemm=false \
--xla_gpu_enable_highest_priority_async_stream=true \
--xla_gpu_all_reduce_combine_threshold_bytes=2147483648 \
--xla_gpu_all_gather_combine_threshold_bytes=2147483648 \
--xla_gpu_reduce_scatter_combine_threshold_bytes=2147483648 \
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
# export LOCAL_IMAGE_NAME=gcr.io/supercomputer-testing/lance-1223-fixed
# export LOCAL_IMAGE_NAME=gcr.io/supercomputer-testing/lance-0106-pinned
export LOCAL_IMAGE_NAME=gcr.io/supercomputer-testing/lance-1204-pinned

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

    local DCN_FSDP=${args[DCN_FSDP]}
    local DCN_PP=${args[DCN_PP]}
    local NUM_LAYERS_PER_PP_STAGE=${args[NUM_LAYERS_PER_PP_STAGE]}

    local REMAT_POLICY=${args[REMAT_POLICY]}

    local ATTENTION=${args[ATTENTION]}

    local WORKLOAD_NAME=$USER-$MODEL_SIZE-${NUM_NODES}n${PER_DEVICE_BATCH_SIZE}b${ICI_TP}tp${DCN_FSDP}fsdp${DCN_PP}pp${NUM_LAYERS_PER_PP_STAGE}l-${RANDOM:0:2}

    echo 'NUM_NODES' ${NUM_NODES} 'PER_DEVICE_BATCH_SIZE' ${PER_DEVICE_BATCH_SIZE} 'ICI_TP' ${ICI_TP} 'DCN_FSDP' ${DCN_FSDP} 'DCN_PP' ${DCN_PP} 'NUM_LAYERS_PER_PP_STAGE' ${NUM_LAYERS_PER_PP_STAGE} 'REMAT_POLICY' ${REMAT_POLICY} 'ATTENTION' ${ATTENTION} WORKLOAD_NAME ${WORKLOAD_NAME}

    COMMAND="python3 MaxText/train.py MaxText/configs/models/gpu/$CONFIG_NAME.yml hardware=gpu run_name=$RUN_NAME steps=10 max_target_length=4096 model_name=$MODEL_NAME enable_checkpointing=false attention=$ATTENTION dataset_type=synthetic async_checkpointing=false base_output_directory=$OUTPUT_BUCKET logits_dot_in_fp32=false use_iota_embed=true scan_layers=true ici_tensor_parallelism=$ICI_TP dcn_fsdp_parallelism=$DCN_FSDP dcn_pipeline_parallelism=$DCN_PP per_device_batch_size=$PER_DEVICE_BATCH_SIZE num_layers_per_pipeline_stage=$NUM_LAYERS_PER_PP_STAGE weight_dtype=bfloat16 remat_policy=$REMAT_POLICY profiler=xplane skip_first_n_steps_for_profiler=5 "; 

    COMMAND='export LD_LIBRARY_PATH=/usr/local/cuda-12.6/compat:$LD_LIBRARY_PATH;'"${COMMAND};""gsutil -m cp -r /tmp/xla_dump/ $OUTPUT_BUCKET"; 

    echo 'COMMAND is:' ${COMMAND}
    python ../xpk/xpk.py workload delete --cluster $CLUSTER_NAME --workload $WORKLOAD_NAME; 
    python ../xpk/xpk.py workload create --cluster $CLUSTER_NAME --workload $WORKLOAD_NAME --command "${COMMAND}" --docker-image=$LOCAL_IMAGE_NAME --device-type=$DEVICE_TYPE --num-nodes=$NUM_NODES --scheduler=gke.io/topology-aware-auto --env-file=env.txt ;
}


# Test run with 2 nodes, cuda kernel fails
# call_config --NUM_LAYERS 126  --NUM_NODES 2 --PER_DEVICE_BATCH_SIZE 1 --ICI_TP 8 --DCN_FSDP 2 --DCN_PP 1 --REMAT_POLICY save_qkv_proj --ATTENTION dot_product --NUM_LAYERS_PER_PP_STAGE 0
# call_config --NUM_LAYERS 126  --NUM_NODES 2 --PER_DEVICE_BATCH_SIZE 1 --ICI_TP 8 --DCN_FSDP 2 --DCN_PP 1 --REMAT_POLICY save_qkv_proj --ATTENTION cudnn_flash_te --NUM_LAYERS_PER_PP_STAGE 0

# PP
# 2 stages, we can have (repeat, layers per stage) as (1, 63), (3, 21), (7, 9), (9, 7), layers per stage needs to be long enough to cover the propogation latency between stages, repeats are auto calculated
# call_config --NUM_LAYERS 126  --NUM_NODES 128 --PER_DEVICE_BATCH_SIZE 1 --ICI_TP 8 --ICI_FSDP 1 --DCN_FSDP 64 --DCN_PP 2 --NUM_LAYERS_PER_PP_STAGE 21 --REMAT_POLICY save_qkv_proj --ATTENTION cudnn_flash_te 

# Non PP, this one should be the best config, but TE has issue with ICI TP
call_config --NUM_LAYERS 126  --NUM_NODES 128 --PER_DEVICE_BATCH_SIZE 1 --ICI_TP 8 --DCN_FSDP 128 --DCN_PP 1 --REMAT_POLICY save_qkv_proj --ATTENTION cudnn_flash_te --NUM_LAYERS_PER_PP_STAGE 0
