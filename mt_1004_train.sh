#!/bin/bash

# Launch llama3.1 405b
export MODEL_NAME=llama3.1-405b

# Common parameters
export CLUSTER_NAME=a3plus-benchmark
export ZONE=australia-southeast1
export WORKLOAD_NAME=lance-$(echo $MODEL_NAME | sed 's/\.//g')

#export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/llama2-xprof_1001_lance # 128 layers
# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/llama2-xprof_1002_126layers_lance
# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/llama2-xprof_1003_nolayers_pinned_lance
export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/llama2-xprof_1004_nolayers_nightly_lance

export DEVICE_TYPE=h100-mega-80gb-8
export OUTPUT_BUCKET=gs://lancewang-dev-supercomputer-testing/maxtext_gpu
export RUN_NAME=maxtext-$MODEL_NAME
CONFIG_NAME=$(echo $MODEL_NAME | sed 's/-/_/g')

export JAX_ENABLE_PGLE=false

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
--xla_gpu_shard_autotuning=false \
--xla_dump_to=$OUTPUT_BUCKET/xla ";


export WORKLOAD_NAME=$USER-$(echo $MODEL_NAME | sed 's/\.//g')-${RANDOM:0:2}

export NUM_NODES=128

export PER_DEVICE_BATCH_SIZE=1
export ICI_TP=8

export DCN_FSDP=$(expr $NUM_NODES / 2)
# export DCN_FSDP=$(expr $NUM_NODES / 4)
#export DCN_FSDP=32

export DCN_PP=1
export NUM_LAYERS_PER_PP_STAGE=$(expr 128 / $DCN_PP) # Layers are modified to 128 for short term solution
# export NUM_LAYERS_PER_PP_STAGE=$(expr 126 / $DCN_PP)

export REMAT_POLICY=save_qkv_proj
# export REMAT_POLICY=full


COMMAND="python3 MaxText/train.py MaxText/configs/models/gpu/$CONFIG_NAME.yml hardware=gpu run_name=$RUN_NAME steps=10 max_target_length=4096 model_name=$MODEL_NAME enable_checkpointing=false attention=cudnn_flash_te dataset_type=synthetic async_checkpointing=false base_output_directory=$OUTPUT_BUCKET logits_dot_in_fp32=false dcn_pipeline_parallelism=$DCN_PP num_layers_per_pipeline_stage=$NUM_LAYERS_PER_PP_STAGE  dcn_fsdp_parallelism=$DCN_FSDP per_device_batch_size=$PER_DEVICE_BATCH_SIZE ici_tensor_parallelism=$ICI_TP weight_dtype=bfloat16 remat_policy=$REMAT_POLICY profiler=xplane skip_first_n_steps_for_profiler=6 base_num_decoder_layers=126"; 

COMMAND='export LD_LIBRARY_PATH=/usr/local/cuda-12.6/compat:$LD_LIBRARY_PATH;'"${COMMAND}"; 

xpk workload delete --cluster $CLUSTER_NAME --workload $WORKLOAD_NAME; xpk workload create --cluster $CLUSTER_NAME --workload $WORKLOAD_NAME --command "${COMMAND}" --docker-image=$LOCAL_IMAGE_NAME --device-type=$DEVICE_TYPE --num-nodes=$NUM_NODES --priority=high --scheduler=gke.io/topology-aware-auto --env NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE=/usr/local/nvidia/lib64/a3plus_guest_config.textproto --env NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS=600000 --env XLA_FLAGS="${XLA_FLAGS}" --env JAX_ENABLE_PGLE="${JAX_ENABLE_PGLE}" --env NCCL_DEBUG=INFO --env NCCL_DEBUG_SUBSYS=INIT,NET,ENV,TUNING,COLL


