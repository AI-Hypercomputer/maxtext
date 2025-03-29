#!/bin/bash

export WORKLOAD_NAME=$USER-$(echo $MODEL_NAME | sed 's/\.//g')-pgle-error-fsdp

export NUM_NODES=32

export PER_DEVICE_BATCH_SIZE=1
# export ICI_TP=8

export DCN_FSDP=$NUM_NODES
#export DCN_FSDP=32

export DCN_PP=1
export NUM_LAYERS_PER_PP_STAGE=$(expr 128 / $DCN_PP) # Layers are modified to 128 for short term solution
# export NUM_LAYERS_PER_PP_STAGE=$(expr 126 / $DCN_PP)

export JAX_ENABLE_PGLE=true
# export REMAT_POLICY=save_qkv_proj
export REMAT_POLICY=full

# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/llama2-xprof_1001_lance
# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/llama2-xprof_1002_126layers_lance
# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/llama2-xprof_1003_nolayers_pinned_lance

#

COMMAND="python3 MaxText/train.py MaxText/configs/models/gpu/$CONFIG_NAME.yml hardware=gpu run_name=$RUN_NAME steps=10 max_target_length=4096 model_name=$MODEL_NAME enable_checkpointing=false attention=cudnn_flash_te dataset_type=synthetic async_checkpointing=false base_output_directory=$OUTPUT_BUCKET logits_dot_in_fp32=false dcn_pipeline_parallelism=$DCN_PP num_layers_per_pipeline_stage=$NUM_LAYERS_PER_PP_STAGE  dcn_fsdp_parallelism=$DCN_FSDP per_device_batch_size=$PER_DEVICE_BATCH_SIZE ici_tensor_parallelism=$ICI_TP weight_dtype=bfloat16 remat_policy=$REMAT_POLICY profiler=xplane skip_first_n_steps_for_profiler=0 base_num_decoder_layers=126"; 

COMMAND='export LD_LIBRARY_PATH=/usr/local/cuda-12.6/compat:$LD_LIBRARY_PATH;'"${COMMAND}"; 

python3 xpk.py workload delete --cluster $CLUSTER_NAME --workload $WORKLOAD_NAME; python3 xpk.py workload create --cluster $CLUSTER_NAME --workload $WORKLOAD_NAME --command "${COMMAND}" --docker-image=$LOCAL_IMAGE_NAME --device-type=$DEVICE_TYPE --num-nodes=$NUM_NODES --priority=high --scheduler=gke.io/topology-aware-auto --env NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE=/usr/local/nvidia/lib64/a3plus_guest_config.textproto --env NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS=600000 --env XLA_FLAGS="${XLA_FLAGS}" --env JAX_ENABLE_PGLE="${JAX_ENABLE_PGLE}"


