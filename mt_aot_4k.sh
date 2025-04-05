#!/bin/bash

export WORKLOAD_NAME=$USER-$(echo $MODEL_NAME | sed 's/\.//g')-aot
export NUM_NODES=1;
export CLUSTER_NAME=a3plus-benchmark
export ZONE=us-central1-b
export PROJECT=gce-gpus-validation-2

export JAX_ENABLE_PGLE=false

# 405B
export MODEL_SIZE=405b
# 1k chips
export TARGET_NUM_NODES=128;
export PER_DEVICE_BATCH_SIZE=1

# 4k chips
# export TARGET_NUM_NODES=512;
# export PER_DEVICE_BATCH_SIZE=2


# export ICI_TP=8

export DCN_FSDP=$TARGET_NUM_NODES
#export DCN_FSDP=32

export DCN_PP=1
export NUM_LAYERS_PER_PP_STAGE=$(expr 126 / $DCN_PP)


# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/llama2-xprof_1002_126layers_lance
# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/llama2-xprof_1003_nolayers_pinned_lance
export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/llama2-1022_405b_lance

export ATTENTION=cudnn_flash_te
# export REMAT_POLICY=minimal
export REMAT_POLICY=save_qkv_proj
# export REMAT_POLICY=full
# export REMAT_POLICY=save_dot_except_mlpwi
# export REMAT_POLICY=save_dot_except_mlp

#70B
export MODEL_SIZE=70b
export TARGET_NUM_NODES=512;
export PER_DEVICE_BATCH_SIZE=3
export ICI_TP=1
export ICI_FSDP=8
export DCN_TP=1
export DCN_FSDP=16
export DCN_PP=1
export NUM_LAYERS_PER_PP_STAGE=$(expr 126 / $DCN_PP)
export REMAT_POLICY=save_qkv_proj



COMMAND="python MaxText/train_compile.py MaxText/configs/models/gpu/llama3.1_$MODEL_SIZE.yml hardware=gpu base_output_directory=$OUTPUT_BUCKET/aot dataset_type=synthetic tokenizer_path=assets/tokenizer_llama3.tiktoken per_device_batch_size=$PER_DEVICE_BATCH_SIZE ici_tensor_parallelism=$ICI_TP ici_fsdp_parallelism=$ICI_FSDP dcn_tensor_parallelism=$DCN_TP dcn_fsdp_parallelism=$DCN_FSDP dcn_pipeline_parallelism=$DCN_PP num_layers_per_pipeline_stage=$NUM_LAYERS_PER_PP_STAGE max_target_length=4096 run_name=runner_finetune steps=10 enable_checkpointing=false model_name=llama3.1-$MODEL_SIZE compile_topology=a3 compile_topology_num_slices=$TARGET_NUM_NODES logits_dot_in_fp32=false weight_dtype=bfloat16 opt_type=adamw attention=$ATTENTION profiler=xplane skip_first_n_steps_for_profiler=0 remat_policy=$REMAT_POLICY base_num_decoder_layers=126";

COMMAND='export LD_LIBRARY_PATH=/usr/local/cuda-12.6/compat:$LD_LIBRARY_PATH;echo yy1$LD_LIBRARY_PATH;'"${COMMAND}";

xpk workload delete --project $PROJECT --cluster $CLUSTER_NAME --zone $ZONE --workload $WORKLOAD_NAME; 

xpk workload create --project $PROJECT --cluster $CLUSTER_NAME --zone $ZONE  --workload $WORKLOAD_NAME --command "${COMMAND}" --docker-image=$LOCAL_IMAGE_NAME --device-type=$DEVICE_TYPE --num-nodes=$NUM_NODES --priority=high --scheduler=gke.io/topology-aware-auto \
--env NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE=/usr/local/nvidia/lib64/a3plus_guest_config.textproto \
--env NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS=600000 --env XLA_FLAGS="${XLA_FLAGS}" --env JAX_ENABLE_PGLE="${JAX_ENABLE_PGLE}"

