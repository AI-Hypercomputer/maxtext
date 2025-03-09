#!/bin/bash

source mt_config.sh

# export WORKLOAD_NAME=$USER-pgle-dot-7b-1n-noxprof-jb1009
# export WORKLOAD_NAME=$USER-pgle-dot-7b-1n-noxprof-jb0830
# export WORKLOAD_NAME=$USER-pgle-dot-7b-1n-noxprof-yw0628
export WORKLOAD_NAME=$USER-pgle-dot-7b-1n-jb1008-tp-pgle

export NUM_NODES=2

export PER_DEVICE_BATCH_SIZE=1
export ICI_TP=8
# export ICI_TP=1

export DCN_FSDP=$NUM_NODES
#export DCN_FSDP=32

export DCN_PP=1
export NUM_LAYERS_PER_PP_STAGE=$(expr 128 / $DCN_PP) # Layers are modified to 128 for short term solution
# export NUM_LAYERS_PER_PP_STAGE=$(expr 126 / $DCN_PP)

# export REMAT_POLICY=save_qkv_proj
export REMAT_POLICY=full
# export REMAT_POLICY=minimal

# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/llama2-xprof_1001_lance
# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/llama2-xprof_1002_126layers_lance
# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/llama2-xprof_1003_nolayers_pinned_lance
# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/llama2-xprof_1010_nolayers_nightly_lance


# export LOCAL_IMAGE_NAME=gcr.io/supercomputer-testing/yangyuwei/maxtext-fastrak:06-12-2024
# export LOCAL_IMAGE_NAME=gcr.io/supercomputer-testing/yangyuwei/maxtext-fastrak:06-28-2024-pgle-nightly

# export LOCAL_IMAGE_NAME=gcr.io/tpu-prod-env-multipod/jonbolin-maxtext-gpu:20241008-1
# export LOCAL_IMAGE_NAME=gcr.io/tpu-prod-env-multipod/jonbolin-maxtext-gpu:no-pp-remat-1


# export ATTENTION=cudnn_flash_te
export ATTENTION=dot_product

# My command
COMMAND="python3 MaxText/train.py MaxText/configs/models/gpu/$CONFIG_NAME.yml hardware=gpu run_name=$RUN_NAME steps=10 max_target_length=4096 model_name=$MODEL_NAME enable_checkpointing=false attention=$ATTENTION dataset_type=synthetic async_checkpointing=false base_output_directory=$OUTPUT_BUCKET logits_dot_in_fp32=false use_iota_embed=true dcn_pipeline_parallelism=$DCN_PP dcn_fsdp_parallelism=$DCN_FSDP per_device_batch_size=$PER_DEVICE_BATCH_SIZE ici_tensor_parallelism=$ICI_TP weight_dtype=bfloat16 remat_policy=$REMAT_POLICY profiler=xplane skip_first_n_steps_for_profiler=5"; 

COMMAND='export LD_LIBRARY_PATH=/usr/local/cuda-12.6/compat:$LD_LIBRARY_PATH;'"${COMMAND}"; 

# xpk workload delete --cluster $CLUSTER_NAME --workload $WORKLOAD_NAME; xpk workload create --cluster $CLUSTER_NAME --workload $WORKLOAD_NAME --command "${COMMAND}" --docker-image=$LOCAL_IMAGE_NAME --device-type=$DEVICE_TYPE --num-nodes=$NUM_NODES --priority=high --scheduler=gke.io/topology-aware-auto --env-file=env.txt ;

# --debug_dump_gcs=$OUTPUT_BUCKET/xla

# python3 MaxText/train.py MaxText/configs/models/gpu/$CONFIG_NAME.yml 
# run_name=maxtext-$MODEL_NAME 
# model_name=llama2-7b 
# attention=cudnn_flash_te 
# use_iota_embed=true 
# per_device_batch_size=1 
# skip_first_n_steps_for_profiler=5 
# profiler=xplane 
# steps=10 
# hardware=gpu 
# enable_checkpointing=false 
# base_output_directory=gs://lancewang-dev-supercomputer-testing/maxtext_gpu 
# dataset_type=synthetic 
# remat_policy=full 
# logits_dot_in_fp32=false 
# dcn_fsdp_parallelism=1 
# max_target_length=4096 
# weight_dtype=bfloat16


# Running fine
# python3 MaxText/train.py MaxText/configs/models/gpu/$CONFIG_NAME.yml hardware=gpu run_name=$RUN_NAME steps=10 max_target_length=4096 model_name=$MODEL_NAME enable_checkpointing=false attention=$ATTENTION dataset_type=synthetic async_checkpointing=false base_output_directory=$OUTPUT_BUCKET logits_dot_in_fp32=false use_iota_embed=true dcn_pipeline_parallelism=$DCN_PP dcn_fsdp_parallelism=$DCN_FSDP per_device_batch_size=$PER_DEVICE_BATCH_SIZE ici_tensor_parallelism=$ICI_TP weight_dtype=bfloat16 remat_policy=$REMAT_POLICY profiler=xplane skip_first_n_steps_for_profiler=5

# --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false --xla_gpu_enable_highest_priority_async_stream=true --xla_gpu_all_reduce_combine_threshold_bytes=134217728 --xla_gpu_all_gather_combine_threshold_bytes=1073741824 --xla_gpu_reduce_scatter_combine_threshold_bytes=33554432 --xla_gpu_enable_pipelined_all_gather=true --xla_gpu_enable_pipelined_reduce_scatter=true --xla_gpu_enable_pipelined_all_reduce=true --xla_gpu_enable_while_loop_double_buffering=true --xla_gpu_enable_triton_softmax_fusion=false --xla_gpu_enable_all_gather_combine_by_dim=false --xla_gpu_enable_reduce_scatter_combine_by_dim=false --xla_disable_hlo_passes=rematerialization

# 7b 2 nodes FSDP
# export WORKLOAD_NAME=$USER-pgle-dot-7b-2n-jb1008-fsdp
# xpk workload delete --cluster a3plus-benchmark --workload $WORKLOAD_NAME;

# xpk workload create --device-type h100-mega-80gb-8 --project supercomputer-testing --zone australia-southeast1 --cluster a3plus-benchmark \
#   --docker-image gcr.io/tpu-prod-env-multipod/jonbolin-maxtext-gpu:20241008-1 \
#   --command 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/compat:$LD_LIBRARY_PATH;'"python3 MaxText/train.py MaxText/configs/models/gpu/$CONFIG_NAME.yml run_name=maxtext-$MODEL_NAME model_name=llama2-7b attention=dot_product use_iota_embed=true per_device_batch_size=1 skip_first_n_steps_for_profiler=5 profiler=xplane steps=10 hardware=gpu enable_checkpointing=false base_output_directory=gs://lancewang-dev-supercomputer-testing/maxtext_gpu dataset_type=synthetic remat_policy=full logits_dot_in_fp32=false ici_tensor_parallelism=1 ici_fsdp_parallelism=8 dcn_fsdp_parallelism=2 max_target_length=4096 weight_dtype=bfloat16" \
#   --num-nodes 2 \
#   --workload $WORKLOAD_NAME \
#   --scheduler=gke.io/topology-aware-auto \
#   --priority high \
#   --env NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE=/usr/local/nvidia/lib64/a3plus_guest_config.textproto \
#   --env NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS=600000 \
#   --env JAX_ENABLE_PGLE=true --env JAX_REMOVE_CUSTOM_PARTITIONING_PTR_FROM_CACHE_KEY=true --env JAX_SHARE_AUTOTUNE_CONFIG_BETWEEN_HOSTS=true --env JAX_DEBUG_LOG_MODULES=compiler \
#   --env XLA_FLAGS='--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false --xla_gpu_enable_highest_priority_async_stream=true --xla_gpu_all_reduce_combine_threshold_bytes=536870912 --xla_gpu_all_gather_combine_threshold_bytes=536870912 --xla_gpu_reduce_scatter_combine_threshold_bytes=536870912 --xla_gpu_enable_pipelined_all_gather=true --xla_gpu_enable_pipelined_reduce_scatter=true --xla_gpu_enable_pipelined_all_reduce=true --xla_gpu_enable_while_loop_double_buffering=true --xla_disable_hlo_passes=rematerialization --xla_gpu_enable_pgle_accuracy_checker=true --xla_gpu_enable_triton_softmax_fusion=false --xla_gpu_enable_all_gather_combine_by_dim=false --xla_gpu_enable_reduce_scatter_combine_by_dim=false'
  
# 70b 16 nodes FSDP
export WORKLOAD_NAME=$USER-pgle-dot-7b-2n-fsdp-nochk-jon

xpk workload delete --cluster a3plus-benchmark --workload $WORKLOAD_NAME;

xpk workload create --device-type h100-mega-80gb-8 --project supercomputer-testing --zone australia-southeast1 --cluster a3plus-benchmark \
  --docker-image gcr.io/tpu-prod-env-multipod/jonbolin-maxtext-gpu:20241008-1 \
  --command 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/compat:$LD_LIBRARY_PATH;'"python3 MaxText/train.py MaxText/configs/models/gpu/$CONFIG_NAME.yml run_name=maxtext-$MODEL_NAME model_name=llama2-70b attention=dot_product use_iota_embed=true per_device_batch_size=1 skip_first_n_steps_for_profiler=5 profiler=xplane steps=10 hardware=gpu enable_checkpointing=false base_output_directory=gs://lancewang-dev-supercomputer-testing/maxtext_gpu dataset_type=synthetic remat_policy=full logits_dot_in_fp32=false ici_tensor_parallelism=1 ici_fsdp_parallelism=8 dcn_fsdp_parallelism=2 max_target_length=4096 weight_dtype=bfloat16" \
  --num-nodes 2 \
  --workload $WORKLOAD_NAME \
  --scheduler=gke.io/topology-aware-auto \
  --priority high \
  --env NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE=/usr/local/nvidia/lib64/a3plus_guest_config.textproto \
  --env NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS=600000 \
  --env JAX_ENABLE_PGLE=true --env JAX_REMOVE_CUSTOM_PARTITIONING_PTR_FROM_CACHE_KEY=true --env JAX_SHARE_AUTOTUNE_CONFIG_BETWEEN_HOSTS=true --env JAX_DEBUG_LOG_MODULES=compiler \
  --env XLA_FLAGS='--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false --xla_gpu_enable_highest_priority_async_stream=true --xla_gpu_all_reduce_combine_threshold_bytes=536870912 --xla_gpu_all_gather_combine_threshold_bytes=536870912 --xla_gpu_reduce_scatter_combine_threshold_bytes=536870912 --xla_gpu_enable_pipelined_all_gather=true --xla_gpu_enable_pipelined_reduce_scatter=true --xla_gpu_enable_pipelined_all_reduce=true --xla_gpu_enable_while_loop_double_buffering=true --xla_disable_hlo_passes=rematerialization --xla_gpu_enable_pgle_accuracy_checker=false --xla_gpu_enable_triton_softmax_fusion=false --xla_gpu_enable_all_gather_combine_by_dim=false --xla_gpu_enable_reduce_scatter_combine_by_dim=false'
  




  # Passing config:
  # python3 MaxText/train.py MaxText/configs/models/gpu/$CONFIG_NAME.yml run_name=maxtext-$MODEL_NAME model_name=llama2-7b attention=cudnn_flash_te use_iota_embed=true per_device_batch_size=1 skip_first_n_steps_for_profiler=5 profiler=xplane steps=10 hardware=gpu enable_checkpointing=false base_output_directory=gs://lancewang-dev-supercomputer-testing/maxtext_gpu dataset_type=synthetic remat_policy=full logits_dot_in_fp32=false dcn_fsdp_parallelism=1 max_target_length=4096 weight_dtype=bfloat16 async_checkpointing=true ici_tensor_parallelism=$ICI_TP 

  
  # My config:
  # 


  # XLA_FLAGS='--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false --xla_gpu_enable_highest_priority_async_stream=true --xla_gpu_all_reduce_combine_threshold_bytes=536870912 --xla_gpu_all_gather_combine_threshold_bytes=536870912 --xla_gpu_reduce_scatter_combine_threshold_bytes=536870912 --xla_gpu_enable_pipelined_all_gather=true --xla_gpu_enable_pipelined_reduce_scatter=true --xla_gpu_enable_pipelined_all_reduce=true --xla_gpu_enable_while_loop_double_buffering=true --xla_disable_hlo_passes=rematerialization --xla_gpu_enable_pgle_accuracy_checker=true --xla_gpu_enable_triton_softmax_fusion=false --xla_gpu_enable_all_gather_combine_by_dim=false --xla_gpu_enable_reduce_scatter_combine_by_dim=false'


# fail1: remove the space between yml and run, still fail
# fail3: remove CUDA_DEVICE_MAX_CONNECTIONS=1, still fail
# fail4: use my command format, but add async_checkpointing=false, failed
# fail5: use Jon's format, but missing async_checkpointing=false, failed
# fail6: setting xla_gpu_enable_pgle_accuracy_checker to true and run Jon's script, failed: https://console.cloud.google.com/kubernetes/service/australia-southeast1/a3plus-benchmark/default/lancewang-pgle-fa-7b-1n-jb1008/details?project=supercomputer-testing
# pass6: setting to false and run Jon's script: https://pantheon.corp.google.com/kubernetes/pod/australia-southeast1/a3plus-benchmark/default/lancewang-pgle-fa-7b-1n-jb1008-p-slice-job-0-0-g6zf4/logs?e=13802955&mods=dataflow_dev&project=supercomputer-testing
# disalbe pgle and run Jon's script: 


