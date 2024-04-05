#!/bin/bash 

echo "Running 2vm.sh"
# This is the config for llama-2 7B model
# for 2 VMs of GPUs using XPK

export CLUSTER_NAME=maxtext-a3-20n
export WORKLOAD_NAME=llama-2-16vm-$(date +%m-%d-%H-%M)
export LOCAL_IMAGE_NAME=yooh/maxtext-tcpx
export DEVICE_TYPE=h100-80gb-8

# Build and upload image
bash docker_build_dependency_image.sh DEVICE=gpu
bash docker_upload_runner.sh CLOUD_IMAGE_NAME=${LOCAL_IMAGE_NAME}

# Write XLA flags as env file
cat << EOF > xpk/env_2.txt
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
CUDA_DEVICE_MAX_CONNECTIONS=1
NVTE_FUSED_ATTN=1
NCCL_DEBUG=VERSION
XLA_FLAGS=--xla_dump_to=gs://runner-maxtext-logs/yooh/llama2-7b-$(date +%Y-%m-%d-%H-%M)/HLO_dumps/ \
--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_async_all_gather=true --xla_gpu_enable_async_reduce_scatter=true \
--xla_gpu_enable_triton_gemm=false --xla_gpu_simplify_all_fp_conversions --xla_gpu_graph_level=0 \
--xla_gpu_enable_async_all_reduce=true --xla_gpu_enable_highest_priority_async_stream=true \
--xla_gpu_all_reduce_combine_threshold_bytes=67108864 --xla_gpu_all_gather_combine_threshold_bytes=134217728 \
--xla_gpu_reduce_scatter_combine_threshold_bytes=67108864 --xla_gpu_enable_pipelined_all_gather=true \
--xla_gpu_enable_pipelined_reduce_scatter=true --xla_gpu_enable_pipelined_all_reduce=true \
--xla_gpu_enable_while_loop_double_buffering=true --xla_gpu_enable_triton_softmax_fusion=false \
--xla_gpu_enable_all_gather_combine_by_dim=false --xla_gpu_enable_reduce_scatter_combine_by_dim=false \
--xla_disable_hlo_passes=rematerialization
EOF


# 2 nodes
python3 xpk/xpk.py workload create --cluster ${CLUSTER_NAME} --workload ${WORKLOAD_NAME} \
    --docker-image=${LOCAL_IMAGE_NAME} --device-type=${DEVICE_TYPE}  --num-slices=2 --env-file=xpk/env_2.txt \
    --command "python MaxText/train.py MaxText/configs/base.yml hardware=gpu \
    steps=30 dcn_data_parallelism=2 ici_fsdp_parallelism=8 per_device_batch_size=4 max_target_length=4096 model_name=llama2-7b \
    enable_checkpointing=false attention=cudnn_flash_te remat_policy=minimal_flash use_iota_embed=true scan_layers=false \
    dataset_type=synthetic async_checkpointing=false base_output_directory=gs://runner-maxtext-logs enable_profiler=true" 

