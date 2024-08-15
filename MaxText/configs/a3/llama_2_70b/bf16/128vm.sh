echo "Running 1vm.sh"

# Example command to invoke this script via XPK on A3 or A3+:
# For A3, you can set DEVICE_TYPE as `h100-80gb-8`.
# For A3+, you can set DEVICE_TYPE as `h100-mega-80gb-8`.
#
# export RUN_NAME="llama2-70b-1vm-$(date +%Y-%m-%d-%H-%M)"
# python3 xpk/xpk.py workload create --cluster ${CLUSTER_NAME} \
# --workload ${WORKLOAD_NAME} --docker-image ${LOCAL_IMAGE_NAME} \
# --device-type ${DEVICE_TYPE} --num-nodes 1 \
# --command "bash MaxText/configs/a3/llama_2_70b/bf16/128vm.sh"

# Stop execution if any command exits with error
set -e

export OUTPUT_PATH="gs://maxtext-experiments-multipod"

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

# The setup accommodates two cases:
# 1) Passing the 'RUN_NAME' variable at runtime
# 2) Propagating the 'M_RUN_NAME' variable within an Airflow sweeping workflow
if [ -n "$RUN_NAME" ];
then
    export M_RUN_NAME=$RUN_NAME
fi

export XLA_FLAGS="--xla_dump_to=$OUTPUT_PATH/$M_RUN_NAME/HLO_dumps/ --xla_dump_hlo_pass_re=.*
 --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_address_computation_fusion=false
 --xla_gpu_enable_triton_gemm=false --xla_gpu_graph_level=0 --xla_gpu_enable_highest_priority_async_stream=true
 --xla_gpu_all_reduce_combine_threshold_bytes=536870912 --xla_gpu_all_gather_combine_threshold_bytes=2147483648
 --xla_gpu_reduce_scatter_combine_threshold_bytes=33554432 --xla_gpu_enable_pipelined_all_gather=true
 --xla_gpu_enable_pipelined_all_reduce=true --xla_gpu_enable_while_loop_double_buffering=true 
 --xla_gpu_enable_triton_softmax_fusion=false --xla_gpu_enable_all_gather_combine_by_dim=false 
 --xla_gpu_enable_reduce_scatter_combine_by_dim=false --xla_disable_hlo_passes=rematerialization
 --xla_gpu_pgle_profile_file_or_directory_path=/app/MaxText/configs/a3/llama_2_70b/bf16/pgle/128vm.pbtxt"

# 128 nodes
python MaxText/train.py MaxText/configs/base.yml hardware=gpu base_output_directory=$OUTPUT_PATH \
    steps=30 model_name=llama2-70b enable_checkpointing=false attention=cudnn_flash_te dataset_type=synthetic \
    async_checkpointing=false profiler=xplane use_iota_embed=true scan_layers=true per_device_batch_size=4 \
    remat_policy=save_qkv_proj logits_dot_in_fp32=false max_target_length=4096 \
    ici_fsdp_parallelism=8 dcn_fsdp_parallelism=16 ici_data_parallelism=1 dcn_data_parallelism=8 \
    ici_tensor_parallelism=1 dcn_tensor_parallelism=1

