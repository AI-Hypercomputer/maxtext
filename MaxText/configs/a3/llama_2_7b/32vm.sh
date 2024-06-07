echo "Running 32vm.sh"

# Example command to invoke this script via XPK on A3 or A3+:
# For A3, you can set DEVICE_TYPE as `h100-80gb-8`.
# For A3+, you can set DEVICE_TYPE as `h100-mega-80gb-8`.
#
# python3 xpk/xpk.py workload create --cluster ${CLUSTER_NAME} \
# --workload ${WORKLOAD_NAME} --docker-image ${LOCAL_IMAGE_NAME} \
# --device-type ${DEVICE_TYPE} --num-nodes 32 \
# --command "bash MaxText/configs/a3/llama_2_7b/32vm.sh"

# Stop execution if any command exits with error
set -e

export OUTPUT_PATH="gs://maxtext-experiments-multipod"
export RUN_NAME="llama-2-32vm-$(date +%Y-%m-%d-%H-%M)"

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

export XLA_FLAGS="--xla_dump_to=$OUTPUT_PATH/$RUN_NAME/HLO_dumps/
 --xla_dump_hlo_pass_re=.* --xla_gpu_all_reduce_contiguous=true
 --xla_gpu_enable_latency_hiding_scheduler=true
 --xla_gpu_enable_triton_gemm=false --xla_gpu_graph_level=0
 --xla_gpu_enable_highest_priority_async_stream=true
 --xla_gpu_all_reduce_combine_threshold_bytes=1073741824 --xla_gpu_all_gather_combine_threshold_bytes=134217728
 --xla_gpu_reduce_scatter_combine_threshold_bytes=134217728 --xla_gpu_enable_pipelined_all_gather=true
 --xla_gpu_enable_pipelined_reduce_scatter=true --xla_gpu_enable_pipelined_all_reduce=true
 --xla_gpu_enable_while_loop_double_buffering=true --xla_gpu_enable_triton_softmax_fusion=false
 --xla_gpu_enable_all_gather_combine_by_dim=false --xla_gpu_enable_reduce_scatter_combine_by_dim=false
 --xla_disable_hlo_passes=rematerialization"

# 32 node, DCN_DP=32, ICI_FSDP=8
python MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME hardware=gpu \
    steps=30 dcn_data_parallelism=32 ici_fsdp_parallelism=8 per_device_batch_size=4 max_target_length=4096 model_name=llama2-7b \
    enable_checkpointing=false attention=cudnn_flash_te remat_policy=minimal_flash use_iota_embed=true scan_layers=false \
    dataset_type=synthetic async_checkpointing=false base_output_directory=$OUTPUT_PATH logits_dot_in_fp32=false profiler=xplane
