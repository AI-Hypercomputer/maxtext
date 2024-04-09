echo "Running 16vm.sh"
# Example command to invoke this script
# bash MaxText/configs/a3/llama_2_7b/16vm.sh
#
# Example command to invoke this script via XPK
# python3 xpk/xpk.py workload create --cluster ${CLUSTER_NAME} \
# --workload ${WORKLOAD_NAME} --docker-image=gcr.io/supercomputer-testing/${LOCAL_IMAGE_NAME} \
# --device-type ${DEVICE_TYPE} --num-slices 16 --priority=high \
# --command "bash MaxText/configs/a3/llama_2_7b/16vm.sh"

# Stop execution if any command exits with error
set -e

export OUTPUT_PATH="gs://maxtext-experiments-multipod"
export RUN_NAME="llama-2-16vm-$(date +%Y-%m-%d-%H-%M)"

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=1
export NCCL_DEBUG=VERSION
export XLA_FLAGS="--xla_dump_to=$OUTPUT_PATH/$RUN_NAME/HLO_dumps/
--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_async_all_gather=true --xla_gpu_enable_async_reduce_scatter=true
--xla_gpu_enable_triton_gemm=false --xla_gpu_simplify_all_fp_conversions --xla_gpu_graph_level=0
--xla_gpu_enable_async_all_reduce=true --xla_gpu_enable_highest_priority_async_stream=true
--xla_gpu_all_reduce_combine_threshold_bytes=1073741824 --xla_gpu_all_gather_combine_threshold_bytes=134217728
--xla_gpu_reduce_scatter_combine_threshold_bytes=134217728 --xla_gpu_enable_pipelined_all_gather=true
--xla_gpu_enable_pipelined_reduce_scatter=true --xla_gpu_enable_pipelined_all_reduce=true
--xla_gpu_enable_while_loop_double_buffering=true --xla_gpu_enable_triton_softmax_fusion=false
--xla_gpu_enable_all_gather_combine_by_dim=false --xla_gpu_enable_reduce_scatter_combine_by_dim=false
--xla_disable_hlo_passes=rematerialization"

# 16 nodes
python MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME hardware=gpu \
    steps=30 dcn_data_parallelism=16 ici_fsdp_parallelism=8 per_device_batch_size=4 max_target_length=4096 model_name=llama2-7b \
    enable_checkpointing=false attention=cudnn_flash_te remat_policy=minimal_flash use_iota_embed=true scan_layers=false \
    dataset_type=synthetic async_checkpointing=false base_output_directory=gs://runner-maxtext-logs enable_profiler=true
