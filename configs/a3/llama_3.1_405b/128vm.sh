echo "Running 128vm.sh"
# Example command to invoke this script via XPK, assume you've installed xpk
# COMMAND="bash MaxText/configs/a3/llama_3.1_405b/128vm.sh"
# COMMAND='export LD_LIBRARY_PATH=/usr/local/cuda-12.6/compat:$LD_LIBRARY_PATH;'"${COMMAND}";
#
# xpk workload create --project=${PROJECT}--cluster=${CLUSTER_NAME} --zone=${ZONE} \
# --workload=${WORKLOAD_NAME} --docker-image=gcr.io/supercomputer-testing/${LOCAL_IMAGE_NAME} \
# --device-type=${DEVICE_TYPE} --num-nodes=2 --priority=high \
# --command="$COMMAND" --env=XLA_FLAGS=$XLA_FLAGS

# Stop execution if any command exits with error
set -e

export OUTPUT_PATH="gs://maxtext-experiments-multipod"
export RUN_NAME="llama-31-128vm-$(date +%Y-%m-%d-%H-%M)"
export EXECUTABLE="train"

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

export XLA_FLAGS="--xla_dump_to=$OUTPUT_PATH/$RUN_NAME/HLO_dumps/
--xla_gpu_enable_latency_hiding_scheduler=true
--xla_gpu_enable_triton_gemm=false --xla_gpu_enable_command_buffer=''
--xla_gpu_enable_highest_priority_async_stream=true
--xla_gpu_all_reduce_combine_threshold_bytes=1073741824 --xla_gpu_all_gather_combine_threshold_bytes=134217728
--xla_gpu_reduce_scatter_combine_threshold_bytes=134217728 --xla_gpu_enable_pipelined_all_gather=true
--xla_gpu_enable_pipelined_reduce_scatter=true --xla_gpu_enable_pipelined_all_reduce=true
--xla_gpu_enable_while_loop_double_buffering=true
--xla_gpu_enable_all_gather_combine_by_dim=false --xla_gpu_enable_reduce_scatter_combine_by_dim=false
--xla_disable_hlo_passes=rematerialization"

# 128 nodes
python3 -m MaxText.$EXECUTABLE MaxText/configs/models/llama3.1_405b.yml run_name=$RUN_NAME \
    base_config=base.yml \
    run_name=gpu_train_test \
    hardware=gpu \
    steps=10 \
    model_name=llama3.1-405b \
    enable_checkpointing: False \
    attention=cudnn_flash_te \
    remat_policy=full \
    use_iota_embed=True \
    scan_layers=True \
    dataset_type=synthetic \
    async_checkpointing=False \
    logits_dot_in_fp32=False \
    per_device_batch_size=1.0 \
    max_target_length=8192 \
    dcn_fsdp_parallelism=128 \
    ici_fsdp_parallelism=8 \
    base_output_directory=$OUTPUT_PATH \
    profiler=xplane

