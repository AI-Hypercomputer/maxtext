echo "Running 1vm.sh"

# Example command to submit this script with Cluster Toolkit (gcluster). See
# docs/run_maxtext/run_maxtext_via_cluster_toolkit.md to install gcluster and set the default
# project, cluster, and location (`gcluster job config set ...`). COMPUTE_TYPE is the GPU
# type, e.g. h100-80gb-8 (a3-highgpu-8g).
# gcluster job submit --name=${WORKLOAD_NAME} \
# --image=gcr.io/${PROJECT_ID}/${LOCAL_IMAGE_NAME} \
# --compute-type=${COMPUTE_TYPE} --num-nodes=1 \
# --command "bash src/maxtext/configs/gpu/a3/llama_2_7b/1vm.sh"

# Stop execution if any command exits with error
set -e

export OUTPUT_PATH="gs://maxtext-experiments-multipod"
export RUN_NAME="llama-2-1vm-$(date +%Y-%m-%d-%H-%M)"

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

export XLA_FLAGS="--xla_dump_to=$OUTPUT_PATH/$RUN_NAME/HLO_dumps/
--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false
 --xla_gpu_enable_command_buffer='' --xla_gpu_enable_highest_priority_async_stream=true
 --xla_gpu_all_reduce_combine_threshold_bytes=134217728 --xla_gpu_all_gather_combine_threshold_bytes=134217728
 --xla_gpu_reduce_scatter_combine_threshold_bytes=67108864 --xla_gpu_pipeline_all_gather=on
 --xla_gpu_pipeline_reduce_scatter=on --xla_gpu_pipeline_all_reduce=on
 --xla_gpu_enable_while_loop_double_buffering=true
 --xla_gpu_enable_all_gather_combine_by_dim=false --xla_gpu_enable_reduce_scatter_combine_by_dim=false
 --xla_disable_hlo_passes=rematerialization"


# 1 node, DATA_DP=1, ICI_FSDP=8
python3 -m maxtext.trainers.pre_train.train "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"/models/gpu/llama2_7b.yml run_name=$RUN_NAME \
    dcn_data_parallelism=1 ici_fsdp_parallelism=8 base_output_directory=$OUTPUT_PATH profiler=xplane
