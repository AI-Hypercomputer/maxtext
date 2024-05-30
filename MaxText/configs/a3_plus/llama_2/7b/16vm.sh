echo "Running 16vm.sh"
# Example command to invoke this script
# bash MaxText/configs/a3_plus/llama_2/7b/16vm.sh
#
# Example command to invoke this script via XPK
# export LOCAL_IMAGE_NAME=us-central1-docker.pkg.dev/gce-ai-infra/maxtext/maxtext_base_image:04_26_2024_flash_release
# python3 xpk/xpk.py workload create --cluster ${CLUSTER_NAME} \
# --workload ${WORKLOAD_NAME} --docker-image ${LOCAL_IMAGE_NAME} \
# --device-type h100-mega-80gb-8 --num-nodes 16 --priority=high \
# --command "bash MaxText/configs/a3_plus/llama_2/7b/16vm.sh"

# Stop execution if any command exits with error
set -e

export OUTPUT_PATH="gs://maxtext-experiments-multipod"
export RUN_NAME="llama-2-16vm-$(date +%Y-%m-%d-%H-%M)"

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

export NCCL_FASTRAK_CTRL_DEV=eth0
export NCCL_FASTRAK_IFNAME=eth1,eth2,eth3,eth4,eth5,eth6,eth7,eth8
export NCCL_SOCKET_IFNAME=eth0
export NCCL_CROSS_NIC=0
export NCCL_ALGO=Ring,Tree
export NCCL_PROTO=Simple
export NCCL_MIN_NCHANNELS=4
export NCCL_DYNAMIC_CHUNK_SIZE=524288
export NCCL_P2P_NET_CHUNKSIZE=524288
export NCCL_P2P_PCI_CHUNKSIZE=524288
export NCCL_P2P_NVL_CHUNKSIZE=1048576
export NCCL_FASTRAK_NUM_FLOWS=2
export NCCL_FASTRAK_USE_SNAP=1
export NCCL_FASTRAK_ENABLE_CONTROL_CHANNEL=0
export NCCL_BUFFSIZE=8388608
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_NET_GDR_LEVEL=PIX
export NCCL_FASTRAK_ENABLE_HOTPATH_LOGGING=0
export NCCL_FASTRAK_USE_LLCM=1
export NCCL_DEBUG=VERSION
export TF_CPP_VMODULE=profile_guided_latency_estimator=10
export TF_CPP_MIN_LOG_LEVEL=0
export TF_CPP_MAX_LOG_LEVEL=100
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=1
export NCCL_NVLS_ENABLE=0
export NCCL_TUNER_PLUGIN=libnccl-tuner.so
export NCCL_TUNER_CONFIG_PATH=/usr/local/nvidia/lib64/a3plus_tuner_config.textproto
export NCCL_FASTRAK_ENABLE_CONTROL_CHANNEL=0
export XLA_FLAGS="--xla_dump_to=$OUTPUT_PATH/$RUN_NAME/HLO_dumps/
--xla_dump_hlo_pass_re=.*
--xla_gpu_cudnn_gemm_fusion_level=3
--xla_gpu_all_reduce_contiguous=true
--xla_gpu_enable_latency_hiding_scheduler=true
--xla_gpu_enable_async_all_gather=true
--xla_gpu_enable_async_reduce_scatter=true
--xla_gpu_enable_triton_gemm=true
--xla_gpu_simplify_all_fp_conversions
--xla_gpu_graph_level=0
--xla_gpu_enable_async_all_reduce=true
--xla_gpu_enable_highest_priority_async_stream=true
--xla_gpu_all_reduce_combine_threshold_bytes=134217728
--xla_gpu_all_gather_combine_threshold_bytes=134217728
--xla_gpu_reduce_scatter_combine_threshold_bytes=67108864
--xla_gpu_enable_pipelined_all_gather=true
--xla_gpu_enable_pipelined_reduce_scatter=true
--xla_gpu_enable_pipelined_all_reduce=true
--xla_gpu_enable_while_loop_double_buffering=true
--xla_gpu_enable_triton_softmax_fusion=false
--xla_gpu_enable_all_gather_combine_by_dim=false
--xla_gpu_enable_reduce_scatter_combine_by_dim=false
--xla_disable_hlo_passes=rematerialization"

# 16 nodes
python MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME hardware=gpu \
    steps=30 dcn_data_parallelism=16 ici_fsdp_parallelism=8 per_device_batch_size=4 max_target_length=4096 model_name=llama2-7b \
    enable_checkpointing=false attention=cudnn_flash_te remat_policy=minimal_flash use_iota_embed=true scan_layers=false \
    dataset_type=synthetic async_checkpointing=false base_output_directory=gs://runner-maxtext-logs enable_profiler=true logits_dot_in_fp32=false
