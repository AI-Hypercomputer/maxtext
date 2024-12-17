
# export LOCAL_IMAGE_NAME=gcr.io/tpu-prod-env-multipod/jonbolin-maxtext-gpu:20241008-1
# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/lance-1113-nv
# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/lance-1203-main
# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/lance-1119-dev-rebased
# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/lance-1209-main-stable
# export LOCAL_IMAGE_NAME=gcr.io/tpu-prod-env-multipod/bvandermoon-maxtext-gpu-nightly-11092024-no-context
# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/lance-1209-main-nightly-v1.11
# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/lance-1209-main-nightly-v1.10
# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/lance-1209-main-nightly-v1.10-jax1008
# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/lance-1209-main-nightly-v1.10-jax1024
# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/lance-1209-main-nightly-v1.10-jax1023
# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/lance-1209-main-nightly-v1.12-jax1109_cuda1125
# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/lance-1209-main-nightly-v1.11-jax1008_cuda1007
# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/lance-1209-main-nightly-constraints
# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/lance-1209-main-nightly-all-constraints
# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/lance-1212_bumpup
# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/yujunzou/llama2-7b
# export LOCAL_IMAGE_NAME=gcr.io/tpu-prod-env-multipod/maxtext_gpu_jax_pinned
# export LOCAL_IMAGE_NAME=gcr.io/supercomputer-testing/us-west1-docker.pkg.dev/supercomputer-testing/lancewang/lance-1216-sts
export LOCAL_IMAGE_NAME=gcr.io/supercomputer-testing/us-west1-docker.pkg.dev/supercomputer-testing/lancewang/lance-1216-bumpup

export MODEL_NAME=llama2-7b
# export MODEL_NAME=llama3.1-405b
# export MODEL_NAME=llama2-70b

export CONFIG_NAME=$(echo $MODEL_NAME | sed 's/-/_/g')
export MODEL_SIZE=$(echo $MODEL_NAME | grep -o '[0-9]\+b')
export NUM_NODES=2

export WORKLOAD_NAME=$USER-${MODEL_SIZE}-${NUM_NODES}n-stable-stack-${RANDOM:0:3}
export ATTENTION=dot_product
# export ATTENTION=cudnn_flash_te
export PGLE=false

python ../xpk/xpk.py  workload delete --cluster a3plus-benchmark --workload $WORKLOAD_NAME;



python ../xpk/xpk.py  workload create --device-type h100-mega-80gb-8 --project supercomputer-testing --zone australia-southeast1 --cluster a3plus-benchmark \
  --docker-image $LOCAL_IMAGE_NAME \
  --command 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/compat:$LD_LIBRARY_PATH;'"python3 MaxText/train.py MaxText/configs/models/gpu/$CONFIG_NAME.yml run_name=maxtext-$MODEL_NAME model_name=$MODEL_NAME attention=$ATTENTION use_iota_embed=true per_device_batch_size=1 skip_first_n_steps_for_profiler=5 profiler=xplane steps=10 hardware=gpu enable_checkpointing=false base_output_directory=gs://lancewang-dev-supercomputer-testing/maxtext_gpu dataset_type=synthetic remat_policy=full logits_dot_in_fp32=false dcn_fsdp_parallelism=$NUM_NODES ici_tensor_parallelism=8 max_target_length=4096 weight_dtype=bfloat16" \
  --num-nodes $NUM_NODES \
  --workload $WORKLOAD_NAME \
  --scheduler=gke.io/topology-aware-auto \
  --env NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE=/usr/local/nvidia/lib64/a3plus_guest_config.textproto \
  --env NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS=600000 --env JAX_ENABLE_PGLE=$PGLE --env JAX_REMOVE_CUSTOM_PARTITIONING_PTR_FROM_CACHE_KEY=true --env XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
  --env XLA_FLAGS='--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false --xla_gpu_enable_highest_priority_async_stream=true --xla_gpu_all_reduce_combine_threshold_bytes=134217728 --xla_gpu_all_gather_combine_threshold_bytes=1073741824 --xla_gpu_reduce_scatter_combine_threshold_bytes=33554432 --xla_gpu_enable_pipelined_all_gather=true --xla_gpu_enable_pipelined_reduce_scatter=true --xla_gpu_enable_pipelined_all_reduce=true --xla_gpu_enable_while_loop_double_buffering=true --xla_gpu_enable_triton_softmax_fusion=false --xla_gpu_enable_all_gather_combine_by_dim=false --xla_gpu_enable_reduce_scatter_combine_by_dim=false --xla_disable_hlo_passes=rematerialization --xla_gpu_graph_level=0'


# python3 ../xpk/xpk.py workload create --cluster a3plus-benchmark \
# --workload $WORKLOAD_NAME --docker-image $LOCAL_IMAGE_NAME \
# --device-type h100-mega-80gb-8 --num-nodes $NUM_NODES \
# --command 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/compat:$LD_LIBRARY_PATH;'"export XLA_FLAGS='--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_shard_autotuning=false
# --xla_gpu_enable_triton_gemm=false --xla_gpu_graph_level=0
# --xla_gpu_enable_highest_priority_async_stream=true
# --xla_gpu_all_reduce_combine_threshold_bytes=67108864 --xla_gpu_all_gather_combine_threshold_bytes=134217728
# --xla_gpu_reduce_scatter_combine_threshold_bytes=67108864 --xla_gpu_enable_pipelined_all_gather=true
# --xla_gpu_enable_pipelined_reduce_scatter=true --xla_gpu_enable_pipelined_all_reduce=true
# --xla_gpu_enable_while_loop_double_buffering=true --xla_gpu_enable_triton_softmax_fusion=false
# --xla_gpu_enable_all_gather_combine_by_dim=false --xla_gpu_enable_reduce_scatter_combine_by_dim=false
# --xla_disable_hlo_passes=rematerialization'; python MaxText/train.py MaxText/configs/models/gpu/llama2_7b.yml run_name=llama-2-2vm-1209 dcn_fsdp_parallelism=2 per_device_batch_size=1 ici_tensor_parallelism=8 base_output_directory=gs://lancewang-dev-supercomputer-testing/maxtext_gpu profiler=xplane" --scheduler=gke.io/topology-aware-auto

