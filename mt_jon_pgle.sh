export WORKLOAD_NAME=$USER-jon-7b-1n-${RANDOM:0:2}
# export LOCAL_IMAGE_NAME=gcr.io/tpu-prod-env-multipod/jonbolin-maxtext-gpu:20241008-1
export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/lance-1113-nv
export CONFIG_NAME=llama2_7b

python ../xpk/xpk.py  workload delete --cluster a3plus-benchmark --workload $WORKLOAD_NAME;



python ../xpk/xpk.py  workload create --device-type h100-mega-80gb-8 --project supercomputer-testing --zone australia-southeast1 --cluster a3plus-benchmark \
  --docker-image $LOCAL_IMAGE_NAME \
  --command 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/compat:$LD_LIBRARY_PATH;'"python3 MaxText/train.py MaxText/configs/models/gpu/$CONFIG_NAME.yml run_name=maxtext-$MODEL_NAME model_name=llama2-7b attention=cudnn_flash_te use_iota_embed=true per_device_batch_size=1 skip_first_n_steps_for_profiler=5 profiler=xplane steps=10 hardware=gpu enable_checkpointing=false base_output_directory=gs://lancewang-dev-supercomputer-testing/maxtext_gpu dataset_type=synthetic remat_policy=full logits_dot_in_fp32=false dcn_fsdp_parallelism=1 max_target_length=4096 weight_dtype=bfloat16" \
  --num-nodes 1 \
  --workload $WORKLOAD_NAME \
  --scheduler=gke.io/topology-aware-auto \
  --env NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE=/usr/local/nvidia/lib64/a3plus_guest_config.textproto \
  --env NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS=600000 --env JAX_ENABLE_PGLE=false --env JAX_REMOVE_CUSTOM_PARTITIONING_PTR_FROM_CACHE_KEY=true \
  --env XLA_FLAGS='--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false --xla_gpu_enable_highest_priority_async_stream=true --xla_gpu_all_reduce_combine_threshold_bytes=134217728 --xla_gpu_all_gather_combine_threshold_bytes=1073741824 --xla_gpu_reduce_scatter_combine_threshold_bytes=33554432 --xla_gpu_enable_pipelined_all_gather=true --xla_gpu_enable_pipelined_reduce_scatter=true --xla_gpu_enable_pipelined_all_reduce=true --xla_gpu_enable_while_loop_double_buffering=true --xla_gpu_enable_triton_softmax_fusion=false --xla_gpu_enable_all_gather_combine_by_dim=false --xla_gpu_enable_reduce_scatter_combine_by_dim=false --xla_disable_hlo_passes=rematerialization'
