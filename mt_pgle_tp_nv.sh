#!/bin/bash

# Common parameters
export CLUSTER_NAME=a3plus-benchmark
export ZONE=australia-southeast1
export DEVICE_TYPE=h100-mega-80gb-8
export WORKLOAD_NAME=lancewang-7b-nv-pgle-tp-${RANDOM:0:2}
export NUM_NODES=1

# Enable PGLE
export JAX_ENABLE_PGLE=false
export STRICT_CHECKER=true

cat <<EOF > env.txt
JAX_ENABLE_COMPILATION_CACHE=True
JAX_ENABLE_PGLE=$JAX_ENABLE_PGLE
JAX_REMOVE_CUSTOM_PARTITIONING_PTR_FROM_CACHE_KEY=$JAX_ENABLE_PGLE
NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE=/usr/local/nvidia/lib64/a3plus_guest_config.textproto
NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS=600000
JAX_DEBUG_LOG_MODULES=jax._src.compiler,jax._src.cache_key,jax._src.interpreters.xla,jax._src.pjit
XLA_FLAGS=--xla_gpu_enable_latency_hiding_scheduler=true \
--xla_gpu_enable_triton_gemm=false \
--xla_gpu_graph_level=0 \
--xla_gpu_enable_highest_priority_async_stream=true \
--xla_gpu_all_reduce_combine_threshold_bytes=536870912 \
--xla_gpu_all_gather_combine_threshold_bytes=536870912 \
--xla_gpu_reduce_scatter_combine_threshold_bytes=536870912 \
--xla_gpu_enable_pipelined_all_gather=true \
--xla_gpu_enable_pipelined_reduce_scatter=true \
--xla_gpu_enable_pipelined_all_reduce=true \
--xla_gpu_enable_while_loop_double_buffering=true \
--xla_disable_hlo_passes=rematerialization \
--xla_gpu_pgle_accuracy_checker=PGLE_STRICTNESS_LEVEL_ERROR \
--xla_gpu_enable_triton_softmax_fusion=false \
--xla_gpu_enable_all_gather_combine_by_dim=false \
--xla_gpu_enable_reduce_scatter_combine_by_dim=false
EOF

# Old flag
# --xla_gpu_enable_pgle_accuracy_checker=$STRICT_CHECKER \

# New flag
# --xla_gpu_pgle_accuracy_checker=PGLE_STRICTNESS_LEVEL_ERROR \
# 

export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/lance-1107-nv
# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/llama2-xprof_1004_nolayers_nightly_lance

COMMAND="python3 MaxText/train.py MaxText/configs/models/gpu/llama2_7b.yml base_config=base.yml hardware=gpu run_name=maxtext_llama2-7b steps=10 max_target_length=4096 model_name=llama2-7b enable_checkpointing=false attention=cudnn_flash_te dataset_type=synthetic async_checkpointing=false base_output_directory=gs://lancewang-dev-supercomputer-testing/maxtext_gpu logits_dot_in_fp32=false use_iota_embed=true scan_layers=true dcn_pipeline_parallelism=1 dcn_fsdp_parallelism=1 per_device_batch_size=1 ici_tensor_parallelism=4 weight_dtype=bfloat16 remat_policy=full profiler=xplane skip_first_n_steps_for_profiler=5 ";

COMMAND='export LD_LIBRARY_PATH=/usr/local/cuda-12.6/compat:$LD_LIBRARY_PATH;'"${COMMAND}"; 

python ../xpk/xpk.py workload delete --cluster $CLUSTER_NAME --workload $WORKLOAD_NAME; 

python ../xpk/xpk.py workload create --cluster $CLUSTER_NAME --workload $WORKLOAD_NAME --command "${COMMAND}" --docker-image=$LOCAL_IMAGE_NAME --device-type=$DEVICE_TYPE --num-nodes=$NUM_NODES --scheduler=gke.io/topology-aware-auto --env-file=env.txt ;