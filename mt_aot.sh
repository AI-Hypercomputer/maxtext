#!/bin/bash
export MODEL_NAME=llama3.1-405b
MODEL_SIZE=$(echo $MODEL_NAME | grep -o '[0-9]\+b')
export WORKLOAD_NAME=$USER-$(echo $MODEL_SIZE | sed 's/\.//g')-aot


export NUM_NODES=1;
export CLUSTER_NAME=a3plus-benchmark
export PROJECT=supercomputer-testing

# 405B
# 1k chips
export TARGET_NUM_NODES=128;
export PER_DEVICE_BATCH_SIZE=1

# 4k chips
# export TARGET_NUM_NODES=512;
# export PER_DEVICE_BATCH_SIZE=2


export ICI_TP=8
export DCN_FSDP=16
# export DCN_FSDP=$TARGET_NUM_NODES


export DCN_PP=1
export NUM_LAYERS_PER_PP_STAGE=$(expr 126 / $DCN_PP)


# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/llama2-xprof_1002_126layers_lance
# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/llama2-xprof_1003_nolayers_pinned_lance
export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/llama2-xprof_1010_nolayers_nightly_lance 

export ATTENTION=cudnn_flash_te
# export REMAT_POLICY=minimal
export REMAT_POLICY=save_qkv_proj
# export REMAT_POLICY=full
# export REMAT_POLICY=save_dot_except_mlpwi
# export REMAT_POLICY=save_dot_except_mlp


export JAX_ENABLE_PGLE=true
# export JAX_ENABLE_PGLE=true
export STRICT_CHECKER=true

export JAX_PGLE_AGGREGATION_PERCENTILE=50
export JAX_SHARE_AUTOTUNE_CONFIG_BETWEEN_HOSTS=true
export JAX_PGLE_PROFILING_RUNS=3

# JAX_PGLE_AGGREGATION_PERCENTILE=$JAX_PGLE_AGGREGATION_PERCENTILE
# JAX_SHARE_AUTOTUNE_CONFIG_BETWEEN_HOSTS=$JAX_SHARE_AUTOTUNE_CONFIG_BETWEEN_HOSTS
# JAX_PGLE_PROFILING_RUNS=$JAX_PGLE_PROFILING_RUNS
# TF_CPP_VMODULE=profile_guided_latency_estimator=10
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
# CUDA_DEVICE_MAX_CONNECTIONS=1

# YY: temp add JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none to bypass the PGLE issue 

cat <<EOF > env.txt

NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE=/usr/local/nvidia/lib64/a3plus_guest_config.textproto
NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS=600000
JAX_ENABLE_PGLE=$JAX_ENABLE_PGLE
JAX_REMOVE_CUSTOM_PARTITIONING_PTR_FROM_CACHE_KEY=true
JAX_DEBUG_LOG_MODULES=compiler
JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none
XLA_FLAGS=--xla_gpu_enable_latency_hiding_scheduler=true \
--xla_gpu_graph_level=0 \
--xla_gpu_enable_triton_gemm=false \
--xla_gpu_enable_highest_priority_async_stream=true \
--xla_gpu_all_reduce_combine_threshold_bytes=134217728 \
--xla_gpu_all_gather_combine_threshold_bytes=1073741824 \
--xla_gpu_reduce_scatter_combine_threshold_bytes=33554432 \
--xla_gpu_enable_pipelined_all_gather=true \
--xla_gpu_enable_pipelined_reduce_scatter=true \
--xla_gpu_enable_pipelined_all_reduce=true \
--xla_gpu_enable_while_loop_double_buffering=true \
--xla_disable_hlo_passes=rematerialization \
--xla_gpu_enable_triton_softmax_fusion=false \
--xla_gpu_enable_all_gather_combine_by_dim=false \
--xla_gpu_enable_reduce_scatter_combine_by_dim=false \
--xla_gpu_use_memcpy_local_p2p=true
EOF


COMMAND="python MaxText/train_compile.py MaxText/configs/models/gpu/llama3.1_405b.yml hardware=gpu base_output_directory=$OUTPUT_BUCKET/aot dataset_type=synthetic tokenizer_path=assets/tokenizer_llama3.tiktoken per_device_batch_size=$PER_DEVICE_BATCH_SIZE ici_tensor_parallelism=$ICI_TP dcn_fsdp_parallelism=$DCN_FSDP dcn_pipeline_parallelism=$DCN_PP num_layers_per_pipeline_stage=$NUM_LAYERS_PER_PP_STAGE max_target_length=8192 run_name=runner_finetune steps=10 enable_checkpointing=false model_name=llama3.1-405b compile_topology=a3 compile_topology_num_slices=$TARGET_NUM_NODES logits_dot_in_fp32=false weight_dtype=bfloat16 opt_type=adamw attention=$ATTENTION profiler=xplane skip_first_n_steps_for_profiler=0 remat_policy=$REMAT_POLICY ";

# base_num_decoder_layers=126

COMMAND='export LD_LIBRARY_PATH=/usr/local/cuda-12.6/compat:$LD_LIBRARY_PATH;echo yy1$LD_LIBRARY_PATH;'"${COMMAND}";

python3 ../xpk/xpk.py workload delete --cluster $CLUSTER_NAME --workload $WORKLOAD_NAME; 

python3 ../xpk/xpk.py workload create --project $PROJECT --cluster $CLUSTER_NAME --workload $WORKLOAD_NAME --command "${COMMAND}" --docker-image=$LOCAL_IMAGE_NAME --device-type=$DEVICE_TYPE --num-nodes=$NUM_NODES --scheduler=gke.io/topology-aware-auto --env-file=env.txt ;

