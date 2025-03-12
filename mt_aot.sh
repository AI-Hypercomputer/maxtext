#!/bin/bash
export MODEL_NAME=llama3.1-405b
MODEL_SIZE=$(echo $MODEL_NAME | grep -o '[0-9]\+b')

export CLUSTER_NAME=a3plus-benchmark
export ZONE=australia-southeast1
export DEVICE_TYPE=h100-mega-80gb-8

export OUTPUT_PATH=lancewang-dev-supercomputer-testing/maxtext_gpu
export OUTPUT_BUCKET=gs://$OUTPUT_PATH

export NUM_NODES=1
# For PP, we need FSDP 2^N, so each stage should be 2^N number of nodes
# PP=2, FSDP=64, NUM_NODES=128
# PP=3, FSDP=32, NUM_NODES=96
# PP=6, FSDP=16, NUM_NODES=96
# PP=7, FSDP=16, NUM_NODES=112
# PP=9, FSDP=8, NUM_NODES=72
# PP=14, FSDP=8, NUM_NODES=112
# PP=18, FSDP=4, NUM_NODES=72
# PP=21, FSDP=4, NUM_NODES=84
# PP=42, FSDP=2, NUM_NODES=84
# PP=63, FSDP=2, NUM_NODES=126


# export TARGET_NUM_NODES=128
# export TARGET_NUM_NODES=112
# export TARGET_NUM_NODES=96


export WORKLOAD_NAME=$USER-$(echo $MODEL_NAME | sed 's/\.//g')-aot-${RANDOM:0:2}

# Non PP setting
# export TARGET_NUM_NODES=64
# export PER_DEVICE_BATCH_SIZE=1
# export ICI_TP=1
# export DCN_PP=1
# export NUM_LAYERS_PER_PP_STAGE=1

# PP setting
# Must turn on
export PER_DEVICE_BATCH_SIZE=1
export ICI_TP=1

# OOM, 80G, TP should help here, but I'd like to isolate TP from FSDP for now https://xprof.corp.google.com/memory_viewer/lancewang-1849170665103313379
# export TARGET_NUM_NODES=128
# export DCN_PP=2 # 126 layers, could be 2, 3, 6, 7, 9, 14, 18, 21
# export NUM_LAYERS_PER_PP_STAGE=9 # 126 layers, could be combo of (3,3,7), so it's (3,7,9,21)

# 72G, https://xprof.corp.google.com/memory_viewer/lancewang-14485792821687062807
# export TARGET_NUM_NODES=128
# export DCN_PP=2 # 126 layers, could be 2, 3, 6, 7, 9, 14, 18, 21
# export NUM_LAYERS_PER_PP_STAGE=7 # (3,3,7), so it's (3,7,9,21), 9 repeats

# OOM, 100G, https://xprof.corp.google.com/memory_viewer/lancewang-8306572723514236220
# export TARGET_NUM_NODES=96
# export DCN_PP=3 # 126 layers, could be 2, 3, 6, 7, 9, 14, 18, 21
# export NUM_LAYERS_PER_PP_STAGE=14 # 126 layers, (2,3,7), so (2,3,7,6,14,21), 3 repeats

# 70.6G, https://xprof.corp.google.com/memory_viewer/lancewang-13350260186473749094
export TARGET_NUM_NODES=96
export DCN_PP=3 # 126 layers, could be 2, 3, 6, 7, 9, 14, 18, 21
export NUM_LAYERS_PER_PP_STAGE=6 # 126 layers, (2,3,7), so (2,3,7,6,14,21), 7 repeats

# OOM, 77G, https://xprof.corp.google.com/memory_viewer/lancewang-14585151666080368961
# export TARGET_NUM_NODES=96
# export DCN_PP=6 # 126 layers, could be 2, 3, 6, 7, 9, 14, 18, 21
# export NUM_LAYERS_PER_PP_STAGE=7 # (3, 7), 3 repeats

# OOM, 89G, https://xprof.corp.google.com/memory_viewer/lancewang-12791364683696204479
# export TARGET_NUM_NODES=112
# export DCN_PP=7 # 126 layers
# export NUM_LAYERS_PER_PP_STAGE=9 # (2,3,3), 2 repeats

# 72G, https://xprof.corp.google.com/memory_viewer/lancewang-4215571770793819877
# Errors out after changing remat_policy=custom decoder_layer_input=offload
# export TARGET_NUM_NODES=112
# export DCN_PP=7 # 126 layers
# export NUM_LAYERS_PER_PP_STAGE=6 # (2,3,3), 3 repeats

# export TARGET_NUM_NODES=72
# export DCN_PP=9 # 126 layers
# export NUM_LAYERS_PER_PP_STAGE=7 # (2,7)

# export TARGET_NUM_NODES=112
# export DCN_PP=14 # 126 layers
# export NUM_LAYERS_PER_PP_STAGE=9 # (3,3)



# DP + TP + PP
# export PER_DEVICE_BATCH_SIZE=2
# export ICI_TP=8

# export TARGET_NUM_NODES=72
# export DCN_PP=9 # 126 layers
# export DCN_DP=8
# export NUM_LAYERS_PER_PP_STAGE=7 # (2,7), 3 repeats

# export TARGET_NUM_NODES=112
# export DCN_PP=14 # 126 layers, could be 2, 3, 6, 7, 9, 14, 18, 21
# export DCN_DP=8
# export NUM_LAYERS_PER_PP_STAGE=9 # 126 layers, (2,3,7), so (2,3,7,6,14,21), 7 repeats

# Must turn on
export DCN_FSDP=$(($TARGET_NUM_NODES / $DCN_PP))
export PP_MBS=$(($DCN_PP))


# export REMAT_POLICY=save_qkv_proj
export REMAT_POLICY=full

# export ATTENTION=dot_product
export ATTENTION=cudnn_flash_te

# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/llama2-xprof_1001_lance
# export LOCAL_IMAGE_NAME=gcr.io/supercomputer-testing/lance-mantaray_maxtext_jsts_gpu_a4_02252025-nv-fix2_xpk
export LOCAL_IMAGE_NAME=gcr.io/supercomputer-testing/maxtext_stable_stack_0305:working
export JAX_ENABLE_PGLE=false

cat <<EOF > env.txt

XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
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
--xla_gpu_enable_all_gather_combine_by_dim=false \
--xla_gpu_enable_reduce_scatter_combine_by_dim=false \
--xla_gpu_threshold_for_windowed_einsum_mib=0 \
--xla_gpu_multi_streamed_windowed_einsum
EOF


COMMAND="python MaxText/train_compile.py MaxText/configs/models/gpu/llama3.1_405b.yml hardware=gpu base_output_directory=$OUTPUT_BUCKET/aot dataset_type=synthetic tokenizer_path=assets/tokenizer_llama3.tiktoken per_device_batch_size=$PER_DEVICE_BATCH_SIZE ici_tensor_parallelism=$ICI_TP dcn_fsdp_parallelism=$DCN_FSDP dcn_pipeline_parallelism=$DCN_PP max_target_length=4096 run_name=runner_aotc steps=10 enable_checkpointing=false model_name=$MODEL_NAME compile_topology=a3 compile_topology_num_slices=$TARGET_NUM_NODES logits_dot_in_fp32=false weight_dtype=bfloat16 opt_type=adamw attention=$ATTENTION profiler=xplane skip_first_n_steps_for_profiler=0 remat_policy=$REMAT_POLICY num_layers_per_pipeline_stage=$NUM_LAYERS_PER_PP_STAGE num_pipeline_microbatches=$PP_MBS";

# remat_policy=custom decoder_layer_input=offload

COMMAND='export LD_LIBRARY_PATH=/usr/local/cuda-12.6/compat:$LD_LIBRARY_PATH;'"${COMMAND}";

echo $COMMAND

python3 ../xpk/xpk.py workload delete --cluster $CLUSTER_NAME --workload $WORKLOAD_NAME;
python3 ../xpk/xpk.py workload create --cluster $CLUSTER_NAME --workload $WORKLOAD_NAME --command "${COMMAND}" --docker-image=$LOCAL_IMAGE_NAME --device-type=$DEVICE_TYPE --num-nodes=$NUM_NODES --scheduler=gke.io/topology-aware-auto --env-file=env.txt
