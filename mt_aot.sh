#!/bin/bash
bash mt_config.sh


export NUM_NODES=1
export TARGET_NUM_NODES=108
# export TARGET_NUM_NODES=512
# export TARGET_NUM_NODES=126
#export TARGET_NUM_NODES=96

export WORKLOAD_NAME=$USER-$(echo $MODEL_NAME | sed 's/\.//g')-aot

# Non PP setting
export PER_DEVICE_BATCH_SIZE=0.125
export ICI_TP=8
export DCN_PP=1 # Could be 2, 3, 6, 9, 18
export DCN_FSDP=32
export DCN_DP=1

# PP setting
# export PER_DEVICE_BATCH_SIZE=5
# export PER_DEVICE_BATCH_SIZE=0.125
# export ICI_TP=8
# export DCN_PP=9 # Could be 2, 3, 6, 9, 18
# # export DCN_PP=3
# #export DCN_FSDP=$TARGET_NUM_NODES
# export DCN_FSDP=4
# export DCN_DP=3

export NUM_LAYERS_PER_PP_STAGE=$(expr 126 / $DCN_PP)
#export NUM_LAYERS_PER_PP_STAGE=14
# export NUM_LAYERS_PER_PP_STAGE=2
export MBS_PER_STAGE=$(expr $PER_DEVICE_BATCH_SIZE * $ICI_TP)

# export REMAT_POLICY=save_qkv_proj
export REMAT_POLICY=full

#export ATTENTION=dot_product
export ATTENTION=cudnn_flash_te

# export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/llama2-xprof_1001_lance
export LOCAL_IMAGE_NAME=gcr.io/supercomputer-testing/lance-mantaray_maxtext_jsts_gpu_a4_02252025-nv-fix2

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


COMMAND="python MaxText/train_compile.py MaxText/configs/models/gpu/llama3.1_405b.yml hardware=gpu base_output_directory=$OUTPUT_BUCKET/aot dataset_type=synthetic tokenizer_path=assets/tokenizer_llama3.tiktoken per_device_batch_size=$PER_DEVICE_BATCH_SIZE ici_tensor_parallelism=$ICI_TP dcn_fsdp_parallelism=$DCN_FSDP dcn_pipeline_parallelism=$DCN_PP max_target_length=4096 run_name=runner_finetune steps=10 enable_checkpointing=false model_name=llama3.1-405b compile_topology=a3 compile_topology_num_slices=$TARGET_NUM_NODES logits_dot_in_fp32=false weight_dtype=bfloat16 opt_type=adamw attention=$ATTENTION profiler=xplane skip_first_n_steps_for_profiler=0 remat_policy=$REMAT_POLICY num_pipeline_microbatches_per_stage=$MBS_PER_STAGE  base_num_decoder_layers=128";

COMMAND='export LD_LIBRARY_PATH=/usr/local/cuda-12.6/compat:$LD_LIBRARY_PATH;'"${COMMAND}";

python3 ../xpk/xpk.py workload delete --cluster $CLUSTER_NAME --workload $WORKLOAD_NAME;
python3 ../xpk/xpk.py workload create --cluster $CLUSTER_NAME --workload $WORKLOAD_NAME --command "${COMMAND}" --docker-image=$LOCAL_IMAGE_NAME --device-type=$DEVICE_TYPE --num-nodes=$NUM_NODES --priority=high --scheduler=gke.io/topology-aware-auto \
--env-file=env.txt
