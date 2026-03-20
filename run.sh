# WORKING SCRIPT!!!!!

#!/bin/bash

# 1. Suppress CUDA warnings (force Jax to only look for TPU/CPU)
export JAX_PLATFORMS=tpu,cpu
# Silence XLA/TF C++ logging spam
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=""
export TENSORSTORE_NUM_THREADS=4 
export LIBTPU_INIT_ARGS='--xla_tpu_scoped_vmem_limit_kib=98304 --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1 --xla_tpu_assign_all_reduce_scatter_layout=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_enable_async_all_gather=auto --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_fuse_all_reduce=true --xla_should_allow_loop_variant_parameter_in_chain=enabled --xla_should_add_loop_invariant_op_in_chain=enabled --xla_tpu_enable_ici_ag_pipelining=true --xla_tpu_enable_dot_strength_reduction=true --xla_tpu_dot_dot_fusion=true --xla_tpu_host_transfer_overlap_limit=24 --xla_tpu_aggressive_opt_barrier_removal=ENABLED --xla_lhs_prioritize_async_depth_over_stall=ENABLED --xla_latency_hiding_scheduler_rerun=2' && \
python3 -m maxtext.trainers.pre_train.train maxtext/configs/base.yml \
model_name='brahmai-27b' \
run_name='${RUN_NAME}' \
dataset_type='grain' \
grain_train_files='/lustre-data/english_dclm/*.arrayrecord*' \
grain_packing_type='best_fit' \
grain_eval_files='/lustre-data/english_dclm/*.arrayrecord*' \
base_output_directory='/lustre-data/${WORKLOAD_NAME}' \
tokenizer_path='src/maxtext/brahmai_tokenizer_v2' \
tokenizer_type='huggingface' \
grain_worker_count=8 \
grain_prefetch_buffer_size=20 \
per_device_batch_size=2 \
max_target_length=4096 \
steps=100 \
attention='flash' \
ici_fsdp_parallelism=-1 \
remat_policy='full' \
decoder_layer_input='offload' \
query_proj='device' \
scan_layers=True \
enable_checkpointing=False \
log_period=50 \
profiler='xplane' \
skip_first_n_steps_for_profiler=20 \
profiler_steps=5 \
upload_all_profiler_results=True \
