# Run command:
#	bash run_v6e_prefill_microbenchmark.sh -x
# Look at profiles: 
# tensorboard --logdir /tmp/mb/profiles/trillium_llama2_70b/tensorboard/prefill_insert_1024 

run_name="trillium_llama2-70b"
dry_run=false
enable_profiler=false
enable_xla_flags=false
prefill_lens="1024"
stages="prefill"

while getopts "npxr:l:" opt
do
  case "$opt" in
      n ) dry_run=true ;;
      p ) enable_profiler=true ;;
      x ) enable_xla_flags=true ;;
      r ) run_name="$OPTARG" ;;
      s ) stages="$OPTARG" ;;
      l ) prefill_lens="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
  esac
done


if "$dry_run"; then
    cmd=echo
else
    cmd=''
fi

PROFILER_OPTION=""
if "$enable_profiler"; then
    PROFILER_OPTION="profiler=xplane upload_all_profiler_results=True"
fi

LIBTPU_INIT_ARGS=""
if "$enable_xla_flags"; then
    export LIBTPU_INIT_ARGS="--xla_jf_auto_cross_replica_sharding=False --xla_tpu_decompose_all_gather_einsum=True --xla_tpu_enable_windowed_einsum_for_reduce_scatter=True --xla_tpu_enable_async_collective_fusion=False --xla_tpu_enable_async_collective_fusion_fuse_all_gather=False --xla_tpu_overlap_compute_collective_tc=False --xla_all_gather_latency_bound_threshold_in_bytes=524290 --xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1 --xla_jf_rematerialization_percent_shared_memory_limit=98 --xla_tpu_allocate_scoped_vmem_at_same_offset=False --xla_tpu_alternate_memory_benefit_scaling_factor_for_large_buffers=NO_SCALE --xla_tpu_async_copy_bandwidth_scaling_factor=1.0044 --xla_tpu_copy_elision_analysis_allowance=239621 --xla_tpu_copy_fusion_pad_unpad_ratio=242.6129 --xla_tpu_copy_insertion_use_region_analysis_limit=3441 --xla_tpu_dot_dot_fusion=True --xla_tpu_dot_dot_fusion_duplicated=False --xla_tpu_enable_aggressive_broadcast_priority_update=True --xla_tpu_enable_dot_strength_reduction=True --xla_tpu_enable_experimental_fusion_cost_model=True --xla_tpu_enable_vmem_to_vmem_dmas=False --xla_tpu_enforce_prefetch_fifo_order=True --xla_tpu_layout_use_dot_grouping=False --xla_tpu_memory_bound_loop_optimizer_options=enabled:true --xla_tpu_msa_inefficient_use_to_copy_ratio=0.471 --xla_tpu_nd_short_transfer_max_chunks=4415 --xla_tpu_order_dot_after_layout=False --xla_tpu_perform_spmd_cse_prevention=True --xla_tpu_prefetch_interval_picker_size_override=26672104 --xla_tpu_reduce_loop_fusion_dup_with_unfusable_user=False --xla_tpu_rwb_fusion=False --xla_tpu_scavenge_vmem_for_fusions=False --xla_tpu_scoped_vmem_limit_kib=19592 --xla_tpu_sliced_prefetch_max_slices=0 --xla_tpu_use_lp_llo_scheduler_for_dot_dot_fusions=True --xla_tpu_use_repeated_instance_for_preferred_prefetch_time=False --xla_tpu_vector_load_fusion_window=644 --xla_tpu_vector_store_fusion_window=1228 --xla_vf_vmem_enable_cross_program_prefetch_freeing=False --xla_vf_vmem_max_outstanding_evictions=136 --xla_vf_vmem_max_outstanding_prefetches=131 --xla_vf_vmem_max_overlap_to_mem_size_async_copy_ratio=16.0009 --xla_vf_vmem_max_repacks=18 --xla_vf_vmem_max_retries=2 --xla_vf_vmem_min_overlap_to_async_copy_ratio=1.4973 --xla_vf_vmem_preferred_overlap_to_async_copy_ratio=8.3221"
fi

export TOKENIZER_PATH=/mnt/disks/persist/maxtext/assets/tokenizer.llama2
export LOAD_PARAMETERS_PATH=gs://patemotter/checkpoints/quant_llama2-70b-chat/int8w_
export LOAD_PARAMETERS_PATH=""
export MAX_PREFILL_PREDICT_LENGTH=1024
export MAX_TARGET_LENGTH=2048
export MODEL_NAME=llama2-70b
export ICI_FSDP_PARALLELISM=1
export ICI_AUTOREGRESSIVE_PARALLELISM=1
export ICI_TENSOR_PARALLELISM=4
export ICI_SEQUENCE_PARALLELISM=2
export SCAN_LAYERS=false
export WEIGHT_DTYPE=bfloat16
export PER_DEVICE_BATCH_SIZE=1
export ICI="FSDP=${ICI_FSDP_PARALLELISM}_AR=${ICI_AUTOREGRESSIVE_PARALLELISM}_TENSOR=${ICI_TENSOR_PARALLELISM}_SEQ=${ICI_SEQUENCE_PARALLELISM}"


export MESH_TYPE="default" 
export RUN_DESC="${run_name}_${stages}_${prefill_lens}_flags_${enable_xla_flags}_${ICI}_${MESH_TYPE}"

mkdir -p /tmp/mb/logs
mkdir -p /tmp/mb/profiles

source /mnt/disks/persist/maxtext/venv/bin/activate && \
python ../inference_microbenchmark.py   \
../configs/base.yml \
base_output_directory=gs://patemotter-env-one-vm/maxtext_prefill_profiles_mb \
tokenizer_path=${TOKENIZER_PATH} \
load_parameters_path=${LOAD_PARAMETERS_PATH}   \
max_prefill_predict_length=${MAX_PREFILL_PREDICT_LENGTH}   \
max_target_length=${MAX_TARGET_LENGTH} \
model_name=${MODEL_NAME}   \
ici_fsdp_parallelism=${ICI_FSDP_PARALLELISM}   \
ici_autoregressive_parallelism=${ICI_AUTOREGRESSIVE_PARALLELISM}   \
ici_tensor_parallelism=${ICI_TENSOR_PARALLELISM}   \
ici_sequence_parallelism=${ICI_SEQUENCE_PARALLELISM} \
scan_layers=false \
weight_dtype=bfloat16 \
per_device_batch_size=${PER_DEVICE_BATCH_SIZE} \
quantization=int8 \
quantize_kvcache=True \
inference_microbenchmark_stages=${stages} \
inference_microbenchmark_prefill_lengths="${prefill_lens}" \
checkpoint_is_quantized=True \
compute_axis_order=0,2,1,3 \
ar_cache_axis_order=0,2,1,3 \
attention=dot_product \
mesh_type=${MESH_TYPE} \
run_name=${RUN_DESC} \
${PROFILER_OPTION} 2>&1 | tee /tmp/mb/logs/${cmd}_${RUN_DESC}