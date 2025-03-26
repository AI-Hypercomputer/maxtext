export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_command_buffer=FUSION --xla_disable_hlo_passes=rematerialization"
export TF_FORCE_GPU_ALLOW_GROWTH=true

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.94

# export CONF_DIR=/scratch/loadgen_run_data

# python -m offline_mode --mlperf_test_mode=accuracy --input_mode tokenized --output_mode tokenized --mlperf_conf $CONF_DIR/mlperf.conf --user_conf $CONF_DIR/user.conf --audit_conf no_audit --total_sample_count 24576 --dataset_path $CONF_DIR/processed-data.pkl --output_log_dir $CONF_DIR/logs/llama70b-llama_offline_benchmarks-full-offline-performance-offline-performance_202502261 --tok_outlen_multiplier 2.5 --maxengine_args "model_name=llama3.1-70b tokenizer_path=/opt/maxtext/assets/tokenizer_llama3.tiktoken load_parameters_path=/scratch/Llama3.1-70B-Instruct/scanned_chkpt/0/items quantize_kvcache=False attention=dot_product scan_layers=true hardware=gpu async_checkpointing=False ici_fsdp_parallelism=1 ici_autoregressive_parallelism=1 ici_tensor_parallelism=8 weight_dtype=bfloat16" --prefill_lengths_and_per_device_batch_sizes 1024,40

# Original, doensn't have the eval, should try eval
# python -m offline_mode --mlperf_test_mode=accuracy --input_mode tokenized --output_mode tokenized --mlperf_conf /scratch/inference/mlperf.conf --user_conf user.conf --audit_conf no_audit --total_sample_count 100 --dataset_path /scratch/loadgen_run_data/processed-data.pkl --output_log_dir /scratch/loadgen_run_data/logs/llama70b-llama_offline_benchmarks-full-offline-performance-offline-performance_202502261 --tok_outlen_multiplier 2.5 --maxengine_args "model_name=llama2-70b tokenizer_path=/opt/maxtext/assets/tokenizer.llama2 load_parameters_path=gs://jwyang/maxtext/direct_generate_param_only_checkpoint_llama2_70b_chat/checkpoints/0/items attention=dot_product scan_layers=false hardware=gpu async_checkpointing=False ici_fsdp_parallelism=1 ici_autoregressive_parallelism=-1 ici_tensor_parallelism=1 weight_dtype=bfloat16 dtype=bfloat16" --prefill_lengths_and_per_device_batch_sizes 1024,80

# kv_quant_dtype=fp8 quantize_kvcache=True quantization=aqt_fp8


DATA_DISK_DIR=/scratch/loadgen_run_data TOKENIZER_PATH=/opt/maxtext/assets/tokenizer.llama2 BASEDIR=/scratch/loadgen_run_data PREFILL_LENS_AND_PER_DEVICE_BATCH_SIZES=1024,64 MAXENGINE_ARGS="model_name=llama2-70b tokenizer_path=/opt/maxtext/assets/tokenizer.llama2 load_parameters_path=gs://jwyang/maxtext/direct_generate_param_only_checkpoint_llama2_70b_chat/checkpoints/0/items attention=dot_product scan_layers=false hardware=gpu async_checkpointing=False ici_fsdp_parallelism=1 ici_autoregressive_parallelism=-1 ici_tensor_parallelism=8 weight_dtype=bfloat16 dtype=bfloat16 attention=dot_product" ./llama_offline_run.sh -a -t
