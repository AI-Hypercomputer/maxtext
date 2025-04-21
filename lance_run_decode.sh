export model_name=llama2-7b
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_command_buffer=FUSION --xla_disable_hlo_passes=rematerialization --xla_disable_hlo_passes=gpu-convert-async-collectives-to-sync"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export BASE_OUTPUT_DIRECTORY=/scratch/temp
export ASYNC_CHECKPOINTING=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.94
export PER_DEVICE_BATCH_SIZE=25
export CHECKPOINT_PATH=gs://jwyang-runner-maxtext-logs/llama2-7b_unscanned_chkpt_2024-04-26-18-28/checkpoints/0/items

python3 MaxText/decode.py MaxText/configs/base.yml  load_parameters_path=$CHECKPOINT_PATH base_output_directory=${BASE_OUTPUT_DIRECTORY}  model_name=$model_name max_prefill_predict_length=1024  max_target_length=2048  attention=dot_product scan_layers=false hardware=gpu async_checkpointing=${ASYNC_CHECKPOINTING} per_device_batch_size=${PER_DEVICE_BATCH_SIZE} run_name=$(date+%Y-%m-%d-%H-%M) ici_fsdp_parallelism=1 ici_autoregressive_parallelism=1 ici_tensor_parallelism=-1 weight_dtype=bfloat16 dtype=bfloat16 weight_dtype=bfloat16  enable_jax_profiler=true profiler=xplane quantize_kvcache=true kv_quant_dtype=fp8 quantization=aqt_fp8



