export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_command_buffer=FUSION --xla_disable_hlo_passes=rematerialization --xla_disable_hlo_passes=gpu-convert-async-collectives-to-sync" # flags from NVidia


export TF_FORCE_GPU_ALLOW_GROWTH=true
export BASE_OUTPUT_DIRECTORY=gs://lancewang/maxtext
export ASYNC_CHECKPOINTING=false

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export PER_DEVICE_BATCH_SIZE=160

python3 MaxText/inference_microbenchmark.py MaxText/configs/base.yml  \
base_output_directory=${BASE_OUTPUT_DIRECTORY}  \
model_name='llama2-70b' \
max_prefill_predict_length=1024  \
max_target_length=2048  \
attention=flash \
scan_layers=false \
hardware=gpu \
async_checkpointing=${ASYNC_CHECKPOINTING} \
per_device_batch_size=${PER_DEVICE_BATCH_SIZE} \
inference_microbenchmark_prefill_lengths=1024  \
inference_microbenchmark_stages=prefill,generate \
inference_microbenchmark_loop_iters=64 \
run_name=$(date +%Y-%m-%d-%H-%M) \
ici_fsdp_parallelism=1 \
ici_autoregressive_parallelism=-1 \
ici_tensor_parallelism=1 \
weight_dtype=bfloat16 \
quantization=aqt_fp8 kv_quant_dtype=fp8 quantize_kvcache=True
