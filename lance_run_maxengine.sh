export model_name=llama2-70b
# export model_name=llama2-7b
export tokenizer_path=/opt/maxtext/assets/tokenizer.llama2
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_command_buffer=FUSION --xla_disable_hlo_passes=rematerialization"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.96

export ici_tensor_parallelism=4
export ici_autoregressive_parallelism=1
export per_device_batch_size=1
# export load_parameters_path_chat=gs://jwyang-runner-maxtext-logs/llama2-7b_unscanned_chkpt_2024-04-26-18-28/checkpoints/0/items
export load_parameters_path_chat=gs://inference-benchmarks/models/llama2-70b-chat/2024-05-08-23-16/param-only-decode-ckpt-maxtext/checkpoints/0/items
# export load_parameters_path=gs://jwyang-runner-maxtext-logs/llama2-7b_unscanned_chkpt_2024-04-26-19-40/checkpoints/0/items
# export load_parameters_path_chat_quantized=gs://jwyang-data/llama7b-chat-quantized-fixed/0/items


# export load_parameters_path_chat=gs://inference-benchmarks/models/llama2-70b-chat/2024-05-08-23-16/param-only-decode-ckpt-maxtext/checkpoints/0/items

export DEBUG_CMD='python3 -m debugpy --listen localhost:5678 --wait-for-client -m'
export CMD='python3 -m'
export JAX_ENABLE_COMPILATION_CACHE=true
export INFERENCE_SERVER=ExperimentalMaxtextDisaggregatedServer_8GPU
# export INFERENCE_SERVER=MaxtextInterleavedServer

python -c "import jax; jax.clear_caches()"
#
$CMD MaxText.maxengine_server \
  MaxText/configs/inference.yml \
  base_output_directory=gs://lancewang-dev-supercomputer-testing/${model_name}/microbenchmark \
  load_parameters_path=${load_parameters_path_chat} \
  run_name=$(date +%Y-%m-%d-%H-%M) \
  save_config_to_gcs=true \
  model_name=${model_name} \
  tokenizer_path=${tokenizer_path} \
  inference_microbenchmark_log_file_path=microbenchmark.json \
  inference_microbenchmark_prefill_lengths=1024 \
  inference_microbenchmark_stages=prefill,generate \
  inference_microbenchmark_loop_iters=1000 \
  max_prefill_predict_length=1024 \
  max_target_length=2048 \
  per_device_batch_size=${per_device_batch_size} \
  ici_fsdp_parallelism=1 \
  ici_tensor_parallelism=${ici_tensor_parallelism} \
  ici_autoregressive_parallelism=${ici_autoregressive_parallelism} \
  scan_layers=false \
  weight_dtype=bfloat16 \
  inference_server=$INFERENCE_SERVER\
  skip_jax_distributed_system=true \
  hardware=gpu \
  optimize_mesh_for_tpu_v6e=false \
  use_iota_embed=false \
  enable_jax_profiler=true \
  quantize_kvcache=true kv_quant_dtype=fp8 quantization=aqt_fp8




