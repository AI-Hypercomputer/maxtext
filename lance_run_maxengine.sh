export model_name=llama2-7b
export tokenizer_path=assets/tokenizer.llama2
export XLA_FLAGS="--xla_disable_hlo_passes=rematerialization"
export ici_tensor_parallelism=4
export ici_autoregressive_parallelism=1
export per_device_batch_size=1
export load_parameters_path_chat=gs://jwyang-runner-maxtext-logs/llama2-7b_unscanned_chkpt_2024-04-26-18-28/checkpoints/0/items
export load_parameters_path=gs://jwyang-runner-maxtext-logs/llama2-7b_unscanned_chkpt_2024-04-26-19-40/checkpoints/0/items
export load_parameters_path_chat_quantized=gs://jwyang-data/llama7b-chat-quantized-fixed/0/items



python3 -m MaxText.maxengine_server \
  MaxText/configs/base.yml \
  base_output_directory=gs://lancewang-dev-supercomputer-testing/maxtext-llama2-7b/microbenchmark \
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
  inference_server=ExperimentalMaxtextDisaggregatedServer_8GPU \
  skip_jax_distributed_system=true \
  hardware=gpu







