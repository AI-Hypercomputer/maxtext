python3 -m benchmarks.lm_eval.maxtext_generator \
  MaxText/configs/base.yml \
  tokenizer_path=assets/tokenizer_llama3.tiktoken \
  load_parameters_path=gs://maxtext-model-checkpoints/llama3.1-8b/2025-01-17-04-13/unscanned/checkpoints/0/items \
  checkpoint_is_quantized=False \
  quantize_kvcache=False \
  tokenizer_type=tiktoken\
  model_name=llama3-8b \
  max_prefill_predict_length=1024 \
  max_target_length=2048 \
  ici_tensor_parallelism=4 \
  per_device_batch_size=8 \
  attention=dot_product \
  ici_fsdp_parallelism=1 \
  ici_autoregressive_parallelism=1 \
  scan_layers=False \
  pagedattn_num_pages=128 \
  pagedattn_tokens_per_page=32 \
  pagedattn_pages_per_compute_block=4