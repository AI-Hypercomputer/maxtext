python3 -m benchmarks.lm_eval.maxtext_generator \
  MaxText/configs/base.yml \
  tokenizer_path=assets/tokenizer.llama2 \
  load_parameters_path=gs://msingh-bkt/checkpoints/quant_llama2-7b-chat/20241120034012/int8_ \
  checkpoint_is_quantized=True \
  quantization=int8 \
  model_name=llama2-7b \
  max_prefill_predict_length=4096 \
  max_target_length=8192 \
  ici_tensor_parallelism=4 \
  per_device_batch_size=8 \
  attention=dot_product \
  pagedattn_num_pages=128 \
  pagedattn_tokens_per_page=32 \
  pagedattn_pages_per_compute_block=4 \
  prompt="The best thing about Kirkland, Washington on a sunny afternoon is" \
  ici_fsdp_parallelism=1 \
  ici_autoregressive_parallelism=1 \
  scan_layers=False \
  return_log_prob=True
