export JAX_TRACEBACK_FILTERING=off
python MaxText/inference_microbenchmark.py \
    MaxText/configs/base.yml \
    model_name=llama2-7b \
    tokenizer_path=assets/tokenizer.llama2 \
    load_parameters_path=gs://msingh-bkt/checkpoints/quant_llama2-7b-chat/20241120034012/int8_\
    checkpoint_is_quantized=True \
    quantization=int8 \
    per_device_batch_size=8 \
    quantize_kvcache=True \
    kv_quant_dtype=int8 \
    attention=paged \
    num_pages=64 \
    page_size=32 \
    block_size=256 \
    weight_dtype=bfloat16 \
    scan_layers=false \
    max_prefill_predict_length=1024 \
    max_target_length=2048 \
    ici_fsdp_parallelism=1 \
    ici_tensor_parallelism=-1 \
    ici_autoregressive_parallelism=1 \
    inference_microbenchmark_prefill_lengths=256 \
    inference_microbenchmark_stages=generate \
    inference_microbenchmark_loop_iters=1 \
    run_name=$(date +%Y-%m-%d-%H-%M) \
    base_output_directory=/mnt/disks/persist/paged_attention_microbenchmark/dot_product \
    profiler=xplane