export JAX_TRACEBACK_FILTERING=off && \
python MaxText/decode.py \
    MaxText/configs/base.yml \
    model_name=llama2-7b \
    tokenizer_path=assets/tokenizer.llama2 \
    per_device_batch_size=1 \
    max_prefill_predict_length=1024 \
    max_target_length=2048 \
    weight_dtype=bfloat16 \
    ici_fsdp_parallelism=1 \
    ici_tensor_parallelism=-1 \
    scan_layers=false \
    load_parameters_path=gs://msingh-bkt/checkpoints/quant_llama2-7b-chat/20241120034012/int8_ \
    quantization=int8 \
    checkpoint_is_quantized=true \
    attention=paged \
    num_pages=64 \
    page_size=32 \
    block_size=256