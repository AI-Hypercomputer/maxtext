python3 -m benchmarks.lm_eval.maxtext_generator \
    MaxText/configs/base.yml \
    model_name="gemma3-4b" \
    tokenizer_path="assets/tokenizer.gemma3" \
    load_parameters_path="gs://maxtext-gemma/unified/gemma3/4b/unscanned/2025-08-05-18-18/0/items" \
    per_device_batch_size=4 \
    ici_tensor_parallelism=4 \
    max_prefill_predict_length=2048 \
    max_target_length=4096 \
    steps=50 \
    async_checkpointing=false \
    scan_layers=false \
    prompt='The Capital of French' \
    attention="dot_product" \
    return_log_prob=True
