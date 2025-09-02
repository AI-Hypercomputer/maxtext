python3 -m benchmarks.lm_eval.maxtext_generator \
    MaxText/configs/base.yml \
    model_name="qwen3-4b" \
    tokenizer_path="Qwen/Qwen3-4B" \
    load_parameters_path="gs://maxtext-qwen/qwen3/4b/unscanned/2025-08-04-21-31/0/items" \
    per_device_batch_size=1 \
    ici_tensor_parallelism=4 \
    max_prefill_predict_length=1024 \
    max_target_length=2048 \
    async_checkpointing=false \
    scan_layers=false \
    attention="dot_product" \
    return_log_prob=True \
    prompt="Hello, please give me a joke."