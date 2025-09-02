python3 -m benchmarks.lm_eval.maxtext_generator \
    MaxText/configs/base.yml \
    model_name="qwen3-235b-a22b" \
    tokenizer_path="Qwen/Qwen3-235B-A22B-Thinking-2507" \
    tokenizer_type="huggingface" \
    load_parameters_path="gs://parambole-qwen3-moe-verification/scanned/07_08_2025/0/items" \
    per_device_batch_size=1 \
    ici_tensor_parallelism=4 \
    ici_expert_parallelism=8 \
    max_prefill_predict_length=1024 \
    max_target_length=2048 \
    steps=50 \
    scan_layers=true \
    async_checkpointing=false \
    attention="dot_product" \
    return_log_prob=True \
    prompt="Hello, please give me a joke."
