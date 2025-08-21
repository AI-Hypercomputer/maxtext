python3 -m benchmarks.lm_eval.maxtext_generator \
    MaxText/configs/base.yml \
    model_name="mixtral-8x7b" \
    tokenizer_path="mistralai/Mixtral-8x7B-Instruct-v0.1" \
    tokenizer_type="huggingface" \
    load_parameters_path="gs://ml-auto-solutions/output/sparsity_diffusion_devx/maxtext/chained_tests_mixtral-8x7b_stable-2025-08-15-01-00-23//unscanned_ckpt/checkpoints/0/items" \
    per_device_batch_size=4 \
    ici_tensor_parallelism=4 \
    max_prefill_predict_length=1024 \
    max_target_length=2048 \
    steps=50 \
    async_checkpointing=false \
    scan_layers=false \
    prompt='The Capital of French' \
    attention="dot_product" \
    return_log_prob=True
