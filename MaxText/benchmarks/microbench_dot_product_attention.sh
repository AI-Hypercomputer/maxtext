python MaxText/inference_microbenchmark.py \
    MaxText/configs/base.yml \
    model_name=llama2-70b \
    tokenizer_path=assets/tokenizer.llama2 \
    load_parameters_path=gs://morgandu-tpu/checkpoints/quantized/aqt/llama2-70b-chat/ \
    checkpoint_is_quantized=True \
    quantization=int8 \
    per_device_batch_size=4 \
    quantize_kvcache=False \
    attention=dot_product \
    weight_dtype=bfloat16 \
    scan_layers=false \
    max_prefill_predict_length=1024 \
    max_target_length=2048 \
    ici_fsdp_parallelism=1 \
    ici_tensor_parallelism=-1 \
    ici_autoregressive_parallelism=1 \
    inference_microbenchmark_prefill_lengths=256 \
    inference_microbenchmark_stages=prefill,generate \
    inference_microbenchmark_loop_iters=1 \
    run_name=$(date +%Y-%m-%d-%H-%M) \
    base_output_directory=gs://morgandu-tpu/test-maxtext-output/dot_product/
