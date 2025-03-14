python MaxText/decode.py MaxText/configs/base.yml tokenizer_path=assets/tokenizer.llama2  \
        load_parameters_path=gs://msingh-bkt/checkpoints/quant_llama2-7b-chat/20241120034012/int8_  \
        max_prefill_predict_length=16 max_target_length=32 model_name=llama2-7b   \
        ici_fsdp_parallelism=1 ici_autoregressive_parallelism=1 ici_tensor_parallelism=-1 \
        scan_layers=false weight_dtype=bfloat16 per_device_batch_size=1   \
        checkpoint_is_quantized=true quantization=int8 \
        attention=paged pagedattn_num_pages=64 pagedattn_tokens_per_page=8 pagedattn_pages_per_compute_block=4
