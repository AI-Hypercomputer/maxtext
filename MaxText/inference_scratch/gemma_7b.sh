export M_LOAD_PARAMETERS_PATH=gs://runner-maxtext-logs/reroll5/checkpoints/10/items/
export M_PER_DEVICE_BATCH_SIZE=24
export M_MAX_PREFILL_PREDICT_LENGTH=1024
export M_MAX_TARGET_LENGTH=2048

#python MaxText/decode.py            MaxText/configs/base.yml tokenizer_path=assets/tokenizer.gemma run_name=runner_2024-03-06-04-17 steps=10 weight_dtype=bfloat16 async_checkpointing=false model_name=gemma-7b ici_fsdp_parallelism=1 ici_autoregressive_parallelism=-1 scan_layers=false

python MaxText/maxengine_server.py MaxText/configs/base.yml tokenizer_path=assets/tokenizer.gemma run_name=runner_2024-03-06-04-17 steps=10 weight_dtype=bfloat16 async_checkpointing=false model_name=gemma-7b ici_fsdp_parallelism=1 ici_autoregressive_parallelism=-1 scan_layers=false
