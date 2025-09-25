export M_LOAD_PARAMETERS_PATH=gs://runner-maxtext-logs/reroll5/checkpoints/10/items/
export M_PER_DEVICE_BATCH_SIZE=24
export M_MAX_PREFILL_PREDICT_LENGTH=1024
export M_MAX_TARGET_LENGTH=2048

#python3 -m MaxText.decode            "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}"/configs/base.yml tokenizer_path="${MAXTEXT_ASSETS_ROOT:-${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText/assets}}"/tokenizer.gemma run_name=runner_2024-03-06-04-17 steps=10 weight_dtype=bfloat16 async_checkpointing=false model_name=gemma-7b ici_fsdp_parallelism=1 ici_autoregressive_parallelism=-1 scan_layers=false

python3 -m MaxText.maxengine_server "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}"/configs/base.yml tokenizer_path="${MAXTEXT_ASSETS_ROOT:-${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText/assets}}"/tokenizer.gemma run_name=runner_2024-03-06-04-17 steps=10 weight_dtype=bfloat16 async_checkpointing=false model_name=gemma-7b ici_fsdp_parallelism=1 ici_autoregressive_parallelism=-1 scan_layers=false
