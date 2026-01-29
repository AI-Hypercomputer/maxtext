# llama2-7b
python3 -m maxtext.inference_microbenchmark \
"${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}"/configs/base.yml \
async_checkpointing=false \
attention=autoselected \
dataset_path=gs://maxtext-dataset \
ici_fsdp_parallelism=1 \
ici_autoregressive_parallelism=-1 \
max_prefill_predict_length=1024 \
max_target_length=2048 \
per_device_batch_size=16 \
quantization=int8 \
quantize_kvcache=True \
steps=10 \
scan_layers=false \
model_name=llama2-7b \
weight_dtype=bfloat16 \
tokenizer_path="${MAXTEXT_ASSETS_ROOT:-${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/assets/tokenizers}}"/tokenizer.llama2
