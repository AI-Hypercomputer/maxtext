# llama2-7b 
python MaxText/inference_microbenchmark.py \
MaxText/configs/base.yml \
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
tokenizer_path=assets/tokenizer.llama2 
