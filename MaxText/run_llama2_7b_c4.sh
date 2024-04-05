python3 MaxText/train.py MaxText/configs/base.yml \
run_name=llama2-7b-c4-$(date +%Y-%m-%d-%H-%M) \
base_output_directory=gs://mazumdera-test-bucket/lg-llama2/hf \
steps=6675 \
model_name=llama2-7b \
per_device_batch_size=2 \
learning_rate=1e-4 \
remat_policy=minimal \
enable_checkpointing=false \
tokenizer_path="assets/vocab_102400" \
tokenizer_loader=AutoTokenizer

#ici_fsdp_transpose_parallelism=16 \