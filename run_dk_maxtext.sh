python MaxText/maxengine_server.py \
/mnt/disks/persist/git/maxtext/MaxText/configs/base.yml  \
max_prefill_predict_length=1024 \
max_target_length=2048 \
per_device_batch_size=1 \
quantize_kvcache=False  \
model_name=deepseek2-16b \
ici_tensor_parallelism=8 \
ici_fsdp_parallelism=1 \
ici_autoregressive_parallelism=1 \
megablox=True \
sparse_matmul=True \
scan_layers=False \
attention=dot_product \
tokenizer_type=huggingface \
tokenizer_path=deepseek-ai/DeepSeek-V2-Lite \
hf_access_token=$HUGGING_FACE_TOKEN \
load_parameters_path=gs://agagik-us/deepseek/maxtext_checkpoints/ds_16b_unscanned_new_3/unscanned_chkpt/checkpoints/0/items \
async_checkpointing=false\
weight_dtype=float32 \
checkpoint_is_quantized=False \
async_checkpointing=True \


