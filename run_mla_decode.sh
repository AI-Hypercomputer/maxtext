export CKPT=load_parameters_path=gs://agagik-us/deepseek/maxtext_checkpoints/ds_16b_unscanned_new_3/unscanned_chkpt/checkpoints/0/items
#export CKPT=""
python MaxText/decode.py MaxText/configs/base.yml run_name=test_dsv2_16b_decode_mla \
mla_naive_kvcache=false max_prefill_predict_length=100 per_device_batch_size=1 model_name=deepseek2-16b \
async_checkpointing=false ici_tensor_parallelism=4 max_target_length=196 ici_fsdp_parallelism=1 \
ici_autoregressive_parallelism=2 \
megablox=False sparse_matmul=False scan_layers=False attention=dot_product tokenizer_type=huggingface \
tokenizer_path=deepseek-ai/DeepSeek-V2-Lite $CKPT scan_layers=False hf_access_token=$HUGGING_FACE_TOKEN
