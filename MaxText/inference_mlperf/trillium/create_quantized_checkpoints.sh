#/bin/bash

# FILES=("int4_weight_only" "int8_weight_only" "dense_llm_weight_only_scale" "dense_llm_subchannel" "a4w4" "a8w4")
# # FILES=("int8_weight_only" "dense_llm_weight_only_scale" "dense_llm_subchannel" "a4w4" "a8w4")

# for FILENAME in "${FILES[@]}"; do
#   JAX_PLATFORMS=cpu python /mnt/disks/persist/maxtext/MaxText/decode.py \
#   /mnt/disks/persist/maxtext/MaxText/configs/base.yml \
#   tokenizer_path=/mnt/disks/persist/maxtext/assets/tokenizer.llama2 \
#   load_parameters_path=gs://inference-benchmarks/models/llama2-70b-chat/2024-05-08-23-16/param-only-decode-ckpt-maxtext/checkpoints/0/items \
#   max_prefill_predict_length=1024 \
#   max_target_length=2048 \
#   model_name=llama2-70b \
#   ici_fsdp_parallelism=1 \
#   ici_autoregressive_parallelism=1 \
#   ici_tensor_parallelism=1 \
#   scan_layers=false \
#   weight_dtype=bfloat16 \
#   per_device_batch_size=1 \
#   attention=dot_product \
#   quantize_kvcache=True \
#   quantization=intmp \
#   quant_cfg_path=/mnt/disks/persist/maxtext/MaxText/configs/quantization/${FILENAME}.json \
#   save_quantized_params_path=gs://patemotter/checkpoints/quantized/llama2-70b-chat/${FILENAME}
# done

JAX_PLATFORMS=cpu python MaxText/decode.py \
MaxText/configs/base.yml \
tokenizer_path=/mnt/disks/persist/maxtext_plain/assets/tokenizer.llama2 \
load_parameters_path=gs://inference-benchmarks/models/llama2-70b-chat/2024-05-08-23-16/param-only-decode-ckpt-maxtext/checkpoints/0/items \
max_prefill_predict_length=1024 \
max_target_length=2048 \
model_name=llama2-70b \
ici_fsdp_parallelism=1 \
ici_autoregressive_parallelism=1 \
ici_tensor_parallelism=1 \
scan_layers=false \
weight_dtype=bfloat16 \
per_device_batch_size=1 \
attention=dot_product \
quantize_kvcache=True \
quantization=intmp \
quant_cfg_path=/mnt/disks/persist/maxtext_plain/MaxText/configs/quantization/int4_weight_only.json \
save_quantized_params_path=gs://patemotter/checkpoints/quantized/llama2-70b-chat/int4_weight_only 