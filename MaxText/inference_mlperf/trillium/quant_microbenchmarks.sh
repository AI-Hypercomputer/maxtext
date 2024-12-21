# int8 baseline
python MaxText/inference_microbenchmark.py \
MaxText/configs/base.yml \
model_name=llama2-70b \
max_prefill_predict_length=1024 \
max_target_length=2048 \
attention=dot_product \
async_checkpointing=false \
ici_fsdp_parallelism=1 \
ici_autoregressive_parallelism=1 \
ici_tensor_parallelism=-1 \
per_device_batch_size=54 \
steps=10 \
scan_layers=false \
weight_dtype=bfloat16 \
tokenizer_path=/mnt/disks/persist/maxtext/assets/tokenizer.llama2 \
checkpoint_is_quantized=True \
compute_axis_order=0,2,1,3 \
ar_cache_axis_order=0,2,1,3 \
profiler=xplane \
base_output_directory=/mnt/disks/persist/quant_microbenchmark_results \
tensorboard_dir=/mnt/disks/persist/quant_microbenchmark_results \
quantize_kvcache=True \
kv_quant_dtype=int8 \
quantization=int8 \
quant_cfg_path= 

# # a16w4
python MaxText/inference_microbenchmark.py \
MaxText/configs/base.yml \
model_name=llama2-70b \
max_prefill_predict_length=1024 \
max_target_length=2048 \
attention=dot_product \
async_checkpointing=false \
ici_fsdp_parallelism=1 \
ici_autoregressive_parallelism=-1 \
per_device_batch_size=54 \
steps=10 \
scan_layers=false \
weight_dtype=bfloat16 \
tokenizer_path=/mnt/disks/persist/maxtext/assets/tokenizer.llama2 \
checkpoint_is_quantized=True \
compute_axis_order=0,2,1,3 \
ar_cache_axis_order=0,2,1,3 \
profiler=xplane \
base_output_directory=/mnt/disks/persist/quant_microbenchmark_results \
tensorboard_dir=/mnt/disks/persist/quant_microbenchmark_results \
quantize_kvcache=True \
kv_quant_dtype=int8 \
quantization=intmp \
quant_cfg_path=/mnt/disks/persist/maxtext/MaxText/configs/quantization/int4_weight_only.json

# # a16w8
python MaxText/inference_microbenchmark.py \
MaxText/configs/base.yml \
model_name=llama2-70b \
max_prefill_predict_length=1024 \
max_target_length=2048 \
attention=dot_product \
async_checkpointing=false \
ici_fsdp_parallelism=1 \
ici_autoregressive_parallelism=-1 \
per_device_batch_size=54 \
steps=10 \
scan_layers=false \
weight_dtype=bfloat16 \
tokenizer_path=/mnt/disks/persist/maxtext/assets/tokenizer.llama2 \
checkpoint_is_quantized=True \
compute_axis_order=0,2,1,3 \
ar_cache_axis_order=0,2,1,3 \
profiler=xplane \
base_output_directory=/mnt/disks/persist/quant_microbenchmark_results \
tensorboard_dir=/mnt/disks/persist/quant_microbenchmark_results \
quantize_kvcache=True \
kv_quant_dtype=int8 \
quantization=intmp \
quant_cfg_path=/mnt/disks/persist/maxtext/MaxText/configs/quantization/int8_weight_only.json

# # a8w4
python MaxText/inference_microbenchmark.py \
MaxText/configs/base.yml \
model_name=llama2-70b \
max_prefill_predict_length=1024 \
max_target_length=2048 \
attention=dot_product \
async_checkpointing=false \
ici_fsdp_parallelism=1 \
ici_autoregressive_parallelism=-1 \
per_device_batch_size=54 \
steps=10 \
scan_layers=false \
weight_dtype=bfloat16 \
tokenizer_path=/mnt/disks/persist/maxtext/assets/tokenizer.llama2 \
checkpoint_is_quantized=True \
compute_axis_order=0,2,1,3 \
ar_cache_axis_order=0,2,1,3 \
profiler=xplane \
base_output_directory=/mnt/disks/persist/quant_microbenchmark_results \
tensorboard_dir=/mnt/disks/persist/quant_microbenchmark_results \
quantize_kvcache=True \
kv_quant_dtype=int8 \
quantization=intmp \
quant_cfg_path=/mnt/disks/persist/maxtext/MaxText/configs/quantization/a8w4.json 

# # a4w4
python MaxText/inference_microbenchmark.py \
MaxText/configs/base.yml \
model_name=llama2-70b \
max_prefill_predict_length=1024 \
max_target_length=2048 \
attention=dot_product \
async_checkpointing=false \
ici_fsdp_parallelism=1 \
ici_autoregressive_parallelism=-1 \
per_device_batch_size=54 \
steps=10 \
scan_layers=false \
weight_dtype=bfloat16 \
tokenizer_path=/mnt/disks/persist/maxtext/assets/tokenizer.llama2 \
checkpoint_is_quantized=True \
compute_axis_order=0,2,1,3 \
ar_cache_axis_order=0,2,1,3 \
profiler=xplane \
base_output_directory=/mnt/disks/persist/quant_microbenchmark_results \
tensorboard_dir=/mnt/disks/persist/quant_microbenchmark_results \
quantize_kvcache=True \
kv_quant_dtype=int8 \
quantization=intmp \
quant_cfg_path=/mnt/disks/persist/maxtext/MaxText/configs/quantization/a4w4.json 

####################################################################################################

# int4 kv cache

# int8 baseline
python MaxText/inference_microbenchmark.py \
MaxText/configs/base.yml \
model_name=llama2-70b \
max_prefill_predict_length=1024 \
max_target_length=2048 \
attention=dot_product \
async_checkpointing=false \
ici_fsdp_parallelism=1 \
ici_autoregressive_parallelism=1 \
ici_tensor_parallelism=-1 \
per_device_batch_size=54 \
steps=10 \
scan_layers=false \
weight_dtype=bfloat16 \
tokenizer_path=/mnt/disks/persist/maxtext/assets/tokenizer.llama2 \
checkpoint_is_quantized=True \
compute_axis_order=0,2,1,3 \
ar_cache_axis_order=0,2,1,3 \
profiler=xplane \
base_output_directory=/mnt/disks/persist/quant_microbenchmark_results \
tensorboard_dir=/mnt/disks/persist/quant_microbenchmark_results \
quantize_kvcache=True \
kv_quant_dtype=int4 \
quantization=int8 \
quant_cfg_path= 

# # a16w4
python MaxText/inference_microbenchmark.py \
MaxText/configs/base.yml \
model_name=llama2-70b \
max_prefill_predict_length=1024 \
max_target_length=2048 \
attention=dot_product \
async_checkpointing=false \
ici_fsdp_parallelism=1 \
ici_autoregressive_parallelism=-1 \
per_device_batch_size=54 \
steps=10 \
scan_layers=false \
weight_dtype=bfloat16 \
tokenizer_path=/mnt/disks/persist/maxtext/assets/tokenizer.llama2 \
checkpoint_is_quantized=True \
compute_axis_order=0,2,1,3 \
ar_cache_axis_order=0,2,1,3 \
profiler=xplane \
base_output_directory=/mnt/disks/persist/quant_microbenchmark_results \
tensorboard_dir=/mnt/disks/persist/quant_microbenchmark_results \
quantize_kvcache=True \
kv_quant_dtype=int4 \
quantization=intmp \
quant_cfg_path=/mnt/disks/persist/maxtext/MaxText/configs/quantization/int4_weight_only.json

# # a16w8
python MaxText/inference_microbenchmark.py \
MaxText/configs/base.yml \
model_name=llama2-70b \
max_prefill_predict_length=1024 \
max_target_length=2048 \
attention=dot_product \
async_checkpointing=false \
ici_fsdp_parallelism=1 \
ici_autoregressive_parallelism=-1 \
per_device_batch_size=54 \
steps=10 \
scan_layers=false \
weight_dtype=bfloat16 \
tokenizer_path=/mnt/disks/persist/maxtext/assets/tokenizer.llama2 \
checkpoint_is_quantized=True \
compute_axis_order=0,2,1,3 \
ar_cache_axis_order=0,2,1,3 \
profiler=xplane \
base_output_directory=/mnt/disks/persist/quant_microbenchmark_results \
tensorboard_dir=/mnt/disks/persist/quant_microbenchmark_results \
quantize_kvcache=True \
kv_quant_dtype=int4 \
quantization=intmp \
quant_cfg_path=/mnt/disks/persist/maxtext/MaxText/configs/quantization/int8_weight_only.json

# # a8w4
python MaxText/inference_microbenchmark.py \
MaxText/configs/base.yml \
model_name=llama2-70b \
max_prefill_predict_length=1024 \
max_target_length=2048 \
attention=dot_product \
async_checkpointing=false \
ici_fsdp_parallelism=1 \
ici_autoregressive_parallelism=-1 \
per_device_batch_size=54 \
steps=10 \
scan_layers=false \
weight_dtype=bfloat16 \
tokenizer_path=/mnt/disks/persist/maxtext/assets/tokenizer.llama2 \
checkpoint_is_quantized=True \
compute_axis_order=0,2,1,3 \
ar_cache_axis_order=0,2,1,3 \
profiler=xplane \
base_output_directory=/mnt/disks/persist/quant_microbenchmark_results \
tensorboard_dir=/mnt/disks/persist/quant_microbenchmark_results \
quantize_kvcache=True \
kv_quant_dtype=int4 \
quantization=intmp \
quant_cfg_path=/mnt/disks/persist/maxtext/MaxText/configs/quantization/a8w4.json 

# # a4w4
python MaxText/inference_microbenchmark.py \
MaxText/configs/base.yml \
model_name=llama2-70b \
max_prefill_predict_length=1024 \
max_target_length=2048 \
attention=dot_product \
async_checkpointing=false \
ici_fsdp_parallelism=1 \
ici_autoregressive_parallelism=-1 \
per_device_batch_size=54 \
steps=10 \
scan_layers=false \
weight_dtype=bfloat16 \
tokenizer_path=/mnt/disks/persist/maxtext/assets/tokenizer.llama2 \
checkpoint_is_quantized=True \
compute_axis_order=0,2,1,3 \
ar_cache_axis_order=0,2,1,3 \
profiler=xplane \
base_output_directory=/mnt/disks/persist/quant_microbenchmark_results \
tensorboard_dir=/mnt/disks/persist/quant_microbenchmark_results \
quantize_kvcache=True \
kv_quant_dtype=int4 \
quantization=intmp \
quant_cfg_path=/mnt/disks/persist/maxtext/MaxText/configs/quantization/a4w4.json 
