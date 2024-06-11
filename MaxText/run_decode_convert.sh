export MODEL_NAME=llama2-7b
# export MODEL_NAME=llama2-70b
export TOKENIZER_PATH=assets/tokenizer.llama2
export LOAD_PARAMETERS_PATH=gs://inference-benchmarks/models/${MODEL_NAME}-chat/2024-05-24-12-39/param-only-decode-ckpt-maxtext/checkpoints/0/items
export MAX_PREFILL_PREDICT_LENGTH=1024
export MAX_TARGET_LENGTH=2048
export ICI_FSDP_PARALLELISM=1
export ICI_AUTOREGRESSIVE_PARALLELISM=1
export ICI_TENSOR_PARALLELISM=-1
export SCAN_LAYERS=false
export WEIGHT_DTYPE=bfloat16
export PER_DEVICE_BATCH_SIZE=11
export QUANTIZATION="int8"
export QUANTIZE_KVCACHE=true
export SAVE_QUANTIZED_CHECKPOINT_PATH=gs://${USER}-bkt/checkpoints/quantized/${MODEL_NAME}-chat

python MaxText/decode.py \
  MaxText/configs/base.yml \
  tokenizer_path=${TOKENIZER_PATH} \
  load_parameters_path=${LOAD_PARAMETERS_PATH} \
  max_prefill_predict_length=${MAX_PREFILL_PREDICT_LENGTH} \
  max_target_length=${MAX_TARGET_LENGTH} \
  model_name=${MODEL_NAME} \
  ici_fsdp_parallelism=${ICI_FSDP_PARALLELISM} \
  ici_autoregressive_parallelism=${ICI_AUTOREGRESSIVE_PARALLELISM} \
  ici_tensor_parallelism=${ICI_TENSOR_PARALLELISM} \
  scan_layers=${SCAN_LAYERS} \
  weight_dtype=${WEIGHT_DTYPE} \
  per_device_batch_size=${PER_DEVICE_BATCH_SIZE} \
  quantization=${QUANTIZATION} \
  quantize_kvcache=${QUANTIZE_KVCACHE} \
  save_quantized_checkpoint=${SAVE_QUANTIZED_CHECKPOINT_PATH} \
