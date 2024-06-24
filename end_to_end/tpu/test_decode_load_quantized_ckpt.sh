#!/bin/sh

dry_run=false
model='llama2-7b'

while getopts "nm:" opt
do
  case "$opt" in
      n ) dry_run=true ;;
      m ) model="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
  esac
done

if "$dry_run"; then
    cmd=echo
else
    cmd=''
fi

export TOKENIZER_PATH=assets/tokenizer.llama2
export MAX_PREFILL_PREDICT_LENGTH=1024
export MAX_TARGET_LENGTH=2048
export MODEL_NAME=${model}
export ICI_FSDP_PARALLELISM=1
export ICI_AUTOREGRESSIVE_PARALLELISM=1
export ICI_TENSOR_PARALLELISM=-1
export SCAN_LAYERS=false
export WEIGHT_DTYPE=bfloat16
export PER_DEVICE_BATCH_SIZE=11
export QUANTIZATION="int8"
export QUANTIZE_KVCACHE=True
export LOAD_PARAMETERS_PATH=gs://${USER}-bkt/checkpoints/quantized/aqt/${MODEL_NAME}-chat

${cmd} python MaxText/decode.py \
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
  checkpoint_is_quantized=True


