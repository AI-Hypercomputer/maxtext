#!/bin/sh

# Example run: bash end_to_end/tpu/test_decode_save_quantized_ckpt.sh -m llama2-70b -r 070924 -n

dry_run=false
model='llama2-7b'
run_name=$(date +'%Y%m%d%H%M%S')

while getopts "nm:r:" opt
do
  case "$opt" in
      n ) dry_run=true ;;
      m ) model="$OPTARG" ;;
      r ) run_name="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
  esac
done

if "$dry_run"; then
    cmd=echo
else
    cmd=''
fi

if [ "$model" = "llama2-7b" ]; then
  checkpoint_ts='2024-05-24-12-39'
fi

if [ "$model" = "llama2-70b" ]; then
  checkpoint_ts='2024-05-08-23-16'
fi

export MODEL_NAME=${model}
export TOKENIZER_PATH=assets/tokenizer.llama2
export LOAD_PARAMETERS_PATH=gs://inference-benchmarks/models/${MODEL_NAME}-chat/${checkpoint_ts}/param-only-decode-ckpt-maxtext/checkpoints/0/items
export MAX_PREFILL_PREDICT_LENGTH=1024
export MAX_TARGET_LENGTH=2048
export ICI_FSDP_PARALLELISM=1
export ICI_AUTOREGRESSIVE_PARALLELISM=1
export ICI_TENSOR_PARALLELISM=-1
export SCAN_LAYERS=false
export WEIGHT_DTYPE=bfloat16
export PER_DEVICE_BATCH_SIZE=11
export QUANTIZATION="intmp"
export QUANT_CFG="mp_scale"
export QUANT_CFG_PATH="MaxText/configs/quantization/${QUANT_CFG}.json"
export QUANTIZE_KVCACHE=False
export CHKPT_SUBDIR="${run_name}/${QUANTIZATION}_${QUANT_CFG}"
export SAVE_QUANTIZED_CHECKPOINT_PATH=gs://${USER}-bkt/checkpoints/quant_${MODEL_NAME}-chat/${CHKPT_SUBDIR}
export OUTDIR="/tmp/${cmd}_res_save_chkpt/${CHKPT_SUBDIR}"
export OUTFILE="${OUTDIR}/decode.txt"
mkdir -p $OUTDIR
echo
# Run command
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
  quant_cfg_path=${QUANT_CFG_PATH} \
  quantize_kvcache=${QUANTIZE_KVCACHE} \
  save_quantized_params_path=${SAVE_QUANTIZED_CHECKPOINT_PATH} \
  | tee -a $OUTFILE
echo
echo "Output directed to: ${OUTFILE}"
echo
echo "Checkpoint saved at:$SAVE_QUANTIZED_CHECKPOINT_PATH"
${cmd} gsutil ls -lh $SAVE_QUANTIZED_CHECKPOINT_PATH >> ${OUTFILE}
echo


