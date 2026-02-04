#!/bin/sh

# Example run: bash tests/end_to_end/tpu/test_decode_load_quantized_ckpt.sh  -m llama2-70b -r test -s decode -n

dry_run=false
model='llama2-7b'
script_name='decode'
run_name="test_quant_ckpt"

while getopts "nm:r:s:" opt
do
  case "$opt" in
      n ) dry_run=true ;;
      m ) model="$OPTARG" ;;
      r ) run_name="$OPTARG" ;;
      s ) script_name="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
  esac
done

if "$dry_run"; then
    cmd=echo
else
    cmd=''
fi

export TOKENIZER_PATH="${MAXTEXT_ASSETS_ROOT:-${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/assets/tokenizers}}"/tokenizer.llama2
export MAX_PREFILL_PREDICT_LENGTH=128
export MAX_TARGET_LENGTH=256
export MODEL_NAME=${model}
export ICI_FSDP_PARALLELISM=1
export ICI_AUTOREGRESSIVE_PARALLELISM=1
export ICI_TENSOR_PARALLELISM=-1
export SCAN_LAYERS=false
export WEIGHT_DTYPE=bfloat16
export PER_DEVICE_BATCH_SIZE=20
export QUANTIZATION="int8"
export QUANTIZE_KVCACHE=True
export CHKPT_SUBDIR="${run_name}/${QUANTIZATION}_"
export LOAD_PARAMETERS_PATH=gs://${USER}-bkt/checkpoints/quant_${MODEL_NAME}-chat/${CHKPT_SUBDIR}
export OUTDIR="/tmp/${cmd}_res_${script_name}_chkpt/${CHKPT_SUBDIR}"
export OUTFILE="${OUTDIR}/${script_name}.txt"
mkdir -p $OUTDIR
echo
# Run script
${cmd} python3 -m MaxText.${script_name} \
  "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}"/configs/base.yml \
  tokenizer_path=${TOKENIZER_PATH} \
  load_parameters_path=${LOAD_PARAMETERS_PATH} \
  checkpoint_is_quantized=True \
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
  | tee -a $OUTFILE
echo
echo "Output directed to: ${OUTFILE}"
echo
echo "Checkpoint loaded from: ${LOAD_PARAMETERS_PATH}"
echo

