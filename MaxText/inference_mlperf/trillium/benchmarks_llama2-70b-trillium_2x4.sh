#!/usr/bin/env bash

# Run command:
# bash benchmarks_llama2-70b-trillium_2x4.sh

run_name="trillium_llama2-70b"
dry_run=false
enable_profiler=false
enable_xla_flags=false
single_bucket=false
token_multiplier=3.0
test_mode=false

while getopts "nptsxr:m:" opt
do
  case "$opt" in
      n ) dry_run=true ;;
      p ) enable_profiler=true ;;
      t ) test_mode=true;;
      s ) single_bucket=true ;;
      x ) enable_xla_flags=true ;;
      r ) run_name="$OPTARG" ;;
      m ) token_multiplier="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
  esac
done


if "$dry_run"; then
    cmd=echo
else
    cmd=''
fi

RUN_OPTIONS=""
if "$enable_profiler"; then
    RUN_OPTIONS=" -p "
fi

if "$test_mode"; then
    RUN_OPTIONS="${RUN_OPTIONS} -t "
fi



if "$single_bucket"; then
    export BATCH_AND_PREFILL_LEN="1024,54"
else
    export BATCH_AND_PREFILL_LEN="256,32|512,16|1024,8"
fi
batch_and_prefill_str=$(echo $BATCH_AND_PREFILL_LEN |tr \|,  _) 

export LIBTPU_INIT_ARGS=""
if "$enable_xla_flags"; then
    TEST_FLAGS=$(python3 select_xla_flags.py)
    export LIBTPU_INIT_ARGS=${TEST_FLAGS}
fi

export TOK_OUTLEN_MULTIPLIER=${token_multiplier}

CHECKPOINT="gs://patemotter/checkpoints/quant_llama2-70b-chat/int8w_"
TOKENIZER_PATH=/mnt/disks/persist/maxtext/assets/tokenizer.llama2
BASE_CFG="model_name=llama2-70b tokenizer_path=${TOKENIZER_PATH} load_parameters_path=${CHECKPOINT}"
QUANT_CFG="quantization=int8 quantize_kvcache=True checkpoint_is_quantized=True"
LAYOUT_CFG="compute_axis_order=0,2,1,3 ar_cache_axis_order=0,2,1,3"
ICI_FSDP_PARALLELISM=1
ICI_AUTOREGRESSIVE_PARALLELISM=1
ICI_TENSOR_PARALLELISM=4
ICI_SEQUENCE_PARALLELISM=2
export ICI_ARGS="ici_tensor_parallelism=${ICI_TENSOR_PARALLELISM} ici_sequence_parallelism=${ICI_SEQUENCE_PARALLELISM} ici_fsdp_parallelism=${ICI_FSDP_PARALLELISM} ici_autoregressive_parallelism=${ICI_AUTOREGRESSIVE_PARALLELISM}"
export MAXENGINE_ARGS="${BASE_CFG} ${QUANT_CFG} ${LAYOUT_CFG} ${ICI_ARGS}"

RUN_DESC=int8_kv_${batch_and_prefill_str}_${token_multiplier}_flags_${enable_xla_flags}

# $cmd  cd ..
# Run mlperf perfromance benchmarks
$cmd bash llama_offline_run.sh  -r benchmarks_performance_${RUN_DESC} ${RUN_OPTIONS}

# Run mlperf audit
# bash llama_offline_run.sh  -r benchmarks_audit_${RUN_DESC} -d

# Run mlperf accuracy run
# bash llama_offline_run.sh  -r benchmarks_accuracy_${RUN_DESC} -a
