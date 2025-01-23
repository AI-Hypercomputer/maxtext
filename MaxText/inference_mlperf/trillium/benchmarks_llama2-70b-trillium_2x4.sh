#!/usr/bin/env bash

# Run command:
# bash benchmarks_llama2-70b-trillium_2x4.sh [-b benchmark_type]
# benchmark_type can be: performance, audit, accuracy, or all (default)

run_name="trillium_llama2-70b"
dry_run=false
enable_profiler=false
enable_batch_prefill=false
enable_xla_flags=false
single_bucket=false
token_multiplier=3.0
test_mode=false
benchmark_type="all"

helpFunction()
{
   echo ""
   echo "Usage: $0 [-n] [-p] [-t] [-s] [-x] [-r run_name] [-m token_multiplier] [-b benchmark_type]"
   echo -e "\t-n Dry run mode"
   echo -e "\t-p Enable profiler"
   echo -e "\t-t Test mode"
   echo -e "\t-s Single bucket mode"
   echo -e "\t-x Enable XLA flags"
   echo -e "\t-r Specify run name"
   echo -e "\t-m Specify token multiplier"
   echo -e "\t-b Specify benchmark type (performance|audit|accuracy|all)"
   exit 1
}


for arg in "$@"; do
    case $arg in
        -n) dry_run=true ;;
        -p) enable_profiler=true ;;
        -t) test_mode=true ;;
        -s) single_bucket=true ;;
        -x) enable_xla_flags=true ;;
        -c) enable_batch_prefill=true ;;
        -r=*|--run=*) run_name="${arg#*=}" ;;
        -r|--run) shift; run_name="$1" ;;
        -m=*|--multiplier=*) token_multiplier="${arg#*=}" ;;
        -m|--multiplier) shift; token_multiplier="$1" ;;
        -b=*|--benchmark=*) benchmark_type="${arg#*=}" ;;
        -b|--benchmark) shift; benchmark_type="$1" ;;
        -h|--help) helpFunction ;;
    esac
    shift
done

# Validate benchmark type
case "$benchmark_type" in
    performance|audit|accuracy|all) ;;
    *) echo "Invalid benchmark type. Must be: performance, audit, accuracy, or all"; exit 1 ;;
esac

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

if "$enable_batch_prefill"; then
    RUN_OPTIONS="${RUN_OPTIONS} -c "
fi

if "$single_bucket"; then
    export BATCH_AND_PREFILL_LEN="1024,54"
else
    export BATCH_AND_PREFILL_LEN="256,216|512,108|1024,54"
fi
batch_and_prefill_str=$(echo $BATCH_AND_PREFILL_LEN |tr \|,  _) 

export LIBTPU_INIT_ARGS=""
if "$enable_xla_flags"; then
    TEST_FLAGS=$(python3 select_xla_flags.py)
    export LIBTPU_INIT_ARGS=${TEST_FLAGS}
fi

export TOK_OUTLEN_MULTIPLIER=${token_multiplier}

if [[ -z ${CHECKPOINT} ]] ; then
  export CHECKPOINT="gs://inference-benchmarks/models/llama2-70b-chat/quant/int8_"
fi

if [[ -z ${TOKENIZER_PATH} ]] ; then
  export TOKENIZER_PATH="/home/${USER}/maxtext/assets/tokenizer.llama2"
fi

BASE_CFG="model_name=llama2-70b tokenizer_path=${TOKENIZER_PATH} load_parameters_path=${CHECKPOINT}"
QUANT_CFG="quantization=int8 quantize_kvcache=True checkpoint_is_quantized=True"
LAYOUT_CFG="compute_axis_order=0,2,1,3 ar_cache_axis_order=0,2,1,3"
export MAXENGINE_ARGS="${BASE_CFG} ${QUANT_CFG} ${LAYOUT_CFG}"

RUN_DESC=int8_kv_${batch_and_prefill_str}_${token_multiplier}_flags_${enable_xla_flags}

$cmd cd ..

run_benchmark() {
    local type=$1
    case "$type" in
        "performance")
            $cmd bash llama_offline_run.sh ${RUN_OPTIONS} -r benchmarks_performance_${RUN_DESC}
            ;;
        "audit")
            $cmd bash llama_offline_run.sh -r benchmarks_audit_${RUN_DESC} -d
            ;;
        "accuracy")
            $cmd bash llama_offline_run.sh -r benchmarks_accuracy_${RUN_DESC} -a  
            ;;
    esac
}

if [ "$benchmark_type" = "all" ]; then
    run_benchmark "performance"
    run_benchmark "audit"
    run_benchmark "accuracy"
else
    run_benchmark "$benchmark_type"
fi