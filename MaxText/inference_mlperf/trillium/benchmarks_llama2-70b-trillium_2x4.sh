#!/usr/bin/env bash

# Run command:
# bash benchmarks_llama2-70b-trillium_2x4.sh [-b benchmark_type]
# benchmark_type can be: performance, audit, accuracy, or all (default)

set -x

run_name="trillium_llama2-70b"
dry_run=false
enable_profiler=false
test_mode=false
benchmark_type="performance"

helpFunction()
{
   echo ""
   echo "Usage: $0 [-n] [-p] [-t] [-s] [-x] [-r run_name] [-m token_multiplier] [-b benchmark_type]"
   echo -e "\t-n Dry run mode"
   echo -e "\t-p Enable profiler"
   echo -e "\t-t Test mode"
   echo -e "\t-r Specify run name"
   echo -e "\t-b Specify benchmark type (performance|audit|accuracy|all)"
   exit 1
}


for arg in "$@"; do
    case $arg in
        -n) dry_run=true ;;
        -p) enable_profiler=true ;;
        -t) test_mode=true ;;
        -g|--paged) paged=true ;;
        -r=*|--run=*) run_name="${arg#*=}" ;;
        -r|--run) shift; run_name="$1" ;;
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


cmd=''
# RUN_OPTIONS=" -c " # Enable prefill packing by default
RUN_OPTIONS=" "
if "$dry_run"; then
    RUN_OPTIONS="${RUN_OPTIONS} -n "
fi

if "$enable_profiler"; then
    RUN_OPTIONS="${RUN_OPTIONS} -p "
fi


if "$test_mode"; then
    RUN_OPTIONS="${RUN_OPTIONS} -t "
fi

if "$paged"; then
    RUN_OPTIONS="${RUN_OPTIONS} -g "
fi

enable_xla_flags=true
export LIBTPU_INIT_ARGS=""
if "$enable_xla_flags"; then
    TEST_FLAGS=$(python3 select_xla_flags.py)
    export LIBTPU_INIT_ARGS=${TEST_FLAGS}
fi
echo XLA_FLAGS: $LIBTPU_INIT_ARGS

if [[ -z ${QUANTIZATION} ]] ; then
  export QUANTIZATION="int8"
  export QUANT_PATH=""
#   export QUANTIZATION="intmp"
#   export QUANT_MP="qkv_subchannel_512"
#   export QUANT_PATH="/home/${USER}/maxtext/MaxText/configs/quantization/${QUANT_MP}.json"
fi

if [[ -z ${KV_QUANT_DTYPE} ]] ; then
  export KV_QUANT_DTYPE="int4"
fi

if [[ -z ${CHECKPOINT} ]] ; then
  export CHECKPOINT="gs://inference-benchmarks/models/llama2-70b-chat/quant/${QUANTIZATION}_${QUANT_MP}"
fi

if [[ -z ${TOKENIZER_PATH} ]] ; then
  export TOKENIZER_PATH="/home/${USER}/maxtext/assets/tokenizer.llama2"
fi

# Default values if not overridden by environment variables
PREFILL_LEN_DEFAULT="2048"
BATCH_SIZE_PER_DEVICE_DEFAULT="7"

# Use environment variables if set, otherwise use defaults.
# These variables will hold the actual values being used, ensuring RUN_DESC is correct.
PREFILL_LEN=${USER_PREFILL_LEN:-$PREFILL_LEN_DEFAULT}
BATCH_SIZE_PER_DEVICE=${USER_BATCH_SIZE_PER_DEVICE:-$BATCH_SIZE_PER_DEVICE_DEFAULT}

export PREFILL_LENS_AND_PER_DEVICE_BATCH_SIZES="${PREFILL_LEN},${BATCH_SIZE_PER_DEVICE}"


BASE_CFG="model_name=llama2-70b tokenizer_path=${TOKENIZER_PATH} load_parameters_path=${CHECKPOINT}"
QUANT_CFG="quantization=${QUANTIZATION} quant_cfg_path=${QUANT_PATH} checkpoint_is_quantized=True"
KV_QUANT_CFG="quantize_kvcache=False" # kv_quant_dtype=${KV_QUANT_DTYPE}"
PAGED_ATTN_CFG=""
if "$paged"; then
    # Use environment variables for paging parameters if set, otherwise use original defaults
    _attention_type=${USER_ATTENTION_TYPE:-paged}
    _num_pages=${USER_PAGEDATTN_NUM_PAGES:-13000}
    _tokens_per_page=${USER_PAGEDATTN_TOKENS_PER_PAGE:-32}
    _pages_per_block=${USER_PAGEDATTN_PAGES_PER_COMPUTE_BLOCK:-4}

    PAGED_ATTN_CFG="attention=${_attention_type} pagedattn_num_pages=${_num_pages} pagedattn_tokens_per_page=${_tokens_per_page} pagedattn_pages_per_compute_block=${_pages_per_block}"
fi
export PAGED_ATTN_CFG=${PAGED_ATTN_CFG}


echo
echo $MAXENGINE_ARGS
echo
RUN_DESC=${run_name}_${PREFILL_LEN}_${BATCH_SIZE_PER_DEVICE}_quant_${QUANTIZATION}_${QUANT_MP}_kv_${KV_QUANT_DTYPE}_opt

$cmd cd ..

run_benchmark() {
    local type=$1
    case "$type" in
        "performance")
            $cmd bash llama_offline_run.sh ${RUN_OPTIONS} --run=benchmarks_performance_${RUN_DESC}
            ;;
        "audit")
            $cmd bash llama_offline_run.sh ${RUN_OPTIONS} -r benchmarks_audit_${RUN_DESC} -d
            ;;
        "accuracy")
            export HF_CKPT="meta-llama/Llama-2-70b-chat-hf"
            $cmd bash llama_offline_run.sh ${RUN_OPTIONS} -r benchmarks_accuracy_${RUN_DESC} -a  
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

