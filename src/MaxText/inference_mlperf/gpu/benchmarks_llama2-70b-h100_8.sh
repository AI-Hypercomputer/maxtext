#!/usr/bin/env bash

# Run command:
# bash benchmarks_llama2-70b-h100_8.sh [-b benchmark_type]
# benchmark_type can be: performance, audit, accuracy, or all (default)

run_name="h100_llama2-70b"
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
RUN_OPTIONS=" -c " # Enable prefill packing by default
if "$dry_run"; then
    RUN_OPTIONS="${RUN_OPTIONS} -n "
fi

if "$enable_profiler"; then
    RUN_OPTIONS="${RUN_OPTIONS} -p -m=${profiler_mode} -x=${profiler_tensorboard_dir} "
fi


if "$test_mode"; then
    RUN_OPTIONS="${RUN_OPTIONS} -t "
fi

export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_command_buffer=FUSION --xla_disable_hlo_passes=rematerialization"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.94
echo XLA_FLAGS: $XLA_FLAGS

if [[ -z ${MAXENGINE_CONFIG_FILEPATH} ]] ; then
    export MAXENGINE_CONFIG_FILEPATH="$(dirname $0)/../../configs/inference.yml"
fi

if [[ -z ${QUANTIZATION} ]] ; then
    export QUANTIZATION="aqt_fp8"
fi

if [[ -z ${KV_QUANT_DTYPE} ]] ; then
    export KV_QUANT_DTYPE="fp8"
    export QUANTIZE_KVCACHE=True
fi

if [[ -z ${CHECKPOINT} ]] ; then
    export CHECKPOINT="gs://inference-benchmarks/models/llama2-70b-chat/2024-05-08-23-16/param-only-decode-ckpt-src/MaxText/checkpoints/0/items"
fi

if [[ -z ${TOKENIZER_PATH} ]] ; then
    export TOKENIZER_PATH="${MAXTEXT_ASSETS_ROOT:-${MAXTEXT_REPO_ROOT:-$PWD}/assets}/tokenizer.llama2"
fi

if [ -z "$PREFILL_LENS_AND_PER_DEVICE_BATCH_SIZES" ];
then
    PREFILL_LEN="1024"
    BATCH_SIZE_PER_DEVICE="190"
    export PREFILL_LENS_AND_PER_DEVICE_BATCH_SIZES="${PREFILL_LEN},${BATCH_SIZE_PER_DEVICE}"
fi


BASE_CFG="model_name=llama2-70b tokenizer_path=${TOKENIZER_PATH} load_parameters_path=${CHECKPOINT} scan_layers=false hardware=gpu async_checkpointing=False ici_tensor_parallelism=-1 weight_dtype=bfloat16"
KV_QUANT_CFG="quantize_kvcache=${QUANTIZE_KVCACHE} kv_quant_dtype=${KV_QUANT_DTYPE} quantization=${QUANTIZATION}"
export MAXENGINE_ARGS="${BASE_CFG} ${KV_QUANT_CFG} optimize_mesh_for_tpu_v6e=false"
echo
echo $MAXENGINE_ARGS
echo
RUN_DESC=${run_name}_${PREFILL_LEN}_${BATCH_SIZE_PER_DEVICE}_quant_${QUANTIZATION}_${QUANT_MP}_kv_${KV_QUANT_DTYPE}_opt
export BASEDIR=/opt/src/MaxText/Maxtext/inference_mlperf/

# Run from repository root
$cmd cd $(dirname $0)/../../../

run_benchmark() {
    local type=$1
    case "$type" in
        "performance")
            $cmd bash ./MaxText/inference_mlperf/llama_offline_run.sh ${RUN_OPTIONS} -r -benchmarks_performance_${RUN_DESC}
            ;;
        "audit")
            $cmd bash ./MaxText/inference_mlperf/llama_offline_run.sh ${RUN_OPTIONS} -r -benchmarks_audit_${RUN_DESC} -d
            ;;
        "accuracy")
            export HF_CKPT="meta-llama/Llama-2-70b-chat-hf"
            $cmd bash ./MaxText/inference_mlperf/llama_offline_run.sh ${RUN_OPTIONS} -r benchmarks_accuracy_${RUN_DESC} -a
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

