# Run command:
# bash deekseekv2-trillium_2x4.sh
# Look at profiles: 
# tensorboard --logdir /tmp/mb/profiles/trillium_deepseekv2/tensorboard/prefill_insert_1024 


run_name="trillium_deepseekv2"
dry_run=false
enable_profiler=false
enable_xla_flags=true
dump_hlo=false
prefill_lens="1024"
stages="prefill,generate"

while getopts "npdr:s:l:" opt
do
  case "$opt" in
      n ) dry_run=true ;;
      p ) enable_profiler=true ;;
      d ) dump_hlo=true ;;
      r ) run_name="$OPTARG" ;;
      s ) stages="$OPTARG" ;;
      l ) prefill_lens="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
  esac
done


if "$dry_run"; then
    cmd=echo
else
    cmd=''
fi

BASEDIR=/tmp/mb
mkdir -p ${BASEDIR}/logs
mkdir -p ${BASEDIR}/profiles
mkdir -p ${BASEDIR}/hlo

PROFILER_OPTION=""
if "$enable_profiler"; then
    PROFILER_OPTION="profiler=xplane upload_all_profiler_results=True"
fi

export XLA_FLAGS=""
if "$dump_hlo"; then
    export XLA_FLAGS="--xla_dump_to=/tmp/mb/hlo"
fi

export LIBTPU_INIT_ARGS=""
if "$enable_xla_flags"; then
    TEST_FLAGS=$(python3 select_xla_flags.py)
    export LIBTPU_INIT_ARGS=${TEST_FLAGS}
fi
echo
echo "LIBTPU_INIT_ARGS:${LIBTPU_INIT_ARGS}"
echo "XLA_FLAGS:${XLA_FLAGS}"
echo

export TOKENIZER_PATH=/mnt/disks/persist/git/maxtext/assets/tokenizer.mistral-v3
#export LOAD_PARAMETERS_PATH=gs://ranran-multipod-dev/deepseek2/conversion/unscanned/0/items
export LOAD_PARAMETERS_PATH=gs://ranran-multipod-dev/deepseek2/conversion/3/0/items
export MAX_PREFILL_PREDICT_LENGTH=64
export MAX_TARGET_LENGTH=256
export MODEL_NAME=deepseek2-16b
export ICI_FSDP_PARALLELISM=1
export ICI_AUTOREGRESSIVE_PARALLELISM=1
export ICI_TENSOR_PARALLELISM=8
export SCAN_LAYERS=true
export WEIGHT_DTYPE=bfloat16
export PER_DEVICE_BATCH_SIZE=16
export MLA_NAIVE_KVCACHE=True
export RUN_DESC="${run_name}_xla_flags_${enable_xla_flags}"

$cmd python3 ../../inference_microbenchmark.py   \
../../configs/base.yml   tokenizer_path=${TOKENIZER_PATH}   \
load_parameters_path=${LOAD_PARAMETERS_PATH}   \
mla_naive_kvcache=${MLA_NAIVE_KVCACHE} \
max_prefill_predict_length=${MAX_PREFILL_PREDICT_LENGTH}   \
max_target_length=${MAX_TARGET_LENGTH}   model_name=${MODEL_NAME}   \
ici_fsdp_parallelism=${ICI_FSDP_PARALLELISM}   \
ici_autoregressive_parallelism=${ICI_AUTOREGRESSIVE_PARALLELISM}   \
ici_tensor_parallelism=${ICI_TENSOR_PARALLELISM}   scan_layers=${SCAN_LAYERS} \
weight_dtype=${WEIGHT_DTYPE}   per_device_batch_size=${PER_DEVICE_BATCH_SIZE} \
inference_microbenchmark_stages=${stages} \
inference_microbenchmark_prefill_lengths="${prefill_lens}" \
attention=dot_product megablox=False sparse_matmul=False base_output_directory="/tmp/mb/profiles" run_name=${RUN_DESC} \
${PROFILER_OPTION} 2>&1 | tee /tmp/mb/logs/${cmd}_${RUN_DESC}
