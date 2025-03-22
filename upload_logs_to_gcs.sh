
# Stop execution if any command exits with error
set -e

export BASE_OUTPUT_DIRECTORY="gs://maxtext-experiments-multipod"

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

export TIMESTAMP=$(date +%s)
export EXP_FOLDER="${BASE_OUTPUT_DIRECTORY}/mlperf/llama31-405b-${TIMESTAMP}"
echo $EXP_FOLDER


if [[ $(grep "MLLOG" /tmp/large_scale_multislice_test_log | wc -l) -gt 0 ]];then
  gsutil -m cp /tmp/large_scale_multislice_test_log "${EXP_FOLDER}/large_scale_multislice_test_log"
  bash parser_metrics.sh 2>&1 | tee /tmp/parser_metrics_log
  gsutil -m cp /tmp/parser_metrics_log "${EXP_FOLDER}/parser_metrics_log"
fi
