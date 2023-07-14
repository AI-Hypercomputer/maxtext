echo "Running 16b.sh"

RUN_NAME=${1}
OUTPUT_PATH="gs://maxtext-experiments-multipod/"
DATASET_PATH="gs://max-datasets-rogue/"

bash rto_setup.sh

TFLOP_THRESHOLD=0 # set to 0 since we are not actually running as a test.
bash end_to_end/test_tflops_16b_params.sh ${RUN_NAME} ${TFLOP_THRESHOLD} ${OUTPUT_PATH} ${DATASET_PATH}
