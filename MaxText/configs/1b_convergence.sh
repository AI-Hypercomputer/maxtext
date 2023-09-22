echo "Running 1b_convergence.sh"

RUN_NAME=${1}
OUTPUT_PATH=${2}
DATASET_PATH=${3}

TFLOP_THRESHOLD=0 # set to 0 since we are not actually running as a test.
bash end_to_end/test_convergence_1b_params.sh ${RUN_NAME} ${TFLOP_THRESHOLD} ${OUTPUT_PATH} ${DATASET_PATH}
