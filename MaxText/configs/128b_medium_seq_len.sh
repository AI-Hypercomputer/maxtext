echo "Running 128b.sh"

RUN_NAME=${1}
OUTPUT_PATH=${2}
DATASET_PATH=${3}

bash rto_setup.sh

TFLOP_THRESHOLD=0 # set to 0 since we are not actually running as a test.
bash end_to_end/test_tflops_128b_params_medium_seq_len.sh ${RUN_NAME} ${TFLOP_THRESHOLD} ${OUTPUT_PATH} ${DATASET_PATH}
