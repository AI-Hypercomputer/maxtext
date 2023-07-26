
echo "Starting conv_gap_test"
SAVE_ACCUMULATOR_ARRAY=("false" "true")
USE_RHS_NOISE_FUNCTION_ARRAY=("false" "true")
PRNG_KEY_ARRAY=(6 7 8 9 10 11)

echo "starting loops"
for SAVE_ACCUMULATOR in ${SAVE_ACCUMULATOR_ARRAY[@]}; do
    for USE_RHS_NOISE_FUNCTION in ${USE_RHS_NOISE_FUNCTION_ARRAY[@]}; do
        for PRNG_KEY in ${PRNG_KEY_ARRAY[@]}; do
            echo "SAVE_ACCUMULATOR: ${SAVE_ACCUMULATOR}, USE_RHS_NOISE_FUNCTION_ARRAY: ${USE_RHS_NOISE_FUNCTION_ARRAY} PRNG_KEY: ${PRNG_KEY}"
            RUN_NAME=mattdavidow-20230725_ACCUMULATOR_${SAVE_ACCUMULATOR}_${USE_RHS_NOISE_FUNCTION}_PRNGKey_${PRNG_KEY}
            echo "Running the above setting with RUN_NAME ${RUN_NAME}"
            python3 multihost_job.py --BUCKET_NAME="mattdavidow-maxtext-br" --RUN_NAME=${RUN_NAME} --TPU_TYPE=v5litepod-256 --NUM_SLICES=1 --VERSION=v2-alpha-tpuv5-lite --COMMAND="bash setup.sh && bash MaxText/configs/1b-experiments-7-25.sh ${SAVE_ACCUMULATOR} ${USE_RHS_NOISE_FUNCTION} ${PRNG_KEY} ${RUN_NAME}" --CQR_EXTRA_ARGS="--reserved" --ZONE=us-west4-a      
        done
    done
done
echo "finished loops"