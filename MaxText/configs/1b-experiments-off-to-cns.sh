
CNS_PATH=/cns/pi-d/home/${USER}/int8-metrics/
fileutil mkdir -p ${CNS_PATH}


REMAT_ARRAY=("full")
USE_INT8_TRAINING_ARRAY=("false")
DTYPE_ARRAY=("bfloat16")
PRNG_KEY_ARRAY=(0 1 2 3 4 5 6 7 8 9 10 11)

for REMAT_POLICY in ${REMAT_ARRAY[@]}; do
    for USE_INT8_TRAINING in ${USE_INT8_TRAINING_ARRAY[@]}; do
        for DTYPE in ${DTYPE_ARRAY[@]}; do
            for PRNG_KEY in ${PRNG_KEY_ARRAY[@]}; do
                echo "Remat: ${REMAT_POLICY}, int8: ${USE_INT8_TRAINING}, dtype: ${DTYPE}, PRNG_KEY: ${PRNG_KEY}"
                RUN_NAME=mattdavidow-a2-20230718_remat_${REMAT_POLICY}_useint8_${USE_INT8_TRAINING}_dtype_${DTYPE}_PRNGKey_${PRNG_KEY}
                OUTPUT_FILE=mattdavidow-maxtext-br/${RUN_NAME}.txt
                /google/data/ro/projects/cloud/bigstore/mpm/fileutil_bs/stable/bin/fileutil_bs cp /bigstore/${OUTPUT_FILE} ${CNS_PATH}/           
            done        
        done
    done
done