#!/bin/bash

# gen the int8 run names
REMAT_ARRAY=("minimal" "full")
USE_INT8_TRAINING_ARRAY=("true")
DTYPE_ARRAY=("bfloat16")
PRNG_KEY_ARRAY=(0 1 2)

RUN_NAMES=()
for REMAT_POLICY in ${REMAT_ARRAY[@]}; do
    for USE_INT8_TRAINING in ${USE_INT8_TRAINING_ARRAY[@]}; do
        for DTYPE in ${DTYPE_ARRAY[@]}; do
            for PRNG_KEY in ${PRNG_KEY_ARRAY[@]}; do
                RUN_NAME=mattdavidow-20230725-a2_remat_${REMAT_POLICY}_useint8_${USE_INT8_TRAINING}_dtype_${DTYPE}_PRNGKey_${PRNG_KEY}
                RUN_NAMES+=(${RUN_NAME})
            done        
        done
    done
done

# get the fwd off run names
REMAT_ARRAY=("full")
USE_INT8_TRAINING_ARRAY=("true")
DTYPE_ARRAY=("bfloat16")
PRNG_KEY_ARRAY=(0 1 2)

for REMAT_POLICY in ${REMAT_ARRAY[@]}; do
    for USE_INT8_TRAINING in ${USE_INT8_TRAINING_ARRAY[@]}; do
        for DTYPE in ${DTYPE_ARRAY[@]}; do
            for PRNG_KEY in ${PRNG_KEY_ARRAY[@]}; do
                RUN_NAME=mattdavidow-20230725-fwd-off_remat_${REMAT_POLICY}_useint8_${USE_INT8_TRAINING}_dtype_${DTYPE}_PRNGKey_${PRNG_KEY}_usefwd_False
                RUN_NAMES+=(${RUN_NAME})
            done        
        done
    done
done

# get the bfloat run names
REMAT_ARRAY=("full")
USE_INT8_TRAINING_ARRAY=("false")
DTYPE_ARRAY=("bfloat16")
PRNG_KEY_ARRAY=(0 1 2)

for REMAT_POLICY in ${REMAT_ARRAY[@]}; do
    for USE_INT8_TRAINING in ${USE_INT8_TRAINING_ARRAY[@]}; do
        for DTYPE in ${DTYPE_ARRAY[@]}; do
            for PRNG_KEY in ${PRNG_KEY_ARRAY[@]}; do
                RUN_NAME=mattdavidow-20230725-a2_remat_${REMAT_POLICY}_useint8_${USE_INT8_TRAINING}_dtype_${DTYPE}_PRNGKey_${PRNG_KEY}
                RUN_NAMES+=(${RUN_NAME})
            done        
        done
    done
done


length=${#RUN_NAMES[@]}
echo "lenth of list is: ${length}"

for item in ${RUN_NAMES[@]}; do
    echo ${item}
    echo ${item} >> mattdavidow-7-25-run-names.txt
done