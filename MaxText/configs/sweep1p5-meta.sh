
int8_training_array=("false" "true")
global_parameter_scale_array=(1)
PRNG_KEY_ARRAY=(0)



echo "starting loops"
for int8_training in ${int8_training_array[@]}; do
    for global_parameter_scale in ${global_parameter_scale_array[@]}; do
        for PRNG_KEY in ${PRNG_KEY_ARRAY[@]}; do
            RUN_NAME=mattdavidow-sweep1p5-a1_int8_${int8_training}_scale_${global_parameter_scale}_PRNGKey_${PRNG_KEY}
            echo ${RUN_NAME}
            echo ${RUN_NAME} >> mattdavidow-sweep1p5-run-names-a1.txt
            python3 multihost_job.py --BUCKET_NAME="mattdavidow-maxtext-br" --RUN_NAME=${RUN_NAME} --TPU_TYPE=v5litepod-256 --NUM_SLICES=1 --VERSION=v2-alpha-tpuv5-lite --COMMAND="bash setup.sh && bash MaxText/configs/sweep1-individual-long.sh ${int8_training} ${global_parameter_scale} ${PRNG_KEY} ${RUN_NAME}" --CQR_EXTRA_ARGS="--reserved --metadata priority-group=medium" --ZONE=us-west4-a      
        done
    done
done
echo "finished loops"