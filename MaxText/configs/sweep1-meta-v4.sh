
fwd_int8_array=("false" "true")
bwd_int8_array=("false" "true")
global_parameter_scale_array=(1 2)
PRNG_KEY_ARRAY=(0)


echo "starting loops"
for fwd_int8 in ${fwd_int8_array[@]}; do
    for bwd_int8 in ${bwd_int8_array[@]}; do
        for global_parameter_scale in ${global_parameter_scale_array[@]}; do
            echo "STEPS: $STEPS"
            for PRNG_KEY in ${PRNG_KEY_ARRAY[@]}; do
                RUN_NAME=mattdavidow-sweep1-a7-v4_fwd_${fwd_int8}_bwd_${bwd_int8}_scale_${global_parameter_scale}_PRNGKey_${PRNG_KEY}
                echo ${RUN_NAME}
                echo ${RUN_NAME} >> mattdavidow-sweep-v4-a7.txt
                python3 multihost_job.py --BUCKET_NAME="mattdavidow-br" --RUN_NAME=${RUN_NAME} --TPU_TYPE=v4-8 --NUM_SLICES=1 --VERSION=tpu-ubuntu2204-base --COMMAND="bash setup.sh && bash MaxText/configs/sweep1-individual-v4.sh ${fwd_int8} ${bwd_int8} ${global_parameter_scale} ${PRNG_KEY} ${RUN_NAME}" --CQR_EXTRA_ARGS="--best-effort --metadata priority-group=medium" --ZONE=us-central2-b --PROJECT=cloud-tpu-multipod-dev     
            done
        done
    done
done
echo "finished loops"