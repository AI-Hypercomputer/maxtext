
fwd_int8_array=("false" "true")
bwd_int8_array=("false" "true")
global_parameter_scale_array=(1 2)
PRNG_KEY_ARRAY=(0 1)



echo "starting loops"
for fwd_int8 in ${fwd_int8_array[@]}; do
    for bwd_int8 in ${bwd_int8_array[@]}; do
        for global_parameter_scale in ${global_parameter_scale_array[@]}; do
            for PRNG_KEY in ${PRNG_KEY_ARRAY[@]}; do
                RUN_NAME=mattdavidow-sweep1-a1_fwd_${fwd_int8}_bwd_${bwd_int8}_scale_${global_parameter_scale}_PRNGKey_${PRNG_KEY}
                echo ${RUN_NAME}
                echo ${RUN_NAME} >> mattdavidow-sweep-run-names-a1.txt
                python3 multihost_job.py --BUCKET_NAME="mattdavidow-maxtext-br" --RUN_NAME=${RUN_NAME} --TPU_TYPE=v5litepod-256 --NUM_SLICES=1 --VERSION=v2-alpha-tpuv5-lite --COMMAND="bash setup.sh && bash MaxText/configs/sweep1-individual.sh ${fwd_int8} ${bwd_int8} ${global_parameter_scale} ${PRNG_KEY} ${RUN_NAME}" --CQR_EXTRA_ARGS="--reserved --metadata priority-group=medium" --ZONE=us-west4-a      
            done
        done
    done
done
echo "finished loops"