
fwd_int8_array=("true")
bwd_int8_array=("true")
global_parameter_scale_array=(1)
prng_key_array=(0 1)


echo "starting loops"
for fwd_int8 in ${fwd_int8_array[@]}; do
    for bwd_int8 in ${bwd_int8_array[@]}; do
        for global_parameter in ${global_parameter_scale_array[@]}; do
            for prng_key in ${prng_key_array[@]}; do
                run_name=mattdavidow-baseline-tpem-a1_${fwd_int8}_bwd_${bwd_int8}_scale_${global_parameter_scale}_PRNGKey_${prng_key}
                echo ${run_name}
                echo ${run_name} >> mattdavidow-baseline-tpem-a1.txt
                python3 multihost_job.py --BUCKET_NAME="mattdavidow-maxtext-br" --RUN_NAME=${RUN_NAME} --TPU_TYPE=v5litepod-256 --NUM_SLICES=1 --VERSION=v2-alpha-tpuv5-lite --COMMAND="bash setup.sh && bash MaxText/configs/gen-baseline-individual.sh ${fwd_int8} ${bwd_int8} ${global_parameter_scale} ${prng_key} ${run_name}" --ZONE=us-east5-b      
            done
        done
    done
done
echo "finished loops"