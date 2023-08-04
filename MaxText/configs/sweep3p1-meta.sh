
lr_array=(0.0004)
scale_array=(1)
num_slice_array=(1)
fwd_int8_array=("true" "false")
bwd_int8_array=("true" "false")
init_key_array=(1 2 3)

echo "starting loops"
for lr in ${lr_array[@]}; do
    for scale in ${scale_array[@]}; do
        for num_slice in ${num_slice_array[@]}; do
            for fwd_int8 in ${fwd_int8_array[@]}; do
                for bwd_int8 in ${bwd_int8_array[@]}; do
                    for key in ${init_key_array[@]}; do
                        run_name=mattdavidow-sweep3p1-a2-params_fwd_${fwd_int8}_bwd_${bwd_int8}_PRNGKey_${key}
                        echo ${run_name}
                        echo ${run_name} >> mattdavidow-sweep3p1-a2.txt
                        python3 multihost_job.py --BUCKET_NAME="mattdavidow-maxtext-br" --RUN_NAME=${run_name} --TPU_TYPE=v5litepod-256 --NUM_SLICES=${num_slice} --VERSION=v2-alpha-tpuv5-lite --COMMAND="bash setup.sh && bash MaxText/configs/sweep3p1-individual.sh ${lr} ${scale} ${num_slice} true ${fwd_int8} ${bwd_int8} ${key} ${run_name}" --ZONE=us-east5-b      
                    done
                done
            done
        done
    done
done
echo "finished loops"