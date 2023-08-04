
lr_array=(0.0004)
scale_array=(1)
num_slice_array=(1)
init_key_array=(1 2 3)

echo "starting loops"
for lr in ${lr_array[@]}; do
    for scale in ${scale_array[@]}; do
        for num_slice in ${num_slice_array[@]}; do
            for key in ${init_key_array[@]}; do
                run_name=mattdavidow-sweep3p1-bf16-a1-params_PRNGKey_${key}
                echo ${run_name}
                echo ${run_name} >> mattdavidow-sweep3p1-baseline-a2.txt
                python3 multihost_job.py --BUCKET_NAME="mattdavidow-maxtext-br" --RUN_NAME=${run_name} --TPU_TYPE=v5litepod-256 --NUM_SLICES=${num_slice} --VERSION=v2-alpha-tpuv5-lite --COMMAND="bash setup.sh && bash MaxText/configs/sweep3p1-individual.sh ${lr} ${scale} ${num_slice} false false false ${key} ${run_name}" --ZONE=us-east5-b      
            done
        done
    done
done
echo "finished loops"