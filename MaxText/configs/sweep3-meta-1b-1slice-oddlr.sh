
lr_array=(8 4 2 1 0.75)
scale_array=(1)
num_slice_array=(1)
key_array=(1)



echo "starting loops"
for lr in ${lr_array[@]}; do
    for scale in ${scale_array[@]}; do
        for num_slice in ${num_slice_array[@]}; do
            for key in ${key_array[@]}; do
                RUN_NAME=mattdavidow-sweep3-a1-params_lr_${lr}_scale_${scale}_slice_${num_slice}_PRNGKey_${key}
                echo ${RUN_NAME}
                echo ${RUN_NAME} >> mattdavidow-sweep3-1b-1slice-odd-a1.txt
                python3 multihost_job.py --BUCKET_NAME="mattdavidow-maxtext-br" --RUN_NAME=${run_name} --TPU_TYPE=v5litepod-256 --NUM_SLICES=${num_slice} --VERSION=v2-alpha-tpuv5-lite --COMMAND="bash setup.sh && bash MaxText/configs/sweep3-individual.sh ${lr} ${scale} ${num_slice} ${key} ${run_name}" --ZONE=us-east5-b      
            done
        done
    done
done
echo "finished loops"