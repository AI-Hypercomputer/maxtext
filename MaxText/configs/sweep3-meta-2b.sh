
lr_array=(.0008 .0006 .0004 .0003 .0002 .00015 .0001 .000075) 
scale_array=(2)
num_slice_array=(1 2)
key_array=(1)


echo "starting loops"
for lr in ${lr_array[@]}; do
    lr_name=$(printf "%.e" "$lr")
    echo "lr_name of $lr_name"
    for scale in ${scale_array[@]}; do
        for num_slice in ${num_slice_array[@]}; do
            for key in ${key_array[@]}; do
                run_name=mattdavidow-sweep3-a4-params_lr_${lr_name}_scale_${scale}_slice_${num_slice}_PRNGKey_${key}
                echo ${run_name}
                echo ${run_name} >> mattdavidow-sweep3-1b-even-a1.txt
                python3 multihost_job.py --BUCKET_NAME="mattdavidow-maxtext-br" --RUN_NAME=${run_name} --TPU_TYPE=v5litepod-256 --NUM_SLICES=${num_slice} --VERSION=v2-alpha-tpuv5-lite --COMMAND="bash setup.sh && bash MaxText/configs/sweep3-individual.sh ${lr} ${scale} ${num_slice} ${key} ${run_name}" --ZONE=us-east5-b      
            done
        done
    done
done
echo "finished loops"