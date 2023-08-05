
lr_array=(.001)
lr_name_array=(10)

#lr_array=(.001 .0015)
#lr_name_array=(10 15)

scale_array=(1)
num_slice_array=(1) #num_slice_array=(1 2)
key_array=(1)

length_lr=${#lr_array[@]}
echo "starting loops"
for ((lr_idx=0; lr_idx<$length_lr; lr_idx++)); do
    lr=${lr_array[lr_idx]}
    lr_name=${lr_name_array[lr_idx]}
    echo "lr_name of $lr_name"
    for scale in ${scale_array[@]}; do
        for num_slice in ${num_slice_array[@]}; do
            for key in ${key_array[@]}; do
                run_name=mattdavidow-canary-a5-sweep4-lr_${lr_name}_scale_${scale}_slice_${num_slice}_PRNGKey_${key}
                echo ${run_name}
                echo ${run_name} >> mattdavidow-canary-a5-sweep4.txt
                python3 multihost_job.py --BUCKET_NAME="mattdavidow-maxtext-br" --RUN_NAME=${run_name} --TPU_TYPE=v5litepod-256 --NUM_SLICES=${num_slice} --VERSION=v2-alpha-tpuv5-lite --COMMAND="bash setup.sh && bash MaxText/configs/sweep3-individual.sh ${lr} ${scale} ${num_slice} ${key} ${run_name}" --ZONE=us-east5-b      
            done
        done
    done
done
echo "finished loops"