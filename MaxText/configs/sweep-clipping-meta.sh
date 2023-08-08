lr=0.001
lr_name=1

int8_array=("True" "False")
int8_name_array=("T" "F")
length_int8=${#int8_array[@]}

clipping_threshold_array=(1.0 10000.0)
clipping_theshold_name_array=("On" "Off")
length_clipping=${#clipping_threshold_array[@]}

key_array=(1 2)

scale=8
num_slice=4

for ((int8_index=0; int8_index<$length_int8; int8_index++)); do
    int8=${int8_array[int8_index]}
    int8_name=${int8_name_array[int8_index]}
    for ((clip_index=0; clip_index<$length_clipping; clip_index++)); do
        clipping_threshold=${clipping_threshold_array[clip_index]}
        clipping_theshold_name=${clipping_theshold_name_array[clip_index]}
        for key in ${key_array[@]}; do
            run_name=mattdavidow-sweep-clipping-a1_int8${int8_name}_size${scale}_pods${num_slice}_clipping${clipping_theshold_name}_key${key}
            echo ${run_name}
            echo ${run_name} >> mattdavidow-sweep-clipping-run-names.txt
        done
    done
done
                
python3 multihost_job.py --BUCKET_NAME="mattdavidow-maxtext-br" --RUN_NAME=${run_name} --TPU_TYPE=v5litepod-256 --NUM_SLICES=${num_slice} --VERSION=v2-alpha-tpuv5-lite --COMMAND="bash setup.sh && bash MaxText/configs/sweep-clipping-individual.sh ${lr} ${int8} ${scale} ${num_slice} ${clipping_threshold} ${key} ${run_name}" --ZONE=us-east5-b      
