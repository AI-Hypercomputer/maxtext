
lr_array=(.001 .0015)
lr_name_array=(10 15) 

length_lr=${#lr_array[@]}
echo "starting loops"
for ((lr_idx=0; lr_idx<$length_lr; lr_idx++)); do
    lr=${lr_array[lr_idx]}
    lr_name=${lr_name_array[lr_idx]}
    echo "lr $lr lr_name:$lr_name"
done