device_name="accel"
pids=$(sudo lsof -w /dev/${device_name}* | grep -E '[0-9]{2,}' | awk 'NF{print $2}')
pids=($(printf '%s\n' "${pids[@]}" | awk '$1 > 0'))
pids_unique=$(printf '%s\n' "${pids[@]}" | sort -u)
echo "pids: $pids"


length=${#pids[@]}
echo "length: $length"

length_unique=${#pids_unique[@]}
echo "length_unique: $length_unique"

echo "deduping..."
g=$(printf '%s\n' "${pids[@]}" | sort -u)
echo "g: $g"

