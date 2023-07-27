import subprocess

s='''
_TPU_VERSION_NAME=${1}
device_name="accel"
if [[ "${_TPU_VERSION_NAME}" =~ ^v5.* ]]; then
  device_name="vfio/"
fi
echo -e "Searching for existing processes on device ${device_name}..."
pids=$(sudo lsof -w /dev/${device_name}* | grep -v '"'"'PID'"'"'| awk '"'"'{print $2}'"'"')
pids_unique=$(printf '"'"'%s\n'"'"' "${pids[@]}" | sort -u)
pids_unique=($(printf '"'"'%s\n'"'"' "${pids_unique[@]}" | grep -v '"'"'^[[:space:]]*$'"'"'))
num_pids=${#pids_unique[@]}

if [[ ! ${num_pids} -eq 0 ]] 
then
    echo "Found ${num_pids} unique PID(s) already running on your TPUs, killing now..."
    for pid in "${pids_unique[@]}"; do
        echo -e "Killing process ${pid}..."
        kill -9 "${pid}"
        tail --pid=$pid -f /dev/null
        echo -e "Existing process ${pid} on your TPU was killed successfully!"
    done
    echo "All existing processes killed, so your TPU is ready to use!"
else
 echo -e "No existing processes found, so your TPU is ready to use!"
fi
sudo rm -f /tmp/libtpu_lockfile'''

command = f"echo '{s}' > matt_a1.txt"