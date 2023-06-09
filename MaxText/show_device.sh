
device_name="accel0"
while [ 1 -le 2 ]
do
    pid=$(sudo lsof -w /dev/${device_name} | awk 'END{print $2}')
    echo "pid: ${pid}"
done