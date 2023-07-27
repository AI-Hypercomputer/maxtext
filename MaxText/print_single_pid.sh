device_name="accel0"
pid=$(sudo lsof -w /dev/${device_name} | awk 'END{print $2}')
echo "pid: $pid"