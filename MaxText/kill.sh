#!/bin/bash
device_name="accel0"
echo -e "Searching for existing processes on device ${device_name}"
pid=$(sudo lsof -w /dev/${device_name} | awk 'END{print $2}')
echo "pid: ${pid}"
if [[ ! -z "${pid}" ]]
then
 echo -e "Existing process found with pid ${pid}"
 echo -e "Killing process ${pid}..."
 kill -9 "${pid}"
 tail --pid=$pid -f /dev/null
 echo -e "Orphaned process ${pid} on your TPU was killed successfully."
else
 echo -e "No existing processes found."
fi
sudo rm -f /tmp/libtpu_lockfile