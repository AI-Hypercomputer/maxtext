#!/bin/bash

docker run \
  -u 0 --network=host \
  --cap-add=IPC_LOCK \
  --userns=host \
  --volume /run/tcpx:/tmp \
  --volume /var/lib/nvidia:/usr/local/nvidia \
  --volume /var/lib/tcpx/lib64:/usr/local/tcpx/lib64 \
  --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
  --device /dev/nvidia0:/dev/nvidia0 \
  --device /dev/nvidia1:/dev/nvidia1 \
  --device /dev/nvidia2:/dev/nvidia2 \
  --device /dev/nvidia3:/dev/nvidia3 \
  --device /dev/nvidia4:/dev/nvidia4 \
  --device /dev/nvidia5:/dev/nvidia5 \
  --device /dev/nvidia6:/dev/nvidia6 \
  --device /dev/nvidia7:/dev/nvidia7 \
  --device /dev/nvidia-uvm:/dev/nvidia-uvm \
  --device /dev/nvidiactl:/dev/nvidiactl \
  --env LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/tcpx/lib64 \
  "$@"
