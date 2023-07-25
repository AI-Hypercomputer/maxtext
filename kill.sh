#!/bin/bash
_TPU_VERSION_NAME=${1}

device_name="accel0"

if [[ "${_TPU_VERSION_NAME}" =~ ^v5.* ]]; then
  device_name="vfio/0"
fi

pid=$(sudo lsof -w /dev/${device_name} | awk 'END{print $2}')
if [[ ! -z "${pid}" ]]
then
  kill -9 "${pid}"
fi