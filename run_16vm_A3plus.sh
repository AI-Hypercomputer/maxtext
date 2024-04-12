#!/bin/bash 

# Build and upload image
# bash docker_build_dependency_image.sh DEVICE=gpu
# bash docker_upload_runner.sh CLOUD_IMAGE_NAME=yooh/maxtext-tcpx

export PROJECT_ID=supercomputer-testing
export ZONE=us-east4-b

gcloud config set project $PROJECT_ID
gcloud config set compute/zone $ZONE

export CLUSTER_NAME=a3plus-pd

export RUN_NAME="yooh-7b-16vm-a3plus-$(date +%m-%d-%H-%M)"
export WORKLOAD_NAME=${RUN_NAME}

export DEVICE_TYPE=h150-80gb-8
# export LOCAL_IMAGE_NAME=yangyuwei/maxtext-tcpx-0327:latest
export LOCAL_IMAGE_NAME=yooh/maxtext-tcpx

python3 xpk/xpk.py workload create --cluster ${CLUSTER_NAME} --workload ${WORKLOAD_NAME} --docker-image=gcr.io/supercomputer-testing/${LOCAL_IMAGE_NAME} --device-type ${DEVICE_TYPE} --num-slices 16 --priority=very-high --command "bash MaxText/configs/a3/llama_2_7b/16vm.sh"
