#!/bin/bash 


# Build and upload image
# export LOCAL_IMAGE_NAME=yooh/maxtext-pinned
# bash docker_build_dependency_image.sh DEVICE=gpu MODE=pinned
# bash docker_upload_runner.sh CLOUD_IMAGE_NAME=${LOCAL_IMAGE_NAME}
export LOCAL_IMAGE_NAME=yooh/maxtext-stable
# bash docker_build_dependency_image.sh DEVICE=gpu MODE=stable
bash docker_upload_runner.sh CLOUD_IMAGE_NAME=${LOCAL_IMAGE_NAME}

export PROJECT_ID=supercomputer-testing
export CLUSTER_NAME=a3plus-benchmark
export ZONE=australia-southeast1
export DEVICE_TYPE=h100-mega-80gb-8
export XPK_IMAGE_NAME=gpu-image
gcloud config set project $PROJECT_ID
gcloud config set compute/zone $ZONE

export RUN_NAME="yooh-oss-$(date +%m-%d-%H-%M)"
export WORKLOAD_NAME=${RUN_NAME}


# python3 xpk/xpk.py workload create --cluster $CLUSTER_NAME \
#     --workload $WORKLOAD_NAME --docker-image gcr.io/supercomputer-testing/$LOCAL_IMAGE_NAME \
#     --device-type $DEVICE_TYPE --num-nodes 1 \
#     --priority=high --scheduler=gke.io/topology-aware-auto \
#     --command "sleep 3600"
gcloud config set project $PROJECT_ID
gcloud config set compute/zone $ZONE
python3 xpk/xpk.py workload create --cluster $CLUSTER_NAME \
    --workload $WORKLOAD_NAME --docker-image gcr.io/supercomputer-testing/$LOCAL_IMAGE_NAME \
    --device-type $DEVICE_TYPE --num-nodes 128 \
    --priority=high --scheduler=gke.io/topology-aware-auto \
    --command "python3 MaxText/train.py MaxText/configs/base.yml \
        base_output_directory=${BASE_OUTPUT_PATH} dataset_path=${DATASET_PATH} \
        model_name=mixtral-8x7b \
        async_checkpointing=false tokenizer_path=assets/tokenizer.mistral-v1 \ 
        dtype=bfloat16 weight_dtype=bfloat16"

