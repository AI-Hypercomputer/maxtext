set -e

# Inputs:
export CLOUD_IMAGE_NAME=${USER}_runner
export PROJECT=$(gcloud config get-value project)

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
    echo "$KEY"="$VALUE"
done

#export LOCAL_IMAGE_NAME_RUNNER=${LOCAL_IMAGE_NAME}__runner

GCR_IMAGE_NAME=gcr.io/${PROJECT}/${CLOUD_IMAGE_NAME}
#docker pull ${GCR_IMAGE_NAME}
#docker run -v $(pwd):/app --rm -it --privileged --entrypoint bash ${CLOUD_IMAGE_NAME}
docker run -v $(pwd):/app --rm -it --privileged --entrypoint bash ${GCR_IMAGE_NAME}

#docker run --rm -it --privileged --entrypoint bash ${GCR_IMAGE_NAME} # remove the [-v $(pwd):/app] so new files are in the containers directory instead of the hosts

# export im_name=mattdavidow_runner
# docker run --rm -it --privileged --entrypoint bash ${GCR_IMAGE_NAME}
# xaot_save_name=xaot_2xv5e-16.pickle
# python3 MaxText/save_xaot.py MaxText/configs/base.yml save_xaot=True xaot_save_name=${xaot_save_name} num_xaot_devices=32 topology=v5e-16 topology_num_slices=2 per_device_batch_size=4

# In separate terminal
# docker ps
# get the ID 
# docker commit -m "Added xaot" -a "mattdavidow" container-id mattdavidow-with-xaot-2:tag
# docker tag ${GCR_IMAGE_NAME} gcr.io/$PROJECT/${CLOUD_IMAGE_NAME}:latest
# docker push gcr.io/$PROJECT/${CLOUD_IMAGE_NAME}:latest

# can be done via 
# export im_name=gcr.io/tpu-prod-env-multipod/mattdavidow_runner
# xaot_save_name=xaot_2xv5e-16.pickle
# bash add_xaot_main_shell.sh IMAGE_NAME=${im_name} xaot_save_name=${xaot_save_name}
# docker run -v $(pwd):/app --rm -it --privileged --entrypoint bash ${GCR_IMAGE_NAME} /bin/bash -c "bash save_xaot_image.sh && docker commit \$(docker ps -lq) mattdavidow-with-xaot:tag"






