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

# bash save_xaot.sh

# In separate terminal
# docker ps
# get the ID 
# docker commit -m "Added xaot" -a "mattdavidow" container-id mattdavidow-with-xaot-2:tag
# docker tag ${GCR_IMAGE_NAME} gcr.io/$PROJECT/${CLOUD_IMAGE_NAME}:latest
# docker push gcr.io/$PROJECT/${CLOUD_IMAGE_NAME}:latest

# docker run -v $(pwd):/app --rm -it --privileged --entrypoint bash ${GCR_IMAGE_NAME} /bin/bash -c "bash save_xaot_image.sh && docker commit \$(docker ps -lq) mattdavidow-with-xaot:tag"






