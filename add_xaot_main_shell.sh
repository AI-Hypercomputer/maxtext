IMAGE_NAME=gcr.io/tpu-prod-env-vlp-2nic/mattdavidow_runner 
NEW_IMAGE_NAME=${IMAGE_NAME}

for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
    echo "$KEY"="$VALUE"
done

container_id=$(docker ps --filter "ancestor=${IMAGE_NAME}" --format "{{.ID}}")
docker commit -m "Added xaot" -a "mattdavidow" ${container_id} mattdavidow-with-xaot-2:tag
docker tag ${IMAGE_NAME} ${NEW_IMAGE_NAME}:latest
docker push ${NEW_IMAGE_NAME}:latest