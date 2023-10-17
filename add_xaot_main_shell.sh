# Input:
IMAGE_NAME=gcr.io/tpu-prod-env-vlp-2nic/mattdavidow_runner 

for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
    echo "$KEY"="$VALUE"
done

echo "Using IMAGE_NAME of ${IMAGE_NAME}"

NEW_IMAGE_NAME=${IMAGE_NAME}

container_id=$(docker ps --filter "ancestor=${IMAGE_NAME}" --format "{{.ID}}")
docker cp setup.sh ${container_id}:/app/x_aot_train_v4-8_num_slices_1.pickle 
docker commit -m "Added xaot" -a "mattdavidow" ${container_id} ${IMAGE_NAME}:latest
docker tag ${IMAGE_NAME} ${NEW_IMAGE_NAME}:latest
docker push ${NEW_IMAGE_NAME}:latest
