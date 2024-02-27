set -e
local_image_name="mattdavidow_local_image"
project=tpu-prod-env-multipod
cloud_image_name="mattdavidow_cloud_image"

bash docker_build_fake_image.sh LOCAL_IMAGE_NAME=$local_image_name

docker tag ${local_image_name} gcr.io/$project/${cloud_image_name}:latest

echo "Trying to push with name $USER"
docker push gcr.io/$project/${cloud_image_name}:latest

echo "All done, check out your artifacts at: gcr.io/$project/${cloud_image_name}"