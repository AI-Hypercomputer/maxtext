local_image_name="image"
project=tpu-prod-env-multipod
cloud_image_name="mattdavidow_cloud_image"

bash docker_build_dependency_image.sh $local_image_name

docker tag ${local_image_name} gcr.io/$project/${cloud_image_name}:latest
docker push gcr.io/$project/${cloud_image_name}:latest

echo "All done, check out your artifacts at: gcr.io/$project/${cloud_image_name}"