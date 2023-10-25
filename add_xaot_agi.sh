#!/bin/bash
set -e 

local_container_name=my_container_2
export PROJECT=$(gcloud config get-value project)


# Local
#CLOUD_IMAGE_NAME=maxtext_base_image

# Repo
export CLOUD_IMAGE_NAME=mattdavidow_runner
export CLOUD_IMAGE_NAME=gcr.io/$PROJECT/${CLOUD_IMAGE_NAME}

# Step 1: Build a Docker Container
docker build -t ${local_container_name} -f ./agi.Dockerfile -t ${CLOUD_IMAGE_NAME} .

# Step 2: Run a Python Script in the Docker Container
#docker run --privileged ${local_container_name} python3 /app/MaxText/train_compile.py MaxText/configs/base.yml compile_topology='v4-8' compile_file_name='save-for-image.pickle'
docker run --privileged ${local_container_name} python3 -c "import jax; print(jax.devices())"
docker run --privileged ${local_container_name} python3 /app/MaxText/train_compile.py MaxText/configs/base.yml compile_topology='v4-128' compiled_trainstep_file='compiled_v4_128.pickle' compile_topology_num_slices=1 per_device_batch_size=1
#docker run --privileged -it ${local_container_name} /bin/bash
#docker run -it --privileged --entrypoint bash maxtext_base_image
# docker run -it --privileged --entrypoint bash gcr.io/tpu-prod-env-multipod/mattdavidow_runner
# docker 

# Step 3: Save the New Docker Container

NEW_IMAGE_NAME=${CLOUD_IMAGE_NAME} # want for full
#NEW_IMAGE_NAME=mattdavidow-out-agi


container_id=$(docker ps -lq)  # Get the ID of the last running container
docker commit "$container_id" ${NEW_IMAGE_NAME}:latest

# Step 4: Push the New Docker Image to GCR
docker push ${NEW_IMAGE_NAME}:latest

# # Clean up: Remove the temporary container
docker rm "$container_id"