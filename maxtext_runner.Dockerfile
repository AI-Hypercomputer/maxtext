# ARG BASEIMAGE=maxtext_base_image
# ARG BASEIMAGE=lancewang-maxtext-gpu
ARG BASEIMAGE==gcr.io/supercomputer-testing/us-west1-docker.pkg.dev/supercomputer-testing/lancewang/lance-1216-bumpup
FROM $BASEIMAGE

#FROM maxtext_base_image

# Set the working directory in the container
WORKDIR /deps

# Copy all files from local workspace into docker container
COPY . .

WORKDIR /deps
