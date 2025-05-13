# syntax=docker.io/docker/dockerfile:1.7-labs

ARG BASEIMAGE=maxtext_base_image
FROM $BASEIMAGE

#FROM maxtext_base_image

# Set the working directory in the container
WORKDIR /deps

# Copy all files from local workspace into docker container. This includes
# MaxText/test_assets which contain some of the reference "golden" data.
COPY . .

