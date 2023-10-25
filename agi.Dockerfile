#ARG BASEIMAGE=gcr.io/tpu-prod-env-multipod/mattdavidow_runner
ARG BASEIMAGE=maxtext_base_image
FROM $BASEIMAGE

#FROM maxtext_base_image

WORKDIR /app


#COPY . .


# Set the working directory in the container
WORKDIR /app