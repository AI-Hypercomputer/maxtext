ARG JAX_AI_IMAGE_BASEIMAGE

# JAX AI Base Image
FROM $JAX_AI_IMAGE_BASEIMAGE
ARG JAX_AI_IMAGE_BASEIMAGE

ARG COMMIT_HASH

ENV COMMIT_HASH=$COMMIT_HASH

RUN mkdir -p /deps

# Set the working directory in the container
WORKDIR /deps

# Copy all files from local workspace into docker container
COPY . .
RUN ls .


# For JAX AI tpu training images 0.4.37 AND 0.4.35
# Orbax checkpoint installs the latest version of JAX,
# but the libtpu version in the base image is older.
# This version mismatch can cause compatibility issues
# and break MaxText.
# Upgrade libtpu version if using either of the old stable images

ARG DEVICE
ENV DEVICE=$DEVICE

RUN if [ "$DEVICE" = "tpu" ] && ([ "$JAX_AI_IMAGE_BASEIMAGE" = "us-docker.pkg.dev/cloud-tpu-images/jax-stable-stack/tpu:jax0.4.37-rev1" ] || [ "$JAX_AI_IMAGE_BASEIMAGE" = "us-docker.pkg.dev/cloud-tpu-images/jax-stable-stack/tpu:jax0.4.35-rev1" ]); then \
        python3 -m pip install --no-cache-dir --upgrade jax[tpu]; fi

# Install Maxtext requirements with Jax AI Image
RUN apt-get update && apt-get install --yes && apt-get install --yes dnsutils
# TODO(bvandermoon, parambole): Remove this when it's added to JAX AI Image
RUN pip install google-cloud-monitoring
RUN python3 -m pip install -r /deps/requirements_with_jax_ai_image.txt

# Run the script available in JAX AI base image to generate the manifest file
RUN bash /jax-stable-stack/generate_manifest.sh PREFIX=maxtext COMMIT_HASH=$COMMIT_HASH
