ARG JAX_AI_IMAGE_BASEIMAGE

# JAX AI Base Image
FROM $JAX_AI_IMAGE_BASEIMAGE
ARG JAX_AI_IMAGE_BASEIMAGE

ARG COMMIT_HASH
ENV COMMIT_HASH=$COMMIT_HASH

RUN mkdir -p /deps

# Set the working directory in the container
WORKDIR /deps

# Copy setup files and dependency files separately for better caching
COPY setup.sh ./
COPY requirements.txt requirements_with_jax_ai_image.txt ./


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

# Now copy the remaining code (source files that may change frequently)
COPY . .
RUN ls .

ARG TEST_TYPE
# Copy over test assets if building image for end-to-end tests or unit tests
RUN if [ "$TEST_TYPE" = "xlml" ] || [ "$TEST_TYPE" = "unit_test" ]; then \
      if ! gcloud storage cp -r gs://maxtext-test-assets/* MaxText/test_assets; then \
        echo "WARNING: Failed to download test assets from GCS. These files are only used for end-to-end tests; you may not have access to the bucket."; \
      fi; \
    fi

# Run the script available in JAX AI base image to generate the manifest file
RUN bash /jax-stable-stack/generate_manifest.sh PREFIX=maxtext COMMIT_HASH=$COMMIT_HASH
