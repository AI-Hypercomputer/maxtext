ARG JAX_AI_IMAGE_BASEIMAGE

# JAX AI Base Image
FROM $JAX_AI_IMAGE_BASEIMAGE
ARG JAX_AI_IMAGE_BASEIMAGE

ARG COMMIT_HASH
ENV COMMIT_HASH=$COMMIT_HASH

ENV MAXTEXT_ASSETS_ROOT=/deps/src/MaxText/assets
ENV MAXTEXT_TEST_ASSETS_ROOT=/deps/src/MaxText/test_assets
ENV MAXTEXT_PKG_DIR=/deps/src/MaxText
ENV MAXTEXT_REPO_ROOT=/deps

# Set the working directory in the container
WORKDIR /deps

# Copy setup files and dependency files separately for better caching
COPY tools/setup tools/setup/
COPY dependencies/requirements/ dependencies/requirements/
COPY src/install_maxtext_extra_deps/extra_deps_from_github.txt src/install_maxtext_extra_deps/

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

# Install requirements file that was generated with pipreqs for JSS 0.6.1 using:
# pipreqs --savepath requirements_with_jax_stable_stack_0_6_1_pipreqs.txt
# Otherwise use general requirements_with_jax_ai_image.txt
RUN if [ "$DEVICE" = "tpu" ] && [ "$JAX_STABLE_STACK_BASEIMAGE" = "us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.6.1-rev1" ]; then \
        python3 -m pip install -r /deps/dependencies/requirements/requirements_with_jax_stable_stack_0_6_1_pipreqs.txt; \
  else \
        python3 -m pip install -r /deps/dependencies/requirements/requirements_with_jax_ai_image.txt; \
  fi

# Install google-tunix for TPU devices, skip for GPU
RUN if [ "$DEVICE" = "tpu" ]; then \
        python3 -m pip install 'google-tunix>=0.1.2'; \
  fi

# Temporarily downgrade to JAX=0.7.2 for GPU images
RUN if [ "$DEVICE" = "gpu" ]; then \
      python3 -m pip install -U "jax[cuda12]==0.8.1"; \
      python3 -m pip install -U "transformer-engine-cu12" "transformer-engine-jax" "transformer-engine"; \
    fi

# Now copy the remaining code (source files that may change frequently)
COPY . .

RUN ls .

ARG TEST_TYPE
# Copy over test assets if building image for end-to-end tests or unit tests
RUN if [ "$TEST_TYPE" = "xlml" ] || [ "$TEST_TYPE" = "unit_test" ]; then \
      if ! gcloud storage cp -r gs://maxtext-test-assets/* "${MAXTEXT_TEST_ASSETS_ROOT}"; then \
        echo "WARNING: Failed to download test assets from GCS. These files are only used for end-to-end tests; you may not have access to the bucket."; \
      fi; \
    fi

# Run the script available in JAX AI base image to generate the manifest file
RUN bash /jax-ai-image/generate_manifest.sh PREFIX=maxtext COMMIT_HASH=$COMMIT_HASH

# Install (editable) MaxText
RUN test -f '/tmp/venv_created' && "$(tail -n1 /tmp/venv_created)"/bin/activate ; pip install --no-dependencies -e .
