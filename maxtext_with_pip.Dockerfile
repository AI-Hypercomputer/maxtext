# Use Python 3.12-slim-bullseye as the base image
FROM python:3.12.3-slim-bullseye

WORKDIR /jax-stable-stack

# Arguments
ARG COMMIT_HASH
ARG DEVICE
ARG TEST_TYPE

# Enviroment variables
ENV COMMIT_HASH=$COMMIT_HASH
ENV DEVICE=$DEVICE
ENV PIP_NO_CACHE_DIR=1
ENV PIP_ROOT_USER_ACTION=ignore
ENV PYTHON_VERSION=3.12
ENV CLOUD_SDK_VERSION=latest
# Ensure apt package installations run without manual intervention
ENV DEBIAN_FRONTEND=noninteractive
# See all env variables
RUN env

# System level dependencies
RUN apt-get update && apt-get install -y apt-utils git curl gnupg procps iproute2 ethtool cmake pkg-config build-essential dnsutils && rm -rf /var/lib/apt/lists/*
# Add the Google Cloud SDK package repository
RUN curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee /etc/apt/sources.list.d/google-cloud-sdk.list
# Install the Google Cloud SDK
RUN apt-get update && apt-get install -y google-cloud-sdk && rm -rf /var/lib/apt/lists/*
# Install gcsfuse
RUN export GCSFUSE_REPO=gcsfuse-bullseye && \
    echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | tee /etc/apt/sources.list.d/gcsfuse.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt update -y && apt -y install gcsfuse && \
    rm -rf /var/lib/apt/lists/*
# See all installed system level packages
RUN apt list --installed

# Set the working directory in the container
WORKDIR /deps

# Now copy the remaining code (source files that may change frequently)
COPY . .
RUN ls .

# Copy over test assets if building image for end-to-end tests or unit tests
RUN if [ "$TEST_TYPE" = "xlml" ] || [ "$TEST_TYPE" = "unit_test" ]; then \
      if ! gcloud storage cp -r gs://maxtext-test-assets/* MaxText/test_assets; then \
        echo "WARNING: Failed to download test assets from GCS. These files are only used for end-to-end tests; you may not have access to the bucket."; \
      fi; \
    fi

# Upgrade pip to the latest version
RUN python -m pip install uv
RUN python -m uv pip install --upgrade pip
RUN rm -rf /root/.cache/pip

# Clean python env
RUN python -m uv pip list

# RUN uv venv --python 3.12 --managed-python --seed
# RUN source .venv/bin/activate
RUN python -m uv pip install seed-env
RUN seed-env --local-requirements=requirements_with_lower_bounds.txt --host-name=maxtext --seed-commit=779c6aefcac3ff0149b28e183770cc1da0879c84 --hardware="$DEVICE" --build-pypi-package
RUN python -m uv pip uninstall seed-env
# RUN deactivate

RUN python -m uv pip list
RUN python -m uv pip install generated_env/dist/*.whl --resolution=lowest
RUN if [ "$DEVICE" = "gpu" ]; then \
      python -m pip install git+https://github.com/NVIDIA/TransformerEngine.git@9d031f; \
    fi
RUN python -m uv pip list

# Run the script to generate the manifest file
RUN bash /deps/generate_manifest.sh PREFIX=maxtext COMMIT_HASH=$COMMIT_HASH
