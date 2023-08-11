# Use Python 3.10 as the base image
FROM python:3.10-slim

# Install system dependencies and Git
RUN apt-get update && apt-get install -y curl gnupg git

# Add the Google Cloud SDK package repository
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# Install the Google Cloud SDK
RUN apt-get update && apt-get install -y google-cloud-sdk

# Set the default Python version to 3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.10 1

# Set environment variables for Google Cloud SDK and Python 3.10
ENV PATH="/usr/local/google-cloud-sdk/bin:/usr/local/bin/python3.10:${PATH}"

ARG MODE
ENV ENV_MODE=$MODE

ARG JAX_VERSION
ENV ENV_JAX_VERSION=$JAX_VERSION

# Set the working directory in the container
WORKDIR /app

RUN git clone https://github.com/google/maxtext.git

RUN echo "Running command: bash setup.sh MODE=$ENV_MODE JAX_VERSION=$ENV_JAX_VERSION"
RUN cd maxtext && bash setup.sh MODE=${ENV_MODE} JAX_VERSION=${ENV_JAX_VERSION}

COPY . .
