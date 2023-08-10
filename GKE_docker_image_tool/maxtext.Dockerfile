FROM python:3.10

ARG MODE
ENV ENV_MODE=$MODE

ARG JAX_VERSION
ENV ENV_JAX_VERSION=$JAX_VERSION

RUN apt-get update && apt-get install -y git

# Set the working directory in the container
WORKDIR /app

RUN git clone https://github.com/google/maxtext.git

RUN echo "Running command: bash setup.sh MODE=$ENV_MODE JAX_VERSION=$ENV_JAX_VERSION"
RUN cd maxtext && bash setup.sh MODE=${ENV_MODE} JAX_VERSION=${ENV_JAX_VERSION}

COPY . .
