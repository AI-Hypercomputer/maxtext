ARG JAX_STABLE_STACK_BASEIMAGE

# JAX Stable Stack Base Image
From $JAX_STABLE_STACK_BASEIMAGE

ARG COMMIT_HASH

ENV COMMIT_HASH=$COMMIT_HASH

RUN mkdir -p /deps

# Set the working directory in the container
WORKDIR /deps

# Copy all files from local workspace into docker container
COPY . .
RUN ls .

ARG MAXTEXT_REQUIREMENTS_FILE

# Install Maxtext requirements
RUN if [ ! -z "${MAXTEXT_REQUIREMENTS_FILE}" ]; then \
        echo "Using Maxtext requirements: ${MAXTEXT_REQUIREMENTS_FILE}" && \
        pip install -r /deps/${MAXTEXT_REQUIREMENTS_FILE}; \
    fi

# Run the script available in JAX Stable Stack base image to generate the manifest file
RUN bash /generate_manifest.sh PREFIX=maxtext COMMIT_HASH=$COMMIT_HASH