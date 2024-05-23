From gcr.io/tpu-prod-env-multipod/jax-ss/tpu:2024-05-23

RUN mkdir -p /deps

# Set the working directory in the container
WORKDIR /deps

# Copy all files from local workspace into docker container
COPY . .
RUN ls .

# Install Python packages from requirements.txt
RUN pip install -r /deps/requirements.txt