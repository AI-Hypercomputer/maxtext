ARG BASEIMAGE=gcr.io/supercomputer-testing/yangyuwei/maxtext-tcpx-0327:latest
FROM $BASEIMAGE

#FROM maxtext_base_image

# Set the working directory in the container
WORKDIR /app

# Copy all files from local workspace into docker container
COPY . .

WORKDIR /app