ARG BASEIMAGE=maxtext_base_image
FROM $BASEIMAGE


RUN sed -i -e "s/main/main contrib/" /etc/apt/sources.list
RUN curl https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.1-1_all.deb -o cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-drivers


WORKDIR /app