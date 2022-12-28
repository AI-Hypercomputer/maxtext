#!/bin/bash
pip3 install -r requirements.txt

# install jax
cd
git clone https://github.com/google/jax.git
cd jax
pip3 install -r build/test-requirements.txt
pip3 install -e .

# install flax
cd
git clone --branch=main https://github.com/google/flax.git
pip3 install -e flax

# nightly libtpu
/usr/bin/docker-credential-gcr configure-docker
sudo bash /var/scripts/docker-login.sh

LIBTPU_DOCKER_IMAGE_PATH=gcr.io/cloud-tpu-v2-images-dev/libtpu_unsanitized:nightly
echo "Pulling libtpu_next from ${LIBTPU_DOCKER_IMAGE_PATH} and copying it to /lib/libtpu.so."
sudo docker pull "${LIBTPU_DOCKER_IMAGE_PATH}"
sudo docker create --name libtpu_next "${LIBTPU_DOCKER_IMAGE_PATH}" "/bin/bash"
sudo docker cp libtpu_next:_libtpu_next.so /lib/libtpu.so

sudo docker rm libtpu_next
echo "export TPU_LIBRARY_PATH=/lib/libtpu.so" >> ~/.profile
source ~/.profile

# install jaxlib
pip3 install --pre -U jaxlib -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html



