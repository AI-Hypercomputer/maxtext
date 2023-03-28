/usr/bin/docker-credential-gcr configure-docker
sudo bash /var/scripts/docker-login.sh

LIBTPU_DOCKER_IMAGE_PATH=gcr.io/cloud-tpu-v2-images-dev/libtpu_v5lite:libtpu_v5lite_202303092321_RC00
echo "Pulling libtpu_next from ${LIBTPU_DOCKER_IMAGE_PATH} and copying it to /lib/libtpu.so."
sudo docker pull "${LIBTPU_DOCKER_IMAGE_PATH}"
sudo docker create --name libtpu_next_2 "${LIBTPU_DOCKER_IMAGE_PATH}" "/bin/bash"
sudo docker cp libtpu_next_2:libtpu.so /lib/libtpu.so
sudo docker cp libtpu_next_2:libtpu.so /lib/libtpu.so.secret
