bash docker_build_dependency_image.sh MODE=stable JAX_VERSION=0.4.13 LIBTPU_GCS_PATH=gs://libtpu_internal/tonyjohnchen/viperlite/2023-07-14-17:22:50-libtpu.so

PROJECT_ID=tpu-prod-env-vlp-2nic
gcloud config set project $PROJECT_ID
bash docker_upload_runner.sh CLOUD_IMAGE_NAME=tonyjohnchen_runner_custom