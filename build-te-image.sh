#!/bin/bash

export LOCAL_IMAGE_NAME=yooh/maxtext-pinned-te
bash docker_build_dependency_image.sh DEVICE=gpu MODE=pinned
bash docker_upload_runner.sh CLOUD_IMAGE_NAME=${LOCAL_IMAGE_NAME}
export LOCAL_IMAGE_NAME=yooh/maxtext-stable-te
bash docker_build_dependency_image.sh DEVICE=gpu MODE=stable
bash docker_upload_runner.sh CLOUD_IMAGE_NAME=${LOCAL_IMAGE_NAME}
