#!/bin/bash

source mt_config.sh

bash docker_build_dependency_image.sh DEVICE=gpu MODE=nightly; 
# bash docker_build_dependency_image.sh DEVICE=gpu MODE=pinned;
# bash docker_build_dependency_image.sh MODE=stable_stack BASEIMAGE=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/llama2-xprof_1004_nolayers_nightly_lance
# bash docker_upload_runner.sh LOCAL_IMAGE_NAME=$LOCAL_IMAGE_NAME CLOUD_IMAGE_NAME=$LOCAL_IMAGE_NAME

docker tag maxtext_base_image $LOCAL_IMAGE_NAME; docker push $LOCAL_IMAGE_NAME;
