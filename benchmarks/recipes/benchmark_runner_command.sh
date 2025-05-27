#!/bin/bash

CLUSTER=bodaborg-v6e-256-tt-c
ZONE=us-west1-c
REGION=us-west1
PROJECT=tpu-prod-env-multipod
TPU_TYPE=v6e-256

NUM_SLICES=1
WORKLOAD_NAME=ksadi-col-py-$RANDOM
BASE_OUTPUT_DIR=gs://trillium-scale-tests-q1-25-west/pw_mcjax_benchmarking/ksadi/
# DATASET_PATH=gs://trillium-scale-datasets-q1-25-west
# DATASET_TYPE=tfds

PROXY_IMAGE=us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/ksadi/unsanitized_proxy_server:latest
SERVER_IMAGE=us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/ksadi/unsanitized_server:latest
COLOCATED_PYTHON_SIDECAR_IMAGE=us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/remote_python_sidecar_server:latest
RUNNER=gcr.io/tpu-prod-env-one-vm/ksadi_runner:latest

python3 -m benchmarks.benchmark_runner xpk --project $PROJECT --zone $ZONE --cluster_name $CLUSTER --device_type v6e-256 --base_output_directory $BASE_OUTPUT_DIR --num_steps 5 --use_pathways True --pathways_server_image "${SERVER_IMAGE}" --pathways_proxy_server_image "${PROXY_IMAGE}" --pathways_runner_image "${RUNNER}" --colocated_python_sidecar_image "${COLOCATED_PYTHON_SIDECAR_IMAGE}"
