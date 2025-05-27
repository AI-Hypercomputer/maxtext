#!/bin/bash
# Constants
# PROJECT=cloud-tpu-multipod-dev
# CLUSTER=v4-128-bodaborg-us-central2-b
# ZONE=us-central2-b
# TPU_TYPE=v4-128

# PROJECT=cloud-tpu-multipod-dev
# CLUSTER=v5p-32-bodaborg-us-east5-a
# ZONE=us-east5-a
# TPU_TYPE=v5p-32

# CLUSTER=v4-8-maxtext
# PROJECT=tpu-prod-env-multipod
# ZONE=us-central2-b
# TPU_TYPE=v4-8

CLUSTER=bodaborg-v6e-256-tt-c
ZONE=us-west1-c
REGION=us-west1
PROJECT=tpu-prod-env-multipod
TPU_TYPE=v6e-256

NUM_SLICES=1
WORKLOAD_NAME=ksadi-col-py-$RANDOM
BASE_OUTPUT_DIR=gs://trillium-scale-tests-q1-25-west/pw_mcjax_benchmarking/ksadi/
DATASET_PATH=gs://trillium-scale-datasets-q1-25-west
DATASET_TYPE=tfds

# Images
PROXY_IMAGE=us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/ksadi/unsanitized_proxy_server:latest
SERVER_IMAGE=us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/ksadi/unsanitized_server:latest
COLOCATED_PYTHON_SIDECAR_IMAGE=us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/remote_python_sidecar_server:latest
xpk workload delete \
  --workload $WORKLOAD_NAME \
  --project="$PROJECT" \
  --cluster=$CLUSTER \
  --zone=$ZONE

xpk workload create-pathways \
  --workload $WORKLOAD_NAME \
  --num-slices=$NUM_SLICES \
  --tpu-type=$TPU_TYPE \
  --project="$PROJECT" \
  --cluster=$CLUSTER \
  --base-docker-image maxtext_base_image \
  --zone=$ZONE \
  --proxy-server-image=$PROXY_IMAGE \
  --server-image=$SERVER_IMAGE \
  --colocated-python-sidecar-image=$COLOCATED_PYTHON_SIDECAR_IMAGE \
  --command "python3 -m MaxText.train MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_DIR} dataset_path=${DATASET_PATH} dataset_type=${DATASET_TYPE} steps=50 per_device_batch_size=1 colocated_python_data_input=True enable_single_controller=True run_name=$WORKLOAD_NAME"

