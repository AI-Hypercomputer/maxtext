#!/bin/bash

# Before running this script
# 1. Install MaxText https://github.com/google/maxtext, since this script relys on
#    MaxText multihost_job.py tool
# 2. Please change the following constants

# Example bash ./run_mlperf_gpt3.sh

# This is a short term script for GCE. GKE will use different approach to run mlperf
# workload.
# Feel free to update this script for your own purpose.

gcloud config set project tpu-prod-env-multipod
gcloud config set compute/zone us-east5-b

TPU_TYPE=v5litepod-16
VERSION=v2-alpha-tpuv5-lite
BUCKET_NAME=mlperf-exp/${USER}
TIMESTAMP=$(date +%y%m%d_%H%M%S)

NODE_COUNT=2
RUN_NAME=${USER}-mlperf-gpt3-${VERSION}-${NODE_COUNT}_v4_$(date +%Y-%m-%d-%H-%M-%S)
PAX_DATE=20230810

EXP=C4SpmdGpt3AdamDataParallel2x4x4 

SCRIPTS_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )


gsutil cp gs://pax-on-cloud-tpu-project/wheels/"${PAX_DATE}"/praxis-*.whl gs://"${BUCKET_NAME}"/mlperf_test_script/
gsutil cp gs://pax-on-cloud-tpu-project/wheels/"${PAX_DATE}"/paxml-*.whl gs://"${BUCKET_NAME}"/mlperf_test_script/
gsutil cp "${SCRIPTS_DIR}"/test_script.sh gs://"${BUCKET_NAME}"/mlperf_test_script
gsutil cp "${SCRIPTS_DIR}"/parser_metrics.sh gs://"${BUCKET_NAME}"/mlperf_test_script

gsutil cp -r "${SCRIPTS_DIR}"/src gs://"${BUCKET_NAME}"/mlperf_test_script/