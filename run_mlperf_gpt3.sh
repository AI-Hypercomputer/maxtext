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
gcloud config set compute/zone us-central2-b

TPU_TYPE=v4-8
VERSION=tpu-ubuntu2204-base
BUCKET_NAME=mlperf-exp/${USER}
TIMESTAMP=$(date +%y%m%d_%H%M%S)

NODE_COUNT=2
RUN_NAME=${USER}-mlperf-gpt3-benchmark-script-test-${NODE_COUNT}_v4_$(date +%Y-%m-%d-%H-%M-%S)
PAX_DATE=20230808

EXP=C4SpmdGpt3AdamDataParallel2x4x4 

SCRIPTS_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

gsutil cp gs://pax-on-cloud-tpu-project/wheels/"${PAX_DATE}"/praxis-*.whl gs://"${BUCKET_NAME}"/mlperf_test_script/
gsutil cp gs://pax-on-cloud-tpu-project/wheels/"${PAX_DATE}"/paxml-*.whl gs://"${BUCKET_NAME}"/mlperf_test_script/
gsutil cp "${SCRIPTS_DIR}"/test_script.sh gs://"${BUCKET_NAME}"/mlperf_test_script
gsutil cp "${SCRIPTS_DIR}"/parser_metrics.sh gs://"${BUCKET_NAME}"/mlperf_test_script

gsutil cp -r "${SCRIPTS_DIR}"/src gs://"${BUCKET_NAME}"/mlperf_test_script/

python3 multihost_job.py --NUM_SLICES=$NODE_COUNT --RUN_NAME="$RUN_NAME"_"$TIMESTAMP" --BUCKET_NAME="$BUCKET_NAME" --TPU_TYPE=$TPU_TYPE --VERSION=$VERSION --CQR_EXTRA_ARGS="--network=mtu9k" \
--COMMAND="bash setup.sh MODE=nightly && gsutil cp -r gs://${BUCKET_NAME}/mlperf_test_script/ /tmp/ && pip install protobuf==3.15 && pip install orbax-checkpoint==0.2.7 && pip install fiddle==0.2.8 \
&& git clone https://github.com/mlperf/logging.git mlperf-logging && pip install -e mlperf-logging \
&& bash /tmp/mlperf_test_script/test_script.sh ${PAX_DATE} ${EXP} ${BUCKET_NAME}"