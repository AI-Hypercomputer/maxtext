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

# python3 multihost_job.py --NUM_SLICES=$NODE_COUNT --RUN_NAME="$RUN_NAME"_"$TIMESTAMP" --BUCKET_NAME="$BUCKET_NAME" --TPU_TYPE=$TPU_TYPE --VERSION=$VERSION --CQR_EXTRA_ARGS="--network=mtu9k" \
# --COMMAND="bash setup.sh MODE=stable && sudo gsutil cp -r gs://${BUCKET_NAME}/mlperf_test_script/ /tmp/ && pip install protobuf==3.15 && pip install orbax-checkpoint==0.2.7 && pip install fiddle==0.2.8 \
# && git clone https://github.com/mlperf/logging.git mlperf-logging && pip install -e mlperf-logging \
# && bash /tmp/mlperf_test_script/test_script.sh ${PAX_DATE} ${EXP} ${BUCKET_NAME}"

# TPU_PREFIX=tonyjohnchen-mlperf-gpt3-v2-alpha-tpuv5-lite-2_v4_2023-08-12-03-28-43_230812_032843

# python3 multihost_runner.py --TPU_PREFIX=$TPU_PREFIX \
# --COMMAND="bash setup.sh MODE=stable && sudo gsutil cp -r gs://${BUCKET_NAME}/mlperf_test_script/ /tmp/ && pip install protobuf==3.15 && pip install orbax-checkpoint==0.2.7 && pip install fiddle==0.2.8 \
# && git clone https://github.com/mlperf/logging.git mlperf-logging && pip install -e mlperf-logging \
# && bash /tmp/mlperf_test_script/test_script.sh ${PAX_DATE} ${EXP} ${BUCKET_NAME}"

# python3 multihost_runner.py --TPU_PREFIX=$TPU_PREFIX \
# --COMMAND="printenv;"