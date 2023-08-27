#!/bin/bash

function deljobs() {
  jobs=$(gcloud alpha compute tpus queued-resources list --zone=$2 --project=$1 | grep "FAILED\|SUSPENDED" | sed "s/ .*//g")
  for job in ${jobs}; do
    cmd="yes | gcloud alpha compute tpus queued-resources delete --zone=$2 --project=$1 $job"
    echo "$cmd"
    # eval "$cmd" &
    # sleep 1
  done
}

deljobs tpu-prod-env-multipod us-east5-b
deljobs tpu-prod-env-multipod us-central2-b
deljobs tpu-prod-env-vlp-2nic us-east5-b


wait

exit

shopt -s expand_aliases

alias tpugcloud_test='CLOUDSDK_API_ENDPOINT_OVERRIDES_TPU=https://test-tpu.sandbox.googleapis.com/ CLOUDSDK_API_ENDPOINT_OVERRIDES_COMPUTE=https://www.googleapis.com/compute/staging_v1/  CLOUDSDK_API_CLIENT_OVERRIDES_COMPUTE=staging_v1 /google/data/ro/teams/cloud-sdk/gcloud'
alias tpugcloud_staging='CLOUDSDK_API_ENDPOINT_OVERRIDES_TPU=https://staging-tpu.sandbox.googleapis.com/ CLOUDSDK_API_ENDPOINT_OVERRIDES_COMPUTE=https://www.googleapis.com/compute/staging_v1/ CLOUDSDK_API_CLIENT_OVERRIDES_COMPUTE=staging_v1 /google/data/ro/teams/cloud-sdk/gcloud'


# Multipod Tests in TPU-TEST (https://testgrid.corp.google.com/TPU-Test#Multipod%20Tests)
function clear_test_env(){
  PROJECT_ID=tpu-test-env-one-vm
  ZONE=us-central1-iw1
  delete_test_qr
}

# Multipod Tests in TPU-STAGING (https://testgrid.corp.google.com/TPU-Staging#Multipod%20Tests)
function clear_staging_env(){
  PROJECT_ID=tpu-staging-env-one-vm
  ZONE=us-central1-iw1
  delete_staging_qr
}

# Multipod Tests in TPU-PROD (https://testgrid.corp.google.com/TPU-Prod#Multipod%20Tests)
function clear_prod_env(){
  PROJECT_ID=cloud-tpu-multipod-dev
  ZONE=us-central2-b
  delete_prod_qr
}

# Multipod Vlp Tests in TPU-PROD (https://testgrid.corp.google.com/TPU-Prod#Multipod%20Vlp%20Tests)
function clear_vlp_in_prod_env(){
  PROJECT_ID=tpu-prod-env-automated
  ZONE=us-east1-c
  delete_prod_qr
}

# Delete v4 QR in prod env
function clear_v4_prod_env(){
  PROJECT_ID=tpu-prod-env-multipod
  ZONE=us-central2-b
  delete_prod_qr
}

# Delete VLP QR in prod env
function clear_vlp_prod_env(){
  PROJECT_ID=tpu-prod-env-multipod
  ZONE=us-west4-a
  delete_prod_qr
}

# Delete VLP burn-in QR in prod env
function clear_vlp_burn_in_prod_env(){
  PROJECT_ID=tpu-prod-env-multipod
  ZONE=us-east5-b
  delete_prod_qr
}

function delete_test_qr(){
  echo "Deleting in test env at PROJECT: $PROJECT_ID ZONE: $ZONE"
  tpugcloud_test alpha compute tpus queued-resources list --zone=$ZONE --project=$PROJECT_ID | grep -E "FAILED|SUSPENDED" | awk '{print $1}' | tail -n +1 | while read -r qr_id
  do
    echo "Deleting $qr_id"
    yes | tpugcloud_test alpha compute tpus queued-resources delete "$qr_id" --async --force --zone=$ZONE --project=$PROJECT_ID
  done
}

function delete_staging_qr(){
  echo "Deleting in staging env at PROJECT: $PROJECT_ID ZONE: $ZONE"
  tpugcloud_staging alpha compute tpus queued-resources list --zone=$ZONE --project=$PROJECT_ID | grep -E "FAILED|SUSPENDED" | awk '{print $1}' | tail -n +1 | while read -r qr_id
  do
    echo "Deleting $qr_id"
    yes | tpugcloud_staging alpha compute tpus queued-resources delete "$qr_id" --async --force --zone=$ZONE --project=$PROJECT_ID
  done
}

function delete_prod_qr(){
  echo "Deleting in prod env at PROJECT: $PROJECT_ID ZONE: $ZONE"
  gcloud alpha compute tpus queued-resources list --zone=$ZONE --project=$PROJECT_ID | grep -E "FAILED|SUSPENDED" | awk '{print $1}' | tail -n +1 | while read -r qr_id
  do
    echo "Deleting $qr_id"
    yes | gcloud alpha compute tpus queued-resources delete "$qr_id" --async --force --zone=$ZONE --project=$PROJECT_ID
  done
}

# Delete All QR for Multipod Tests
clear_test_env
clear_staging_env
clear_prod_env
clear_vlp_in_prod_env

# Delete all v4 and v5 QRs in project `tpu-prod-env-multipod`
clear_v4_prod_env
clear_vlp_prod_env
clear_vlp_burn_in_prod_env