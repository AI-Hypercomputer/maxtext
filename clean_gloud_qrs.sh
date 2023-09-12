#!/bin/bash

function deljobs() {
  echo "JOBS  $1 / $2"
  jobs=$(gcloud alpha compute tpus queued-resources list --zone=$2 --project=$1)
  echo "$jobs"
  grepped=$(echo "$jobs" | grep "FAILED\|SUSPENDED" | sed "s/ .*//g")
  echo
  echo "run these:"
  for job in ${grepped}; do
    cmd="yes | gcloud alpha compute tpus queued-resources delete --zone=$2 --project=$1 $job &"
    echo "$cmd"
    # eval "$cmd"
    # sleep 1
  done
  echo
}

deljobs tpu-prod-env-multipod us-east5-b
deljobs tpu-prod-env-multipod us-central2-b
deljobs tpu-prod-env-vlp-2nic us-east5-b


wait

exit
