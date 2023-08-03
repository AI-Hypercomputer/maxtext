
#!/bin/bash
PROJECT=tpu-prod-env-multipod
ZONE=us-east5-b

QUEUED_RESOURCE_NAME=startup_time_2slice_v5litepod-256_2023-08-01-23-03-19
POLLING_FREQUENCY=100
TPU_NODE=$QUEUED_RESOURCE_NAME-0 # The ID of the 0-th node in the queued resource.
while true
do
  PID=$(gcloud compute tpus tpu-vm ssh root@${TPU_NODE} --worker=0 --command="sudo lsof -w /dev/vfio/0" --project=${PROJECT} --zone=${ZONE})
  if [[ -z "$PID" ]]; then
    echo "TPU is not in use, resetting the queued resource."
    yes | (gcloud alpha compute tpus queued-resources reset "$QUEUED_RESOURCE_NAME" --project=${PROJECT} --zone=${ZONE})
    # Additional sleep after resetting the queued resource, to give the TPU time to start.
    sleep 300
  else
    echo "TPU is in use."
  fi
  echo "Sleeping $POLLING_FREQUENCY seconds"
  sleep "$POLLING_FREQUENCY"
done