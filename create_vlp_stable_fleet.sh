# PROJECT=tpu-prod-env-multipod
PROJECT=tpu-prod-env-vlp-2nic
ZONE=us-east5-b
NUM_SLICES=2
TPU_TYPE=v5litepod-256
VERSION=v2-alpha-tpuv5-lite
BUCKET_NAME=tonyjohnchen-maxtext

for (( i=0; i<$iteration; i++ ));
do
    RUN_NAME=startup_time_${NUM_SLICES}slice_${TPU_TYPE}_$(date +%Y-%m-%d-%H-%M-%S)_stable_fleet
    QR_ID=$RUN_NAME
    python3 multihost_job.py --COMMAND_TYPE=curl --NUM_SLICES=$NUM_SLICES --RUN_NAME=$RUN_NAME --BUCKET_NAME=$BUCKET_NAME --PROJECT=${PROJECT} --ZONE=${ZONE}\
    --TPU_TYPE=$TPU_TYPE --VERSION=$VERSION \
    --COMMAND="echo hello"
done

gcloud alpha compute tpus queued-resources list --project=${PROJECT} --zone=${ZONE} --filter=startup_time 
