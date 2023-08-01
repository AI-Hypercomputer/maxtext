PROJECT=tpu-prod-env-multipod
ZONE=us-east5-b
NUM_SLICES=2
TPU_TYPE=v5litepod-256
VERSION=v2-alpha-tpuv5-lite
BUCKET_NAME=tonyjohnchen-maxtext


RUN_NAME=startup_time_${NUM_SLICES}slice_${TPU_TYPE}_$(date +%Y-%m-%d-%H-%M-%S)
QR_ID=$RUN_NAME
python3 multihost_job.py --NUM_SLICES=$NUM_SLICES --RUN_NAME=$RUN_NAME --BUCKET_NAME=$BUCKET_NAME --PROJECT=${PROJECT} --ZONE=${ZONE} \
--TPU_TYPE=$TPU_TYPE --VERSION=$VERSION \
--COMMAND="bash setup.sh MODE=stable; echo \"Sleeping for 60s\" && sleep 60; \
        echo 'ready for test!' "
        
gcloud alpha compute tpus queued-resources list --project=${PROJECT} --zone=${ZONE} --filter=startup_time 
