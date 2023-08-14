PROJECT=tpu-prod-env-multipod
ZONE=us-east5-b
NUM_SLICES=2
TPU_TYPE=v5litepod-256
VERSION=v2-alpha-tpuv5-lite
BUCKET_NAME=tonyjohnchen-maxtext

# PROJECT=tpu-prod-env-vlp-2nic
# ZONE=us-east5-b
# NUM_SLICES=2
# TPU_TYPE=v5litepod-256
# VERSION=v2-alpha-tpuv5-lite
# BUCKET_NAME=tonyjohnchen-maxtext

iteration=1

# for (( i=0; i<$iteration; i++ ));
# do
#     RUN_NAME=startup_time_${NUM_SLICES}slice_${TPU_TYPE}_$(date +%Y-%m-%d-%H-%M-%S)
#     QR_ID=$RUN_NAME
#     python3 multihost_job.py --COMMAND_TYPE=gcloud --NUM_SLICES=$NUM_SLICES --RUN_NAME=$RUN_NAME --BUCKET_NAME=$BUCKET_NAME --PROJECT=${PROJECT} --ZONE=${ZONE}\
#     --TPU_TYPE=$TPU_TYPE --VERSION=$VERSION \
#     --COMMAND="bash setup.sh MODE=stable JAX_VERSION=0.4.13 LIBTPU_GCS_PATH=gs://libtpu_internal/tonyjohnchen/viperlite/2023-07-14-17:22:50-libtpu.so; echo \"Sleeping for 60s\" && sleep 60; \
#     TPU_LIBRARY_PATH=\$HOME/custom_libtpu/libtpu.so TPU_NAME=local JAX_USE_PJRT_C_API_ON_TPU=1 \
#     TPU_STDERR_LOG_LEVEL=0 TPU_MIN_LOG_LEVEL=0 TPU_VMODULE=tpu_configuration_ops_impl=3 TF_CPP_MIN_LOG_LEVEL=0 \
#     EMIT_MEGASCALE_METRICS=True \
#     python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME \
#     base_output_directory=gs://maxtext-experiments-tpem/ \
#     dataset_path=gs://max-datasets-rogue \
#     steps=100 per_device_batch_size=1"
# done

for (( i=0; i<$iteration; i++ ));
do
    RUN_NAME=startup_time_${NUM_SLICES}slice_${TPU_TYPE}_$(date +%Y-%m-%d-%H-%M-%S)_stable_fleet
    QR_ID=$RUN_NAME
    python3 multihost_job.py --COMMAND_TYPE=curl --NUM_SLICES=$NUM_SLICES --RUN_NAME=$RUN_NAME --BUCKET_NAME=$BUCKET_NAME --PROJECT=${PROJECT} --ZONE=${ZONE}\
    --TPU_TYPE=$TPU_TYPE --VERSION=$VERSION \
    --COMMAND="bash setup.sh MODE=stable JAX_VERSION=0.4.13 LIBTPU_GCS_PATH=gs://libtpu_internal/tonyjohnchen/viperlite/2023-07-14-17:22:50-libtpu.so; echo \"Sleeping for 60s\" && sleep 60; \
    TPU_LIBRARY_PATH=\$HOME/custom_libtpu/libtpu.so TPU_NAME=local JAX_USE_PJRT_C_API_ON_TPU=1 \
    TPU_STDERR_LOG_LEVEL=0 TPU_MIN_LOG_LEVEL=0 TPU_VMODULE=tpu_configuration_ops_impl=3 TF_CPP_MIN_LOG_LEVEL=0 \
    EMIT_MEGASCALE_METRICS=True \
    python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME \
    base_output_directory=gs://maxtext-experiments-tpem/ \
    dataset_path=gs://max-datasets-rogue \
    steps=100 per_device_batch_size=1"
done

gcloud alpha compute tpus queued-resources list --project=${PROJECT} --zone=${ZONE} --filter=startup_time 
