export PROJECT=tpu-prod-env-multipod
export ZONE=us-central2-b	
export NUM_SLICES=2
export TPU_TYPE=v4-8
export VERSION=tpu-ubuntu2204-base
export BUCKET_NAME=tonyjohnchen-maxtext
export RUN_NAME=tonyjohnchen_${NUM_SLICES}slice_${TPU_TYPE}_$(date +%Y-%m-%d-%H-%M-%S)
export QR_ID=$RUN_NAME

gcloud config set project ${PROJECT}
gcloud config set compute/zone ${ZONE}

python3 multihost_job.py --NUM_SLICES=$NUM_SLICES --RUN_NAME=$RUN_NAME --BUCKET_NAME=$BUCKET_NAME --CQR_EXTRA_ARGS="--best-effort" \
--TPU_TYPE=$TPU_TYPE --VERSION=$VERSION \
--COMMAND="bash setup.sh MODE=stable LIBTPU_GCS_PATH=gs://libtpu_internal/tonyjohnchen/viperlite/2023-07-14-17:22:50-libtpu.so; echo \"Sleeping for 60s\" && sleep 60;\
for ((i = 1; i <= 100; i++))
do
    echo \"Running test \$i\"; \
    TPU_LIBRARY_PATH=\$HOME/custom_libtpu/libtpu.so TPU_NAME=local JAX_USE_PJRT_C_API_ON_TPU=1 \
    TPU_STDERR_LOG_LEVEL=0 TPU_MIN_LOG_LEVEL=0 TPU_VMODULE=tpu_configuration_ops_impl=3 TF_CPP_MIN_LOG_LEVEL=0 \
    EMIT_MEGASCALE_METRICS=True \
    python3 MaxText/train.py MaxText/configs/base.yml run_name=\$i \
    base_output_directory=gs://max-experiments/ \
    dataset_path=gs://maxtext-dataset/ \
    steps=100 per_device_batch_size=1; \
    echo \"Sleeping for 100s\" && sleep 100; \
done"