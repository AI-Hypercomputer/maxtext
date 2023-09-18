PROJECT_ID=tpu-prod-env-vlp-2nic
gcloud config set project $PROJECT_ID
ZONE=us-east5-b
gcloud config set compute/zone $ZONE

NUM_SLICES=2
TPU_TYPE=v5litepod-256

iteration=1

for (( i=0; i<$iteration; i++ ));
do
    WORKLOAD_NAME=${USER}-debug-$(date +%Y-%m-%d-%H-%M-%S)
    RUN_NAME=$WORKLOAD_NAME

    python3 ../experimental/users/vbarr/multipod/xpk/xpk.py workload create \
    --cluster bodaborgprivate5 \
    --docker-image gcr.io/${PROJECT_ID}/tonyjohnchen_runner_nightly \
    --workload ${WORKLOAD_NAME} \
    --priority low \
    --tpu-type=${TPU_TYPE} \
    --num-slices=${NUM_SLICES}  \
    --command "echo 'libtpu' && echo \$TPU_LIBRARY_PATH; EMIT_MEGASCALE_METRICS=true TPU_NAME=local JAX_USE_PJRT_C_API_ON_TPU=1 TPU_STDERR_LOG_LEVEL=0 TPU_MIN_LOG_LEVEL=0 TPU_VMODULE=tpu_configuration_ops_impl=3 TF_CPP_MIN_LOG_LEVEL=0 python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME base_output_directory=gs://maxtext-experiments-tpem/ dataset_path=gs://max-datasets-rogue steps=100 per_device_batch_size=1"
done