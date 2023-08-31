# PROJECT=tpu-prod-env-multipod
# PROJECT=tpu-prod-env-vlp-2nic
PROJECT=tpu-burn-in-prod-env-multipod
ZONE=us-east5-b
NUM_SLICES=1
TPU_TYPE=v5litepod-256
VERSION=v2-alpha-tpuv5-lite
BUCKET_NAME=tonyjohnchen-maxtext

iteration=1

for (( i=0; i<$iteration; i++ ));
do
    RUN_NAME=startup_time_${NUM_SLICES}slice_${TPU_TYPE}_$(date +%Y-%m-%d-%H-%M-%S)
    QR_ID=$RUN_NAME
    python3 multihost_job.py --COMMAND_TYPE=gcloud --NUM_SLICES=$NUM_SLICES --RUN_NAME=$RUN_NAME --BUCKET_NAME=$BUCKET_NAME --PROJECT=${PROJECT} --ZONE=${ZONE}\
    --TPU_TYPE=$TPU_TYPE --VERSION=$VERSION \
    --COMMAND="bash setup.sh MODE=stable JAX_VERSION=0.4.13 LIBTPU_GCS_PATH=gs://libtpu_internal/tonyjohnchen/viperlite/2023-07-14-17:22:50-libtpu.so; echo \"Sleeping for 60s\" && sleep 60; \
    TPU_LIBRARY_PATH=\$HOME/custom_libtpu/libtpu.so TPU_NAME=local JAX_USE_PJRT_C_API_ON_TPU=1 \
    TPU_STDERR_LOG_LEVEL=0 TPU_MIN_LOG_LEVEL=0 TPU_VMODULE=tpu_configuration_ops_impl=3 TF_CPP_MIN_LOG_LEVEL=0 \
    EMIT_MEGASCALE_METRICS=False \
    LIBTPU_INIT_ARGS='xla_tpu_sdc_check_halt_on_detection=false --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true \
    --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true --xla_tpu_enable_sdc_checker=true --xla_tpu_sdc_check_repeat_count=3' \
    python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME steps=110000 per_device_batch_size=6 enable_checkpointing=false enable_profiler=false base_output_directory=gs://maxtext-experiments-tpem/ dataset_path=gs://max-datasets-rogue \
    remat_policy=full base_emb_dim=6144 base_mlp_dim=24576 base_num_heads=24 base_num_decoder_layers=36 head_dim=256 max_target_length=2048 enable_dropout=False enable_data_shuffling=False
    "
done

# for (( i=0; i<$iteration; i++ ));
# do
#     RUN_NAME=startup_time_${NUM_SLICES}slice_${TPU_TYPE}_$(date +%Y-%m-%d-%H-%M-%S)_stable_fleet
#     QR_ID=$RUN_NAME
#     python3 multihost_job.py --COMMAND_TYPE=curl --NUM_SLICES=$NUM_SLICES --RUN_NAME=$RUN_NAME --BUCKET_NAME=$BUCKET_NAME --PROJECT=${PROJECT} --ZONE=${ZONE}\
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

gcloud alpha compute tpus queued-resources list --project=${PROJECT} --zone=${ZONE} --filter=startup_time 
