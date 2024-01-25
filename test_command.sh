PROJECT_ID=tpu-prod-env-multipod
ZONE=us-east5-b
# gcloud config:
gcloud config set project $PROJECT_ID
gcloud config set compute/zone $ZONE
# xpk arguments

bash docker_upload_runner.sh CLOUD_IMAGE_NAME=${USER}_powertest_workload

xpk cluster list

xpk workload list \
--cluster v5p-12288-yucmhac-c-2

export OUTPUT_PATH="gs://maxtext-experiments-multipod"
export DATASET_PATH="gs://maxtext-dataset/"
DATETIME=$(date +%Y-%m-%d-%H-%M-%S)
RUN_NAME=power-test-${DATETIME}

xpk workload create \
--cluster v5p-12288-yucmhac-c-2 \
--tpu-type=v5p-12288 \
--priority=high \
--workload tony-powertest-workload \
--base-docker-image=gcr.io/tpu-prod-env-multipod/tonyjohnchen_powertest_workload \
--command "export LIBTPU_INIT_ARGS=\"--xla_tpu_megacore_fusion_allow_ags=false --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true\";\
python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME\
    steps=20 per_device_batch_size=2 enable_checkpointing=false\
    enable_profiler=false remat_policy=full global_parameter_scale=1024\
    max_target_length=2048 base_output_directory=$OUTPUT_PATH\
    dataset_path=$DATASET_PATH use_iota_embed=true reuse_example_batch=1\
    dataset_type=synthetic gcs_metrics=true attention='flash' int8_training=false"


xpk workload delete \
--workload tony-powertest-workload --cluster v5p-12288-yucmhac-c-2
