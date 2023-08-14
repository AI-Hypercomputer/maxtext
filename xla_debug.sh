export PROJECT=tpu-prod-env-multipod
export ZONE=us-east5-b
export NUM_SLICES=1
export TPU_TYPE=v5litepod-256
export VERSION=v2-alpha-tpuv5-lite
export BUCKET_NAME=tonyjohnchen-maxtext
export RUN_NAME=XLA_debug_$(date +%Y-%m-%d-%H-%M-%S)
export QR_ID=$RUN_NAME

python3 multihost_job.py --COMMAND_TYPE=gcloud --NUM_SLICES=$NUM_SLICES --RUN_NAME=$RUN_NAME --BUCKET_NAME=$BUCKET_NAME --PROJECT=${PROJECT} --ZONE=${ZONE} \
--TPU_TYPE=$TPU_TYPE --VERSION=$VERSION \
--COMMAND="bash setup.sh; \
export XLA_FLAGS='--xla_dump_hlo_as_proto --xla_dump_to=/tmp/xla_dump_file'; \
export LIBTPU_INIT_ARGS='--xla_tpu_enable_data_parallel_all_reduce_opt=true \
--xla_tpu_data_parallel_opt_different_sized_ops=true \
--xla_tpu_enable_async_collective_fusion=true \
--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true \
--xla_tpu_enable_async_collective_fusion_multiple_steps=true \
--xla_tpu_overlap_compute_collective_tc=true \
--xla_enable_async_all_gather=true \
--xla_tpu_enable_sdc_checker=true \
--xla_tpu_sdc_check_repeat_count=3'; \
TPU_NAME=local JAX_USE_PJRT_C_API_ON_TPU=1 \
TPU_STDERR_LOG_LEVEL=0 TPU_MIN_LOG_LEVEL=0 TPU_VMODULE=tpu_configuration_ops_impl=3 TF_CPP_MIN_LOG_LEVEL=0 \
python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME \
base_output_directory=gs://maxtext-experiments-tpem/ \
dataset_path=gs://max-datasets-rogue \
steps=100 per_device_batch_size=1;"