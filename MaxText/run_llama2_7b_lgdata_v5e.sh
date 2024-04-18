set -ex

BUCKET_NAME=mazumdera-test-bucket-us-west4
LEARNING_RATE=1e-4
STEPS=20001
CHECKPOINT_PERIOD=5000

PER_DEVICE_BATCH_SIZE=2

bash setup_gcsfuse.sh DATASET_GCS_BUCKET=mazumdera-test-bucket MOUNT_PATH=/tmp/gcsfuse

gsutil cp -r gs://mazumdera-test-bucket/lg/vocab_102400 assets

export LIBTPU_INIT_ARGS="--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"


python3 MaxText/train.py MaxText/configs/base.yml \
run_name=mazumdera-llama27b-lg-$(date +%Y-%m-%d-%H-%M) base_output_directory=gs://${BUCKET_NAME}/lg-llama2/hf \
dataset_path=gs://mazumdera-test-bucket/lg/jsonl-data \
steps=${STEPS} \
dataset_name=json \
dataset_dir=None \
dataset_files=/tmp/gcsfuse/lg/jsonl-data/*.jsonl \
eval_dataset_name='' \
eval_split='' \
dataset_type=hf \
model_name=llama2-7b \
ici_fsdp_transpose_parallelism=16 \
per_device_batch_size=${PER_DEVICE_BATCH_SIZE} \
learning_rate=${LEARNING_RATE} \
remat_policy=minimal \
checkpoint_period=${CHECKPOINT_PERIOD} \
async_checkpointing=False