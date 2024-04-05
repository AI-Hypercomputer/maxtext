set -ex

BUCKET_NAME=aireenmei-multipod
LEARNING_RATE=3e-4

#bash setup_gcsfuse.sh DATASET_GCS_BUCKET=mazumdera-test-bucket MOUNT_PATH=/tmp/gcsfuse

gsutil cp -r gs://mazumdera-test-bucket/lg/vocab_102400 assets
#gsutil cp -r gs://maxtext-dataset/hf/llama2-tokenizer assets

python3 MaxText/train.py MaxText/configs/base.yml \
run_name=llama2-7b-lg-$(date +%Y-%m-%d-%H-%M) \
base_output_directory=gs://${BUCKET_NAME}/hf \
dataset_name='parquet' \
dataset_path=gs://maxtext-dataset/hf/c4/c4-train-*.parquet \
steps=20 \
model_name=llama2-7b \
per_device_batch_size=2 \
learning_rate=${LEARNING_RATE} \
remat_policy=minimal \
enable_checkpointing=false \
tokenizer_loader=AutoTokenizer \
tokenizer_path="assets/llama2-tokenizer"


#tokenizer_path="assets/vocab_102400" \

# dataset_path=/tmp/gcsfuse/lg/jsonl-data/*.jsonl \
#dataset_path=gs://mazumdera-test-bucket/lg/jsonl-data/*.jsonl \