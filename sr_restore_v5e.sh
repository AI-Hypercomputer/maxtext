export CLUSTER=roshanin-shared-cluster
export PROJECT=tpu-prod-multislice-feature
export ZONE=europe-west4-b
export BASE_OUTPUT_DIR=gs://ssusie-maxtext-v5e
export DATASET_PATH=gs://ssusie-maxtext

# gcloud config set project $PROJECT
# gcloud config set compute/zone $ZONE


for i in {1..10}; do
  echo "Running qkv256 d 256; " $i
  xpk workload create --cluster ${CLUSTER} --zone ${ZONE} --base-docker-image  ssusie-jun10-v5e-v2  --tpu-type=v5litepod-256 --num-slices=13  --command "python3 MaxText/train.py MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_DIR} dataset_path=${DATASET_PATH} steps=5 checkpoint_period=3  per_device_batch_size=1 base_num_query_heads=256 base_num_kv_heads=256 base_num_decoder_layers=256 enable_single_replica_ckpt_restoring=true run_name=ss-v5e-qkvd256" --workload ss-v5eqkv256d256-$i
  echo "--------------------"
done

echo '**********************************************************************'
for i in {1..10}; do
  echo "Running qkv256 d 196; " $i
  xpk workload create --cluster ${CLUSTER} --zone ${ZONE} --base-docker-image  ssusie-jun10-v5e-v2  --tpu-type=v5litepod-256 --num-slices=13  --command "python3 MaxText/train.py MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_DIR} dataset_path=${DATASET_PATH} steps=5 checkpoint_period=3  per_device_batch_size=1 base_num_query_heads=256 base_num_kv_heads=256 base_num_decoder_layers=196 enable_single_replica_ckpt_restoring=true run_name=ss-v5e-qkv256d196" --workload ss-v5eqkv256d196-$i
  echo "--------------------"
done
