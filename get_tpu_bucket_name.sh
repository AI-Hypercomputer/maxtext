#!/bin/bash

get_tpu_name() {
  local zone ip name
  zone=$(curl -s -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F'/' '{print $NF}')
  ip=$(curl -s -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/ip)

  # Match against any semicolon-separated IP list
  name=$(gcloud compute tpus tpu-vm list --zone="$zone" \
    --format="value(name,networkEndpoints.ipAddress)" \
    | awk -v ip="$ip" 'index($2, ip) {print $1; exit}')

  echo "$name"
}

get_zone() {
  zone=$(curl -s -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F'/' '{print $NF}')
  echo "$zone"
}

get_bucket_name() {
  MOUNT_DIR="/home/zephyr/gcs-bucket" 
  BUCKET_NAME=$(mount | grep "on ${MOUNT_DIR}" | awk '{print $1}')
  echo "${BUCKET_NAME#gs://}"
}

get_num_hosts() {
  ZONE=$(get_zone)
  TPU_NAME=$(get_tpu_name)
  HOSTS=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone="$ZONE" --format="value(networkEndpoints)" | grep -o "ipAddress" | wc -l) 
  echo "$HOSTS"
}

TPU_NAME=$(get_tpu_name)
echo "✅ Detected TPU name: ${TPU_NAME:-unknown}"

BUCKET_NAME=$(get_bucket_name)
echo "✅ Detected Bucket name: ${BUCKET_NAME:-unknown}"

NUM_HOSTS=$(get_num_hosts)
echo "✅ Detected number of hosts: ${NUM_HOSTS:-unknown}"