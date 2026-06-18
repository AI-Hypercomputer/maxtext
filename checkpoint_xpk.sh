#!/bin/bash
set -e

WORKLOAD_NAME="snehalv-dsv4-0616-h"
CLUSTER_NAME="mlperf-v5p"
ZONE="europe-west4-b"
PROJECT="cloud-tpu-multipod-dev"
RESERVATION_NAME="cloudtpu-20240716121201-595617744"

echo "1. Cleaning up any existing workload (waiting for complete deletion)..."
kubectl delete jobset "${WORKLOAD_NAME}" -n default --ignore-not-found --wait=true

echo "2. Creating workload via xpk (this builds & uploads container image)..."
./xpk_venv/bin/xpk workload create \
  --cluster="${CLUSTER_NAME}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --tpu-type="v5p-256" \
  --num-slices=1 \
  --priority=very-high \
  --workload="${WORKLOAD_NAME}" \
  --base-docker-image="gcr.io/tpu-prod-env-multipod/maxtext_jax_nightly:latest" \
  --command="PYTHONPATH=src:\$PYTHONPATH python3 src/maxtext/trainers/pre_train/train_compile.py src/maxtext/configs/base.yml run_name=test_compile metrics_file=metrics.txt base_output_directory=/tmp/maxtext_logs dataset_path=gs://maxtext-dataset dataset_type=synthetic steps=1 per_device_batch_size=1 max_target_length=2048 enable_checkpointing=False model_name=deepseek4-284b compile_topology=v5p-256 compile_topology_num_slices=1 ici_fsdp_parallelism=-1 opt_type=sgd attention=dot_product" \
  --output-manifest-file=workload_manifest.yaml

echo "3. Deleting un-schedulable base workload from cluster..."
kubectl delete jobset "${WORKLOAD_NAME}" -n default --ignore-not-found --wait=true

echo "4. Patching manifest with GKE reservation details & compute-class tolerations..."
./xpk_venv/bin/python3 /usr/local/google/home/snehalv/.gemini/jetski/brain/6d130342-8f6e-4abd-a4c5-dbbfd7180ae5/scratch/fix_manifest.py \
  workload_manifest.yaml \
  workload_manifest_patched.yaml \
  "${RESERVATION_NAME}"

echo "5. Submitting patched workload to cluster..."
kubectl apply -f workload_manifest_patched.yaml

echo "Workload submitted successfully!"