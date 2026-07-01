#!/bin/bash
set -e

WORKLOAD_NAME="snehalv-dsv4-layercheck-v2"
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
  --tpu-type="v5p-64" \
  --num-slices=1 \
  --priority=very-high \
  --workload="${WORKLOAD_NAME}" \
  --base-docker-image="gcr.io/tpu-prod-env-multipod/maxtext_jax_nightly:latest" \
  --command="gsutil cp gs://snehalv-data/golden-logits/deepseek-ai/DeepSeek-V4-Flash/golden_dsv4_flash.jsonl /tmp/golden.jsonl && python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu && PYTHONPATH=src:\$PYTHONPATH python3 -m tests.utils.forward_pass_logit_checker src/maxtext/configs/base.yml run_name=dsv4-layer-checker base_output_directory=gs://pb-maxtext-logs/ load_parameters_path=gs://snehalv-data/deepseek_v4-flash/scanned/deepseek4-284b_2026-06-17-18-16/checkpoints/0/items per_device_batch_size=1 max_target_length=2048 model_name=deepseek4-284b weight_dtype=bfloat16 tokenizer_type=huggingface tokenizer_path=deepseek-ai/DeepSeek-V4-Flash ici_fsdp_parallelism=1 ici_expert_parallelism=-1 attention=dot_product checkpoint_storage_concurrent_gb=256 --compare_layerwise_hidden_states --golden_logits_path=/tmp/golden.jsonl --atol=0.02 --output_logits_path=/tmp/maxtext_layerwise_logits.jsonl" \
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
