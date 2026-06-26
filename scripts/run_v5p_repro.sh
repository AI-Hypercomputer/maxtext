#!/bin/bash
# This script launches a vLLM serving and benchmark workload on a GKE TPU v5p-64 cluster using XPK.

set -e

# --- Environment Setup ---
if ! /usr/local/google/home/mohitkhatwani/max_venv/bin/pip show xpk &> /dev/null; then
    echo "xpk not found in the environment. Please install it."
    exit 1
fi

# --- Environment Variables ---
export PROJECT_ID="${PROJECT_ID:-cloud-tpu-multipod-dev}"
export CLUSTER_NAME="${CLUSTER_NAME:-mlperf-v5p}"
export ZONE="${ZONE:-europe-west4}"
export BASE_OUTPUT_DIRECTORY="${BASE_OUTPUT_DIRECTORY:-gs://runner-maxtext-logs/}"
export DOCKER_IMAGE="${DOCKER_IMAGE:-gcr.io/tpu-prod-env-multipod/maxtext_post_training_nightly:2026-06-25}"
export WORKLOAD_NAME="mohit-v5p-repro-$RANDOM"
export TPU_TYPE="v5p-64"

# --- Variable Validation ---
if [ -z "$PROJECT_ID" ] || [ -z "$CLUSTER_NAME" ] || [ -z "$ZONE" ] || [ -z "$BASE_OUTPUT_DIRECTORY" ] || [ -z "$DOCKER_IMAGE" ]; then
    echo "Error: Required environment variables are not set."
    exit 1
fi

XLA_FLAGS="--xla_tpu_dvfs_p_state=7 \
--xla_tpu_scoped_vmem_limit_kib=65536 \
--xla_tpu_num_sparse_cores_for_gather_offloading=1 \
--xla_tpu_bf16_emission_mode=NATIVE_EMISSION \
--xla_tpu_enable_sparse_core_reduce_scatter_v2=true \
--xla_tpu_enable_sparse_core_collective_offload_all_gather=true \
--xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true \
--xla_tpu_use_tc_device_shape_on_sc=True \
--xla_sc_disable_megacore_partitioning=True \
--xla_tpu_enable_async_collective_fusion_fuse_all_gather=false \
--xla_enable_async_all_gather=true \
--xla_tpu_prefer_async_allgather_to_allreduce=true \
--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true \
--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true \
--xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true \
--xla_tpu_use_single_sparse_core_for_all_gather_offload=true \
--xla_tpu_enable_concurrent_sparse_core_offloading=true \
--xla_tpu_enable_offloading_gather_to_sparsecore=true \
--xla_tpu_sparse_core_all_gather_latency_multiplier=1 \
--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3 \
--xla_tpu_enable_sparse_core_collective_aggregator=true \
--xla_tpu_enable_latency_hiding_layer_scheduler=true \
--xla_tpu_scheduler_percent_shared_memory_limit=150 \
--xla_tpu_enable_layer_scheduler_for_dependent_collectives=true \
--xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true \
--xla_tpu_pcie_bandwidth_multiplier=0.03 \
--xla_tpu_enable_sparse_core_offload_queuing_in_lhs=true \
--xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=false \
--xla_tpu_enable_3d_reduce_scatter_decomposer=false"

# 1. Run live submit to compile and upload the container image
echo "Compiling and uploading container image via xpk..."
/usr/local/google/home/mohitkhatwani/max_venv/bin/xpk workload create-pathways \
  --cluster="${CLUSTER_NAME}" \
  --project="${PROJECT_ID}" \
  --tpu-type="${TPU_TYPE}" \
  --zone="${ZONE}" \
  --priority=very-high \
  --num-slices=1 \
  --base-docker-image="${DOCKER_IMAGE}" \
  --server-image="us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:20260608-jax_0.10.1" \
  --proxy-server-image="us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:20260608-jax_0.10.1" \
  --workload="${WORKLOAD_NAME}-build" \
  --custom-pathways-proxy-server-args="${XLA_FLAGS}" \
  --custom-pathways-server-args="" \
  --env PHASED_PROFILING_DIR="${BASE_OUTPUT_DIRECTORY}${WORKLOAD_NAME}/tensorboard" \
  --command="echo build-stage" 2>&1 | tee xpk_build.log

# 2. Extract the dynamically built container tag
BUILT_IMAGE=$(grep -o -E "gcr.io/cloud-tpu-multipod-dev/mohitkhatwani-runner:[a-zA-Z0-9._-]+" xpk_build.log | head -n 1)
echo "Successfully built and uploaded image: ${BUILT_IMAGE}"

# 3. Immediately clean up the build workload in GKE
echo "Cleaning up build workload..."
kubectl delete jobset "${WORKLOAD_NAME}-build" || true
rm -f xpk_build.log

# 4. Run dry-run using the built image to generate manifest
echo "Generating workload manifest using built image..."
/usr/local/google/home/mohitkhatwani/max_venv/bin/xpk workload create-pathways \
  --cluster="${CLUSTER_NAME}" \
  --project="${PROJECT_ID}" \
  --tpu-type="${TPU_TYPE}" \
  --zone="${ZONE}" \
  --priority=very-high \
  --num-slices=1 \
  --docker-image="${BUILT_IMAGE}" \
  --server-image="us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:20260608-jax_0.10.1" \
  --proxy-server-image="us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:20260608-jax_0.10.1" \
  --workload="${WORKLOAD_NAME}" \
  --custom-pathways-proxy-server-args="${XLA_FLAGS}" \
  --custom-pathways-server-args="" \
  --env PHASED_PROFILING_DIR="${BASE_OUTPUT_DIRECTORY}${WORKLOAD_NAME}/tensorboard" \
  --env PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR=50 \
  --env PHASED_PROFILER_NUM_DECODE_STEPS_TO_SKIP=10 \
  --env PHASED_PROFILER_DECODE_ONLY_KV_LEN_THRESHOLD=2000 \
  --command="cd /app; pip install -e . --no-deps; bash scripts/run_all.sh" \
  --dry-run \
  --output-manifest-file=generated_repro_manifest.yaml

# Get the GKE internal DNS service IP
KUBE_DNS_IP=$(kubectl get service -n kube-system kube-dns -o jsonpath='{.spec.clusterIP}' 2>/dev/null || echo "34.118.224.10")
echo "Using Kube-DNS ClusterIP: ${KUBE_DNS_IP}"

# 5. Patch the manifest
echo "Applying patch_manifest.py with dnsConfig fix..."
python3 scripts/patch_manifest.py generated_repro_manifest.yaml "${BUILT_IMAGE}" "${KUBE_DNS_IP}"

# 6. Deploy
echo "Deploying the patched workload manifest..."
kubectl apply -f generated_repro_manifest.yaml

# Clean up temp log
rm -f xpk_submit.log

echo "Workload ${WORKLOAD_NAME} successfully submitted!"
