#!/usr/bin/env bash
# Run an OLMo 3.5 rung on the Shared Pathways Service (tpu7x-8, real Ironwood).
#
# The Pathways controller runs locally, so the model comes from THIS worktree's
# source. No image build and no config injection: PYTHONPATH below is what
# defines the model that the remote TPU workers execute.
#
# Usage:
#   scripts/sps_olmo35.sh                       # olmo35-tiny, 20 steps, seq 8192
#   EXTRA="per_device_batch_size=2" scripts/sps_olmo35.sh
#   MODEL=olmo35-small STEPS=12 scripts/sps_olmo35.sh
set -uo pipefail
WT=/home/agagik_google_com/olmo35/maxtext
export PATH="/home/agagik_google_com/venv-maxtext/bin:${PATH}"
export PYTHONPATH="$WT/src"
unset JAX_PLATFORMS              # must NOT pin tpu; the run uses the remote proxy backend
export USER="${SPS_USER:-agagik}"  # k8s names forbid the underscores in agagik_google_com
# Keep the default kubeconfig untouched: get-credentials below would otherwise
# retarget the current-context that the y6k watcher reads with bare kubectl.
export KUBECONFIG="${KUBECONFIG:-/tmp/kc-sps-olmo35.yaml}"
cd "$WT"

GKE_CLUSTER=bodaborg-tpu7x-sps
PROJECT=cloud-tpu-shared-capacity
REGION=us-central1
GCS_BUCKET=gs://cloud-pathways-staging
SERVICE_JOBSET_NAME=${SPS_SERVICE:-sps-j6080103}
PROXY_IMAGE=us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:20260901-jax_0.11.1

# Ironwood XLA flags from run_olmo3_7b_stage1.sh, which took OLMo3-7B from 27% to
# 44.5% MFU. Set XLA=0 to measure without them. NOTE: with Pathways the workers are
# pre-deployed, so whether these reach the compiler is itself something to verify.
if [ "${XLA:-1}" = "1" ]; then
  export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-} \
    --xla_tpu_scoped_vmem_limit_kib=65536 \
    --xla_tpu_bf16_emission_mode=NATIVE_EMISSION \
    --xla_tpu_dvfs_p_state=7 \
    --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true \
    --xla_tpu_enable_sparse_core_collective_offload_all_gather=true \
    --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true \
    --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true \
    --xla_tpu_use_tc_device_shape_on_sc=True \
    --xla_sc_disable_megacore_partitioning=True \
    --xla_tpu_enable_async_collective_fusion_fuse_all_gather=false"
fi

# Patched-tokamax KDA knobs (olmo35/tokamax-kda-patched). All default off, which
# reproduces PR #1103 bit-for-bit. Exported so the local controller, which builds
# the kernel, sees them.
for v in TOKAMAX_KDA_BF16_FWD TOKAMAX_KDA_BF16_BWD TOKAMAX_KDA_DENSE_PAIRS TOKAMAX_KDA_CHUNK_SIZE; do
  [ -n "${!v:-}" ] && export "$v"
done

MODEL="${MODEL:-olmo35-tiny}"
STEPS="${STEPS:-20}"
SEQ="${SEQ:-8192}"
PDB="${PDB:-1}"
RUN="${RUN:-o35$(date +%H%M%S)}"
EXTRA="${EXTRA:-}"

# MoE and KDA flags only apply to the olmoe3/olmo35 family; a dense model like
# olmo3-7b takes neither, and use_tokamax_kda needs the patched tokamax op.
case "$MODEL" in
  olmo35-*|olmoe3-*) FAMILY_FLAGS="megablox=True sparse_matmul=True use_tokamax_kda=${KDA:-True} num_vocab_tiling=8" ;;
  *)                 FAMILY_FLAGS="" ;;
esac

CMD="python3 -m maxtext.trainers.pre_train.train $WT/src/maxtext/configs/base.yml
 model_name=$MODEL run_name=$RUN steps=$STEPS
 dataset_type=synthetic enable_checkpointing=false async_checkpointing=false
 per_device_batch_size=$PDB max_target_length=$SEQ
 dtype=bfloat16 weight_dtype=float32
 ici_fsdp_parallelism=-1 remat_policy=${REMAT:-full}
 $FAMILY_FLAGS
 enable_single_controller=true
 base_output_directory=$GCS_BUCKET $EXTRA"
CMD=$(echo "$CMD" | tr '\n' ' ' | tr -s ' ')

echo "[sps] model=$MODEL seq=$SEQ pdb=$PDB steps=$STEPS run=$RUN"
echo "[sps] extra: ${EXTRA:-<none>}"
python3 -m pathwaysutils.experimental.shared_pathways_service.run_workload \
  --cluster=$GKE_CLUSTER --project=$PROJECT --region=$REGION \
  --gcs_bucket=$GCS_BUCKET \
  --pathways_service="$SERVICE_JOBSET_NAME-pathways-head-0-0.$SERVICE_JOBSET_NAME:29001" \
  --tpu_type="tpu7x:2x2x1" --tpu_count=1 \
  --proxy_server_image=$PROXY_IMAGE \
  --command "$CMD"
RC=$?
echo "=== run_workload exit RC=$RC ==="
gcloud container clusters get-credentials $GKE_CLUSTER --region=$REGION --project=$PROJECT >/dev/null 2>&1
if kubectl get jobs 2>/dev/null | grep -q "isc-proxy-${USER}"; then
  echo "WARNING leftover proxy job: kubectl delete job \$(kubectl get jobs -o name | grep isc-proxy-${USER})"
else
  echo "cleanup OK"
fi
exit $RC
