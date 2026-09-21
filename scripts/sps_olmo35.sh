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
SERVICE_JOBSET_NAME=sps-j6080103
PROXY_IMAGE=us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/unsanitized_proxy_server:cloud_pathways.runtime_20260720_0_RC00

MODEL="${MODEL:-olmo35-tiny}"
STEPS="${STEPS:-20}"
SEQ="${SEQ:-8192}"
PDB="${PDB:-1}"
RUN="${RUN:-o35$(date +%H%M%S)}"
EXTRA="${EXTRA:-}"

CMD="python3 -m maxtext.trainers.pre_train.train $WT/src/maxtext/configs/base.yml
 model_name=$MODEL run_name=$RUN steps=$STEPS
 dataset_type=synthetic enable_checkpointing=false async_checkpointing=false
 per_device_batch_size=$PDB max_target_length=$SEQ
 dtype=bfloat16 weight_dtype=float32
 ici_fsdp_parallelism=-1 megablox=True sparse_matmul=True
 use_tokamax_kda=True remat_policy=full num_vocab_tiling=8
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
  --proxy_server_image=$PROXY_IMAGE --collect_service_metrics \
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
