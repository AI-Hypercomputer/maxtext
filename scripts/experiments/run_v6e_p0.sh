#!/bin/bash
# P0 decomposition: plain single-slice train.py (NO DiLoCo) on 1 x v6e-8.
#
# Purpose: attribute the +0.44 s/step gap between SPMD main (v6e-spm-m2,
# 1.3824 s/step wall) and non-SPMD branch (v6e-nscc-n2, 1.82 s/step wall) to
#   * model implementation (NNX vs Linen),
#   * source tree (branch vs main),
#   * runtime/compiler (Pathways server:latest vs McJAX libtpu),
# before touching DiLoCo itself.
#
# Workload is byte-identical to the fair harness except DiLoCo is off and
# NUM_SLICES=1: qwen3-8b, bf16, pdbs=4, seq 2048, synthetic, no XLA flags.
#
# Usage: run_v6e_p0.sh <ID> <IMAGE_TAG> <mcjax|pw> <nnx|linen|none> [extra k=v ...]
#   none = pass no nnx keys at all (for main, which has no such config keys).
set -euo pipefail

ID="$1"; IMG_TAG="$2"; RUNTIME="$3"; MODEL_IMPL="$4"; shift 4
EXTRA="$*"

CLUSTER=bodaborg-v6e-nap
PROJECT=tpu-prod-env-one-vm
ZONE=southamerica-west1
DEVICE_TYPE=v6e-8
NUM_SLICES=1
MY_IMAGE="gcr.io/cloud-tpu-multipod-dev/jzuo-runner:${IMG_TAG}"
STEPS="${STEPS:-40}"
XLA_FLAGS="${XLA_FLAGS:-}"

case "${MODEL_IMPL}" in
  nnx)   NNX_ARGS="enable_nnx=true pure_nnx=true pure_nnx_decoder=true" ;;
  linen) NNX_ARGS="enable_nnx=false pure_nnx=false pure_nnx_decoder=false" ;;
  none)  NNX_ARGS="" ;;
  *) echo "bad model impl ${MODEL_IMPL}"; exit 2 ;;
esac

RUNNAME="v6e-p0-${ID}"
SC_ARG=""
SUB="workload create"
if [[ "${RUNTIME}" == "pw" ]]; then
  SUB="workload create-pathways"
  SC_ARG="enable_single_controller=true"
fi

CMD="export PYTHONPATH=/app/src:\$PYTHONPATH && export JAX_NUM_CPU_DEVICES=8 \
&& export LIBTPU_INIT_ARGS='${XLA_FLAGS}' && cd /app/src/ \
&& python3 maxtext/trainers/pre_train/train.py maxtext/configs/base.yml \
     run_name=${RUNNAME} \
     base_output_directory=gs://chriszuo-maxtext-logs \
     dataset_type=synthetic \
     model_name=qwen3-8b \
     weight_dtype=bfloat16 dtype=bfloat16 \
     per_device_batch_size=4 max_target_length=2048 \
     steps=${STEPS} log_period=1 enable_checkpointing=false \
     ${SC_ARG} ${NNX_ARGS} ${EXTRA}; echo EXIT_CODE=\$?"

echo "=== ${RUNNAME}: image=${MY_IMAGE} runtime=${RUNTIME} impl=${MODEL_IMPL} extra='${EXTRA}' ==="

export CLOUDSDK_CORE_PROJECT="${PROJECT}"
export GOOGLE_CLOUD_PROJECT="${PROJECT}"
export GOOGLE_CLOUD_QUOTA_PROJECT="${PROJECT}"

MANIFEST="MyStuff/Data/manifests/${RUNNAME}.yaml"
mkdir -p "$(dirname "${MANIFEST}")"
/usr/local/google/home/jzuo/xpk_venv/bin/xpk ${SUB} --workload "${RUNNAME}" \
  --docker-image "${MY_IMAGE}" --command "${CMD}" \
  --num-slices=${NUM_SLICES} --priority medium \
  --cluster "${CLUSTER}" --tpu-type "${DEVICE_TYPE}" \
  --project "${PROJECT}" --zone "${ZONE}" \
  --dry-run --output-manifest-file "${MANIFEST}"
[ -s "${MANIFEST}" ] || { echo "ERROR: no manifest"; exit 1; }
python3 MyStuff/scripts/patch_manifest.py "${MANIFEST}" \
  --proxy-mem=70G --rm-mem=16G --head-mem=130G --head-cpu=20 \
  --libtpu-init-args="${XLA_FLAGS}"
gcloud container clusters get-credentials "${CLUSTER}" --location="${ZONE}" \
  --project="${PROJECT}" >/dev/null 2>&1
kubectl apply -f "${MANIFEST}"
echo "applied ${MANIFEST}"
