#!/usr/bin/env bash
# Launch one olmoe3-3p5b hill-climb series on a v4-128 slice (v4-128-bodaborg-us-central2-b) and
# archive everything needed to rerun it under gs://agagik-us/olmo35/v4/runs/$RUN/:
#   src.tgz      the exact source the pods run (put first on PYTHONPATH)
#   arms.txt     the arms, `name|pdb|seq|maxtext flags|env`, separated by `;`
#   jobset.yaml  the rendered JobSet (image, libtpu flags, full command)
# After the series ends, scripts/olmoe3_v4_harvest.sh $RUN adds the per-arm logs, results and profiles.
#
#   scripts/olmoe3_v4_launch.sh k benchmarks/olmoe3_v4/v4_arms_k.txt
#   SRC_ROOT=/path/to/other/tree BASE_FLAGS="" scripts/olmoe3_v4_launch.sh p arms.txt   # that tree, only its flags
set -euo pipefail
SERIES=$1
ARMS_FILE=$2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export RUN=${RUN:-o3v4${SERIES}$(date +%d%H%M)}
ARCH=gs://agagik-us/olmo35/v4/runs/$RUN

# SRC_ROOT ships another tree (e.g. a PR checkout) through the same launcher.
SRC_ROOT=${SRC_ROOT:-$ROOT}
EXTRA_SRC=""
[ -f "$SRC_ROOT/scripts/kda_wy_algebra_patch.py" ] && EXTRA_SRC=scripts/kda_wy_algebra_patch.py
tar czf "/tmp/src-$RUN.tgz" -C "$SRC_ROOT" --exclude='__pycache__' --exclude='*.pyc' src $EXTRA_SRC
# Per-run source paths so two series in flight never read each other's tree. multipod-dev pods
# may not read agagik-us, hence the staging-bucket copy the pods fall back to.
gcloud storage cp -q "/tmp/src-$RUN.tgz" "$ARCH/src.tgz"
gcloud storage cp -q "/tmp/src-$RUN.tgz" "gs://cloud-pathways-staging/agagik/olmo35-src-$RUN.tgz"
gcloud storage cp -q "$ARMS_FILE" "$ARCH/arms.txt"

export CLUSTER=v4-128-bodaborg-us-central2-b PROJECT=cloud-tpu-multipod-dev REGION=us-central2
export KUBECONFIG=${KUBECONFIG:-/tmp/kc-v4.yaml}
export ACCEL=tpu-v4-podslice TOPOLOGY=4x4x4 NODES=16 PLACEMENT_POLICY= RESERVATION= QUEUE=multislice-queue
export OUT=/tmp/out MODELS=olmoe3-3p5b STEPS=${STEPS:-20} HOLD_S=${HOLD_S:-2400} DURATION=${DURATION:-300}
export SRC=$ARCH/src.tgz SRC2=gs://cloud-pathways-staging/agagik/olmo35-src-$RUN.tgz
export LIBTPU_ARGS=${LIBTPU_ARGS:---xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_bf16_emission_mode=NATIVE_EMISSION --xla_tpu_scoped_vmem_limit_kib=16384}
ARMS="$(cat "$ARMS_FILE")" bash "$ROOT/scripts/olmo35_xpk_4x8x8.sh"
gcloud storage cp -q "/tmp/$RUN.yaml" "$ARCH/jobset.yaml"
echo "archived to $ARCH"
