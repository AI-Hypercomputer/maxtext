#!/usr/bin/env bash
# Copy a finished olmoe3-3p5b v4 series off its pods and archive it next to what
# scripts/olmoe3_v4_launch.sh stored, under gs://agagik-us/olmo35/v4/runs/$RUN/:
#   hc/results.tsv            one row per arm (median of the last 10 steps)
#   hc/olmoe3-3p5b-<arm>.log  full MaxText log per arm
#   hc/*.xplane.pb            profiles (only the JAX process-0 pod has them)
#   pod.log                   the leader pod's stdout
# Pods cannot write GCS on this cluster, so this runs from the workstation while HOLD_S keeps them up.
#
#   scripts/olmoe3_v4_harvest.sh o3v4k080512
set -euo pipefail
RUN=$1
export KUBECONFIG=${KUBECONFIG:-/tmp/kc-v4.yaml}
ARCH=gs://agagik-us/olmo35/v4/runs/$RUN
L=/tmp/v4runs/$RUN
mkdir -p "$L"
PODS=$(kubectl get pods -o name | grep "$RUN" | sed 's|pod/||')
LEAD=$(echo "$PODS" | head -1)
for P in $PODS; do
  if kubectl exec "$P" -- sh -c 'ls /tmp/hc/*.xplane.pb' >/dev/null 2>&1; then LEAD=$P; break; fi
done
echo "leader $LEAD"
# Warn loudly if the series is still running: deleting the JobSet now loses the remaining arms.
kubectl logs "$LEAD" | grep -q '^END ' || echo "WARNING: $RUN has not finished (no END in the leader log); harvest again before deleting it"
kubectl cp "$LEAD:/tmp/hc" "$L/hc" >/dev/null
kubectl logs "$LEAD" > "$L/pod.log"
gcloud storage cp -q -r "$L/hc" "$L/pod.log" "$ARCH/"
cat "$L/hc/results.tsv"
echo "archived to $ARCH"
