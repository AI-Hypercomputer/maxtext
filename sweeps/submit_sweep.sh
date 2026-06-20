#!/bin/bash
# Submit every manifest row. Kueue serializes them on the single 4x8x8 slot,
# ordered by PRIORITY. Usage: bash sweeps/submit_sweep.sh [sweeps/manifest.tsv]
set -uo pipefail
cd ~/maxtext
MANIFEST="${1:-sweeps/manifest.tsv}"
while IFS=$'\t' read -r TAG IMAGE FLAGS EXTRA PRIO; do
  case "$TAG" in ''|\#*) continue ;; esac
  [ "$EXTRA" = "-" ] && EXTRA=""
  echo ">>> submit $TAG  img=${IMAGE##*:}  flags=$FLAGS  args='$EXTRA'  prio=$PRIO"
  WORKLOAD_IMAGE="$IMAGE" RUN_TAG="$TAG" FLAGS_FILE="$FLAGS" EXTRA_MAXTEXT_ARGS="$EXTRA" PRIORITY="$PRIO" \
    bash repro_variant.sh 2>&1 | grep -iE "created|terminated with code|Error" | head -2
done < "$MANIFEST"
echo "All rows submitted. Watch: kubectl get jobsets | grep ds-"
