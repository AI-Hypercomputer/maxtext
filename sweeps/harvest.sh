#!/bin/bash
# Harvest steady-state step time + profile path for finished runs into results.md.
# Usage: bash sweeps/harvest.sh TAG1 TAG2 ...
# Reads the GCS train log each run copies to gs://ubench-logs/<tag>/logs/.
set -uo pipefail
cd ~/maxtext
RESULTS=sweeps/results.md
if [ ! -f "$RESULTS" ]; then
  printf "| tag | step_s (median 3..8) | Δ vs 16.24 | profile dir | note |\n|---|---|---|---|---|\n" > "$RESULTS"
fi
for TAG in "$@"; do
  log="gs://ubench-logs/$TAG/logs/train-0.log"
  steps=$(gcloud storage cat "$log" 2>/dev/null \
    | grep -oE "completed step: [0-9]+, seconds: [0-9.]+" \
    | sed -nE 's/.*step: ([0-9]+), seconds: ([0-9.]+)/\1 \2/p')
  med=$(echo "$steps" | awk '$1>=3 && $1<=8 && $2>1 {print $2}' | sort -n \
    | awk '{a[NR]=$1} END{if(NR)print a[int((NR+1)/2)]}')
  prof=$(gcloud storage ls "gs://ubench-logs/$TAG/tensorboard/plugins/profile/" 2>/dev/null | tail -1)
  delta=$(awk -v m="$med" 'BEGIN{if(m!="")printf "%+.2f", m-16.24}')
  printf "| %s | %s | %s | %s | |\n" "$TAG" "${med:-CRASH/none}" "${delta:-?}" "${prof:-none}" >> "$RESULTS"
  echo "harvested $TAG: step=${med:-none}s  prof=${prof:-none}"
done
echo "--- $RESULTS ---"; cat "$RESULTS"
