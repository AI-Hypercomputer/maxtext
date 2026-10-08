#!/usr/bin/env bash
# Pull an xplane capture for one hill-climb arm out of GCS and run the xla-shell
# analysis over it, writing everything under a local report directory.
#
# MaxText writes the capture to
#   <base_output_directory>/<run_name>/tensorboard/plugins/profile/<ts>/*.xplane.pb
# and run_name here is "<RUN>-<model>-<arm>", so the arm name is enough to find it.
#
# Usage: olmo35_profile_report.sh <gcs-base> <run> <arm> [outdir]
#   olmo35_profile_report.sh gs://cloud-pathways-staging/agagik/olmo35-out o35fle241730 c1_prof_p2
set -uo pipefail

BASE=${1:?gcs base output directory}
RUN=${2:?run name}
ARM=${3:?arm name}
OUTDIR=${4:-/tmp/olmo35_profiles/$RUN-$ARM}
XLASHELL=${XLASHELL:-/home/agagik_google_com/olmo35/xla-shell}
MODEL=${MODEL:-olmo35-tiny}

mkdir -p "$OUTDIR"

echo "== locating xplane for $RUN-$MODEL-$ARM"
PB=$(gcloud storage ls -r "$BASE/$RUN-$MODEL-$ARM/**" 2>/dev/null | grep -E '\.xplane\.pb$' | head -1)
if [ -z "$PB" ]; then
  echo "no xplane found under $BASE/$RUN-$MODEL-$ARM/"
  exit 2
fi
echo "   $PB"
gcloud storage cp "$PB" "$OUTDIR/capture.xplane.pb" >/dev/null 2>&1 || { echo "download failed"; exit 3; }
echo "   $(du -h "$OUTDIR/capture.xplane.pb" | cut -f1) downloaded"

run_cmd() {  # label, xla-shell command string
  local label=$1 cmd=$2
  echo "== $label"
  ( cd "$XLASHELL" && timeout 900 python3 -m xla_shell \
      -c "read_xplane $OUTDIR/capture.xplane.pb; $cmd" 2>&1 | grep -vE '^(WARNING|I0000)' ) \
    | tee "$OUTDIR/$label.txt" | head -60
  echo
}

run_cmd analyze_profile "analyze_profile"
run_cmd roadmap_all     "roadmap --all"
run_cmd roadmap_kernels "roadmap --kernels"
run_cmd roadmap_collective "roadmap --collective"
run_cmd roadmap_remat   "roadmap --remat"
run_cmd roadmap_relayout "roadmap --relayout"

echo "== wrote $OUTDIR"
ls -la "$OUTDIR"
