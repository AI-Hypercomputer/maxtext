#!/usr/bin/env bash
# Keep an OLMo 3.5 hill-climb job queued on every Ironwood route we can reach,
# overnight, and harvest results the moment one lands.
#
# Why a supervisor rather than a single submission: a queued Kueue workload here
# is not durable. `check-capacity-prov` deactivates a workload with "Capacity
# reservation time is expired" and it never retries, spot pools lose their nodes
# mid-run, and a JobSet that fails is not resubmitted by anything. Each of those
# happened at least once over 2026-09-22..23. So this loop re-arms each route.
#
# It never holds more than one job per route, and it stops re-arming a route once
# that route has produced a results.tsv.
set -uo pipefail
cd "$(dirname "$0")/.."

LOG=${LOG:-/tmp/olmo35_overnight.log}
HOURS=${HOURS:-14}
PERIOD=${PERIOD:-300}
HARVEST=${HARVEST:-/tmp/olmo35_results}
mkdir -p "$HARVEST"

# route|kubeconfig|env prefix for the launcher
ROUTES=(
  "nap4x4x4|/tmp/kc-nap-olmo35.yaml|TOPOLOGY=4x4x4 NODES=16"
  "flexspot4x4x4|/tmp/kc-flex2.yaml|CLUSTER=tpu7x-cluster-flex PROJECT=cloud-tpu-multipod-dev NAMESPACE=default QUEUE= TOPOLOGY=4x4x4 NODES=16 PLACEMENT_POLICY= RESERVATION= OUT=gs://cloud-pathways-staging/agagik/olmo35-out EXTRA_SELECTORS=NODEPOOL:tpu7x-full-pod-spot"
  "flexdws2x4x4|/tmp/kc-flex2.yaml|CLUSTER=tpu7x-cluster-flex PROJECT=cloud-tpu-multipod-dev NAMESPACE=default QUEUE= TOPOLOGY=2x4x4 NODES=8 PLACEMENT_POLICY= RESERVATION= OUT=gs://cloud-pathways-staging/agagik/olmo35-out EXTRA_SELECTORS=NODEPOOL:tpu7x-half-cube-2x4x4"
)

say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }

outdir_for() { case "$1" in nap*) echo "gs://agagik-us/olmo35/4x4x4";; *) echo "gs://cloud-pathways-staging/agagik/olmo35-out";; esac; }

submit() {  # route kubeconfig envs -> prints run name
  local route=$1 kc=$2 envs=$3
  local run="o35$(echo "$route" | cut -c1-3)$(date +%d%H%M)"
  local sel=""
  case "$envs" in *EXTRA_SELECTORS=NODEPOOL:*)
    local np="${envs##*EXTRA_SELECTORS=NODEPOOL:}"; np="${np%% *}"
    sel="              cloud.google.com/gke-nodepool: $np"$'\n'
    envs="${envs//EXTRA_SELECTORS=NODEPOOL:$np/}" ;;
  esac
  if [ -n "${ARMS_OVERRIDE:-}" ]; then
    KUBECONFIG=$kc EXTRA_SELECTORS="$sel" RUN=$run MODELS=olmo35-tiny STEPS=20 ARMS="$ARMS_OVERRIDE" \
      env $envs bash scripts/olmo35_xpk_4x8x8.sh >>"$LOG" 2>&1
  else
    KUBECONFIG=$kc EXTRA_SELECTORS="$sel" RUN=$run MODELS=olmo35-tiny STEPS=20 \
      env $envs bash scripts/olmo35_xpk_4x8x8.sh >>"$LOG" 2>&1
  fi
  echo "$run"
}

declare -A RUN NS DONE
RUN[nap4x4x4]=$(grep -oP 'RUNN=\K.*' /tmp/olmo35_run.txt 2>/dev/null)
RUN[flexspot4x4x4]=$(grep -oP 'RUNF=\K.*' /tmp/olmo35_run.txt 2>/dev/null)
RUN[flexdws2x4x4]=$(grep -oP 'RUNH=\K.*' /tmp/olmo35_run.txt 2>/dev/null)

say "supervisor start, ${HOURS}h, period ${PERIOD}s, routes: ${!RUN[*]}"
END=$(( $(date +%s) + HOURS*3600 ))

while [ "$(date +%s)" -lt "$END" ]; do
  for entry in "${ROUTES[@]}"; do
    IFS='|' read -r route kc envs <<< "$entry"
    [ "${DONE[$route]:-0}" = "1" ] && continue
    run=${RUN[$route]:-}
    out="$(outdir_for "$route")"

    # Harvest first: results.tsv means this route delivered.
    if [ -n "$run" ] && gcloud storage cp "$out/$run/hc/results.tsv" "$HARVEST/$route-$run.tsv" >/dev/null 2>&1; then
      say "HARVEST $route $run -> $HARVEST/$route-$run.tsv"
      gcloud storage cp -r "$out/$run/hc" "$HARVEST/$route-$run-logs" >/dev/null 2>&1
      DONE[$route]=1; continue
    fi

    # Health: resubmit if the jobset is gone, failed, or Kueue deactivated it.
    state=$(KUBECONFIG=$kc timeout 90 kubectl get jobset -n default "$run" -o json 2>/dev/null | python3 -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: print('MISSING'); raise SystemExit
cs={c['type']:c for c in d.get('status',{}).get('conditions',[])}
if cs.get('Failed',{}).get('status')=='True': print('FAILED')
elif cs.get('Completed',{}).get('status')=='True': print('COMPLETED')
else:
    rj=(d.get('status',{}).get('replicatedJobsStatus') or [{}])[0]
    print(f\"ALIVE active={rj.get('active',0)} ready={rj.get('ready',0)} succ={rj.get('succeeded',0)}\")
" 2>/dev/null)
    [ -z "$state" ] && state=MISSING

    evicted=$(KUBECONFIG=$kc timeout 90 kubectl get workload -n default -o json 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin)
for w in d['items']:
    if '$run' in w['metadata']['name']:
        for c in w.get('status',{}).get('conditions',[]):
            if c['type']=='Evicted' and c['status']=='True': print('EVICTED')
" 2>/dev/null | head -1)

    case "$state$evicted" in
      MISSING*|FAILED*|*EVICTED)
        say "$route: state=$state$evicted, re-arming"
        KUBECONFIG=$kc timeout 90 kubectl delete jobset -n default "$run" >/dev/null 2>&1
        newrun=$(submit "$route" "$kc" "$envs")
        RUN[$route]=$newrun; say "$route: resubmitted as $newrun" ;;
      *) say "$route: $run $state" ;;
    esac
  done
  sleep "$PERIOD"
done
say "supervisor end"
