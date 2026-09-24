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
  "flexspot4x4x4|/tmp/kc-flex2.yaml|CLUSTER=tpu7x-cluster-flex PROJECT=cloud-tpu-multipod-dev NAMESPACE=default QUEUE= TOPOLOGY=4x4x4 NODES=16 PLACEMENT_POLICY= RESERVATION= OUT=/tmp/out EXTRA_SELECTORS=NODEPOOL:tpu7x-full-pod-spot"
  "flexdws2x4x4|/tmp/kc-flex2.yaml|CLUSTER=tpu7x-cluster-flex PROJECT=cloud-tpu-multipod-dev NAMESPACE=default QUEUE= TOPOLOGY=2x4x4 NODES=8 PLACEMENT_POLICY= RESERVATION= OUT=/tmp/out EXTRA_SELECTORS=NODEPOOL:tpu7x-half-cube-2x4x4"
)

say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }

outdir_for() { case "$1" in nap*) echo "gs://agagik-us/olmo35/4x4x4";; *) echo "POD";; esac; }

# multipod-dev pods can read gs://agagik-us but cannot WRITE any bucket, which
# blocks the summary writer and loses every result. Those routes keep /tmp/hc
# inside the leader pod and are copied out over the API instead.
harvest_pod() {  # kubeconfig run dest -> 0 if a results.tsv came back
  local kc=$1 run=$2 dest=$3
  local pod
  pod=$(KUBECONFIG=$kc timeout 60 kubectl get pods -n default --no-headers 2>/dev/null \
        | grep "$run" | awk '$3=="Running"{print $1}' | head -1)
  [ -z "$pod" ] && return 1
  KUBECONFIG=$kc timeout 120 kubectl exec -n default "$pod" -- cat /tmp/hc/results.tsv > "$dest" 2>/dev/null
  [ -s "$dest" ] || return 1
  mkdir -p "${dest%.tsv}-logs"
  for f in $(KUBECONFIG=$kc timeout 60 kubectl exec -n default "$pod" -- ls /tmp/hc 2>/dev/null); do
    KUBECONFIG=$kc timeout 900 kubectl exec -n default "$pod" -- cat "/tmp/hc/$f" > "${dest%.tsv}-logs/$f" 2>/dev/null
  done
  return 0
}

submit() {  # route kubeconfig envs -> prints run name
  local route=$1 kc=$2 envs=$3
  # One letter per route. Both flex routes start "fle", and a shared prefix let
  # them claim the same jobset name in the same minute and delete each other.
  local code; case "$route" in nap*) code=n;; flexspot*) code=s;; flexdws*) code=h;; *) code=x;; esac
  local run="o35$code$(date +%d%H%M)"
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

ARMS_N=$(echo "${ARMS_OVERRIDE:-}" | tr ';' '\n' | grep -c . ); [ "$ARMS_N" -lt 1 ] && ARMS_N=9
say "supervisor start, ${HOURS}h, period ${PERIOD}s, routes: ${!RUN[*]}"
END=$(( $(date +%s) + HOURS*3600 ))

while [ "$(date +%s)" -lt "$END" ]; do
  for entry in "${ROUTES[@]}"; do
    IFS='|' read -r route kc envs <<< "$entry"
    [ "${DONE[$route]:-0}" = "1" ] && continue
    run=${RUN[$route]:-}
    out="$(outdir_for "$route")"

    # Harvest first: results.tsv means this route delivered.
    # Snapshot every poll. Do NOT retire the route on the first results.tsv:
    # the runner pushes after each arm, so an early file holds one row and
    # retiring on it loses the other eight (that is what truncated o35nap240641).
    got=1
    if [ "$out" = "POD" ]; then
      harvest_pod "$kc" "$run" "$HARVEST/$route-$run.tsv" && got=0
    else
      gcloud storage cp "$out/$run/hc/results.tsv" "$HARVEST/$route-$run.tsv" >/dev/null 2>&1 && got=0
    fi
    if [ -n "$run" ] && [ "$got" = "0" ]; then
      rows=$(( $(wc -l < "$HARVEST/$route-$run.tsv") - 1 ))
      say "HARVEST $route $run rows=$rows/$ARMS_N"
      [ "$out" != "POD" ] && gcloud storage cp -r "$out/$run/hc" "$HARVEST/$route-$run-logs" >/dev/null 2>&1
      if [ "$rows" -ge "$ARMS_N" ]; then say "$route COMPLETE"; DONE[$route]=1; fi
      continue
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
