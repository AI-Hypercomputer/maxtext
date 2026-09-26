#!/usr/bin/env bash
# Run OLMo 3.5 rungs on a 4x8x8 Ironwood slice (64 nodes, 256 chips, 512 devices)
# on the shared-capacity cluster bodaborg-tpu7x-nap.
#
# Why this exists: SPS only offers a 2x2x1 (8 devices), where olmo35-tiny is
# comm-bound and every kernel lever measures neutral. perfsim says the shipped
# geometry reaches 35-51% MFU at 128+ devices and pdb>=4, and this is the run
# that checks that against hardware.
#
# The image predates the current worktree by hundreds of commits, so the rebased
# source is shipped via GCS and put first on PYTHONPATH rather than baked in.
#
# The JobSet shape (network block, exclusive-topology, priority-class label,
# declared-duration, reservation toleration) is copied from
# scripts/arch_ablation_bench.sh, where each field was established the hard way.
set -uo pipefail

CLUSTER=${CLUSTER:-bodaborg-tpu7x-nap}
PROJECT=${PROJECT:-cloud-tpu-shared-capacity}
REGION=${REGION:-us-central1}
# `default` is the namespace this account can actually create in; `priority-dev`
# holds more chips but denies create. Checked with `kubectl auth can-i`.
NAMESPACE=${NAMESPACE:-default}
QUEUE=${QUEUE-multislice-queue}
PRIORITY=${PRIORITY:-medium}
DURATION=${DURATION:-90}

# Read straight off the idle pool's node labels; do not guess these.
TOPOLOGY=${TOPOLOGY:-4x4x4}
NODES=${NODES:-16}
PLACEMENT_POLICY=${PLACEMENT_POLICY-tpu7x-128-4x4x4-placement-policy}
RESERVATION=${RESERVATION-cloudtpu-20260710003900-159478293}

IMAGE=${IMAGE:-gcr.io/cloud-tpu-multipod-dev/agagik-olmoe3:kdaj24}
# Two buckets because pod identity differs by cluster: shared-capacity pods can
# read agagik-us, multipod-dev pods 403 on it and need cloud-pathways-staging.
# The container tries them in order rather than guessing.
SRC=${SRC:-gs://agagik-us/olmo35/src.tgz}
SRC2=${SRC2:-gs://cloud-pathways-staging/agagik/olmo35-src.tgz}
OUT=${OUT:-gs://agagik-us/olmo35/4x4x4}
RUN=${RUN:-o35x$(date +%m%d%H%M)}
MODELS=${MODELS:-olmo35-tiny}
STEPS=${STEPS:-20}

# The hill climb, as `name|pdb|seq|extra maxtext flags`, arms separated by `;`.
# One admission runs all of it, because getting a 4x4x4 is the scarce resource
# and a slice must not be spent on a single data point.
#
# Stage A finds the pdb knee. perfsim at 128 devices puts it at pdb=4 (14.5% ->
# 29.0% -> 35.7% MFU at pdb 1/2/4) with the collective share of the step falling
# 50% -> 12.5% -> 0%, so a1..a4 are there to confirm that curve on hardware.
# Stage B re-tests the levers at the knee, because the 8-device ranking does not
# transfer once comm stops binding. b2 in particular: expert parallelism measured
# 0.79-0.97x at 8 devices and perfsim claims 2.6-3.8x, and 128 devices is where
# that disagreement gets settled.
# z_profile captures an xplane for xla-shell at the best-known config.
ARMS=${ARMS:-\
a1_p1s8k|1|8192|;\
a2_p2s8k|2|8192|;\
a3_p4s8k|4|8192|;\
a4_p2s16k|2|16384|;\
b1_rematcustom|4|8192|remat_policy=custom decoder_layer_input=offload mlpwo=offload;\
b2_ep4|4|8192|ici_expert_parallelism=4;\
b3_noshardexp|4|8192|shard_exp_on_fsdp=False;\
b4_megablox|4|8192|megablox=True use_tokamax_gmm=False use_gmm_v2=False;\
z_profile|4|8192|profiler=xplane skip_first_n_steps_for_profiler=5 profiler_steps=3}

export KUBECONFIG=${KUBECONFIG:-/tmp/kc-nap-olmo35.yaml}
gcloud container clusters get-credentials $CLUSTER --region=$REGION --project=$PROJECT >/dev/null 2>&1

# Ironwood XLA flags from run_olmo3_7b_stage1.sh (27% -> 44.5% on OLMo3-7B).
# Unlike SPS, here the workers are ours, so these actually reach the compiler.
LIBTPU='--xla_tpu_scoped_vmem_limit_kib=65536 --xla_tpu_bf16_emission_mode=NATIVE_EMISSION --xla_tpu_dvfs_p_state=7 --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_use_tc_device_shape_on_sc=True --xla_sc_disable_megacore_partitioning=True --xla_tpu_enable_async_collective_fusion_fuse_all_gather=false'

# Single-host pools carry no placement policy; only emit the selector when set.
PP_LINE=""
[ -n "$PLACEMENT_POLICY" ] && PP_LINE="              cloud.google.com/placement-policy-name: $PLACEMENT_POLICY"$'\n'
EXTRA_SEL="${EXTRA_SELECTORS:-}"
RES_SEL=""; RES_TOL=""
if [ -n "$RESERVATION" ]; then
  RES_SEL="              cloud.google.com/reservation-name: $RESERVATION"$'\n'
  RES_TOL=$'            - key: cloud.google.com/reservation-name\n              operator: Equal\n              value: '"$RESERVATION"$'\n              effect: NoSchedule\n'
fi

QLABELS=""
if [ -n "$QUEUE" ]; then
  # Queue name only. The admitted jobsets on this cluster carry no
  # `priority-class` label, and Kueue rejects one that names no WorkloadPriorityClass.
  QLABELS="    kueue.x-k8s.io/queue-name: $QUEUE"
else
  QLABELS="    olmo35: bench"
fi

YAML=/tmp/$RUN.yaml
cat > "$YAML" <<YAMLEOF
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  name: $RUN
  namespace: $NAMESPACE
  labels:
$QLABELS
  annotations:
    alpha.jobset.sigs.k8s.io/exclusive-topology: cloud.google.com/gke-nodepool
spec:
  network:
    enableDNSHostnames: true
    publishNotReadyAddresses: true
    subdomain: $RUN
  failurePolicy:
    maxRestarts: 0
  successPolicy:
    operator: All
  ttlSecondsAfterFinished: 43200
  replicatedJobs:
  - name: slice-job
    replicas: 1
    template:
      spec:
        backoffLimit: 0
        completionMode: Indexed
        completions: $NODES
        parallelism: $NODES
        template:
          metadata:
            labels:
              declared-duration-minutes: "$DURATION"
            annotations:
              cloud.google.com/gke-tpu-slice-topology: $TOPOLOGY
          spec:
            hostNetwork: true
            dnsPolicy: ClusterFirstWithHostNet
            restartPolicy: Never
            priorityClassName: $PRIORITY
            nodeSelector:
              cloud.google.com/gke-tpu-accelerator: tpu7x
              cloud.google.com/gke-tpu-topology: $TOPOLOGY
$PP_LINE$RES_SEL$EXTRA_SEL
            tolerations:
            - key: google.com/tpu
              operator: Exists
            - key: google.com/tpu
              operator: Exists
              effect: NoSchedule
$RES_TOL            - key: cloud.google.com/gke-spot
              operator: Exists
              effect: NoSchedule
            # flex-start (DWS) pools taint their nodes gke-queued until the
            # provisioning request lands. Harmless on non-flex pools.
            - key: cloud.google.com/gke-queued
              operator: Exists
              effect: NoSchedule
            volumes:
            - name: dshm
              emptyDir:
                medium: Memory
            containers:
            - name: jax-tpu
              image: $IMAGE
              imagePullPolicy: Always
              securityContext:
                privileged: true
              resources:
                limits:
                  google.com/tpu: "4"
              volumeMounts:
              - mountPath: /dev/shm
                name: dshm
              command:
              - bash
              - -c
              - |
                set -o pipefail
                echo START \$(date);
                # Ship the rebased worktree source; the image is hundreds of
                # commits behind and has no olmo35 configs.
                mkdir -p /wt
                for B in $SRC $SRC2; do
                  gcloud storage cp \$B /wt/src.tgz >/dev/null 2>&1 && { echo "src from \$B"; break; }
                done
                tar xzf /wt/src.tgz -C /wt && echo "src ok: \$(ls /wt/src/maxtext | wc -l) entries" || { echo "SRC FETCH FAILED"; exit 1; }
                export PYTHONPATH=/wt/src
                export LIBTPU_INIT_ARGS='$LIBTPU'
                export TMPDIR=/dev/shm
                # Measured 1.34x together on tpu7x; see olmo35-ironwood-plan.md phase 5.
                export TOKAMAX_KDA_DENSE_PAIRS=${DENSE_PAIRS:-1}
                export TOKAMAX_KDA_BF16_FWD=${KDA_BF16:-1} TOKAMAX_KDA_BF16_BWD=${KDA_BF16:-1}
                mkdir -p /tmp/hc
                RES=/tmp/hc/results.tsv
                printf 'model\tarm\tpdb\tseq\tsteps_ok\tmed_tflops_dev\tmfu_pct\tmed_step_s\texit\n' > \$RES
                # Only the leader uploads; every pod runs the same command.
                IDX=\${JOB_COMPLETION_INDEX:-0}
                # Only push when the output is GCS AND this pod can write it.
                # multipod-dev pods can READ gs://agagik-us but cannot write any
                # bucket, which hangs the summary writer and silently loses results;
                # those routes use a local $OUT and are harvested with kubectl.
                case "$OUT" in
                  gs://*) push() { [ "\$IDX" = "0" ] && gcloud storage cp -r /tmp/hc/* $OUT/$RUN/hc/ >/dev/null 2>&1; } ;;
                  *)      push() { :; }; mkdir -p $OUT ;;
                esac
                for M in $MODELS; do
                  echo "$ARMS" | tr ';' '\n' | while IFS='|' read -r NAME P S XTRA ENVX; do
                    [ -z "\$NAME" ] && continue
                    echo "=== ARM \$NAME model=\$M pdb=\$P seq=\$S extra='\$XTRA' \$(date) ===";
                    LOG=/tmp/hc/\$M-\$NAME.log
                    # Optional 5th field: per-arm env overrides, e.g. TOKAMAX_KDA_BF16_FWD=0.
                    # Tokens starting with + are appended to LIBTPU_INIT_ARGS instead.
                    EV=""; LX="$LIBTPU"
                    for T in \$ENVX; do case "\$T" in +*) LX="\$LX \${T#+}";; *) EV="\$EV \$T";; esac; done
                    # KDA_BC=<n> sets the tokamax KDA intra-chunk sub-block (installed value 4,
                    # an overflow patch; safe while (n/2) x decay floor x log2(e) < 127).
                    KD=\$(python3 -c "import tokamax,os;print(os.path.dirname(tokamax.__file__))")/_src/ops/experimental/kda
                    [ -f /tmp/kda_fwd.bak ] || { cp \$KD/pallas_mosaic_tpu_fwd_kernel.py /tmp/kda_fwd.bak; cp \$KD/pallas_mosaic_tpu_bwd_kernel.py /tmp/kda_bwd.bak; }
                    cp /tmp/kda_fwd.bak \$KD/pallas_mosaic_tpu_fwd_kernel.py; cp /tmp/kda_bwd.bak \$KD/pallas_mosaic_tpu_bwd_kernel.py
                    BC=\$(echo "\$ENVX" | tr ' ' '\\n' | sed -n 's/^KDA_BC=//p')
                    if [ -n "\$BC" ]; then
                      sed -i "s/^  BC = 4  # kda8/  BC = \$BC  # kda8/" \$KD/pallas_mosaic_tpu_fwd_kernel.py
                      sed -i "s/^  BC = min(4, BT)/  BC = min(\$BC, BT)/" \$KD/pallas_mosaic_tpu_bwd_kernel.py
                      echo "KDA_BC=\$BC: \$(grep -c "BC = \$BC  # kda8" \$KD/pallas_mosaic_tpu_fwd_kernel.py) fwd, \$(grep -c "BC = min(\$BC, BT)" \$KD/pallas_mosaic_tpu_bwd_kernel.py) bwd"
                    fi
                    env LIBTPU_INIT_ARGS="\$LX" \$EV python3 -m maxtext.trainers.pre_train.train \
                      /wt/src/maxtext/configs/base.yml \
                      model_name=\$M run_name=$RUN-\$M-\$NAME steps=$STEPS \
                      dataset_type=synthetic enable_checkpointing=False async_checkpointing=False \
                      per_device_batch_size=\$P max_target_length=\$S \
                      dtype=bfloat16 weight_dtype=float32 \
                      ici_fsdp_parallelism=-1 remat_policy=full \
                      sparse_matmul=True use_tokamax_kda=True \
                      megablox=False use_tokamax_gmm=True use_gmm_v2=True \
                      shard_exp_on_fsdp=True num_vocab_tiling=8 \
                      base_output_directory=$OUT \$XTRA > \$LOG 2>&1
                    EX=\$?
                    # Pod-local output dies with the pod; park the capture where the harvester looks.
                    find $OUT -path "*$RUN-\$M-\$NAME*" -name '*.xplane.pb' -exec cp {} /tmp/hc/\$M-\$NAME.xplane.pb \; 2>/dev/null
                    # Median of the last 10 steps, so compile and warmup do not count.
                    python3 - "\$LOG" "\$M" "\$NAME" "\$P" "\$S" "\$EX" >> \$RES <<'PYEOF'
                import re, statistics, sys
                log, model, arm, pdb, seq, ex = sys.argv[1:7]
                tf, st = [], []
                for line in open(log, errors='ignore'):
                    m = re.search(r'TFLOP/s/device:\s*([0-9.]+)', line)
                    if m: tf.append(float(m.group(1)))
                    m = re.search(r'seconds:\s*([0-9.]+)', line)
                    if m: st.append(float(m.group(1)))
                mt = statistics.median(tf[-10:]) if tf else 0.0
                ms = statistics.median(st[-10:]) if st else 0.0
                print(f"{model}\t{arm}\t{pdb}\t{seq}\t{len(tf)}\t{mt:.1f}\t{100*mt/1153.5:.2f}\t{ms:.3f}\t{ex}")
                PYEOF
                    tail -3 \$RES
                    tail -12 \$LOG
                    push
                  done
                done
                echo "=== RESULTS ==="; cat \$RES; push
                echo END \$(date)
YAMLEOF

if [ "${DRYRUN:-0}" = "1" ]; then echo "rendered $YAML"; exit 0; fi
kubectl apply -f "$YAML" 2>&1 | tail -2
echo "run=$RUN namespace=$NAMESPACE queue=$QUEUE nodes=$NODES ($TOPOLOGY, $((NODES*4)) chips, $((NODES*8)) devices)"
echo "watch:  KUBECONFIG=$KUBECONFIG kubectl get pods -n $NAMESPACE | grep $RUN"
echo "logs:   KUBECONFIG=$KUBECONFIG kubectl logs -n $NAMESPACE -l jobset.sigs.k8s.io/jobset-name=$RUN --tail=40"
echo "kill:   KUBECONFIG=$KUBECONFIG kubectl delete jobset -n $NAMESPACE $RUN"
