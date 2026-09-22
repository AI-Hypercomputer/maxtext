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

CLUSTER=bodaborg-tpu7x-nap
PROJECT=cloud-tpu-shared-capacity
REGION=us-central1
NAMESPACE=${NAMESPACE:-priority-dev}
QUEUE=${QUEUE:-multislice-queue}
PRIORITY=${PRIORITY:-medium}
DURATION=${DURATION:-90}

# Read straight off the idle pool's node labels; do not guess these.
TOPOLOGY=${TOPOLOGY:-4x8x8}
NODES=${NODES:-64}
PLACEMENT_POLICY=${PLACEMENT_POLICY:-tpu7x-512-4x8x8-placement-policy}
RESERVATION=cloudtpu-20260710003900-159478293

IMAGE=${IMAGE:-gcr.io/cloud-tpu-multipod-dev/agagik-olmoe3:kdaj24}
SRC=${SRC:-gs://agagik-us/olmo35/src.tgz}
OUT=${OUT:-gs://agagik-us/olmo35/4x8x8}
RUN=${RUN:-o35x$(date +%m%d%H%M)}
MODELS=${MODELS:-olmo35-tiny olmo35-small}
PDB=${PDB:-4}
SEQ=${SEQ:-8192}
STEPS=${STEPS:-20}

export KUBECONFIG=${KUBECONFIG:-/tmp/kc-nap-olmo35.yaml}
gcloud container clusters get-credentials $CLUSTER --region=$REGION --project=$PROJECT >/dev/null 2>&1

# Ironwood XLA flags from run_olmo3_7b_stage1.sh (27% -> 44.5% on OLMo3-7B).
# Unlike SPS, here the workers are ours, so these actually reach the compiler.
LIBTPU='--xla_tpu_scoped_vmem_limit_kib=65536 --xla_tpu_bf16_emission_mode=NATIVE_EMISSION --xla_tpu_dvfs_p_state=7 --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_use_tc_device_shape_on_sc=True --xla_sc_disable_megacore_partitioning=True --xla_tpu_enable_async_collective_fusion_fuse_all_gather=false'

# Single-host pools carry no placement policy; only emit the selector when set.
PP_LINE=""
[ -n "$PLACEMENT_POLICY" ] && PP_LINE="              cloud.google.com/placement-policy-name: $PLACEMENT_POLICY"$'\n'

YAML=/tmp/$RUN.yaml
cat > "$YAML" <<YAMLEOF
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  name: $RUN
  namespace: $NAMESPACE
  labels:
    kueue.x-k8s.io/queue-name: $QUEUE
    kueue.x-k8s.io/priority-class: $PRIORITY
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
$PP_LINE              cloud.google.com/reservation-name: $RESERVATION
            tolerations:
            - key: google.com/tpu
              operator: Exists
            - key: google.com/tpu
              operator: Exists
              effect: NoSchedule
            - key: cloud.google.com/reservation-name
              operator: Equal
              value: $RESERVATION
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
                mkdir -p /wt && gcloud storage cp $SRC /wt/src.tgz >/dev/null 2>&1 \
                  && tar xzf /wt/src.tgz -C /wt && echo "src ok: \$(ls /wt/src/maxtext | wc -l) entries";
                export PYTHONPATH=/wt/src
                export LIBTPU_INIT_ARGS='$LIBTPU'
                export TMPDIR=/dev/shm
                # Measured 1.34x together on tpu7x; see olmo35-ironwood-plan.md phase 5.
                export TOKAMAX_KDA_DENSE_PAIRS=${DENSE_PAIRS:-1}
                export TOKAMAX_KDA_BF16_FWD=${KDA_BF16:-1} TOKAMAX_KDA_BF16_BWD=${KDA_BF16:-1}
                for M in $MODELS; do
                  echo "=== MODEL=\$M pdb=$PDB seq=$SEQ ===";
                  python3 -m maxtext.trainers.pre_train.train \
                    /wt/src/maxtext/configs/base.yml \
                    model_name=\$M run_name=$RUN-\$M steps=$STEPS \
                    dataset_type=synthetic enable_checkpointing=False async_checkpointing=False \
                    per_device_batch_size=$PDB max_target_length=$SEQ \
                    dtype=bfloat16 weight_dtype=float32 \
                    ici_fsdp_parallelism=-1 remat_policy=full \
                    sparse_matmul=True use_tokamax_kda=True \
                    megablox=False use_tokamax_gmm=True use_gmm_v2=True \
                    shard_exp_on_fsdp=True num_vocab_tiling=8 \
                    base_output_directory=$OUT 2>&1 | tail -30;
                  echo "=== \$M exit=\${PIPESTATUS[0]} ===";
                done
                echo END \$(date)
YAMLEOF

if [ "${DRYRUN:-0}" = "1" ]; then echo "rendered $YAML"; exit 0; fi
kubectl apply -f "$YAML" 2>&1 | tail -2
echo "run=$RUN namespace=$NAMESPACE queue=$QUEUE nodes=$NODES ($TOPOLOGY, $((NODES*4)) chips, $((NODES*8)) devices)"
echo "watch:  KUBECONFIG=$KUBECONFIG kubectl get pods -n $NAMESPACE | grep $RUN"
echo "logs:   KUBECONFIG=$KUBECONFIG kubectl logs -n $NAMESPACE -l jobset.sigs.k8s.io/jobset-name=$RUN --tail=40"
echo "kill:   KUBECONFIG=$KUBECONFIG kubectl delete jobset -n $NAMESPACE $RUN"
