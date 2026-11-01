# OLMo 3.5 Small on one Cloud TPU v4 `8x8x48` slice (3,072 chips)

How to reproduce the best measured OLMo 3.5 Small (Option A, `olmoe3-3p5b-e256k8-sh2x`: 63.1B total / 3.74B
active, E=256, K=8) pre-training throughput on a single TPU v4 slice with topology `8x8x48`, using only this
branch. The run uses strict FP32 numerics (FP32 master weights, router logits, KDA state and 4-tap conv),
sequence length 8,192, and 25.17M tokens per step.

## Measured results

| Runtime  | Mesh (`ici_data` x `ici_expert`)        | Optimizer sharding | Data           | Step time (ms) | MFU    |
| -------- | --------------------------------------- | ------------------ | -------------- | -------------- | ------ |
| Pathways | 48 x 64 (closed `8x8` X-Y planes)       | Zero-1             | Dolma 3.5      | 2,117          | 34.26% |
| McJAX    | 48 x 64 (closed `8x8` X-Y planes)       | Zero-1             | Dolma 3.5      | 2,150          | 33.74% |
| Pathways | 24 x 128 (`8x8x2` slabs)                | none               | Dolma 3.5      | 2,195          | 33.06% |
| McJAX    | 24 x 128 (`8x8x2` slabs)                | none               | Dolma 3.5      | 2,253          | 32.20% |

Step times are steady-state means after compilation. The best config (row 1) runs at 11.89M tokens/s (McJAX row 2 runs at 11.71M tokens/s).

## Files

| File                                       | Purpose                                                                  |
| ------------------------------------------ | ------------------------------------------------------------------------ |
| `run_olmoe3_3p5b_v4.sh`                    | Training launcher. `TOPOLOGY=8x8x48` pins every flag of the runs above.  |
| `k8s/olmoe3_v4_8x8x48_pathways_jobset.yaml` | Pathways JobSet (head pod + 768 Pathways workers).                       |
| `k8s/olmoe3_v4_8x8x48_mcjax_jobset.yaml`    | McJAX JobSet (768 pods, one MaxText process per TPU VM host).            |

Launcher knobs for `TOPOLOGY=8x8x48`:

| Variable  | Values                  | Default    | Meaning                                                              |
| --------- | ----------------------- | ---------- | -------------------------------------------------------------------- |
| `RUNTIME` | `pathways`, `mcjax`     | `pathways` | Single controller (Pathways) or one process per host (McJAX).       |
| `DATA`    | `real`, `synthetic`     | `real`     | Dolma 3.5 through `dataset_type=olmo_grain`, or synthetic batches.   |
| `MODE`    | `bench`, `hero`         | `bench`    | `bench`: 20 steps, no checkpoints. `hero`: long run, async checkpoints every 1,000 steps plus a final one (needs your own `OUTPUT_DIR=gs://...`). |
| `EP`      | `64`, `128`             | `64`       | `64`: DP=48 x EP=64 + Zero-1 (best). `128`: DP=24 x EP=128.         |

Other overrides: `HERO_STEPS` (default 100,000), `CHECKPOINT_PERIOD` (1,000), `GRAIN_WORKER_COUNT`
(8 on Pathways, 1 on McJAX), `INDEX_PATH`, `GCS_BASE`, `LOCAL_MOUNT`, `GLOBAL_INDICES_PATH`, `DATA_SHUFFLE_SEED`.
Any extra `key=value` arguments to the script are passed to MaxText last and win.

## Prerequisites

1. A GKE cluster with a TPU v4 node pool of topology `8x8x48` (machine type `ct4p-hightpu-4t`, 768 nodes),
   JobSet and Kueue installed, and a Kueue `LocalQueue` that admits `google.com/tpu: 3072`.
2. The GKE Cloud Storage FUSE CSI driver enabled, and a Kubernetes service account with Workload Identity access to:
   * the index bucket: `indices/dolma3p5-14t_seq8192.json` and
     `indices/global_indices_dataset_size1708983195_epoch1_seed928543231_v1.npy` (mounted at `/mnt/olmo-3p5-maxtext`),
   * the Dolma 3.5 bucket (`dolma3p5-14t-e6c5a51bd6b9/`, mounted at `/mnt/dolma3p5-14t`),
   * a writable bucket for `OUTPUT_DIR` (checkpoints, metrics) and, for Pathways, `PATHWAYS_GCS_SCRATCH`.
3. A MaxText TPU image built from this branch (MaxText is installed under `/deps` in the image). From the
   repository root, follow the MaxText image build guide, for example:

   ```bash
   bash src/dependencies/scripts/docker_build_dependency_image.sh DEVICE=tpu MODE=stable
   bash src/dependencies/scripts/docker_upload_runner.sh CLOUD_IMAGE_NAME=<image-name>
   # -> gcr.io/<project>/<image-name>; use it as MAXTEXT_IMAGE below.
   ```

   The image must contain the code of this branch: the 8x8x48 expert-group mesh mapping, the on-device Zero-1
   reshard for Pathways, and the launcher above are not in older images.

## Launch

```bash
export NAMESPACE=<namespace> KUEUE_QUEUE=<local-queue> SERVICE_ACCOUNT=<k8s-service-account>
export MAXTEXT_IMAGE=<registry>/<image>:<tag>
export SLICE_NODEPOOL=<name of the 8x8x48 node pool>      # Pathways head-pod placement
export INDEX_BUCKET=<bucket> INDEX_DIR=<dir containing indices/>
export DATA_BUCKET=<bucket> DATA_DIR=dolma3p5-14t-e6c5a51bd6b9
export PATHWAYS_GCS_SCRATCH=gs://<bucket>/pathways-scratch/olmo35-v4-8x8x48
export OUTPUT_DIR=gs://<bucket>/olmo35-v4-8x8x48
export JOBSET_NAME=olmo35-v4-8848-hero MODE=hero DATA=real EP=64 DECOUPLE_GCLOUD=FALSE

VARS='${NAMESPACE} ${JOBSET_NAME} ${KUEUE_QUEUE} ${SERVICE_ACCOUNT} ${MAXTEXT_IMAGE} ${SLICE_NODEPOOL}
${PATHWAYS_GCS_SCRATCH} ${OUTPUT_DIR} ${INDEX_BUCKET} ${INDEX_DIR} ${DATA_BUCKET} ${DATA_DIR} ${MODE} ${DATA} ${EP}
${DECOUPLE_GCLOUD}'
envsubst "${VARS}" < k8s/olmoe3_v4_8x8x48_pathways_jobset.yaml | kubectl apply -f -
```

Pass the explicit variable list to `envsubst`; the manifests also contain `$(...)` Kubernetes expansions.
For a short benchmark use `MODE=bench` (20 steps, no checkpoints; `DECOUPLE_GCLOUD=TRUE` is fine). The index and
Dolma paths inside the image are fixed (`/mnt/olmo-3p5-maxtext/...`, `/mnt/dolma3p5-14t/`); the JobSets mount
the buckets there. If your GCS layout differs, change the `INDEX_PATH`, `GCS_BASE` (the `gs://` prefix the index
was built with), `LOCAL_MOUNT`, and `GLOBAL_INDICES_PATH` env vars of the driver container.

McJAX instead of Pathways:

```bash
envsubst "${VARS}" < k8s/olmoe3_v4_8x8x48_mcjax_jobset.yaml | kubectl apply -f -
```

EP=128 variant: set `EP=128` before `envsubst` (no Zero-1; 2,195 ms on Pathways, 2,253 ms on McJAX).

## Monitor

```bash
kubectl -n ${NAMESPACE} get jobset ${JOBSET_NAME}
# Pathways: training logs come from the jax-driver container of the head pod.
kubectl -n ${NAMESPACE} logs -f -c jax-driver \
  $(kubectl -n ${NAMESPACE} get pods -l jobset.sigs.k8s.io/jobset-name=${JOBSET_NAME},jobset.sigs.k8s.io/replicatedjob-name=pathways-head -o name)
# McJAX: logs of worker 0.
kubectl -n ${NAMESPACE} logs -f \
  $(kubectl -n ${NAMESPACE} get pods -l jobset.sigs.k8s.io/jobset-name=${JOBSET_NAME},batch.kubernetes.io/job-completion-index=0 -o name)
```

Each step logs `seconds`, `TFLOP/s/device`, `Tokens/s/device`, and loss. Expect a cold compile of about 8
minutes on Pathways and 15-20 minutes on McJAX before step 0; the first 2-3 steps are slower. Loss starts near
11.87 and is about 10.9 after 15 steps with `learning_rate=1e-5` (the launcher default; set the real schedule for a
production run with extra `key=value` arguments).

## Troubleshooting

* **Pathways head pod OOM-killed** (driver exits, `OOMKilled`, or the run never reaches compilation): at 3,072
  chips the head pod needs much more host RAM than a small CPU node has. Keep it on a TPU VM host of the slice
  node pool as in the template (nodeSelector on the node pool, `google.com/tpu` toleration, no TPU request, and
  no `exclusive-topology` annotation on the worker job), or move it to a CPU node with >= 128 GB RAM.
* **Zero-1 (`shard_optimizer_over_data=true`) on Pathways**: the train state is resharded on the TPUs with a
  jitted identity (`sharding.reshard_train_state`). Older MaxText code uses `jax.device_put`, which stages the
  whole state through the controller host (driver RSS > 187 GB) and never reaches compilation.
* **Slice with repaired or degraded hosts**: the resource manager runs with
  `temporary_flag_for_debugging_enable_late_binding=false`; keep it when Pathways workers fail to bind.
* **Profiling**: XProf tracing of 768 Pathways workers currently crashes the slice; do not set `profiler` for
  Pathways hero runs. Profile with McJAX (`profiler=xplane skip_first_n_steps_for_profiler=8 profiler_steps=2`).
* **Real-data stalls on Pathways**: one driver process loads every batch; keep `GRAIN_WORKER_COUNT=8` and the
  head pod on a large host.
* **Expert-group layout**: with `ici_expert_parallelism=64` the expert groups are whole `8x8` X-Y planes by default
  on `8x8x48`; `OLMOE3_EP64_SUBMESH=4x4x4` selects `4x4x4` cubes instead (slower in our runs).
