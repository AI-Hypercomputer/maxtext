<!--
 Copyright 2026 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

(run-snapshotting)=

# In-memory snapshotting on GKE with Pathways

Launch and verify MaxText **in-memory snapshotting** on GKE across three cases:

| Case       | Scenario                                        | Key Flags                                                                                                         | What It Validates                                                                                                            |
| :--------- | :---------------------------------------------- | :---------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------- |
| **Case 1** | **Pause/Resume with Snapshotting**              | `elastic_min_slice_count=-1`, `colocated_python_data_input=true`                                                  | Pauses on slice loss and restores from pinned TPU host memory with Grain `ElasticIterator` on TPU worker sidecars            |
| **Case 2** | **Replica Resize + Scaled Global Batch Size**   | `elastic_min_slice_count=2`, `elastic_global_batch_size_mode=scaled`, `colocated_python_data_input=true`          | Continues immediately on surviving slices (`per_device_batch_size` fixed) and scales up once capacity is repaired            |
| **Case 3** | **Replica Resize + Constant Global Batch Size** | `elastic_min_slice_count=2`, `elastic_global_batch_size_mode=constant_padded`, `colocated_python_data_input=true` | Continues on surviving slices while keeping the full-job global batch size constant, and scales up once capacity is repaired |

## 1. Prerequisites & container images

- **GKE Cluster & Client Tools:**
  - A GKE cluster configured for [Pathways on Cloud](https://cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/create-gke-cluster) with multi-slice Cloud TPU node pools (`v5e`, `v5p`, or `v6e`), JobSet (`jobset.x-k8s.io/v1alpha2`), and Kueue (`gcluster` submits workloads to Kueue). See [Running MaxText with Cluster Toolkit](run_maxtext_via_cluster_toolkit.md).
  - An Artifact Registry repository with push permissions (`roles/artifactregistry.writer`).
  - Local client tools: Cluster Toolkit (`gcluster >= v1.108.0`), `gcloud`, `kubectl`, Docker (with a running non-root daemon), and `jq` (used by `build_upload_colocated_python_image.sh`).
- **GCS Bucket & ArrayRecord Dataset:**
  - **Checkpoints & Pathways scratch:** A GCS bucket in the same region as your GKE cluster (`gs://<YOUR_BUCKET>`).
  - **ArrayRecord training shards:** Either the public C4 ArrayRecord dataset at `gs://maxtext-dataset/array-record/c4/en/3.0.1/c4-train.array_record*` (US multi-region, readable by GKE Workload Identity) or shards staged in your bucket (`gs://<YOUR_BUCKET>/array-record/c4/en/3.0.1/c4-train.array_record*`; see [Data input pipeline](../guides/data_input_pipeline.md)).
- **MaxText Container Images:** Requires `AI-Hypercomputer/maxtext` at or after commit `32eb86f05` (includes `e66c28c` / [PR #6215](https://github.com/AI-Hypercomputer/maxtext/pull/6215) and `e600234` / [PR #6113](https://github.com/AI-Hypercomputer/maxtext/pull/6113)). Build **both** `DOCKER_IMAGE` and `SIDECAR_IMAGE` from the **exact same MaxText Git commit and `jax`/`jaxlib` version** (controller and sidecar must not straddle `e66c28c`; if overriding `JAX_VERSION=<version>`, pass it to both build commands; see [Build and upload MaxText Docker images](../tutorials/build_maxtext.md)):

```bash
git clone https://github.com/AI-Hypercomputer/maxtext.git
cd maxtext
git checkout 32eb86f05  # optional: pin validated commit (or omit for latest main >= 32eb86f05)

uv venv --python 3.12 --seed maxtext_venv && source maxtext_venv/bin/activate
uv pip install -e ".[runner]" --resolution=lowest

export AR_PROJECT_ID=<YOUR_ARTIFACT_REGISTRY_PROJECT_ID>
export AR_REGION=<YOUR_ARTIFACT_REGISTRY_REGION>   # e.g., us
export AR_REPO=<YOUR_ARTIFACT_REGISTRY_REPO>       # e.g., maxtext-images
export IMAGE_TAG="snapshot-$(git rev-parse --short HEAD)"

export DOCKER_IMAGE="${AR_REGION}-docker.pkg.dev/${AR_PROJECT_ID}/${AR_REPO}/maxtext-controller:${IMAGE_TAG}"
export SIDECAR_IMAGE="${AR_REGION}-docker.pkg.dev/${AR_PROJECT_ID}/${AR_REPO}/maxtext-colocated-sidecar:${IMAGE_TAG}"

gcloud auth configure-docker "${AR_REGION}-docker.pkg.dev" --quiet

# 1. Build and push the MaxText controller image:
build_maxtext_docker_image
upload_maxtext_docker_image CLOUD_IMAGE_NAME="${DOCKER_IMAGE?}"

# 2. Build and push the matched Colocated Python sidecar image:
bash src/dependencies/scripts/build_upload_colocated_python_image.sh \
  IMAGE_LOCATION="${SIDECAR_IMAGE?}"

# 3. Verify matching jax/jaxlib versions across both images (orbax/pathwaysutils patch versions may differ):
for IMG in "${DOCKER_IMAGE}" "${SIDECAR_IMAGE}"; do
  docker run --rm --entrypoint python3 "${IMG}" -c \
    "import jax, jaxlib, orbax.checkpoint as ocp, pathwaysutils; print('${IMG}', jax.__version__, jaxlib.__version__, ocp.__version__, pathwaysutils.__version__)"
done
```

## 2. Required GKE & MaxText snapshotting flags

### 2.1 Cluster Toolkit (`gcluster job submit`) flags

| Flag                                                                                  | Value                                              | Purpose                                                                                                                                                                                                                                                                                                                                                                  |
| :------------------------------------------------------------------------------------ | :------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--pathways`                                                                          | Present (boolean)                                  | Launches the workload on Pathways (`pathways-rm`, `pathways-proxy`, and `pathways-worker`).                                                                                                                                                                                                                                                                              |
| `--name`                                                                              | `<RUN_NAME>` (`<= 22` chars)                       | Lowercase alphanumeric/hyphen DNS label. `gcluster` enforces `<= 22` chars because `jobset.sigs.k8s.io/coordinator=<RUN_NAME>-pathways-head-0-0.<RUN_NAME>` embeds the name twice (`<= 63` chars).                                                                                                                                                                       |
| `--num-slices`                                                                        | `${NUM_SLICES}` (e.g., `4`)                        | Total allocated TPU slices in the `JobSet`.                                                                                                                                                                                                                                                                                                                              |
| `--pathways-gcs-location`                                                             | `gs://<BUCKET>/pathways/...`                       | GCS scratch directory for Pathways coordinator and worker pods.                                                                                                                                                                                                                                                                                                          |
| `--pathways-elastic-slices`                                                           | `${NUM_SLICES}`                                    | Enables in-process slice failure handling in `pathways-proxy`.                                                                                                                                                                                                                                                                                                           |
| `--pathways-colocated-python-sidecar-image`                                           | `${SIDECAR_IMAGE}`                                 | Injects `colocated-python-sidecar` into each TPU worker pod and configures sidecar IPC automatically.                                                                                                                                                                                                                                                                    |
| `--pathways-max-slice-restarts`                                                       | Optional (defaults to `4000` when elastic enabled) | Upper bound on per-pod-index Kubernetes restarts (`backoffLimitPerIndex`). Leave unset so `gcluster`'s default `4000` avoids premature `JobSet` restarts.                                                                                                                                                                                                                |
| `--queue` / `--priority`                                                              | Optional (`<LOCAL_QUEUE>` / `<PRIORITY_CLASS>`)    | Selects the Kueue `LocalQueue` and workload priority on shared clusters. Prefer a non-preemptible `WorkloadPriorityClass`: a **Kueue workload eviction** restarts the whole `JobSet` (clearing host-RAM snapshots), whereas **pod-priority preemption** of a worker slice is handled elastically in-process (unless only part of a slice is evicted; see Section 5).     |
| `--pathways-server-image`, `--pathways-proxy-server-image`, `--pathways-worker-image` | Optional (`<IMAGE_URI>`)                           | Pins explicit Pathways release tags matching your JAX version instead of `:latest` (validated with `20261111-jax_0.11.1`: `us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:20261111-jax_0.11.1` for `--pathways-server-image` and `--pathways-worker-image`, and `us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:20261111-jax_0.11.1` for the proxy). |

```{note}
**Using custom GKE `JobSet` YAML instead of `gcluster`:**
- **Sidecar wiring:** Set `--num_elastic_instances=${NUM_SLICES}` and `--sidecar_name=external` on `pathways-proxy`, set `--cloud_pathways_sidecar_shm_directory=/tmp/sidecar` on `pathways-worker`, and run `colocated-python-sidecar` as a native sidecar (`initContainers` with `restartPolicy: Always`, `GRPC_SERVER_ADDRESS=0.0.0.0:50051`, and `CLOUD_PATHWAYS_SIDECAR_SHM_DIRECTORY=/tmp/sidecar`) with both `pathways-worker` and `colocated-python-sidecar` mounting `/tmp/sidecar` (`emptyDir: {medium: Memory}`). (`gcluster --pathways-colocated-python-sidecar-image` configures these sidecar settings automatically.)
- **Worker `ReplicatedJob` policy:** Set `podReplacementPolicy: Failed`, `maxFailedIndexes: 0`, and a high `backoffLimitPerIndex` (e.g., `4000`).
- **Optional Pathways persistence for fallback GCS checkpoints (`enable_checkpointing=true`):** Set `ENABLE_PATHWAYS_PERSISTENCE=1` on the MaxText controller container (`--env="ENABLE_PATHWAYS_PERSISTENCE=1"` in `gcluster`) **and** pass `checkpoint_storage_use_ocdbt=false checkpoint_storage_use_zarr3=false` in the MaxText command (`ENABLE_PATHWAYS_PERSISTENCE=1` raises `ValueError` unless both OCDBT and Zarr3 are disabled).
```

### 2.2 MaxText snapshotting flags

| Flag                             | Required Setting                                      | Notes                                                                                                                                                                                                                   |
| :------------------------------- | :---------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `enable_single_controller`       | `true`                                                | Mandatory for Pathways elastic training.                                                                                                                                                                                |
| `colocated_python_data_input`    | `true`                                                | Recommended at scale (experimental); runs Grain `ElasticIterator` on TPU worker sidecars (can also run on the controller when `false`).                                                                                 |
| `colocated_python_checkpointing` | `false`                                               | Keep `false` with Replica Resize so fallback GCS saves do not block waiting on slice-0 worker metadata creation while slice 0 is down.                                                                                  |
| `eval_interval`                  | `-1`                                                  | Keep evaluation disabled when `colocated_python_data_input=true` to avoid advancing a closed iterator across elastic retries.                                                                                           |
| `elastic_enabled`                | `true`                                                | Enables in-process slice recovery (`elastic_retry`).                                                                                                                                                                    |
| `elastic_backup_kind`            | `snapshot`                                            | Saves and restores train state and iterator cursor in pinned TPU host RAM.                                                                                                                                              |
| `elastic_snapshot_interval`      | `10` (or `5`–`25`)                                    | Steps between async host-memory snapshots (skipped if a fallback GCS checkpoint save is currently in flight).                                                                                                           |
| `elastic_min_slice_count`        | `-1` (Pause/Resume) or `1 .. NUM_SLICES - 1` (Resize) | `-1` waits for all slices; `< NUM_SLICES` starts and continues once `>= elastic_min_slice_count` slices are up (at launch, training may start on the first ready slices and immediately scale up once the rest join).   |
| `elastic_global_batch_size_mode` | `scaled` or `constant_padded`                         | `scaled` (default) scales GBS with active slices; `constant_padded` requires an integer `per_device_batch_size >= 1`.                                                                                                   |
| `elastic_timeout_seconds`        | `900`                                                 | Max seconds to wait for `elastic_min_slice_count` healthy slices on GKE.                                                                                                                                                |
| `elastic_max_retries`            | `50`                                                  | Caps total controller attempts (`attempt < elastic_max_retries`, so `50` allows up to `49` transitions; each resize cycle costs `2` attempts—slice-down plus scale-up—in addition to startup scale-up and preemptions). |
| `shard_optimizer_over_data`      | `false`                                               | Keep `false` so each slice holds a full optimizer-state replica in host RAM (`true` logs a warning and falls back to GCS checkpoints on slice loss).                                                                    |
| `enable_checkpointing`           | `true` (`checkpoint_period=250`)                      | Keeps periodic GCS checkpoints as a fallback if all slices or the controller restart.                                                                                                                                   |
| `profiler`                       | `""` (unset)                                          | Leave profiling disabled during elastic runs.                                                                                                                                                                           |
| `dataset_type`                   | `grain` (or `synthetic`)                              | Real-data runs require `grain_file_type=arrayrecord`, `grain_use_elastic_iterator=true`, `packing=false`, `use_truncation=true`, and a single `grain_train_files` glob pattern (no `;` mixtures).                       |

## 3. Launch commands by case

### Step 0: Configure environment & shared variables

Export your cluster, bucket, dataset, and container image variables (preserving
`DOCKER_IMAGE` and `SIDECAR_IMAGE` if already exported in Section 1):

```bash
export PROJECT_ID=<YOUR_GKE_CLUSTER_PROJECT_ID>
export LOCATION=<YOUR_CLUSTER_ZONE_OR_REGION>        # e.g., us-central2-b or us-south1
export GKE_CLUSTER=<YOUR_GKE_CLUSTER_NAME>
export COMPUTE_TYPE=ct5lp-hightpu-4t                 # v5e: ct5lp-hightpu-4t | v5p: ct5p-hightpu-4t | v6e: ct6e-standard-4t
export TOPOLOGY=4x8                                  # e.g., 4x8 (v5e-32) or 4x4x4 (v5p-128)
export NUM_SLICES=4

export BASE_OUTPUT_DIRECTORY=gs://<YOUR_BUCKET>/snapshot-runs
# Public US multi-region C4 ArrayRecord dataset (or gs://<YOUR_BUCKET>/array-record/c4/en/3.0.1/c4-train.array_record*):
export GRAIN_TRAIN_FILES="gs://maxtext-dataset/array-record/c4/en/3.0.1/c4-train.array_record*"

: "${DOCKER_IMAGE:=<YOUR_CONTROLLER_MAXTEXT_IMAGE>}"
: "${SIDECAR_IMAGE:=<YOUR_COLOCATED_PYTHON_SIDECAR_IMAGE>}"

# Update active kubeconfig context for kubectl commands in Section 4:
gcloud container clusters get-credentials "${GKE_CLUSTER?}" \
  --location="${LOCATION?}" --project="${PROJECT_ID?}"

# Shared baseline MaxText model, snapshotting & colocated Grain ElasticIterator flags
# (For private GKE clusters without Hugging Face Hub access, append the in-sidecar path:
#  tokenizer_type=huggingface tokenizer_path=/app/maxtext/src/maxtext/assets/tokenizers/qwen3-tokenizer)
export COMMON_SNAPSHOT_FLAGS="model_name=qwen3-0.6b \
  override_model_config=true \
  normalize_embedding_logits=true \
  per_device_batch_size=1 \
  max_target_length=2048 \
  attention=flash \
  remat_policy=full \
  ici_fsdp_parallelism=-1 \
  dcn_data_parallelism=-1 \
  shard_optimizer_over_data=false \
  steps=2000 \
  log_period=1 \
  enable_single_controller=true \
  colocated_python_data_input=true \
  colocated_python_checkpointing=false \
  eval_interval=-1 \
  elastic_enabled=true \
  elastic_backup_kind=snapshot \
  elastic_snapshot_interval=10 \
  elastic_timeout_seconds=900 \
  elastic_max_retries=50 \
  enable_checkpointing=true \
  checkpoint_period=250 \
  dataset_type=grain \
  grain_file_type=arrayrecord \
  grain_train_files=${GRAIN_TRAIN_FILES?} \
  grain_use_elastic_iterator=true \
  packing=false \
  use_truncation=true"
```

### Case 1: Pause/Resume with snapshotting (`elastic_min_slice_count=-1`, `colocated_python_data_input=true`)

Pauses on slice loss and restores from host RAM once capacity recovers (on
shared Kueue clusters, pass `--queue=<LOCAL_QUEUE>` and
`--priority=<PRIORITY_CLASS>` as needed):

```bash
export RUN_NAME=case1-coloc-pause

gcluster job submit \
  --project="${PROJECT_ID?}" \
  --cluster="${GKE_CLUSTER?}" \
  --location="${LOCATION?}" \
  --image="${DOCKER_IMAGE?}" \
  --name="${RUN_NAME?}" \
  --pathways \
  --compute-type="${COMPUTE_TYPE?}" \
  --topology="${TOPOLOGY?}" \
  --num-slices="${NUM_SLICES?}" \
  --pathways-gcs-location="${BASE_OUTPUT_DIRECTORY?}/pathways/${RUN_NAME?}" \
  --pathways-elastic-slices="${NUM_SLICES?}" \
  --pathways-colocated-python-sidecar-image="${SIDECAR_IMAGE?}" \
  --command="python3 -u -m maxtext.trainers.pre_train.train \
    base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    run_name=${RUN_NAME?} \
    ${COMMON_SNAPSHOT_FLAGS?} \
    elastic_min_slice_count=-1"
```

### Case 2: Replica resize + scaled global batch size (`elastic_global_batch_size_mode=scaled`, `colocated_python_data_input=true`)

Continues on `>= 2` slices with scaled GBS, scaling up once repaired:

```bash
export RUN_NAME=case2-coloc-scaled

gcluster job submit \
  --project="${PROJECT_ID?}" \
  --cluster="${GKE_CLUSTER?}" \
  --location="${LOCATION?}" \
  --image="${DOCKER_IMAGE?}" \
  --name="${RUN_NAME?}" \
  --pathways \
  --compute-type="${COMPUTE_TYPE?}" \
  --topology="${TOPOLOGY?}" \
  --num-slices="${NUM_SLICES?}" \
  --pathways-gcs-location="${BASE_OUTPUT_DIRECTORY?}/pathways/${RUN_NAME?}" \
  --pathways-elastic-slices="${NUM_SLICES?}" \
  --pathways-colocated-python-sidecar-image="${SIDECAR_IMAGE?}" \
  --command="python3 -u -m maxtext.trainers.pre_train.train \
    base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    run_name=${RUN_NAME?} \
    ${COMMON_SNAPSHOT_FLAGS?} \
    elastic_min_slice_count=2 \
    elastic_global_batch_size_mode=scaled"
```

### Case 3: Replica resize + constant global batch size (`elastic_global_batch_size_mode=constant_padded`, `colocated_python_data_input=true`)

Continues on `>= 2` slices at constant GBS, scaling up once repaired:

```bash
export RUN_NAME=case3-coloc-const

gcluster job submit \
  --project="${PROJECT_ID?}" \
  --cluster="${GKE_CLUSTER?}" \
  --location="${LOCATION?}" \
  --image="${DOCKER_IMAGE?}" \
  --name="${RUN_NAME?}" \
  --pathways \
  --compute-type="${COMPUTE_TYPE?}" \
  --topology="${TOPOLOGY?}" \
  --num-slices="${NUM_SLICES?}" \
  --pathways-gcs-location="${BASE_OUTPUT_DIRECTORY?}/pathways/${RUN_NAME?}" \
  --pathways-elastic-slices="${NUM_SLICES?}" \
  --pathways-colocated-python-sidecar-image="${SIDECAR_IMAGE?}" \
  --command="python3 -u -m maxtext.trainers.pre_train.train \
    base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    run_name=${RUN_NAME?} \
    ${COMMON_SNAPSHOT_FLAGS?} \
    elastic_min_slice_count=2 \
    elastic_global_batch_size_mode=constant_padded"
```

```{tip}
**Optional gradient accumulation:** In MaxText, `global_batch_size = num_devices × per_device_batch_size × gradient_accumulation_steps`. Appending `gradient_accumulation_steps=2 use_tunix_gradient_accumulation=false` to Case 3 therefore **doubles** the global batch size (and `global_next_index` / `total_weights` per step) while keeping the per-device microbatch at the rounded-up `per_device_batch_size`. To reduce per-device HBM at `elastic_min_slice_count` slices while keeping the same global batch size, lower `per_device_batch_size` and increase `gradient_accumulation_steps` proportionally (verify that `per_device_batch_size × NUM_SLICES / elastic_min_slice_count` fits in TPU HBM).
```

## 4. Simulate a slice failure, verify snapshot recovery & clean up

### Step 1: Stream controller logs and wait past first snapshot

Wait for Kueue to admit the `JobSet` and for the head pod to become `Ready`,
then stream `workload-container` logs (`jax-tpu` on custom `JobSet`):

```bash
until kubectl get pods \
  -l "jobset.sigs.k8s.io/jobset-name=${RUN_NAME?},jobset.sigs.k8s.io/replicatedjob-name=pathways-head" \
  -o name | grep -q .; do
  sleep 5
done
kubectl wait --for=condition=Ready pod \
  -l "jobset.sigs.k8s.io/jobset-name=${RUN_NAME?},jobset.sigs.k8s.io/replicatedjob-name=pathways-head" \
  --timeout=900s

HEAD_POD=$(kubectl get pods \
  -l "jobset.sigs.k8s.io/jobset-name=${RUN_NAME?},jobset.sigs.k8s.io/replicatedjob-name=pathways-head" \
  -o jsonpath='{.items[0].metadata.name}')
kubectl logs -f "${HEAD_POD?}" -c workload-container
```

Wait until all `${NUM_SLICES}` slices are active (`live slice count: 4`; if a
slice stays stuck below `NUM_SLICES` with `Pending` pods after a partial
eviction, see Section 5) **and** at least one
`Saving in-memory snapshot at step <N>...` line has appeared (early intervals
such as steps `0` and `10` may be skipped while the step-`0` GCS checkpoint save
is in progress). Open a second terminal with `RUN_NAME`, `NUM_SLICES`,
`PROJECT_ID`, `LOCATION`, and `GKE_CLUSTER` exported for Steps 2 and 4 (if the
`JobSet` restarts, re-run Step 1 to resolve the new `HEAD_POD`).

### Step 2: Inject a slice failure

#### Option A: Immediate pod deletion (Pause/Resume in Case 1, or automatic resize + scale-up in Cases 2 & 3)

`kubectl delete ... --force` can take ~15–60s to return while pods terminate,
and replacement slice pods typically rejoin within ~15–75s (on the same node
pool or another free pool). Because `maybe_back_up_and_scale_up` checks for
recovered slices after every step, Cases 2 & 3 run anywhere from `1` to `~60+`
reduced-slice steps (~25–75s) before scaling back up:

```bash
TARGET_SLICE=$(( NUM_SLICES - 1 ))
kubectl delete pods \
  -l "jobset.sigs.k8s.io/jobset-name=${RUN_NAME?},jobset.sigs.k8s.io/replicatedjob-name=worker,jobset.sigs.k8s.io/job-index=${TARGET_SLICE}" \
  --grace-period=0 --force
```

#### Option B: Hold reduced slice count before scale-up (dedicated clusters, Cases 2 & 3)

Cordon one node of the target slice, delete all pods of that slice so
multi-host workers exit together, then uncordon to scale back up. Use Option B
**only on a dedicated cluster** and always uncordon promptly: on a shared
cluster, cordoning blocks other workloads from scheduling onto that node pool
(and if spare pools or Node Auto-Provisioning exist, Kubernetes schedules
replacement pods onto a free pool instead of holding at `NUM_SLICES - 1`; do
not cordon spare pools on shared clusters):

```bash
TARGET_SLICE=$(( NUM_SLICES - 1 ))
VICTIM_POD=$(kubectl get pods \
  -l "jobset.sigs.k8s.io/jobset-name=${RUN_NAME?},jobset.sigs.k8s.io/replicatedjob-name=worker,jobset.sigs.k8s.io/job-index=${TARGET_SLICE}" \
  -o jsonpath='{.items[0].metadata.name}')
VICTIM_NODE=$(kubectl get pod "${VICTIM_POD?}" -o jsonpath='{.spec.nodeName}')

kubectl cordon "${VICTIM_NODE?}"
kubectl delete pods \
  -l "jobset.sigs.k8s.io/jobset-name=${RUN_NAME?},jobset.sigs.k8s.io/replicatedjob-name=worker,jobset.sigs.k8s.io/job-index=${TARGET_SLICE}" \
  --grace-period=0 --force

# After observing reduced-slice steps in controller logs, uncordon to scale back up:
kubectl uncordon "${VICTIM_NODE?}"
```

### Step 3: Verify expected snapshot log lines

Confirm the following log excerpts in `HEAD_POD` (measured on `4 × v5e-32` at
`GBS=128`):

- **Case 1 excerpt — Pause/Resume (`elastic_min_slice_count=-1`, `4 → pause → 4`):**

```text
Saving in-memory snapshot at step 30...
completed step: 31, ..., live slice count: 4, ..., total_weights: 49665, loss: 11.314
Elastic event detected
Elastic attempt 2
Waiting for 4 slices. Poll interval: 10, Timeout: 900
3 slices active. Wanting at least 4. Active indices: frozenset({0, 2, 3})
Sufficient slices active: 4 >= 4. Active indices: frozenset({0, 1, 2, 3})
Restoring the in-memory snapshot at step 30...
Restored the data iterator state: {'global_next_index': 3968}.
Restored the in-memory snapshot at step 30.
completed step: 31, ..., live slice count: 4, ..., total_weights: 49665, loss: 11.314
```

- **Cases 2 & 3 excerpt — Slice-down restore (`4 → 3`; Case 3 `constant_padded` values shown):**

```text
Saving in-memory snapshot at step 30...
completed step: 31, ..., live slice count: 4, ..., total_weights: 49665, loss: 11.314
Elastic event detected
Elastic attempt 2
Waiting for 2 slices. Poll interval: 10, Timeout: 900
Sufficient slices active: 3 >= 2. Active indices: frozenset({0, 1, 3})
Restoring the in-memory snapshot at step 30...
Restored the data iterator state: {'global_next_index': 3968}.
Restored the in-memory snapshot at step 30.
completed step: 31, ..., live slice count: 3, ..., total_weights: 49665, loss: 11.314
```

- **Cases 2 & 3 excerpt — Zero-lost-step scale-up when slice returns (`3 → 4`):**

```text
New slices are available. Snapshotting step 31 before scaling up.
Saving in-memory snapshot at step 31...
New slices are available. Waiting for any checkpoint save in progress before interrupting.
Interrupting the attempt to scale up.
completed step: 31, ..., live slice count: 3, ...
Scale up requested.
Elastic attempt 3
Waiting for 2 slices. Poll interval: 10, Timeout: 900
Sufficient slices active: 4 >= 2. Active indices: frozenset({0, 1, 2, 3})
Restoring the in-memory snapshot at step 31...
Restored the data iterator state: {'global_next_index': 4096}.
Restored the in-memory snapshot at step 31.
completed step: 32, ..., live slice count: 4, ..., total_weights: 44350, loss: 11.258
```

```{note}
- **Case 3 (`constant_padded`) vs. Case 2 (`scaled`):** In Case 3, re-executed step `31` logs the exact same `total_weights` (`49665`) and `loss` (`11.314`) on `3` slices as on `4` slices (`global_next_index` advances by `128` per step to `4096`). In Case 2, `total_weights` and `global_next_index` stride scale proportionally with the active slice count (`37879` / `+96` to `4064` per step on `3` of `4` slices).
- **Slice numbering & replay determinism:** Pathways `Active indices` (e.g., `frozenset({0, 1, 3})`) are runtime slice IDs and need not match the Kubernetes `JobSet` `job-index`. Up to `elastic_snapshot_interval` steps since the last completed snapshot are recomputed; the first replayed step matches pre-failure loss, while subsequent replayed steps may differ in the 3rd decimal place due to floating-point collective non-determinism. `Saving in-memory snapshot at step N...` during scale-up prints only when step `N` was not already the latest snapshot.
- **GCS checkpoint interactions (`enable_checkpointing=true`):** If a slice returns while a periodic GCS checkpoint save is in flight (e.g., step `250`), scale-up triggers from the checkpoint path (`maybe_elastic_scale_up`) without `Snapshotting step N before scaling up.` and logs `Restoring the checkpoint at step 250 rather than the in-memory snapshot at step 240, which is older.`—this is lossless and expected whenever the committed GCS checkpoint is newer than the latest snapshot. Conversely, if a slice fails during an in-flight GCS save (note that GCS saves can pause step progression for ~60–90s while finalizing), recovery cleans up the uncommitted directory (`No commit_success file found. Deleting .../checkpoints/<step>/...`) and restores the latest host-RAM snapshot normally.
```

### Step 4: Clean up

```bash
# Cancels and deletes the JobSet:
gcluster job cancel "${RUN_NAME?}" \
  --project="${PROJECT_ID?}" \
  --cluster="${GKE_CLUSTER?}" \
  --location="${LOCATION?}"

# If Option B was used, verify VICTIM_NODE is uncordoned (Ready, without SchedulingDisabled):
if [[ -n "${VICTIM_NODE:-}" ]]; then
  kubectl get node "${VICTIM_NODE}"
fi
```

## 5. Troubleshooting

| Error / Symptom                                                                                                                                                                                                                                                                                                                                                                              | Resolution                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Error: workload name cannot exceed 22 characters for Pathways jobs` / `jobset.sigs.k8s.io/coordinator` label `must be no more than 63 characters`                                                                                                                                                                                                                                           | Keep `RUN_NAME` at **22 characters or fewer** (lowercase alphanumeric and hyphens).                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `error: container jax-tpu is not valid for pod <RUN_NAME>-pathways-head-0-0-<suffix>`                                                                                                                                                                                                                                                                                                        | Use `-c workload-container` for `gcluster` jobs (`kubectl logs -f "${HEAD_POD}" -c workload-container`).                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Job stays stuck below `NUM_SLICES` with some worker pods `Pending` (`node(s) didn't match ...` / `didn't have free ports`)                                                                                                                                                                                                                                                                   | Partial pod-priority preemption evicted one pod of a multi-host slice while its peers stayed pinned to the old node pool by `JobSet` exclusive placement. Force-delete all pods of that `jobset.sigs.k8s.io/job-index` (Section 4 Step 2 Option A) so `JobSet` re-places the slice together, and submit with a non-preemptible `--priority`.                                                                                                                                                                                                                 |
| `` elastic_backup_kind='snapshot' requires `dataset_type=grain` with `grain_use_elastic_iterator=True` (or synthetic data) ``                                                                                                                                                                                                                                                                | Pass `dataset_type=grain grain_file_type=arrayrecord grain_use_elastic_iterator=true packing=false use_truncation=true` (and a single `grain_train_files` glob without `;`), or `dataset_type=synthetic`.                                                                                                                                                                                                                                                                                                                                                    |
| Snapshot falls back to GCS checkpoint on slice loss (`Skipping the in-memory snapshot at step N because a checkpoint save is in progress.`, `Restoring the checkpoint at step X rather than the in-memory snapshot at step Y, which is older.`, `WARNING: elastic_backup_kind=snapshot with shard_optimizer_over_data=True...`, or `Restoring the in-memory snapshot at step N failed: ...`) | Ensure `shard_optimizer_over_data=false` and wait until a `Saving in-memory snapshot at step <N>...` line completes after any in-flight GCS checkpoint save before injecting failure (`Restoring the checkpoint at step X ...` is expected and harmless right after a periodic GCS checkpoint save when `X > Y`).                                                                                                                                                                                                                                            |
| `` `elastic_global_batch_size_mode='constant_padded'` requires an integer `per_device_batch_size >= 1`, got <N>. ``                                                                                                                                                                                                                                                                          | Set an integer `per_device_batch_size >= 1` (`1`, `2`, `4`, etc.).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `RESOURCE_EXHAUSTED: ... CompileTimeHbmOom: Ran out of memory on HBM` at launch or after scaling down in `constant_padded` mode                                                                                                                                                                                                                                                              | An HBM OOM terminates the controller and triggers a `JobSet` restart from the GCS checkpoint (clearing in-memory snapshots; `gcluster` defaults to `maxRestarts: 1` via `--restarts`, so a second OOM permanently fails the `JobSet`). Verify HBM fit at `per_device_batch_size × NUM_SLICES / elastic_min_slice_count` before launch/relaunch; raise `elastic_min_slice_count`, keep `remat_policy=full`, or lower `per_device_batch_size` (and increase `gradient_accumulation_steps` with `use_tunix_gradient_accumulation=false` to keep effective GBS). |
| `Trying to advance a closed iterator`                                                                                                                                                                                                                                                                                                                                                        | Keep `eval_interval=-1` when `colocated_python_data_input=true`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `Timed out waiting for array_metadatas base directory creation ... primary_process=0`                                                                                                                                                                                                                                                                                                        | Keep `colocated_python_checkpointing=false` when running Replica Resize.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Sidecar `AttributeError` / deserialization error on startup                                                                                                                                                                                                                                                                                                                                  | Rebuild `DOCKER_IMAGE` and `SIDECAR_IMAGE` from the exact same MaxText commit and `jax`/`jaxlib` version (Section 1).                                                                                                                                                                                                                                                                                                                                                                                                                                        |

## More information

- [Elastic training with Pathways](run_maxtext_elastic_training.md)
- [Running MaxText via Pathways](run_maxtext_via_pathways.md)
- [Pathways Resilient Training documentation](https://cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/resilient-training)
