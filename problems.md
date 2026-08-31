# Issues and Friction Points Encountered During Raiden Sync E2E Verification

This document catalogs the issues, potential customer friction points, and recommendations identified while testing the end-to-end integration between `tunix` (`mohit/raiden-maxtext-rlvllm`), `maxtext` (`igorts/merge_anisha_mohit`), and `tpu-inference` (`mohit/rl-sampler`) on GKE cluster `auto-v5p-8-bodaborg`.

---

## 1. `k8s_launcher.sh` Hardcodes `python` Instead of `python3`

* **File**: `tunix/experimental/examples/math_gsm8k_dist/k8s_launcher.sh` (lines 119, 150)
* **Symptom**:
  Executing `k8s_launcher.sh` on standard Linux host environments where `python` is not aliased or symlinked to `python3` fails immediately with:
  ```
  k8s_launcher.sh: line 119: python: command not found
  error: no objects passed to apply
  ```
* **Root Cause**:
  The launcher invokes `python tunix/experimental/distributed/deployment/yaml_generator.py ...` directly instead of using `python3` or the active virtual environment Python interpreter.
* **Impact**:
  Any user or CI runner attempting to launch the demo without an active `python` binary in `$PATH` fails at step 1.
* **Recommendation**:
  Update `k8s_launcher.sh` to use `${PYTHON:-python3}`:
  ```bash
  ${PYTHON:-python3} tunix/experimental/distributed/deployment/yaml_generator.py ...
  ```

---

## 2. Default `CPU_MACHINE=n2-standard-64` Causes Indefinite Pending Pods

* **File**: `tunix/experimental/examples/math_gsm8k_dist/k8s_launcher.sh` (line 77)
* **Symptom**:
  The orchestrator pod remains stuck in `Pending` with `FailedScheduling`:
  ```
  Warning  FailedScheduling  0/23 nodes are available: 15 node(s) didn't match Pod's node affinity/selector
  ```
* **Root Cause**:
  `k8s_launcher.sh` hardcodes `CPU_MACHINE=n2-standard-64` as default, but standard TPU clusters (like `auto-v5p-8-bodaborg`) often have `n2-standard-16` or `n2d-standard-224` node pools instead.
* **Impact**:
  Users running on existing shared clusters fail to schedule the orchestrator unless they know to manually override `CPU_MACHINE`.
* **Recommendation**:
  Allow automatic detection of available CPU node instance types or default to `n2-standard-16`.

---

## 3. Container `/app` Directory Shadowing Layered Packages

* **File**: `Dockerfile.maxtext` / `Dockerfile` in `tunix`
* **Symptom**:
  When launching with the latest Tunix launcher, the orchestrator crashed on startup:
  ```
  main.py: error: unrecognized arguments: --num_rollout_workers=1
  ```
* **Root Cause**:
  In the base image (`gcr.io/tpu-prod-env-multipod/mohitkhatwani-rl:maxtext-raiden-0824-v3`), `tunix` was installed in editable mode from `/app`. Even if an updated package is copied to `/workspace/tunix`, `sys.path` and `WORKDIR /app` cause Python to load the older `/app/tunix` code.
* **Impact**:
  Users attempting to layer changes on top of the base image without replacing `/app` will run stale code and encounter argument parsing or version mismatch crashes.
* **Recommendation**:
  Document that `COPY tunix /app` and `pip install --no-deps -e /app` must be used when updating Tunix in derived Dockerfiles.

---

## 4. `run_trainer_node.py` Inflexible Micro-Batch Size Assert

* **File**: `tunix/experimental/examples/math_gsm8k_dist/run_trainer_node.py` (line 285)
* **Symptom**:
  Running the trainer with launcher defaults (`TRAIN_MICRO_BATCH_SIZE=1` and `TRAINER_MESH_FSDP=4`) immediately crashes with:
  ```
  ValueError: --train_micro_batch_size=1 must be a multiple of --mesh_fsdp=4; MaxText shards the batch dimension across it.
  ```
* **Root Cause**:
  `run_trainer_node.py` enforces `args.train_micro_batch_size % args.mesh_fsdp == 0`. However, `MaxTextTrainingEngine` already implements dynamic dimension replication in `_batch_data_shardings` to support `train_micro_batch_size < mesh_fsdp` (e.g. sequence-packed inputs).
* **Impact**:
  Users following default configurations are blocked unless they increase `TRAIN_MICRO_BATCH_SIZE` to match `mesh_fsdp`.
* **Recommendation**:
  Relax or remove the assertion in `run_trainer_node.py` when using `TRAINER_BACKEND=maxtext`.

---

## 5. Multi-Chunk Raiden Synchronizers Triggering Socket Connection Resets

* **File**: `src/maxtext/training_engine/maxtext_engine.py` (lines 1064-1085)
* **Symptom**:
  When `RAIDEN_WEIGHT_SYNC_CHUNKS=4` was used on a standard TPU slice, weight transfer failed during step 1 with:
  ```
  RuntimeError: Raiden remote native execution failed: tcp socket recv failed: errno=104 (Connection reset by peer)
  ```
* **Root Cause**:
  When 4 separate `RaidenSynchronizer` instances execute `PushWeightsResharded` concurrently to a single rollout listener, the concurrent incoming TCP streams cause socket connection drops on the receiver.
* **Impact**:
  Weight sync fails during RL iterations on direct TPU setups.
* **Recommendation**:
  Default `RAIDEN_WEIGHT_SYNC_CHUNKS` to `1` (and assign `worker_index=jax.process_index()`), reserving multi-chunk splitting only for large-model Pathways environments where host memory pressure requires chunked staging.

---

## 6. Outdated MaxText Commit Pinned in `special_requirements.txt`

* **File**: `tunix/requirements/special_requirements.txt`
* **Symptom**:
  Building `Dockerfile.maxtext` from scratch pulls an outdated MaxText commit before recent critical fixes (such as `None` leaf dynamic filtering, multi-host `process_index` assignment, `_split_into_chunks` dictionary flattening, and `release_weight_sync` cleanup).
* **Impact**:
  Users following the build instructions from scratch will produce an image that contains known bugs.
* **Recommendation**:
  Update `special_requirements.txt` in the Tunix branch to point to the latest merged commit on MaxText.

---

## 7. Unauthenticated Hugging Face Hub Rate Limiting Warning

* **File**: `tunix/experimental/examples/math_gsm8k_dist/run_gsm8k_dist_grpo.py` (Orchestrator node)
* **Symptom**:
  The orchestrator and trainer nodes log warnings on every startup:
  ```
  Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
  ```
* **Impact**:
  In multi-pod environments or high-frequency CI testing, unauthenticated requests can hit Hugging Face IP rate limits (HTTP 429), causing startup to fail.
* **Recommendation**:
  Allow passing `HF_TOKEN` via environment variable or Kubernetes secret into the orchestrator and trainer containers.

---

## 8. `k8s_launcher.sh` Empty `--reward_mode=` Argument Causes Argparse Failure

* **File**: `tunix/experimental/examples/math_gsm8k_dist/k8s_launcher.sh` (line 141)
* **Symptom**:
  Orchestrator crashes immediately upon container start with:
  ```
  main.py: error: argument --reward_mode: invalid choice: '' (choose from synthetic, exact)
  ```
* **Root Cause**:
  `k8s_launcher.sh` unconditionally passes `--reward_mode=${REWARD_MODE}` into the orchestrator startup command. When `$REWARD_MODE` is unset or empty, bash passes `--reward_mode=""`, which `argparse` rejects because empty string is not in `('synthetic', 'exact')`.
* **Impact**:
  Workloads launched without explicitly exporting `REWARD_MODE=synthetic` or `REWARD_MODE=exact` fail at startup.
* **Recommendation**:
  Use parameter expansion to only pass the flag when non-empty:
  ```bash
  ${REWARD_MODE:+--reward_mode=${REWARD_MODE}}
  ```

---

## 9. `special_requirements.txt` Git Shallow Clone Fails on Unadvertised Commit SHAs

* **File**: `tunix/requirements/special_requirements.txt`
* **Symptom**:
  Building Docker images with `uv pip install` fails when pinning a raw commit SHA from an external branch:
  ```
  error: Server does not allow request for unadvertised object <COMMIT_SHA>
  ```
* **Root Cause**:
  GitHub and standard Git servers disallow shallow fetching (`git fetch --depth 1`) of arbitrary commit SHAs that are not the head of an advertised branch or tag.
* **Impact**:
  Automated builds and developers trying to pin specific experimental commits fail during dependency installation.
* **Recommendation**:
  Pin advertised branch names (e.g. `@branch_name`) or tags in `special_requirements.txt` instead of raw unmerged commit SHAs.

---

## 10. Docker BuildKit OCI Manifest Incompatibility with Container Registry (GCR)

* **File**: `Dockerfile.maxtext` / build pipelines
* **Symptom**:
  Image pushes with `docker buildx build --push` succeed and return a digest, but Kubernetes nodes fail to pull the image with:
  ```
  Failed to pull image "gcr.io/...": rpc error: code = NotFound desc = failed to resolve reference "...": not found
  ```
* **Root Cause**:
  BuildKit creates OCI multi-platform manifest lists and provenance attestations by default (`--provenance=mode=max`), which older Container Registry (`gcr.io`) endpoints do not index into their tag catalog.
* **Impact**:
  Pods fail with `ErrImagePull` / `ImagePullBackOff` even though build logs show successful push.
* **Recommendation**:
  Pass `--provenance=false` when building with BuildKit, or load the image locally and push via `docker push`.

---

## 11. Multi-Host FSDP ICI Partitioning on TPU v5p Slices

* **File**: `tunix/experimental/examples/math_gsm8k_dist/run_trainer_node.py` (line 330)
* **Symptom**:
  When scaling the trainer to a 2-host TPU slice (`tpuv5:2x2x2`, 8 chips across 2 hosts = 4 chips/host) with `TRAINER_MESH_FSDP=8`, MaxText initialization fails if `ici_fsdp` exceeds 4.
* **Root Cause**:
  In a multi-host slice, ICI (Inter-Chip Interconnect) mesh axes cannot exceed the per-host device count (4). FSDP=8 must be split into `ici_fsdp=4` and `dcn_fsdp=2` (Data Center Network across hosts).
* **Impact**:
  Distributed multi-host training crashes during mesh construction.
* **Recommendation**:
  Ensure `run_trainer_node.py` dynamically computes:
  ```python
  ici_fsdp = min(args.mesh_fsdp, 4)
  dcn_fsdp = max(1, args.mesh_fsdp // ici_fsdp)
  ```

---

## 12. GKE TPU Slice Fragmentation and Exclusive Topology Requirement

* **File**: `tunix/experimental/distributed/deployment/yamls/jobset.tpu.yaml`, `jobset.pathways.yaml`
* **Symptom**:
  Multi-host TPU workloads (e.g. `2x2x2` with 2 hosts) either fail with:
  ```
  Terminating the libtpu controller process because an anomalous TPUworker process is detected: SLICE_FAILURE_INIT_ERROR
  ```
  or have worker pods stuck in `Pending` with `0/N nodes available: Insufficient google.com/tpu`.
* **Root Cause**:
  By default, standard Kubernetes scheduling does not guarantee that indexed pods of a multi-host TPU slice land on nodes within the *same* physical ICI slice reservation (`cloud.google.com/gke-nodepool`). If pod 0 is placed on node 1 of slice A and pod 1 is placed on node 2 of slice B, inter-chip optical interconnects cannot communicate, causing libtpu to abort immediately on initialization. Furthermore, single-host or incorrectly placed workloads leave physical slices partially occupied (fragmented), blocking whole-slice allocations.
* **Impact**:
  Multi-host distributed training and multi-host rollouts fail unpredictably on shared clusters.
* **Recommendation**:
  1. Add `alpha.jobset.sigs.k8s.io/exclusive-topology: cloud.google.com/gke-nodepool` to the pod template metadata annotations in all multi-host TPU JobSet templates (`jobset.tpu.yaml`, `jobset.pathways.yaml`).
  2. Ensure rollout workers with single-host replicas use `ROLLOUT_TPU_SLICE=tpuv5:2x2x1` with `ROLLOUT_REPLICAS=N` so each replica requests an exact single-host slice.

---

## 13. Multi-Host Trainer Discovery Overwrite Race

* **File**: `tunix/experimental/examples/math_gsm8k_dist/run_trainer_node.py` (line 612), `run_gsm8k_dist_grpo.py`
* **Symptom**:
  In multi-host training setups (e.g. `2x2x2` with 2 hosts), the orchestrator connects to a secondary worker pod (e.g. `proc-0-1`) instead of the primary rank 0 pod (`proc-0-0`), causing weight synchronization and step coordination to hang indefinitely during `Broadcasting initial weights from Trainer to Rollout workers...`.
* **Root Cause**:
  All trainer worker pods in a multi-host JobSet unconditionally called `context.ipc.discovery.register(...)` under the same `service_type: "trainer"` and `worker_id`. If a secondary host pod (`process_index != 0`) contacted discovery first, the orchestrator resolved the trainer endpoint to that secondary host. Since JAX distributed execution and Orbax checkpointing require primary host 0 coordination, driving RPCs into secondary hosts leads to deadlocks.
* **Impact**:
  Multi-host distributed training deadlocks on initial weight broadcast.
* **Recommendation**:
  In `run_trainer_node.py`, restrict `context.ipc.discovery.register` to `if jax.process_index() == 0:`. Secondary hosts participate in JAX distributed execution via JAX coordination on port 8482 without registering as independent gRPC endpoints.

---

## 14. Multi-Controller JAX (McJax) Multi-Host Collective Execution vs Pathways Single-Controller

* **File**: `tunix/experimental/examples/math_gsm8k_dist/run_gsm8k_dist_grpo.py`, `run_trainer_node.py`
* **Symptom**:
  When scaling multi-host training in McJax mode (`jobset.tpu.yaml`), calling trainer RPCs (`prepare_weight_sync`, `train_step`, `compute_logps`) against only one host's gRPC server hangs if operations invoke JAX SPMD/collective routines on distributed arrays without the other hosts concurrently participating.
* **Root Cause**:
  In single-controller Pathways (`jobset.pathways.yaml`), a single client Python process (`proc`) orchestrates TPU computation across remote Pathways workers (`pw-node`). In multi-controller JAX (McJax), each host runs its own independent Python interpreter and SPMD collective operations require concurrent execution across all host processes.
* **Impact**:
  Multi-host McJax training requires either fan-out RPC coordination across all trainer hosts or Pathways single-controller deployment.
* **Recommendation**:
  For distributed multi-host training on GKE, either use the Pathways configuration with `exclusive-topology` enabled on `pw-node` or implement orchestrator fan-out to all participant processes in the multi-host trainer jobset.

---

## 15. Pathways Default GCS Scratch Location Authorization Failure

* **File**: `tunix/experimental/examples/math_gsm8k_dist/k8s_launcher.sh` (line 79), `jobset.pathways.yaml`
* **Symptom**:
  Pathways compilation crashes with:
  ```
  jax.errors.JaxRuntimeError: ABORTED: Compilation service has crashed or shut down while compilation was pending for computation with key: ..., computation name: jit_stage, original error: PERMISSION_DENIED: Permanent error, with a last message of Provided scope(s) are not authorized
  ```
* **Root Cause**:
  `k8s_launcher.sh` defaults `GCS_SCRATCH_LOCATION` to `gs://cloud-pathways-staging/tmp`, a staging bucket that is not accessible to cluster service accounts in external or development projects.
* **Impact**:
  Model compilation and trainer startup crash immediately under Pathways.
* **Recommendation**:
  Set `GCS_SCRATCH_LOCATION` to a project-accessible bucket such as `gs://tunix_maxtext/pathways_tmp` or `gs://tunix-artifacts-dev/tmp`.

---

## 16. Kueue Multi-JobSet Resource Throttling and Suspension

* **File**: `tunix/experimental/distributed/deployment/yamls/jobset.tpu.yaml`, `jobset.pathways.yaml`
* **Symptom**:
  One or more rollout or trainer JobSets stay in `Suspended: true` with events:
  ```
  Normal   SuspendedJobs    jobset is suspended
  Normal   CreatedWorkload  Created Workload: default/jobset-igorts-...
  ```
  while orchestrator logs `Waiting for workers to connect...` until timing out.
* **Root Cause**:
  When multiple independent JobSets (`orch`, `train`, `roll-0`, `roll-1`) are launched simultaneously on a shared cluster managed by Kueue, the total requested TPU quotas across all slices may exceed the available quota or concurrency limits of the LocalQueue/ClusterQueue. Kueue admits some JobSets and suspends others until quota becomes available.
* **Impact**:
  Workloads with multiple rollout replicas hang waiting for all replicas to register.
* **Recommendation**:
  1. Inspect `kubectl get workloads` and `kubectl get clusterqueue` to check pending workloads and available quota.
  2. Reduce `ROLLOUT_REPLICAS` or ensure cluster quota permits all replicas concurrently before launching.







