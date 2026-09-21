# Running MaxText on the Shared Pathways Service (SPS)

The Shared Pathways Service is a managed Pathways pool: you run from your cloudtop/dev box, the controller runs locally on CPU, and TPU compute runs on a shared, long-lived TPU cluster via the Pathways `proxy` backend. No SSH, no container build, no slice provisioning.

&nbsp;

Use it for **small-scale dev runs and smoke tests** (quick numerical checks, config bring-up, single-slice debugging) without queueing for a dedicated XPK workload. It is not a replacement for large multi-slice training.

&nbsp;

Because compute lands on the remote pool, an SPS run **does not touch the local TPU**, so it is safe to launch alongside a local v4-8 job.

&nbsp;

Verified working on `tpu7x-8` from this box on 2026-06-16 (20-step synthetic smoke test, RC=0; see [§6](#6-what-success-looks-like)). Upstream source guide: `go/sps-internal-user-guide`, tpu7x section `go/sps-tpu7x-userguide`.

&nbsp;

---

## 1\. Available services

Two service instances are already deployed and shared. Do **not** change the service name (`SERVICE_JOBSET_NAME`) or the proxy image; they must match the deployed head pod.

&nbsp;

|  | tpu7x-8 (default) | v5p-8 |
| :---- | :---- | :---- |
| Cluster | `bodaborg-tpu7x-sps` | `auto-v5p-8-bodaborg` |
| Region | `us-central1` | `europe-west4` |
| Project | `cloud-tpu-multipod-dev` | `cloud-tpu-multipod-dev` |
| Service name | `sps-j6080103x15` | `sps8-0512jx10` |
| `tpu_type:topology` | `tpu7x:2x2x1` | `tpuv5:2x2x1` |
| Devices | 8 (4 chips × 2/chip) | 8 |
| Proxy image | `…/pathways/proxy_server:20260608-jax_0.10.1` | `…/pathways/proxy_server:20260512_RC00-jax_0.10.0` |
| GCS scratch | `gs://cloud-pathways-staging` | `gs://akshu-v5e` |

&nbsp;

Proxy image prefix is `us-docker.pkg.dev/cloud-tpu-v2-images`.

&nbsp;

**Capacity note (2026-06-30):** the tpu7x SPS pool was reassigned to a customer; use v5p-8 meanwhile and check the chat space for current tpu7x availability.

&nbsp;

The rest of this guide uses **tpu7x-8**. The reusable runner `scripts/sps_run.sh` is hardcoded to it.

&nbsp;

## 2\. One-time setup

Install the client library into the py3.12 venv (`venv-maxtext`):

&nbsp;

```sh
/home/agagik_google_com/venv-maxtext/bin/pip install --upgrade \
  "git+https://github.com/AI-Hypercomputer/pathways-utils" portpicker google-cloud-monitoring
```

&nbsp;

Notes:

&nbsp;

- `pathwaysutils` is at head and updated often; reinstall if you hit a protocol mismatch. Verified version: `v0.1.10`.  
- Claude Code's auto classifier blocks the `git+https://…` install as untrusted external code. Run it yourself with the `!` prefix (`! <pip command>`) or add a Bash allow rule.  
- The JAX version in your venv can differ from the proxy image; the proxy carries its own JAX (here 0.10.0/0.10.1).

&nbsp;

---

## 3\. Run a workload

The simplest path is the reusable runner; no flags are needed for the smoke test:

&nbsp;

```sh
scripts/sps_run.sh                  # 20-step synthetic smoke test
scripts/sps_run.sh "python3 -m maxtext.trainers.pre_train.train <your config…>"
```

&nbsp;

It bakes in the static service config, the `$USER` fix (§4), `unset JAX_PLATFORMS`, and a post-run cleanup check.

&nbsp;

The raw command it wraps (CLI mode):

&nbsp;

```sh
python3 -m pathwaysutils.experimental.shared_pathways_service.run_workload \
  --cluster=bodaborg-tpu7x-sps \
  --project=cloud-tpu-multipod-dev \
  --region=us-central1 \
  --gcs_bucket=gs://cloud-pathways-staging \
  --pathways_service="sps-j6080103x15-pathways-head-0-0.sps-j6080103x15:29001" \
  --tpu_type="tpu7x:2x2x1" \
  --tpu_count=1 \
  --proxy_server_image=us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:20260608-jax_0.10.1 \
  --collect_service_metrics \
  --command "python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
    base_output_directory=gs://cloud-pathways-staging dataset_path=gs://maxtext-dataset/ \
    per_device_batch_size=1 enable_checkpointing=false remat_policy=full \
    global_parameter_scale=4 steps=20 max_target_length=2048 use_iota_embed=true \
    reuse_example_batch=1 dataset_type=synthetic attention=dot_product \
    enable_single_controller=true"
```

&nbsp;

Required workload flags: **`enable_single_controller=true`** (Pathways single controller). Keep `enable_checkpointing=false` for smoke tests.

&nbsp;

&nbsp;

## 4\. Two musts (or it fails)

1. **`unset JAX_PLATFORMS`**: do not pin `tpu`. The run uses the remote `proxy` backend; pinning the local platform breaks it.  
2. **k8s-safe `$USER`**: the proxy job is named `isc-proxy-${USER}-<rand>`. Our `$USER` is `agagik_google_com`, and the underscores are illegal in k8s names (RFC 1123), so deploy fails with *"a lowercase RFC 1123 subdomain must consist of…"*. Fix: `export USER=agagik` before the run. The runner does this automatically; override with `SPS_USER=<name>`.

&nbsp;

## 5\. Sidecar (colocated Python) for large data / checkpoints

If your run loads big checkpoints or lots of data, the default path streams it through your cloudtop and can OOM or run slow. Use the colocated-Python sidecar to keep data on the TPU VMs. Requires `jax-0.10.0` \+ `python-3.12` locally. Add:

&nbsp;

```
--proxy_options=sidecar:true
```

&nbsp;

and append to the workload command:

&nbsp;

```
colocated_python_checkpointing=true colocated_python_data_input=true
```

&nbsp;

For smoke tests with synthetic data, skip the sidecar.

&nbsp;

---

## 6\. What success looks like

The run streams proxy setup, then MaxText logs. Confirm:

&nbsp;

- `System Information: Jax Backend: Pathways` (running remote, not local).  
- `Num_devices: 8` (the `2x2x1` tpu7x slice: 4 chips × 2 devices/chip).  
- `assignment_time` of a few to 20 s (slice acquired).  
- `completed step: N … loss: …` lines through your `steps`.

&nbsp;

Reference smoke-test result (this box, 2026-06-16): 20 steps, loss collapses 10.87 → 0.003 (expected: `reuse_example_batch=1` overfits one synthetic batch), \~0.41 s/step at \~120–126 TFLOP/s/device, `run_workload exit RC=0`, proxy job auto-deleted.

&nbsp;

Benign log noise to ignore: `cuInit … UNKNOWN ERROR (303)` (JAX probing for an absent GPU), `[NO-OP] mldiagnostics: dependency missing`, and a possible segfault at exit after the job finishes.

&nbsp;

## 7\. Cleanup

On a clean exit the proxy job/pod auto-delete (`Successfully deleted job …`). `scripts/sps_run.sh` verifies this and warns if something lingers. To check or clean manually:

&nbsp;

```sh
gcloud container clusters get-credentials bodaborg-tpu7x-sps --region=us-central1 --project=cloud-tpu-multipod-dev
kubectl get jobs | grep "isc-proxy-${USER:-agagik}"
kubectl delete job <your-isc-proxy-job>
```

&nbsp;

Always clean up leftover proxy jobs; they hold a slot on the shared pool.

&nbsp;

## 8\. Troubleshooting

| Symptom | Cause / fix |
| :---- | :---- |
| `Invalid value … RFC 1123 subdomain` on proxy deploy | `$USER` has underscores. `export USER=agagik` (§4). |
| `Backend 'proxy' is not in the list of known backends` | Add `import pathwaysutils; pathwaysutils.initialize()` to your workload's main, or use `run_workload` (handles it). Also check `JAX_PLATFORMS` isn't pinned. |
| `JAX Program Stuck` / "Waiting to acquire placements" | Slice is busy; another user holds it. The run starts when freed. |
| `Connection to IFRT proxy server was terminated: GrpcClientSession: writes no longer allowed` (run dies mid-flight, esp. under concurrent bursts) | Client connect deadline lapsed while the proxy pod was slow to come up. Upstream `pathwaysutils/proxy_backend.py` passes no timeout. Patch it to read `connection_timeout_in_seconds` (int) from `PATHWAYS_PROXY_CONN_TIMEOUT_S` (default 1200s); the sweep harness exports it. **Lost on `pathwaysutils` reinstall; re-apply.** |
| `FAILED_PRECONDITION (Protocol Mismatch)` | Proxy image ≠ deployed head image. Reinstall `pathwaysutils` at head and use the proxy image from §1. |
| `UNAVAILABLE: Socket closed` at the end | Expected: the proxy socket closes after the job completes. |
| `ImportError: cannot import name 'timegm' from 'calendar'` | Don't run inside a Fig workspace (`google3` calendar conflict). |
| OOM on the client | Controller data/ckpt loading is local. Use synthetic/less data, or the sidecar (§5). |
| `ALREADY_EXISTS: Another profiling session active` | Only one profiling session per slice at a time. |

&nbsp;

For deeper proxy debugging, the run prints a Cloud Logging link (`console.cloud.google.com/logs/…isc-proxy-…`).

&nbsp;

## 9\. Caveats

- **Subslicing**: you can request a subslice (e.g. `v5p-8` on a `v5p-16`), but the failure domain is the whole physical slice; if any host fails, every client on the siblings restarts.  
- **Shared capacity**: automated tests and other users share the pool; a run can wait for placement. Keep dev runs small.  
- **No image upgrade on the fly**: a new Pathways image needs the service redeployed (rare; the service owner handles it).

&nbsp;

Support: chat space *Shared Pathways Service Users*; bug hotlist `b/7726698`.

## 10\. Feedback

Collected from our June–July 2026 usage (mostly tpu7x-8, some v5p-8).

&nbsp;

**What worked well.** The core promise held: no SSH, no image build, controller local, and a smoke test reached RC=0 quickly. Runs don't touch the local TPU, subslicing is useful, and the sponge-tee tip is a good default.

&nbsp;

**Issues, in priority order:**

&nbsp;

1. **`$USER` with underscores breaks deploy (RFC 1123).** Proxy jobs are named `isc-proxy-${USER}-…`; on GCE VMs the default user is `<ldap>_google_com`, so every such user hits a cryptic "lowercase RFC 1123 subdomain" failure. Fix in `pathwaysutils`: sanitize the name (lowercase, `_` to `-`, truncate) instead of making each user discover `export USER=<ldap>`.  
2. **No client connect timeout, so runs die mid-flight.** `proxy_backend.py` passes no timeout to the gRPC session; when the proxy pod is slow to come up (concurrent bursts), the run dies with `GrpcClientSession: writes no longer allowed`. We patched it locally to read `PATHWAYS_PROXY_CONN_TIMEOUT_S` (default 1200s), but the patch is lost on every head reinstall, which the guide itself recommends daily. Please upstream a configurable timeout.  
3. **Stale proxy jobs silently hold slots.** When a run dies (see item 2, or a killed terminal), the `isc-proxy` job lingers and holds placement on the shared pool; other users just see "Waiting to acquire placements" with no owner information. Suggest `ttlSecondsAfterFinished` (or client-liveness GC) on proxy jobs, plus some "who holds the slices" visibility.  
4. **Placement waits are opaque.** No queue position, ETA, or holder shown; diagnosing requires knowing the pod-naming convention. A status subcommand or richer `run_workload` output would remove most of the guesswork.  
5. **Install-at-head plus pinned images causes protocol-mismatch roulette.** Daily head installs of `pathwaysutils` against fixed proxy/server images periodically produce `FAILED_PRECONDITION (Protocol Mismatch)` with no hint of which combination is valid. A compatibility check in `run_workload` (print the expected image for the deployed head) or a version matrix in the guide would fix it.  
6. **`JAX_PLATFORMS=tpu` pinned locally breaks the proxy backend confusingly.** Common on dev boxes; `run_workload` could detect and warn or unset.  
7. **Capacity changes travel by word of mouth.** The tpu7x pool went to a customer at end of June while the guide still listed it; a pinned status line (chat or the go/ page) for pool availability would save failed runs.

&nbsp;