---
name: gcluster-tpu-runner
description: >-
  Validates, packages, and submits MaxText workloads on GKE TPU clusters with
  static syntax/config checks, TPU AOT HBM compilation pre-flight, topology-aware
  JobSet generation, and stage-scoped Cloud Logging verification.
---

# MaxText GKE TPU Workload Pre-Flight & Runner

Submitting multi-slice TPU jobs without local validation wastes cluster hours on
preventable syntax errors, missing container dependencies, `.dockerignore` leaks,
or HBM out-of-memory (OOM) failures. Run the 4-stage pipeline below before every
cluster submission.

| Script | Purpose |
| :--- | :--- |
| [`scripts/preflight_check.py`](scripts/preflight_check.py) | AST syntax check, YAML validation, `.dockerignore` audit, and optional Docker import check |
| [`scripts/aot_preflight.py`](scripts/aot_preflight.py) | CPU-only TPU AOT compilation (`forward` or `train`) with topology-aware HBM limit verification |
| [`scripts/submit_gsc_jobset.py`](scripts/submit_gsc_jobset.py) | Generates and submits Kueue dynamic-slicing `JobSet` specs for any 2D/3D TPU topology |
| [`scripts/submit_workload.py`](scripts/submit_workload.py) | Wraps `gcluster job submit` with pre-flight validation and `/deps` overlay setup |
| [`scripts/make_stage_log_links.py`](scripts/make_stage_log_links.py) | Builds stage-scoped Cloud Logging URLs and verifies log slices via `gcloud logging read` |

---

## Stage 1: Static & Container Pre-Flight (`preflight_check.py`)

Verify Python syntax (`ast.parse`), YAML configs, `.dockerignore` exclusions
(preventing multi-GB `.git` or `venv` uploads during image builds), and
container imports before packaging:

```bash
python3 .agents/skills/gcluster-tpu-runner/scripts/preflight_check.py \
  --maxtext-root . \
  --docker-image gcr.io/<project>/<base-image>:latest \
  --check-imports maxtext.train,maxtext.checkpoint_conversion.to_maxtext
```

---

## Stage 2: TPU AOT Compilation & HBM Pre-Flight (`aot_preflight.py`)

CPU unit tests do not compile Pallas/Mosaic TPU kernels or check device HBM
allocation. Run AOT compilation on CPU targeting the exact TPU topology (`v4`,
`v5e`, `v5p`, `v6e`, `tpu7x`) and parallelism mesh before submitting to the
cluster:

```bash
# Training step AOT check (wraps maxtext.train_compile)
python3 .agents/skills/gcluster-tpu-runner/scripts/aot_preflight.py \
  --mode train \
  --config src/maxtext/configs/base.yml \
  --model-name llama3.1-8b \
  --topology v5p-128 \
  --num-slices 1 \
  --extra-args "per_device_batch_size=1 max_target_length=4096 remat_policy=full"

# Forward-only AOT check (for logit verification or inference)
python3 .agents/skills/gcluster-tpu-runner/scripts/aot_preflight.py \
  --mode forward \
  --config src/maxtext/configs/base.yml \
  --model-name llama3.1-8b \
  --topology v6e-16 \
  --num-slices 1
```

`aot_preflight.py` automatically infers the per-device HBM ceiling from the
topology prefix (`v4`: 31.7 GiB, `v5e`: 15.7 GiB, `v5p`: 94.7 GiB, `v6e`: 31.2
GiB, `tpu7x`: 191.5 GiB; override with `--hbm-limit-gib`) and exits non-zero if
peak compiled memory exceeds the limit.

---

## Stage 3: Workload Submission (`submit_gsc_jobset.py` / `submit_workload.py`)

### Option A: Kueue Dynamic-Slicing JobSet (`submit_gsc_jobset.py`)

Computes `nodes_per_slice` automatically from any 2D (`AxB`) or 3D (`AxBxC`)
TPU topology string:

```bash
python3 .agents/skills/gcluster-tpu-runner/scripts/submit_gsc_jobset.py \
  --job-name my-e2e-run \
  --context gke_<project>_<region>_<cluster> \
  --image gcr.io/<project>/<image>:latest \
  --topology 4x4x4 \
  --num-slices 1 \
  --accelerator tpu-v5p-slice \
  --command "bash tests/end_to_end/tpu/run_e2e.sh" \
  --dry-run
```

### Option B: `gcluster` CLI Wrapper (`submit_workload.py`)

When submitting against a base image where `/deps` is mounted from a lightweight
code-overlay layer, `submit_workload.py` automatically prepends
`cp -a /src/. /deps/src/ && cd /deps/src` so the container executes the overlaid
source tree rather than stale files in `/deps`.

---

## Stage 4: Stage-Scoped Cloud Logging Verification (`make_stage_log_links.py`)

For multi-stage E2E runs (`forward_logit`, `pre_train`, `decode`), generate a
Cloud Logging URL scoped to each stage's `[start, end]` UTC window on worker 0
and verify that `gcloud logging read` returns the expected anchor lines:

```bash
python3 .agents/skills/gcluster-tpu-runner/scripts/make_stage_log_links.py \
  --project <gcp-project> \
  --cluster <cluster-name> \
  --pod <job-name>-main-job-0-0-xxxxx \
  --stage "pre_train|2026-10-12T06:51:00Z|2026-10-12T07:10:00Z|completed step"
```
