# Checkpoint-Load Benchmark v2: Design & Implementation Plan

> **Branch:** `docs/checkpoint-load-benchmark` (rebased onto `origin/main` `5b6773c9b`)
> **Tracking Sheet:** `mesa-checkpoint-load-benchmark-v2 (safetensors_dynamic vs to_maxtext)` — ID [`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`](https://docs.google.com/spreadsheets/d/1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I/edit)
> **GCS Root:** `gs://mesa-maxtext/mesa-bench-v2/` (code bundle under `gs://mesa-maxtext/mesa-bench-v2/code/`)

**Goal:** Measure where every second goes in getting a HuggingFace checkpoint loaded into MaxText on TPU across **v5e**, **v6e**, and **v7x**, comparing `safetensors_dynamic` with `to_maxtext`. Record full per-host stage timings, memory, software/git metadata, and publish everything in the v2 Google Sheet.

---

## 1. Decided Principles & Constraints

| Topic | Decision |
| :--- | :--- |
| Code under test | Every experiment runs `docs/checkpoint-load-benchmark`, packaged via `scripts/build_code_tarball.py` from a clean, pushed commit with `CODE_VERSION.json`. No runtime string patches on `src/`. |
| Load device | Every checkpoint load (`to_maxtext` Step 3 and `safetensors_dynamic`) runs on **TPU**, including `qwen3-0.6b`, and passes `check_tpu_backend` (fails with `failure_category: "NOT-TPU"` before reading weights if CPU fallback or wrong device count occurs). Only `to_maxtext` Step 2 runs on CPU. |
| Step 1 | Every model gets a measured cold `HF -> GCS` download in every region it runs in (no synthetic DeepSeek files or `0.00 (local stage)` placeholders). |
| Resource naming | Every newly created resource across Kubernetes, GCS, and Drive starts with **`mesa-`** (`app: mesa-bench-v2`, JobSets `mesa-smoke-...` / `mesa-...`, KSA `mesa-bench`, secret `mesa-hf-token`, GCS prefix `gs://mesa-maxtext/mesa-bench-v2/`). Existing `mesakh_tpu7x_bench/` resources stay untouched. |
| Report scope | Show every number produced by the end-to-end process without assuming customer/reader priorities. |

---

## 2. Current Live Status (as of 2026-11-12)

### Completed (Committed & Pushed)
- **Phase 0 (T1–T5)**:
  - Fixed sheet markdown table separator and stale `README.md` / `handoff.md` docs.
  - Persisted `worker_upload_max_s`, made `sitecustomize` once-per-process, removed broken legacy shims, fixed `global_devices` docs.
  - Added `scripts/build_code_tarball.py` (`CODE_VERSION.json`, dirty/unpushed guard, `code_version` written into every phase JSON).
  - Updated `bootstrap_tpu7x.py`, `scripts/tpu7x_task_queue.py`, `scripts/run_tpu7x_cluster.sh` so pods extract only `/tmp/maxtext` from the pushed tarball and keep the 16-worker download cap without editing `src/`.
- **Smoke Tests & TPU Verification (S0–S4)**:
  - **S0 Preflight**: Verified Kueue, quotas, image pulls (`yaoyuchen-runner:ds41-2026-10-22-16-03-48`), HF reachability, and GCS access across all 3 clusters; created v5e KSA `mesa-bench` with `roles/storage.objectAdmin` + `roles/storage.legacyBucketReader` on `gs://mesa-maxtext`.
  - **S1 Bundle Upload**: Initial bundle pushed at `82cc1c701`, then rebuilt/uploaded from `c64b44244` (with multi-host gRPC fix) and `site_packages_overlay.tar.gz` to `gs://mesa-maxtext/mesa-bench-v2/code/`.
  - **S2 Code**: Added `bench/platforms.py` (`v7x`, `v6e`, `v5e`), `scripts/make_smoke_jobset.py`, `scripts/run_smoke.sh`, and mandatory TPU backend/device-count verification (`check_tpu_backend`) in `benchmark_ckpt_load_time.py`.
  - **Multi-host `v5e` `SliceBuilder` gRPC fix (`c64b44244`)**: First `v5e` `16x16` smoke timed out in 64-host `SliceBuilder` ALTS gRPC handshake between GKE Workload Identity pods. Added `_ensure_tpu_insecure_grpc_init_args` in `init_worker_runtime` (appends `--slicebuilder_use_insecure_grpc=true` when `total_procs > 1` and `--megascale_use_insecure_grpc=true` when `num_slices > 1` if supported by `libtpu.so`; disable via `KEEP_ALTS_GRPC=1`), unbuffered worker streaming (`PYTHONUNBUFFERED=1`), and remote-abort error attribution (`62` unit tests pass).
  - **S3/S4 Smoke Results (`qwen3-0.6b`, `scanned` & `unscanned`, both pipelines, 0 failures, all JobSets deleted)**:
    - **v7x** (`2x2x1`, `1` host, `8` `TPU7x` devices @ `82cc1c701`, `gs://mesa-maxtext/mesa-bench-v2/smoke/run_smoke-v7x/`): cold Step 1 `17.64s` (`local 15.65s`); `scanned` Step 2 `17.16s`, Step 3 `2.53s`, `safetensors_dynamic` `15.34s` (Columns e2e `52.99s` vs `32.99s`, `1.61x`; CSV `37.34s` vs `32.99s`); `unscanned` Step 2 `63.06s`, Step 3 `3.14s`, `safetensors_dynamic` `7.04s` (Columns e2e `99.49s` vs `24.68s`, `4.03x`; CSV `83.84s` vs `24.68s`).
    - **v6e** (`2x4`, `1` host, `8` `TPU v6 lite` devices @ `82cc1c701`, `gs://mesa-maxtext/mesa-bench-v2/smoke/run_smoke-v6e/`): cold Step 1 `102.24s` (`local 83.06s`); `scanned` Step 2 `79.13s`, Step 3 `23.66s`, `safetensors_dynamic` `30.00s` (Columns e2e `288.09s` vs `132.24s`, `2.18x`; CSV `205.03s` vs `132.24s`); `unscanned` Step 2 `237.41s`, Step 3 `33.49s`, `safetensors_dynamic` `23.19s` (Columns e2e `456.20s` vs `125.43s`, `3.64x`; CSV `373.13s` vs `125.43s`).
    - **v5e** (`16x16`, `64` hosts, `256` `TPU v5 lite` devices @ `c64b44244`, `gs://mesa-maxtext/mesa-bench-v2/smoke/run_smoke-v5e-r2/`): cold Step 1 `81.96s` (`local 67.32s`); `scanned` Step 2 `64.82s`, Step 3 `23.64s`, `safetensors_dynamic` `54.78s` (Columns e2e `237.74s` vs `136.74s`, `1.74x`; CSV `170.42s` vs `136.74s`); `unscanned` Step 2 `161.60s`, Step 3 `21.53s`, `safetensors_dynamic` `45.23s` (Columns e2e `332.40s` vs `127.18s`, `2.61x`; CSV `265.09s` vs `127.18s`). *(Note: remote host 1 raised `-6` during coordinator teardown after writing valid JSONs on `safetensors_dynamic`.)*
- **Google Sheet v2**: Created `1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I` (`11` tabs), marked `S3` & `S4` `Done`, and populated `Summary` with the `12` Tier S rows (`12/20` Tier S = `60.0%`; `12/352` overall = `3.4%`).

### Verified Cluster Facts (`bench/platforms.py`)

| Platform | Cluster & Context | Region | Queue / TAS | Pod / Placement Specifics | Topologies |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **v7x** | `bodaborg-tpu7x-nap`<br>`gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap` | `us-central1` | `default` / TAS `True` (`podset-slice-size`) | Spot selectors + `reservation-name DoesNotExist` affinity, `gitimg` pull secret, 4 chips/host × 2 devices = 8/host, 960 GiB RAM, 180 min limit | `2x2x1`, `2x2x2`, `2x2x4`, `2x4x4` |
| **v6e** | `bodaborg-v6e-nap`<br>`gke_cloud-tpu-shared-capacity_southamerica-west1_bodaborg-v6e-nap` | `southamerica-west1` | `default` / TAS `False` | `priorityClassName: medium`, NO spot selectors/affinity, NO TAS annotations, node SA pulls image, `2x4` = 8 chips (`ct6e-standard-8t`, 1,440 GiB) or `2x2` = 4 chips (`ct6e-standard-4t`, 720 GiB), 1 device/chip, 180 min limit | `2x4`, `2x2`, `4x4`, `8x8`, `16x16` |
| **v5e** | `v5e-256-bodaborg-europe-west4`<br>`gke_tpu-prod-env-multipod_europe-west4_v5e-256-bodaborg-europe-west4` | `europe-west4` | `None` (TAS `gke-v5e-tas-2` broken) | Omit `kueue.x-k8s.io/queue-name`, set `serviceAccountName: mesa-bench`, 4 chips/host × 1 device = 4/host, 192 GiB RAM, no time limit | `16x16` (64 hosts) |

---

## 3. Model Matrix (21 Models, 352 Planned Benchmark Rows)

Execution tier order per platform: **S** (smoke) -> **A** (core) -> **C** (>200B where they fit) -> **B** (rest of 16-model suite).

- **Tier S (smoke):** `qwen3-0.6b` (v7x `1/2 x 2x2x1`, v6e `1/2 x 2x4`, v5e `1 x 16x16`)
- **Tier A (core across all 3 platforms):** `llama3.1-8b`, `qwen3-30b-a3b`, `llama3.1-70b`, `gpt-oss-120b`, `qwen3-235b-a22b`
- **Tier B (v7x & v6e only, 8–93 GB):** `gemma2-2b` (`F32` HF), `qwen3-4b`, `gemma3-4b`, `qwen3-8b`, `qwen3-14b`, `qwen3-32b`, `deepseek2-16b`, `gpt-oss-20b`, `mixtral-8x7b`
- **Tier C (>200B):** `deepseek4-284b` (FP8+FP4 -> BF16), `qwen3.5-397b-a17b`, `qwen3-480b-a35b`, `deepseek3-671b` (FP8 -> BF16), `deepseek3.2-671b` (FP8 -> BF16), `deepseek4-1600b` (FP8+FP4 -> BF16)

---

## 4. Remaining Tasks (Backlog to Pick Up)

### Immediate Next (Start Here)
- **T0.3–T0.6 Cluster & Storage Prereqs**:
  - **T0.3**: Create regional buckets `gs://mesa-maxtext-southamerica-west1` and `gs://mesa-maxtext-europe-west4` with Workload Identity permissions so v6e and v5e loads do not read across continents from the US multi-region bucket.
  - **T0.4**: Create `mesa-hf-token` secret on v7x/v6e/v5e for gated HuggingFace repos (`meta-llama/*`, `google/*`).
  - **T0.5 / T0.6**: Confirm budget/TPU chip-hours & check v6e `8x8`/`16x16` + v5e 2x`16x16` capacity constraints.

### Phase 1: Schema, Stage Recorder, Metadata & Step 1 Download Job (Pure Python, Test-First)
- **T1.1 `bench/schema_v2.py`**: Canonical v2 column definitions (`step1_*`, `step1b_deepseek_*`, `setup_*`, `safetensors_dynamic_*`, `to_maxtext_*`, `memory_*`, `speedup_*`), record validation, status/failure enums (`OK`, `FAILED`, `NOT RUN`, `BLOCKED`, `SKIPPED`, `NOT-TPU`, `OOM-HBM`, `OOM-HOST`, `TIMEOUT`, `PREEMPTED`, `ERROR`).
- **T1.2 `bench/stages.py`**: `stage()`, `wrap()`, `jax.monitoring` listeners, barrier timer, stage-tagged RAM/HBM sampler, per-host `BENCH_STAGES_JSON`, min/median/max/slowest-host merger.
- **T1.3 `bench/legacy_v1.py`**: Adapter mapping the 259 old v7x result JSONs (`gs://mesa-maxtext/mesakh_tpu7x_bench/...`) to v2 schema rows with `LOCAL` / pre-rebase annotations.
- **T1.4 `bench/metadata.py`**: Container/package versions, `CODE_VERSION.json`, downward-API pod/node metadata, safetensors header dtype counter.
- **T1.5 `configs/models_v2.json`**: Full 21-model & topology matrix (§3) consumed by manifest/queue generators.
- **T1.6 `bench/step1_download_job.py`**: Standalone CPU-node cold `HF -> GCS` downloader + DeepSeek Step 1b FP8/FP4 -> BF16 dequantization wrapper (`deepseek_dequantize.py`).

### Phase 2: Harness Instrumentation (Validated on v7x)
- **T2.1–T2.6**: Instrument `setup_*`, `safetensors_dynamic_*`, `to_maxtext_step{1,2,3}_*`, per-host RAM/RSS/HBM (`memory_stats()`), rich FAILED/SKIPPED/NOT_RUN records, and full metadata.
- **T2.7**: Validate on v7x (`qwen3-0.6b`, `llama3.1-8b`, 1 & 2 slices): unattributed time <= 2% (or <= 5 s), totals within ±5%, per-host stages present.
- **T2.8–T2.9**: Tier A parameter checksum correctness comparison between pipelines; fresh-pod / empty JAX compilation cache isolation.

### Phase 3: Full Multi-Platform Integration (`v7x`, `v6e`, `v5e`)
- **T3.1–T3.4**: Expand `bench/platforms.py` (regional buckets, RAM margins, concurrency caps, OOM floors); replace legacy `generate_tpu7x_manifests.py` & `scripts/run_tpu7x_cluster.sh` hardcoding with profile-driven JobSet + CPU Step-1 Job generation (`mesa-` named) and `hosts_per_slice` awareness in `scripts/tpu7x_task_queue.py`.
- **T3.5**: Multi-platform Tier S smoke validation with full v2 instrumentation.

### Phase 4: Sheet v2 Builder & Population
- **T4.1–T4.4 `scripts/build_sheet_v2.py`**: Generate CSVs + `gsheets` batch JSONs from v2 JSONs + `legacy_v1` rows, populate all 11 tabs of `1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`, populate `README` and Tier/Platform summaries, keeping formatting rules (numbers in numeric columns, legend codes, per-phase log links).

### Phase 5: Execute 352-Row Matrix & Final Report
- Run regional Step 1 downloads -> DeepSeek Step 1b -> TPU loads (`S` -> `A` -> `C` -> `B`), sync sheet after each tier, update reports and handoff.
