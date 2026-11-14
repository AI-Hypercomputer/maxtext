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

## 2. Final Execution Status (Completed 2026-11-14)

### All Phases `0–5` (`T0.1–T5.3`) Completed & Validated (`100.0%`)
- **Phase 0 (`T0.1–T0.8`)**:
  - Added [`scripts/build_code_tarball.py`](../scripts/build_code_tarball.py) (`CODE_VERSION.json`, dirty/unpushed guard, `code_version` recorded in every phase JSON).
  - Updated [`bootstrap_tpu7x.py`](../bootstrap_tpu7x.py) and [`scripts/tpu7x_task_queue.py`](../scripts/tpu7x_task_queue.py) so pods extract only `/tmp/maxtext` from the pushed tarball with clean `/ramdisk` teardown between tasks.
  - Verified regional buckets `gs://mesa-maxtext-{us-central1,southamerica-west1,europe-west4}/mesa-bench-v2/`, `mesa-hf-token` secrets, Kueue queues, and KSA `mesa-bench`.
- **Phase 1 (`T1.1–T1.6`)**:
  - Implemented and unit-tested [`bench/schema_v2.py`](../bench/schema_v2.py), [`bench/stages.py`](../bench/stages.py), [`bench/legacy_v1.py`](../bench/legacy_v1.py), [`bench/metadata.py`](../bench/metadata.py), [`configs/models_v2.json`](../configs/models_v2.json), and [`bench/step1_download_job.py`](../bench/step1_download_job.py) (`96` unit tests passing, `pylint 10.00/10`).
- **Phase 2 (`T2.1–T2.10`)**:
  - Instrumented per-stage timings (`setup_*`, `safetensors_dynamic_step{1,2}_*`, `to_maxtext_step{1,2,3}_*`), per-host RAM/RSS & TPU HBM milestones, `check_tpu_backend` enforcement, synchronized multi-host teardown (`worker_teardown_results_written`), Tier A parameter checksum verification, and cold JAX compilation cache isolation (`jax_compilation_cache_state = empty`).
- **Phase 3 (`T3.1–T3.5`)**:
  - Completed [`bench/platforms.py`](../bench/platforms.py), [`bench/manifests_v2.py`](../bench/manifests_v2.py), [`scripts/generate_v2_manifests.py`](../scripts/generate_v2_manifests.py), [`scripts/run_v2_cluster.sh`](../scripts/run_v2_cluster.sh), and [`scripts/run_v2_orchestrator.py`](../scripts/run_v2_orchestrator.py) across `v7x` (`2x2x1`..`2x4x4`), `v6e` (`2x4`..`16x16`), and `v5e` (`16x16`, 64 hosts).
- **Phase 4 & Phase 5 (`T4.1–T5.3`) — Full 352-Row Matrix Executed & Validated**:
  - **Cold Step 1 (`HF -> GCS`) & DeepSeek Step 1b (`FP8/FP4 -> BF16`):** **`54 / 54` (`100%`)** completed cold across `us-central1` (`21`), `southamerica-west1` (`21`), and `europe-west4` (`12`), plus all 12 regional DeepSeek Step 1b BF16 conversions/transfers.
  - **TPU Benchmark Matrix (`136 / 136` queue tasks = `352 / 352` planned `v2` `Summary` rows, `0` `ERROR`, `0` `TIMEOUT`):**
    - **`v7x` (`us-central1`):** `61 / 61` tasks (`164 / 164` rows: `142` `OK`, `22` genuine `OOM-HOST` in single-host CPU `to_maxtext` Step 2, `0` `ERROR`, `0` `TIMEOUT`, `3/3` Tier A repetitions).
    - **`v6e` (`southamerica-west1`):** `53 / 53` tasks (`140 / 140` rows: `127` `OK`, `13` genuine `OOM-HOST`, `0` `ERROR`, `0` `TIMEOUT`, `3/3` Tier A repetitions).
    - **`v5e` (`europe-west4`):** `22 / 22` tasks (`48 / 48` rows: `22` `OK`, `24` genuine `OOM-HOST` [`18` `to_maxtext` S2 + `6` unfused Qwen3 MoE `safetensors_dynamic` on `192 GiB` hosts], `1` genuine `OOM-HBM` on `deepseek4-1600b` `2x16x16`, `1` `SKIPPED`, `0` `ERROR`, `0` `TIMEOUT`, `3/3` Tier A repetitions).
  - **Google Sheet v2 & Summary Report:** Synced all 11 tabs of [`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`](https://docs.google.com/spreadsheets/d/1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I/edit) via [`scripts/build_sheet_v2.py`](../scripts/build_sheet_v2.py) with clickable `gpaste` logs on every executed row, hidden internal `Experiment Tier` column on `Summary`/`Runs`, `README` findings & limits block, and executive summary in [`reports/benchmark_v2_summary.md`](../reports/benchmark_v2_summary.md).

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

## 4. Completed Deliverables (`T0.3–T5.3`)

### Cluster & Storage Prerequisites (`Done`)
- **T0.3–T0.6**: Regional buckets `gs://mesa-maxtext-{us-central1,southamerica-west1,europe-west4}/mesa-bench-v2/`, `mesa-hf-token` secrets across all 3 clusters, KSA `mesa-bench` on `v5e`, and topology capacity (`v6e` up to `16x16`, `v5e` up to `2×16x16`, `v7x` up to `1×2x4x4` / `2×2x2x2`) verified and cleaned up after completion.

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
