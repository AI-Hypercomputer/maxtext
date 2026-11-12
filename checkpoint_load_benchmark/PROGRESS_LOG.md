# Checkpoint-Load Benchmark v2 — Live Progress & Team Handoff Log

- **Last Updated (UTC):** `2026-11-12 14:09:21 UTC`
- **Git Branch:** `docs/checkpoint-load-benchmark`
- **Repository Path on Cloudtop:** `/usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3`
- **This Progress File:** `/usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3/checkpoint_load_benchmark/PROGRESS_LOG.md`
- **Handoff Runbook:** `/usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3/checkpoint_load_benchmark/handoff.md`
- **Design & Task Plan:** `/usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3/checkpoint_load_benchmark/plans/benchmark_v2_plan.md`
- **Live Google Sheet v2:** https://docs.google.com/spreadsheets/d/1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I/edit
- **Background Orchestrator Script:** `checkpoint_load_benchmark/scripts/run_v2_orchestrator.py`

## 1. Live Cluster & Experiment Progress Summary

| Platform | Region | Step 1 Done | TPU Queue Tasks Done | Active TPU JobSets | Active CPU Jobs |
|---|---|---:|---:|---|---|
| `v7x` | `us-central1` | **21 / 21** | **6 / 61** | `mesa-q-v7x-2x2x1-1s-w2`, `mesa-q-v7x-2x2x1-2s-w2` | `mesa-step1b-deepseek4-1600b-v2-s1b-v7x` |
| `v6e` | `southamerica-west1` | **21 / 21** | **8 / 53** | `mesa-q-v6e-2x4-2s-w2` | _none_ |
| `v5e` | `europe-west4` | **12 / 12** | **17 / 22** | `mesa-q-v5e-16x16-1s-w2` | _none_ |

## 2. Chat Session Summary & Handoff Guide for Other Engineers

1. **Summary of What Has Been Completed in This Session:**
   - **Code & Tests (`docs/checkpoint-load-benchmark`):** Completed all Phase 0 through Phase 4 deliverables (`bench/schema_v2.py`, `bench/harness_v2.py`, `bench/platforms.py`, `bench/stages.py`, `bench/manifests_v2.py`, `scripts/generate_v2_manifests.py`, `scripts/run_v2_cluster.sh`, `scripts/build_sheet_v2.py`, `scripts/run_v2_orchestrator.py`). Fixed 64-host stage timing aggregation on `v5e` and TPU HBM metric keys on `v5e`/`v6e`/`v7x`. All 94 unit tests pass (`pytest`) and all modular files score `10.00/10` on `pylint`.
   - **Cross-Platform Validation (`T3.5`) & Sheet Sync (`T4.1-T4.3`):** Verified end-to-end on `v5e` (`16x16` = 64 hosts / 256 chips in `europe-west4`), `v6e` (`2x4` and `4x4` in `southamerica-west1`), and `v7x` (`2x2x1` in `us-central1`). Populated Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with all 555 Summary rows (352 planned v2 rows + 203 legacy v1 rows).
   - **Background Execution (`Phase 5`):** To avoid blocking the interactive Jetski chat session while multi-hour HuggingFace downloads and TPU benchmarks run, `scripts/run_v2_orchestrator.py` runs in the background (`nohup` daemon logging to `/tmp/mesa_v2_orchestrator.log`). Every 60 seconds it polls all 3 GKE clusters and regional GCS buckets, cleans up finished K8s Jobs/JobSets, launches the next Step 1 / Step 1b / TPU Queue Worker waves in `S -> A -> C -> B` tier order, syncs Google Sheet v2, and updates this Markdown file.
2. **How Another Engineer Can Inspect or Take Over on Cloudtop:**
   ```bash
   # 1. Go to the repo and verify branch is docs/checkpoint-load-benchmark
   cd /usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3
   git status
   git pull origin docs/checkpoint-load-benchmark

   # 2. Read this live progress file and tail the background orchestrator log
   cat checkpoint_load_benchmark/PROGRESS_LOG.md
   ps -ef | grep run_v2_orchestrator.py
   tail -f /tmp/mesa_v2_orchestrator.log

   # 3. Check live K8s jobs/jobsets/pods across all 3 clusters (v7x, v6e, v5e)
   ./checkpoint_load_benchmark/scripts/run_v2_cluster.sh --platform v7x --mode status
   ./checkpoint_load_benchmark/scripts/run_v2_cluster.sh --platform v6e --mode status
   ./checkpoint_load_benchmark/scripts/run_v2_cluster.sh --platform v5e --mode status
   ```

## 3. Chronological Event Log (Newest Last)

- [2026-11-12 12:40:57 UTC] Background orchestrator started on branch `docs/checkpoint-load-benchmark`.
- [2026-11-12 12:43:29 UTC] Updated `v5e` queue `gs://mesa-maxtext-europe-west4/mesa-bench-v2/queue/v2/tasks.json` with 21/22 staged tasks.
- [2026-11-12 12:43:34 UTC] `v6e` CPU job `mesa-step1-v6e-deepseek4-1600b-v2-s1-v6e` Completed and cleaned up.
- [2026-11-12 12:43:41 UTC] Updated `v6e` queue `gs://mesa-maxtext-southamerica-west1/mesa-bench-v2/queue/v2/tasks.json` with 52/53 staged tasks.
- [2026-11-12 12:43:50 UTC] Updated `v7x` queue `gs://mesa-maxtext-us-central1/mesa-bench-v2/queue/v2/tasks.json` with 60/61 staged tasks.
- [2026-11-12 12:44:14 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 258 mirrored JSONs.
- [2026-11-12 12:48:07 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 266 mirrored JSONs.
- [2026-11-12 12:51:59 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 269 mirrored JSONs.
- [2026-11-12 12:55:54 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 281 mirrored JSONs.
- [2026-11-12 13:00:09 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 294 mirrored JSONs.
- [2026-11-12 13:04:13 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 302 mirrored JSONs.
- [2026-11-12 13:08:32 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 312 mirrored JSONs.
- [2026-11-12 13:12:41 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 316 mirrored JSONs.
- [2026-11-12 13:16:53 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 326 mirrored JSONs.
- [2026-11-12 13:21:05 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 328 mirrored JSONs.
- [2026-11-12 13:25:17 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 338 mirrored JSONs.
- [2026-11-12 13:29:24 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 339 mirrored JSONs.
- [2026-11-12 13:33:33 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 342 mirrored JSONs.
- [2026-11-12 13:37:39 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 349 mirrored JSONs.
- [2026-11-12 13:41:53 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 354 mirrored JSONs.
- [2026-11-12 13:45:54 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 356 mirrored JSONs.
- [2026-11-12 13:50:10 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 363 mirrored JSONs.
- [2026-11-12 13:54:18 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 365 mirrored JSONs.
- [2026-11-12 14:05:17 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 366 mirrored JSONs.
- [2026-11-12 14:09:21 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 368 mirrored JSONs.
