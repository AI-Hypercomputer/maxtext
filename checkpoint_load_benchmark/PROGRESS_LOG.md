# Checkpoint-Load Benchmark v2 — Live Progress & Team Handoff Log

- **Last Updated (UTC):** `2026-11-12 18:15:37 UTC`
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
| `v7x` | `us-central1` | **21 / 21** | **16 / 61** | `mesa-q-v7x-2x2x1-1s-w2`, `mesa-q-v7x-2x2x1-2s-w3` | `mesa-step1b-deepseek4-1600b-v2-s1b-v7x` |
| `v6e` | `southamerica-west1` | **21 / 21** | **12 / 53** | `mesa-q-v6e-2x4-2s-w2` | _none_ |
| `v5e` | `europe-west4` | **12 / 12** | **21 / 22** | _none_ | _none_ |

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

- [2026-11-12 16:35:56 UTC] Background orchestrator started on branch `docs/checkpoint-load-benchmark`.
- [2026-11-12 16:38:20 UTC] Updated `v5e` queue `gs://mesa-maxtext-europe-west4/mesa-bench-v2/queue/v2/tasks.json` with 21/22 staged tasks.
- [2026-11-12 16:38:31 UTC] Updated `v6e` queue `gs://mesa-maxtext-southamerica-west1/mesa-bench-v2/queue/v2/tasks.json` with 52/53 staged tasks.
- [2026-11-12 16:38:39 UTC] Updated `v7x` queue `gs://mesa-maxtext-us-central1/mesa-bench-v2/queue/v2/tasks.json` with 60/61 staged tasks.
- [2026-11-12 16:40:22 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 484 mirrored JSONs.
- [2026-11-12 16:44:15 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 485 mirrored JSONs.
- [2026-11-12 16:48:45 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 487 mirrored JSONs.
- [2026-11-12 16:52:34 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 489 mirrored JSONs.
- [2026-11-12 16:59:33 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 491 mirrored JSONs.
- [2026-11-12 17:03:35 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 493 mirrored JSONs.
- [2026-11-12 17:07:30 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 495 mirrored JSONs.
- [2026-11-12 17:11:18 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 496 mirrored JSONs.
- [2026-11-12 17:21:41 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 498 mirrored JSONs.
- [2026-11-12 17:28:50 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 501 mirrored JSONs.
- [2026-11-12 17:32:50 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 503 mirrored JSONs.
- [2026-11-12 17:36:38 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 505 mirrored JSONs.
- [2026-11-12 17:40:23 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 506 mirrored JSONs.
- [2026-11-12 17:47:23 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 507 mirrored JSONs.
- [2026-11-12 17:50:40 UTC] `v7x` TPU JobSet `mesa-q-v7x-2x2x1-1s-w2` `Failed` and cleaned up.
- [2026-11-12 17:50:43 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x1-1s-w2` (`1x2x2x1`, min_slices=1) for 45 runnable tasks.
- [2026-11-12 17:54:31 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 508 mirrored JSONs.
- [2026-11-12 17:58:19 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 510 mirrored JSONs.
- [2026-11-12 18:02:09 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 514 mirrored JSONs.
- [2026-11-12 18:09:07 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 517 mirrored JSONs.
