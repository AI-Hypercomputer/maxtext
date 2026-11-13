# Checkpoint-Load Benchmark v2 — Live Progress & Team Handoff Log

- **Last Updated (UTC):** `2026-11-13 12:33:16 UTC`
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
| `v7x` | `us-central1` | **21 / 21** | **38 / 61** | `mesa-q-v7x-2x2x1-1s-w23`, `mesa-q-v7x-2x2x1-1s-w24`, `mesa-q-v7x-2x2x1-1s-w25`, `mesa-q-v7x-2x2x1-2s-w20`, `mesa-q-v7x-2x2x1-2s-w21`, `mesa-q-v7x-2x2x1-2s-w22`, `mesa-q-v7x-2x2x2-2s-w10`, `mesa-q-v7x-2x4x4-1s-w11` | _none_ |
| `v6e` | `southamerica-west1` | **21 / 21** | **49 / 53** | `mesa-q-v6e-16x16-1s-w6`, `mesa-q-v6e-8x8-1s-w4` | _none_ |
| `v5e` | `europe-west4` | **12 / 12** | **22 / 22** | _none_ | _none_ |

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

- [2026-11-13 10:11:31 UTC] Background orchestrator started on branch `docs/checkpoint-load-benchmark`.
- [2026-11-13 10:13:59 UTC] Updated `v5e` queue `gs://mesa-maxtext-europe-west4/mesa-bench-v2/queue/v2/tasks.json` with 22/22 staged tasks.
- [2026-11-13 10:14:09 UTC] Updated `v6e` queue `gs://mesa-maxtext-southamerica-west1/mesa-bench-v2/queue/v2/tasks.json` with 53/53 staged tasks.
- [2026-11-13 10:14:19 UTC] Updated `v7x` queue `gs://mesa-maxtext-us-central1/mesa-bench-v2/queue/v2/tasks.json` with 61/61 staged tasks.
- [2026-11-13 10:14:23 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x1-2s-w20` (`2x2x2x1`, min_slices=2) for 35 runnable tasks.
- [2026-11-13 10:14:25 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x1-2s-w21` (`2x2x2x1`, min_slices=2) for 35 runnable tasks.
- [2026-11-13 10:14:28 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x1-2s-w22` (`2x2x2x1`, min_slices=2) for 35 runnable tasks.
- [2026-11-13 10:14:30 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x1-1s-w23` (`1x2x2x1`, min_slices=1) for 35 runnable tasks.
- [2026-11-13 10:14:32 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x1-1s-w24` (`1x2x2x1`, min_slices=1) for 35 runnable tasks.
- [2026-11-13 10:14:34 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x1-1s-w25` (`1x2x2x1`, min_slices=1) for 35 runnable tasks.
- [2026-11-13 10:15:04 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 940 mirrored JSONs.
- [2026-11-13 10:19:04 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 946 mirrored JSONs.
- [2026-11-13 10:23:05 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 948 mirrored JSONs.
- [2026-11-13 10:27:08 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 954 mirrored JSONs.
- [2026-11-13 10:31:19 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 956 mirrored JSONs.
- [2026-11-13 10:35:21 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 959 mirrored JSONs.
- [2026-11-13 10:38:52 UTC] `v7x` TPU JobSet `mesa-q-v7x-2x2x2-1s-w18` `Completed` and cleaned up.
- [2026-11-13 10:38:55 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x2-1s-w26` (`1x2x2x2`, min_slices=1) for 35 runnable tasks.
- [2026-11-13 10:39:25 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 961 mirrored JSONs.
- [2026-11-13 10:43:30 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 964 mirrored JSONs.
- [2026-11-13 10:47:33 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 967 mirrored JSONs.
- [2026-11-13 10:51:39 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 969 mirrored JSONs.
- [2026-11-13 10:55:35 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 970 mirrored JSONs.
- [2026-11-13 10:59:43 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 973 mirrored JSONs.
- [2026-11-13 11:04:22 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 977 mirrored JSONs.
- [2026-11-13 11:13:18 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 980 mirrored JSONs.
- [2026-11-13 11:17:33 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 982 mirrored JSONs.
- [2026-11-13 11:21:50 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 992 mirrored JSONs.
- [2026-11-13 11:26:59 UTC] `v7x` TPU JobSet `mesa-q-v7x-2x2x4-1s-w19` `Completed` and cleaned up.
- [2026-11-13 11:27:01 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x4-1s-w27` (`1x2x2x4`, min_slices=1) for 30 runnable tasks.
- [2026-11-13 11:27:34 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 998 mirrored JSONs.
- [2026-11-13 11:32:14 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1009 mirrored JSONs.
- [2026-11-13 11:36:33 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1021 mirrored JSONs.
- [2026-11-13 11:41:36 UTC] `v6e` TPU JobSet `mesa-q-v6e-16x16-1s-w5` `Completed` and cleaned up.
- [2026-11-13 11:41:40 UTC] Launched `v6e` TPU queue worker `mesa-q-v6e-16x16-1s-w6` (`1x16x16`, min_slices=1) for 5 runnable tasks.
- [2026-11-13 11:42:21 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1025 mirrored JSONs.
- [2026-11-13 11:46:58 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1042 mirrored JSONs.
- [2026-11-13 11:51:22 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1054 mirrored JSONs.
- [2026-11-13 11:55:39 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1061 mirrored JSONs.
- [2026-11-13 11:59:57 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1065 mirrored JSONs.
- [2026-11-13 12:04:19 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1069 mirrored JSONs.
- [2026-11-13 12:08:40 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1076 mirrored JSONs.
- [2026-11-13 12:12:56 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1077 mirrored JSONs.
- [2026-11-13 12:17:10 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1078 mirrored JSONs.
