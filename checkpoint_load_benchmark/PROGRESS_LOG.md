# Checkpoint-Load Benchmark v2 — Live Progress & Team Handoff Log

- **Last Updated (UTC):** `2026-11-13 08:05:59 UTC`
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
| `v7x` | `us-central1` | **21 / 21** | **25 / 61** | `mesa-q-v7x-2x2x1-1s-w14`, `mesa-q-v7x-2x2x1-1s-w17`, `mesa-q-v7x-2x2x1-1s-w20`, `mesa-q-v7x-2x2x1-2s-w12`, `mesa-q-v7x-2x2x1-2s-w15`, `mesa-q-v7x-2x2x1-2s-w16`, `mesa-q-v7x-2x2x2-1s-w18`, `mesa-q-v7x-2x2x2-2s-w10`, `mesa-q-v7x-2x2x4-1s-w19`, `mesa-q-v7x-2x4x4-1s-w11` | _none_ |
| `v6e` | `southamerica-west1` | **21 / 21** | **44 / 53** | `mesa-q-v6e-16x16-1s-w5`, `mesa-q-v6e-8x8-1s-w4` | _none_ |
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

- [2026-11-13 02:39:30 UTC] Background orchestrator started on branch `docs/checkpoint-load-benchmark`.
- [2026-11-13 02:41:51 UTC] Updated `v5e` queue `gs://mesa-maxtext-europe-west4/mesa-bench-v2/queue/v2/tasks.json` with 22/22 staged tasks.
- [2026-11-13 02:42:01 UTC] Updated `v6e` queue `gs://mesa-maxtext-southamerica-west1/mesa-bench-v2/queue/v2/tasks.json` with 53/53 staged tasks.
- [2026-11-13 02:42:06 UTC] Launched `v6e` TPU queue worker `mesa-q-v6e-8x8-1s-w4` (`1x8x8`, min_slices=1) for 30 runnable tasks.
- [2026-11-13 02:42:09 UTC] Launched `v6e` TPU queue worker `mesa-q-v6e-16x16-1s-w5` (`1x16x16`, min_slices=1) for 30 runnable tasks.
- [2026-11-13 02:42:15 UTC] Updated `v7x` queue `gs://mesa-maxtext-us-central1/mesa-bench-v2/queue/v2/tasks.json` with 61/61 staged tasks.
- [2026-11-13 02:42:46 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 667 mirrored JSONs.
- [2026-11-13 02:46:40 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 674 mirrored JSONs.
- [2026-11-13 02:50:10 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x1-2s-w12` (`2x2x2x1`, min_slices=2) for 42 runnable tasks.
- [2026-11-13 02:50:40 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 687 mirrored JSONs.
- [2026-11-13 02:54:29 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 691 mirrored JSONs.
- [2026-11-13 02:58:26 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 697 mirrored JSONs.
- [2026-11-13 03:02:22 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 700 mirrored JSONs.
- [2026-11-13 03:06:23 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 706 mirrored JSONs.
- [2026-11-13 03:10:29 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 710 mirrored JSONs.
- [2026-11-13 03:14:28 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 716 mirrored JSONs.
- [2026-11-13 03:18:08 UTC] `v7x` TPU JobSet `mesa-q-v7x-2x2x1-1s-w3` `Failed` and cleaned up.
- [2026-11-13 03:18:10 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x1-1s-w13` (`1x2x2x1`, min_slices=1) for 41 runnable tasks.
- [2026-11-13 03:18:42 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 722 mirrored JSONs.
- [2026-11-13 03:22:58 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 725 mirrored JSONs.
- [2026-11-13 03:27:07 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 733 mirrored JSONs.
- [2026-11-13 03:31:08 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 739 mirrored JSONs.
- [2026-11-13 03:35:04 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 741 mirrored JSONs.
- [2026-11-13 03:39:11 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 743 mirrored JSONs.
- [2026-11-13 03:43:16 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 747 mirrored JSONs.
- [2026-11-13 03:47:24 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 755 mirrored JSONs.
- [2026-11-13 03:51:22 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 758 mirrored JSONs.
- [2026-11-13 03:55:18 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 759 mirrored JSONs.
- [2026-11-13 03:59:20 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 763 mirrored JSONs.
- [2026-11-13 04:03:20 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 766 mirrored JSONs.
- [2026-11-13 04:07:23 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 769 mirrored JSONs.
- [2026-11-13 04:11:25 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 774 mirrored JSONs.
- [2026-11-13 04:15:24 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 776 mirrored JSONs.
- [2026-11-13 04:23:03 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 777 mirrored JSONs.
- [2026-11-13 04:30:18 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 778 mirrored JSONs.
- [2026-11-13 04:34:17 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 781 mirrored JSONs.
- [2026-11-13 04:37:46 UTC] `v7x` TPU JobSet `mesa-q-v7x-2x2x1-1s-w6` `Failed` and cleaned up.
- [2026-11-13 04:37:48 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x1-1s-w14` (`1x2x2x1`, min_slices=1) for 39 runnable tasks.
- [2026-11-13 05:29:36 UTC] `v7x` TPU JobSet `mesa-q-v7x-2x2x1-1s-w7` `Suspended/Deactivated by Kueue` and cleaned up.
- [2026-11-13 05:29:38 UTC] `v7x` TPU JobSet `mesa-q-v7x-2x2x1-2s-w4` `Suspended/Deactivated by Kueue` and cleaned up.
- [2026-11-13 05:29:39 UTC] `v7x` TPU JobSet `mesa-q-v7x-2x2x1-2s-w5` `Suspended/Deactivated by Kueue` and cleaned up.
- [2026-11-13 05:29:40 UTC] `v7x` TPU JobSet `mesa-q-v7x-2x2x2-1s-w8` `Suspended/Deactivated by Kueue` and cleaned up.
- [2026-11-13 05:29:43 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x1-2s-w15` (`2x2x2x1`, min_slices=2) for 36 runnable tasks.
- [2026-11-13 05:29:45 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x1-2s-w16` (`2x2x2x1`, min_slices=2) for 36 runnable tasks.
- [2026-11-13 05:29:48 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x1-1s-w17` (`1x2x2x1`, min_slices=1) for 36 runnable tasks.
- [2026-11-13 05:29:50 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x2-1s-w18` (`1x2x2x2`, min_slices=1) for 36 runnable tasks.
- [2026-11-13 05:36:40 UTC] `v7x` TPU JobSet `mesa-q-v7x-2x2x4-1s-w9` `Suspended/Deactivated by Kueue` and cleaned up.
- [2026-11-13 05:36:43 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x4-1s-w19` (`1x2x2x4`, min_slices=1) for 36 runnable tasks.
- [2026-11-13 06:18:15 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 820 mirrored JSONs.
- [2026-11-13 06:21:41 UTC] `v7x` TPU JobSet `mesa-q-v7x-2x2x1-1s-w13` `Suspended/Deactivated by Kueue` and cleaned up.
- [2026-11-13 06:21:44 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x1-1s-w20` (`1x2x2x1`, min_slices=1) for 36 runnable tasks.
- [2026-11-13 06:29:05 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 822 mirrored JSONs.
- [2026-11-13 06:33:13 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 823 mirrored JSONs.
- [2026-11-13 06:37:24 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 830 mirrored JSONs.
- [2026-11-13 06:41:30 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 833 mirrored JSONs.
- [2026-11-13 06:45:45 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 840 mirrored JSONs.
- [2026-11-13 06:49:29 UTC] Launched `v6e` TPU queue worker `mesa-q-v6e-4x4-1s-w6` (`1x4x4`, min_slices=1) for 17 runnable tasks.
- [2026-11-13 06:50:03 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 852 mirrored JSONs.
- [2026-11-13 06:54:12 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 858 mirrored JSONs.
- [2026-11-13 06:58:27 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 866 mirrored JSONs.
- [2026-11-13 07:02:31 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 870 mirrored JSONs.
- [2026-11-13 07:06:43 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 873 mirrored JSONs.
- [2026-11-13 07:10:42 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 877 mirrored JSONs.
- [2026-11-13 07:14:47 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 885 mirrored JSONs.
- [2026-11-13 07:22:06 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 889 mirrored JSONs.
- [2026-11-13 07:26:06 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 893 mirrored JSONs.
- [2026-11-13 07:30:03 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 895 mirrored JSONs.
- [2026-11-13 07:34:09 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 897 mirrored JSONs.
- [2026-11-13 07:38:08 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 900 mirrored JSONs.
- [2026-11-13 07:45:38 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 903 mirrored JSONs.
- [2026-11-13 07:49:38 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 906 mirrored JSONs.
- [2026-11-13 07:53:10 UTC] `v6e` TPU JobSet `mesa-q-v6e-4x4-1s-w6` `Completed` and cleaned up.
- [2026-11-13 07:53:45 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 909 mirrored JSONs.
- [2026-11-13 07:57:50 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 911 mirrored JSONs.
- [2026-11-13 08:01:18 UTC] `v6e` TPU JobSet `mesa-q-v6e-2x4-2s-w2` `Completed` and cleaned up.
- [2026-11-13 08:01:52 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 913 mirrored JSONs.
- [2026-11-13 08:05:59 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 914 mirrored JSONs.
