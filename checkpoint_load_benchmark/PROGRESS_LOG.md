# Checkpoint-Load Benchmark v2 — Live Progress & Team Handoff Log

- **Last Updated (UTC):** `2026-11-12 10:51:32 UTC`
- **Git Branch:** `docs/checkpoint-load-benchmark`
- **Repository Path on Cloudtop:** `/usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3`
- **This Progress File:** `checkpoint_load_benchmark/PROGRESS_LOG.md`
- **Handoff Runbook:** `/usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3/checkpoint_load_benchmark/handoff.md`
- **Design & Task Plan:** `/usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3/checkpoint_load_benchmark/plans/benchmark_v2_plan.md`
- **Live Google Sheet v2:** https://docs.google.com/spreadsheets/d/1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I/edit
- **Background Orchestrator Script:** `checkpoint_load_benchmark/scripts/run_v2_orchestrator.py`

## 1. Live Cluster & Experiment Progress Summary

| Platform | Region | Step 1 Done | TPU Queue Tasks Done | Active TPU JobSets | Active CPU Jobs |
|---|---|---:|---:|---|---|
| `v7x` | `us-central1` | **7 / 21** | **1 / 61** | `mesa-q-v7x-2x2x1-1s-w1` | `mesa-copy-v7x-deepseek3-671b-v2-cp-v7x`, `mesa-step1-v7x-deepseek3-671b-v2-s1-v7x`, `mesa-step1-v7x-deepseek3p2-671b-v2-s1-v7x`, `mesa-step1-v7x-qwen3-480b-a35b-v2-s1-v7x`, `mesa-step1-v7x-qwen3p5-397b-a17b-v2-s1-v7x` |
| `v6e` | `southamerica-west1` | **7 / 21** | **5 / 53** | `mesa-q-v6e-2x4-2s-w2` | `mesa-copy-v6e-deepseek3-671b-v2-cp-v6e`, `mesa-step1-v6e-deepseek3-671b-v2-s1-v6e`, `mesa-step1-v6e-deepseek3p2-671b-v2-s1-v6e`, `mesa-step1-v6e-qwen3-480b-a35b-v2-s1-v6e`, `mesa-step1-v6e-qwen3p5-397b-a17b-v2-s1-v6e` |
| `v5e` | `europe-west4` | **9 / 12** | **4 / 22** | `mesa-q-v5e-16x16-1s-w2` | `mesa-copy-v5e-deepseek3-671b-v2-cp-v5e`, `mesa-step1-v5e-deepseek3p2-671b-v2-s1-v5e`, `mesa-step1-v5e-qwen3-480b-a35b-v2-s1-v5e` |

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

- [2026-11-12 10:23:22 UTC] Background orchestrator started on branch `docs/checkpoint-load-benchmark`.
- [2026-11-12 10:25:13 UTC] `v5e` CPU job `mesa-copy-v5e-deepseek4-284b-v2-cp-v5e` Completed and cleaned up.
- [2026-11-12 10:25:16 UTC] Launched `v5e` Step 1 CPU job `mesa-step1-v5e-deepseek3-671b-v2-s1-v5e` for `deepseek3-671b`.
- [2026-11-12 10:25:22 UTC] Updated `v5e` queue `gs://mesa-maxtext-europe-west4/mesa-bench-v2/queue/v2/tasks.json` with 17/22 staged tasks.
- [2026-11-12 10:25:27 UTC] `v6e` CPU job `mesa-step1-v6e-gpt-oss-120b-v2-s1-v6e` Completed and cleaned up.
- [2026-11-12 10:25:29 UTC] `v6e` CPU job `mesa-step1-v6e-llama3p1-70b-v2-s1-v6e` Completed and cleaned up.
- [2026-11-12 10:25:31 UTC] Launched `v6e` Step 1 CPU job `mesa-step1-v6e-deepseek3-671b-v2-s1-v6e` for `deepseek3-671b`.
- [2026-11-12 10:25:34 UTC] Launched `v6e` BF16 regional copy job `mesa-copy-v6e-deepseek3-671b-v2-cp-v6e` for `deepseek3-671b`.
- [2026-11-12 10:25:41 UTC] Updated `v6e` queue `gs://mesa-maxtext-southamerica-west1/mesa-bench-v2/queue/v2/tasks.json` with 29/53 staged tasks.
- [2026-11-12 10:25:49 UTC] Updated `v7x` queue `gs://mesa-maxtext-us-central1/mesa-bench-v2/queue/v2/tasks.json` with 32/61 staged tasks.
- [2026-11-12 10:26:02 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 109 mirrored JSONs.
- [2026-11-12 10:29:52 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 118 mirrored JSONs.
- [2026-11-12 10:33:10 UTC] `v6e` CPU job `mesa-copy-v6e-deepseek4-284b-v2-cp-v6e` Completed and cleaned up.
- [2026-11-12 10:33:12 UTC] Launched `v6e` Step 1 CPU job `mesa-step1-v6e-deepseek3p2-671b-v2-s1-v6e` for `deepseek3.2-671b`.
- [2026-11-12 10:33:19 UTC] Updated `v6e` queue `gs://mesa-maxtext-southamerica-west1/mesa-bench-v2/queue/v2/tasks.json` with 30/53 staged tasks.
- [2026-11-12 10:33:36 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 126 mirrored JSONs.
- [2026-11-12 10:36:49 UTC] `v5e` CPU job `mesa-step1-v5e-qwen3p5-397b-a17b-v2-s1-v5e` Completed and cleaned up.
- [2026-11-12 10:36:52 UTC] Launched `v5e` BF16 regional copy job `mesa-copy-v5e-deepseek3-671b-v2-cp-v5e` for `deepseek3-671b`.
- [2026-11-12 10:37:00 UTC] Updated `v5e` queue `gs://mesa-maxtext-europe-west4/mesa-bench-v2/queue/v2/tasks.json` with 18/22 staged tasks.
- [2026-11-12 10:37:08 UTC] `v7x` CPU job `mesa-copy-v7x-deepseek4-284b-v2-cp-v7x` Completed and cleaned up.
- [2026-11-12 10:37:11 UTC] Launched `v7x` Step 1 CPU job `mesa-step1-v7x-deepseek3p2-671b-v2-s1-v7x` for `deepseek3.2-671b`.
- [2026-11-12 10:37:15 UTC] Updated `v7x` queue `gs://mesa-maxtext-us-central1/mesa-bench-v2/queue/v2/tasks.json` with 34/61 staged tasks.
- [2026-11-12 10:37:35 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 129 mirrored JSONs.
- [2026-11-12 10:41:02 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 131 mirrored JSONs.
- [2026-11-12 10:44:31 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 133 mirrored JSONs.
- [2026-11-12 10:47:39 UTC] `v5e` CPU job `mesa-step1-v5e-deepseek3-671b-v2-s1-v5e` Completed and cleaned up.
- [2026-11-12 10:47:42 UTC] Launched `v5e` Step 1 CPU job `mesa-step1-v5e-deepseek3p2-671b-v2-s1-v5e` for `deepseek3.2-671b`.
- [2026-11-12 10:48:03 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 134 mirrored JSONs.
- [2026-11-12 10:51:32 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 136 mirrored JSONs.
