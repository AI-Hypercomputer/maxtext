# Checkpoint-Load Benchmark v2 — Live Progress & Team Handoff Log

- **Last Updated (UTC):** `2026-11-14 07:24:45 UTC`
- **Git Branch:** `docs/checkpoint-load-benchmark`
- **Repository Path on Cloudtop:** `/usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3`
- **This Progress File:** `/usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3/checkpoint_load_benchmark/PROGRESS_LOG.md`
- **Handoff Runbook:** `/usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3/checkpoint_load_benchmark/handoff.md`
- **Executive Summary Report:** `/usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3/checkpoint_load_benchmark/reports/benchmark_v2_summary.md`
- **Design & Task Plan:** `/usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3/checkpoint_load_benchmark/plans/benchmark_v2_plan.md`
- **Live Google Sheet v2:** https://docs.google.com/spreadsheets/d/1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I/edit (`100.0%` complete & validated)
- **Background Orchestrator Script:** `checkpoint_load_benchmark/scripts/run_v2_orchestrator.py` (stopped cleanly after `100.0%` completion)

## 1. Live Cluster & Experiment Progress Summary

| Platform | Region | Step 1 Done | TPU Queue Tasks Done | Active TPU JobSets | Active CPU Jobs |
|---|---|---:|---:|---|---|
| `v7x` | `us-central1` | **21 / 21** | **61 / 61** | _none_ | _none_ |
| `v6e` | `southamerica-west1` | **21 / 21** | **53 / 53** | _none_ | _none_ |
| `v5e` | `europe-west4` | **12 / 12** | **22 / 22** | _none_ | _none_ |

## 2. Chat Session Summary & Handoff Guide for Other Engineers

1. **Summary of What Has Been Completed (`100.0%` Done, Validated & Cleaned Up):**
   - **Code & Tests (`docs/checkpoint-load-benchmark`):** Completed all Phase 0 through Phase 5 deliverables (`bench/schema_v2.py`, `bench/harness_v2.py`, `bench/platforms.py`, `bench/stages.py`, `bench/manifests_v2.py`, `bench/step1_download_job.py`, `scripts/generate_v2_manifests.py`, `scripts/run_v2_cluster.sh`, `scripts/build_sheet_v2.py`, `scripts/run_v2_orchestrator.py`). All 96 unit tests pass (`pytest`) and all modular files score `10.00/10` on `pylint`.
   - **Full Multi-Platform Matrix (`352 / 352` Planned `v2` Rows, `0` `ERROR`, `0` `TIMEOUT`):** Completed `54 / 54` cold regional Step 1 downloads + `12 / 12` DeepSeek Step 1b BF16 conversions/regional transfers, and `136 / 136` TPU queue tasks (`352` planned `v2` `Summary` rows: `291` `OK`, `59` genuine `OOM-HOST`, `1` genuine `OOM-HBM`, `1` `SKIPPED`, `3/3` Tier A repetitions).
   - **Live Google Sheet v2 & Executive Summary (`0 ERRORS, 0 WARNINGS`):** Synced and forensically validated all 11 tabs (`180,453` live cells) of Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`: `396` `Summary` rows = `352` `v2` + `44` non-overlapping `legacy|`, `54` `Step 1 downloads` rows, `205` `Stages: safetensors_dynamic` rows, `191` `Stages: to_maxtext` rows, `1,498` `Runs` phase records, clickable `gpaste` URLs on every executed row, internal `Experiment Tier` column hidden on `Summary` & `Runs`, `README!A20:B26` findings & limits populated), wrote `checkpoint_load_benchmark/reports/benchmark_v2_summary.md`, and cleaned up all K8s resources across `v7x`, `v6e`, and `v5e`.
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

- [2026-11-13 14:59:09 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1209 mirrored JSONs.
- [2026-11-13 15:07:18 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1211 mirrored JSONs.
- [2026-11-13 15:11:08 UTC] `v6e` TPU JobSet `mesa-q-v6e-8x8-1s-w7` `Failed` and cleaned up.
- [2026-11-13 15:11:48 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1213 mirrored JSONs.
- [2026-11-13 15:23:02 UTC] `v6e` TPU JobSet `mesa-q-v6e-8x8-1s-w8` `Completed` and cleaned up.
- [2026-11-13 15:23:39 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1214 mirrored JSONs.
- [2026-11-13 15:28:08 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1215 mirrored JSONs.
- [2026-11-13 16:28:10 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1216 mirrored JSONs.
- [2026-11-13 16:39:19 UTC] `v7x` TPU JobSet `mesa-q-v7x-2x2x2-2s-w29` `Completed` and cleaned up.
- [2026-11-13 16:43:03 UTC] `v7x` TPU JobSet `mesa-q-v7x-2x2x1-2s-w27` `Suspended/Deactivated by Kueue` and cleaned up.
- [2026-11-13 16:43:05 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x1-2s-w30` (`2x2x2x1`, min_slices=2) for 11 runnable tasks.
- [2026-11-13 16:43:51 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1222 mirrored JSONs.
- [2026-11-13 16:47:39 UTC] `v7x` TPU JobSet `mesa-q-v7x-2x2x1-2s-w21` `Failed` and cleaned up.
- [2026-11-13 16:47:42 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x1-2s-w31` (`2x2x2x1`, min_slices=2) for 10 runnable tasks.
- [2026-11-13 16:48:16 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1231 mirrored JSONs.
- [2026-11-13 16:52:00 UTC] `v7x` TPU JobSet `mesa-q-v7x-2x2x1-2s-w28` `Suspended/Deactivated by Kueue` and cleaned up.
- [2026-11-13 16:52:03 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x1-2s-w32` (`2x2x2x1`, min_slices=2) for 9 runnable tasks.
- [2026-11-13 16:52:41 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1235 mirrored JSONs.
- [2026-11-13 16:57:09 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1245 mirrored JSONs.
- [2026-11-13 17:01:31 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1248 mirrored JSONs.
- [2026-11-13 17:05:57 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1253 mirrored JSONs.
- [2026-11-13 17:09:50 UTC] `v7x` TPU JobSet `mesa-q-v7x-2x2x1-2s-w26` `Failed` and cleaned up.
- [2026-11-13 17:09:51 UTC] `v7x` TPU JobSet `mesa-q-v7x-2x2x1-2s-w30` `Failed` and cleaned up.
- [2026-11-13 17:09:52 UTC] `v7x` TPU JobSet `mesa-q-v7x-2x2x1-2s-w32` `Failed` and cleaned up.
- [2026-11-13 17:09:54 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x1-2s-w33` (`2x2x2x1`, min_slices=2) for 5 runnable tasks.
- [2026-11-13 17:10:33 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1260 mirrored JSONs.
- [2026-11-13 17:14:26 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x1-2s-w34` (`2x2x2x1`, min_slices=2) for 4 runnable tasks.
- [2026-11-13 17:15:04 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1267 mirrored JSONs.
- [2026-11-13 17:18:56 UTC] `v7x` TPU JobSet `mesa-q-v7x-2x2x1-2s-w34` `Failed` and cleaned up.
- [2026-11-13 17:19:33 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1272 mirrored JSONs.
- [2026-11-13 17:23:57 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1273 mirrored JSONs.
- [2026-11-13 17:28:23 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1275 mirrored JSONs.
- [2026-11-13 17:32:14 UTC] `v7x` TPU JobSet `mesa-q-v7x-2x2x1-2s-w31` `Failed` and cleaned up.
- [2026-11-13 17:32:56 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1278 mirrored JSONs.
- [2026-11-13 17:36:48 UTC] `v7x` TPU JobSet `mesa-q-v7x-2x2x1-2s-w20` `Completed` and cleaned up.
- [2026-11-13 17:36:50 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x1-2s-w35` (`2x2x2x1`, min_slices=2) for 2 runnable tasks.
- [2026-11-13 17:37:28 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1280 mirrored JSONs.
- [2026-11-13 17:41:22 UTC] `v7x` TPU JobSet `mesa-q-v7x-2x2x1-2s-w35` `Completed` and cleaned up.
- [2026-11-13 17:41:25 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x1-2s-w36` (`2x2x2x1`, min_slices=2) for 3 runnable tasks.
- [2026-11-13 17:41:28 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x1-2s-w37` (`2x2x2x1`, min_slices=2) for 3 runnable tasks.
- [2026-11-13 17:42:08 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1264 mirrored JSONs.
- [2026-11-13 17:45:54 UTC] Launched `v5e` TPU queue worker `mesa-q-v5e-16x16-1s-w2` (`1x16x16`, min_slices=1) for 3 runnable tasks.
- [2026-11-13 17:46:02 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x1-2s-w38` (`2x2x2x1`, min_slices=2) for 13 runnable tasks.
- [2026-11-13 17:46:04 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x1-1s-w39` (`1x2x2x1`, min_slices=1) for 13 runnable tasks.
- [2026-11-13 17:46:06 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x2-1s-w40` (`1x2x2x2`, min_slices=1) for 13 runnable tasks.
- [2026-11-13 17:46:43 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1150 mirrored JSONs.
- [2026-11-13 17:51:05 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1109 mirrored JSONs.
- [2026-11-13 17:55:23 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1111 mirrored JSONs.
- [2026-11-13 17:59:09 UTC] `v7x` TPU JobSet `mesa-q-v7x-2x2x1-1s-w39` `Completed` and cleaned up.
- [2026-11-13 17:59:12 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x1-1s-w41` (`1x2x2x1`, min_slices=1) for 12 runnable tasks.
- [2026-11-13 17:59:42 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1114 mirrored JSONs.
- [2026-11-13 18:03:35 UTC] `v7x` TPU JobSet `mesa-q-v7x-2x2x1-1s-w39` `Completed` and cleaned up.
- [2026-11-13 18:03:36 UTC] `v7x` TPU JobSet `mesa-q-v7x-2x2x1-1s-w41` `Completed` and cleaned up.
- [2026-11-13 18:03:39 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x1-2s-w42` (`2x2x2x1`, min_slices=2) for 11 runnable tasks.
- [2026-11-13 18:03:42 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x1-2s-w43` (`2x2x2x1`, min_slices=2) for 11 runnable tasks.
- [2026-11-13 18:03:44 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x1-2s-w44` (`2x2x2x1`, min_slices=2) for 11 runnable tasks.
- [2026-11-13 18:04:16 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1123 mirrored JSONs.
- [2026-11-13 18:09:30 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1132 mirrored JSONs.
- [2026-11-13 18:14:16 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1133 mirrored JSONs.
- [2026-11-13 18:18:37 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1134 mirrored JSONs.
- [2026-11-13 18:22:58 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1142 mirrored JSONs.
- [2026-11-13 18:26:39 UTC] `v7x` TPU JobSet `mesa-q-v7x-2x2x1-2s-w36` `Failed` and cleaned up.
- [2026-11-13 18:26:41 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x1-2s-w45` (`2x2x2x1`, min_slices=2) for 10 runnable tasks.
- [2026-11-13 18:27:26 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1143 mirrored JSONs.
- [2026-11-13 18:31:48 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1147 mirrored JSONs.
- [2026-11-13 18:35:59 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1148 mirrored JSONs.
- [2026-11-13 18:40:22 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1157 mirrored JSONs.
- [2026-11-13 18:44:50 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1172 mirrored JSONs.
- [2026-11-13 18:49:22 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1185 mirrored JSONs.
- [2026-11-13 18:54:00 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1196 mirrored JSONs.
- [2026-11-13 18:57:48 UTC] `v7x` TPU JobSet `mesa-q-v7x-2x2x1-2s-w38` `Failed` and cleaned up.
- [2026-11-13 18:57:50 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x1-2s-w46` (`2x2x2x1`, min_slices=2) for 9 runnable tasks.
- [2026-11-13 18:58:22 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1199 mirrored JSONs.
- [2026-11-13 19:06:29 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1202 mirrored JSONs.
- [2026-11-13 19:10:49 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1208 mirrored JSONs.
- [2026-11-13 19:14:48 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x1-2s-w47` (`2x2x2x1`, min_slices=2) for 9 runnable tasks.
- [2026-11-13 19:14:51 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x1-2s-w48` (`2x2x2x1`, min_slices=2) for 9 runnable tasks.
- [2026-11-13 19:14:53 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x1-2s-w49` (`2x2x2x1`, min_slices=2) for 9 runnable tasks.
- [2026-11-13 19:15:25 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1215 mirrored JSONs.
- [2026-11-13 19:19:52 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1217 mirrored JSONs.
- [2026-11-13 19:24:10 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1218 mirrored JSONs.
- [2026-11-13 19:29:23 UTC] `v5e` TPU JobSet `mesa-q-v5e-16x16-1s-w2` `Failed` and cleaned up.
- [2026-11-13 19:30:06 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1221 mirrored JSONs.
- [2026-11-13 19:34:05 UTC] `v5e` TPU JobSet `mesa-q-v5e-16x16-1s-w99` `Completed` and cleaned up.
- [2026-11-13 19:34:44 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1227 mirrored JSONs.
- [2026-11-13 19:38:28 UTC] Launched `v5e` TPU queue worker `mesa-q-v5e-16x16-1s-w100` (`1x16x16`, min_slices=1) for 2 runnable tasks.
- [2026-11-13 19:39:00 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1209 mirrored JSONs.
- [2026-11-13 19:42:54 UTC] `v7x` TPU JobSet `mesa-q-v7x-2x2x1-2s-w43` `Failed` and cleaned up.
- [2026-11-13 19:42:56 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x1-2s-w50` (`2x2x2x1`, min_slices=2) for 8 runnable tasks.
- [2026-11-13 19:43:30 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1214 mirrored JSONs.
- [2026-11-13 19:47:47 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1223 mirrored JSONs.
- [2026-11-13 19:52:25 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1229 mirrored JSONs.
- [2026-11-13 19:56:11 UTC] `v7x` TPU JobSet `mesa-q-v7x-2x2x1-2s-w47` `Failed` and cleaned up.
- [2026-11-13 19:56:14 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x1-2s-w51` (`2x2x2x1`, min_slices=2) for 8 runnable tasks.
- [2026-11-13 19:56:45 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1230 mirrored JSONs.
- [2026-11-13 20:00:39 UTC] `v5e` TPU JobSet `mesa-q-v5e-16x16-1s-w100` `Failed` and cleaned up.
- [2026-11-13 20:01:17 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1239 mirrored JSONs.
- [2026-11-13 20:05:44 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1243 mirrored JSONs.
- [2026-11-13 20:10:04 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1247 mirrored JSONs.
- [2026-11-13 20:13:57 UTC] `v5e` TPU JobSet `mesa-q-v5e-16x16-1s-w101` `Completed` and cleaned up.
- [2026-11-13 20:14:34 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1251 mirrored JSONs.
- [2026-11-13 20:18:53 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1255 mirrored JSONs.
- [2026-11-13 20:26:23 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x1-2s-w52` (`2x2x2x1`, min_slices=2) for 7 runnable tasks.
- [2026-11-13 20:26:56 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1258 mirrored JSONs.
- [2026-11-13 20:31:19 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1262 mirrored JSONs.
- [2026-11-13 20:35:06 UTC] `v7x` TPU JobSet `mesa-q-v7x-2x2x2-1s-w40` `Completed` and cleaned up.
- [2026-11-13 20:35:40 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1265 mirrored JSONs.
- [2026-11-13 20:47:00 UTC] `v7x` TPU JobSet `mesa-q-v7x-2x4x4-1s-w11` `Completed` and cleaned up.
- [2026-11-13 20:47:33 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1269 mirrored JSONs.
- [2026-11-13 20:51:52 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1270 mirrored JSONs.
- [2026-11-13 21:03:11 UTC] `v7x` TPU JobSet `mesa-q-v7x-2x2x1-2s-w42` `Failed` and cleaned up.
- [2026-11-13 21:07:36 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1274 mirrored JSONs.
- [2026-11-13 21:11:28 UTC] `v7x` TPU JobSet `mesa-q-v7x-2x2x1-2s-w52` `Failed` and cleaned up.
- [2026-11-13 21:11:31 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x1-2s-w53` (`2x2x2x1`, min_slices=2) for 3 runnable tasks.
- [2026-11-13 21:12:02 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1276 mirrored JSONs.
- [2026-11-13 21:15:55 UTC] `v7x` TPU JobSet `mesa-q-v7x-2x2x1-2s-w53` `Failed` and cleaned up.
- [2026-11-13 21:16:26 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1283 mirrored JSONs.
- [2026-11-13 21:20:45 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1285 mirrored JSONs.
- [2026-11-13 21:36:28 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1287 mirrored JSONs.
- [2026-11-13 21:40:50 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1288 mirrored JSONs.
- [2026-11-13 21:45:24 UTC] `v7x` TPU JobSet `mesa-q-v7x-2x2x1-2s-w49` `Failed` and cleaned up.
- [2026-11-13 21:45:54 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1270 mirrored JSONs.
- [2026-11-13 21:51:01 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1271 mirrored JSONs.
- [2026-11-13 21:54:55 UTC] `v5e` TPU JobSet `mesa-q-v5e-16x16-1s-w30` `Completed` and cleaned up.
- [2026-11-13 21:54:57 UTC] `v5e` TPU JobSet `mesa-q-v5e-16x16-1s-w31` `Completed` and cleaned up.
- [2026-11-13 21:59:24 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1273 mirrored JSONs.
- [2026-11-13 22:03:45 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1275 mirrored JSONs.
- [2026-11-13 22:08:11 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1281 mirrored JSONs.
- [2026-11-13 22:12:39 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1283 mirrored JSONs.
- [2026-11-13 22:21:48 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1282 mirrored JSONs.
- [2026-11-13 22:56:49 UTC] `v5e` TPU JobSet `mesa-fix-v5e-30b-r12` `Failed` and cleaned up.
- [2026-11-13 23:08:11 UTC] `v5e` TPU JobSet `mesa-v5e-r2-s12-np1` `Completed` and cleaned up.
- [2026-11-13 23:08:48 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1283 mirrored JSONs.
- [2026-11-13 23:27:45 UTC] `v5e` TPU JobSet `mesa-v5e-r1-s3-np0` `Completed` and cleaned up.
- [2026-11-13 23:27:47 UTC] `v5e` TPU JobSet `mesa-v5e-r2-s3-np1` `Completed` and cleaned up.
- [2026-11-13 23:28:21 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1282 mirrored JSONs.
- [2026-11-13 23:32:37 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1283 mirrored JSONs.
- [2026-11-13 23:40:41 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1284 mirrored JSONs.
- [2026-11-13 23:44:59 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1285 mirrored JSONs.
- [2026-11-14 00:03:58 UTC] `v7x` TPU JobSet `mesa-q-v7x-2x2x1-2s-w65` `Failed` and cleaned up.
- [2026-11-14 00:08:20 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1286 mirrored JSONs.
- [2026-11-14 00:16:58 UTC] `v7x` TPU JobSet `mesa-sd-v7x-397b-1x2x2x2` `Completed` and cleaned up.
- [2026-11-14 00:22:28 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1288 mirrored JSONs.
- [2026-11-14 00:54:34 UTC] `v7x` TPU JobSet `mesa-q-v7x-2x2x2-1s-w63` `Completed` and cleaned up.
- [2026-11-14 00:55:07 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1289 mirrored JSONs.
- [2026-11-14 02:40:20 UTC] `v7x` TPU JobSet `mesa-q-v7x-2x2x1-2s-w66` `Suspended/Deactivated by Kueue` and cleaned up.
- [2026-11-14 04:19:17 UTC] `v7x` TPU JobSet `mesa-tm2-v7x-397b-2h` `Completed` and cleaned up.
- [2026-11-14 04:23:12 UTC] Launched `v7x` TPU queue worker `mesa-q-v7x-2x2x1-2s-w67` (`2x2x2x1`, min_slices=2) for 1 runnable tasks.
- [2026-11-14 04:23:42 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1288 mirrored JSONs.
- [2026-11-14 04:27:57 UTC] Synced Google Sheet v2 (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`) with 1289 mirrored JSONs.
- [2026-11-14 05:31:14 UTC] `v7x` TPU JobSet `mesa-s3b-v7x-397b-2s` (`qwen3.5-397b-a17b` `2x2x2x1` `unscanned` `to_maxtext` Step 3 = `1952.53s` `OK`) completed (`61/61` on `v7x`, `136/136` overall) and cleaned up; `mesa-v2-orchestrator.service` stopped cleanly.
- [2026-11-14 07:05:18 UTC] `v5e` CPU Job `mesa-copy-v5e-deepseek4-1600b-v2-cp-v5e-cold` (`5300.98s` cold BF16 regional copy) completed and cleaned up.
- [2026-11-14 07:14:52 UTC] `v6e` CPU Job `mesa-copy-v6e-deepseek4-1600b-v2-cp-v6e-cold` (`5803.30s` cold BF16 regional copy) completed and cleaned up (`0` active resources across `v7x`, `v6e`, `v5e`).
- [2026-11-14 07:20:53 UTC] Final GCS mirror (`1291` JSONs), Google Sheet v2 sync (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`), `README!A20:B26` summary update, and `reports/benchmark_v2_summary.md` completed.
- [2026-11-14 07:23:13 UTC] Completed 10-step live Google Sheet & CSV forensic validation across `180,453` live cells (`FINAL RESULT: 0 ERRORS, 0 WARNINGS`).
