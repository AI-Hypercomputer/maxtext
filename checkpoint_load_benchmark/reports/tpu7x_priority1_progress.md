# Priority 1 (>200B models) progress on TPU v7x (`bodaborg-tpu7x-nap`)

> **Final:** 2026-10-29 (all runs finished; cluster cleaned up) · **Branch:** `docs/checkpoint-load-benchmark` · **Live queue:**
> `gs://mesa-maxtext/mesakh_tpu7x_bench/queue/p1/` (task ids below refer to
> [`configs/p1_completion_tasks.json`](../configs/p1_completion_tasks.json))

**Hardware per slice:** `tpu7x-standard-4t`, topology `2x2x1` = 4 chips = 8 TPU cores, **813.8 GB HBM**,
**1,014.5 GB host RAM** (986.7 GB allocatable to pods), spot capacity. **1 slice** = 8 cores; **2 slices** = 16 cores
(multi-slice over DCN, `dcn_fsdp_parallelism=2`); **4 slices** = 32 cores, 3.26 TB HBM (`dcn_fsdp_parallelism=4`;
`deepseek3-671b`, plus `qwen3.5-397b-a17b` scanned).

**Experiments**
- **1A**: `safetensors_dynamic` BEFORE (`origin/main`, `f596f5369`) vs WITH Lance's 3 PRs (#5706, #5828, #5705).
- **1B**: `to_maxtext` (S1 HF→GCS, S2 CPU conversion to Orbax, S3 Orbax load into HBM) vs `safetensors_dynamic`
  (S1 HF→GCS, S2 sharded load + in-memory conversion).

**Multi-host slices** (`2x2x2`, `2x2x4`; Lance Wang's runs, `safetensors_dynamic` only):
[`tpu7x_multihost_results.md`](tpu7x_multihost_results.md). Their rows are also in the row-by-row table below.

**Legend:** ✅ done · ❌ failed or not possible (reason) · ⚠️ inconclusive · ➖ not applicable or skipped

## Summary by model

| Model | BF16 size / shards | S1 HF→GCS (cold) | 1A WITH PRs | 1A BEFORE (`origin/main`) | 1B `to_maxtext` |
| :--- | :--- | ---: | :--- | :--- | :--- |
| `qwen3-235b-a22b` | 470.4 GB / 118 | 583.3 s | ✅ 4/4 | ❌ scanned · 1 and · 2: **timeout at 1800 s**, still loading (WITH: 757.5 s / 792.4 s) | ❌ scanned: S2 can't run on one host · ✅ unscanned (shards fetched as 16 parallel range reads): S2 2,625.5 s, S3 545.5 s on 1 slice and 659.1 s on 2 slices (the single-stream first run: S2 5,150.1 s) |
| `deepseek4-284b` | 581.9 GB / 146 *(synthetic sparse files)* | – | ✅ 4/4 | ❌ scanned · 1 and · 2: **timeout at 1800 s**, still loading (WITH: 453.3 s / 440.7 s) | ❌ S2 OOM-killed at 989.6 GB host RAM (scanned and unscanned) |
| `qwen3.5-397b-a17b` | 795.5 GB / 94 | 814.5 s | ✅ unscanned · 2 (1,311.9 s; 1,272.2 s with fix) · ✅ scanned · 4 (1,399.7 s) · ❌ scanned · 1 and · 2 (HBM OOM; also with `hbmfix` on 2 slices) · ✅ scanned **with fix** (U-1): **1,608.3 s** on 2 × `2x2x1`, **1,475.2 s** on 1 × `2x2x2` | ❌ unscanned · 2 and scanned · 4: **timeout at 1800 s**, still in the load phase (WITH: 1,311.9 s and 1,399.7 s) | ❌ scanned: S2 can't run on one host · ✅ unscanned **with save-buffer cap** (U-8, 16 parallel shard reads): S2 **8,096.5 s** (884.2 GB peak RAM), S3 **2,025.8 s** on 2 × `2x2x1` and **1,838.3 s** on 1 × `2x2x2` (❌ OOM-killed at the start of the Orbax save without the cap) |
| `qwen3-480b-a35b` | 961.0 GB / 241 | 1,046.1 s | ✅ 2/2 that fit (1 slice ❌ HBM) | ❌ `KeyError` (no mapping before #5706); conclusive | ❌ scanned: S2 can't run on one host · ❌ unscanned: the 960 GB converted state doesn't fit host RAM (fails fast) |
| `deepseek3-671b` | 1,342.4 GB / 336 *(synthetic sparse files)* | – | ❌ doesn't fit 1 or 2 slices · ✅ **4 slices**: scanned 649.4 s, unscanned 531.0 s | ❌ scanned · 4: **timeout at 1800 s**, still loading (WITH: 649.4 s) | ➖ |

## `to_maxtext` vs `safetensor`, row by row (the "tpu-v7x" tab of the benchmark sheet)

This section is the content of the "tpu-v7x" tab of the benchmark Google Sheet. It adds Lance Wang's multi-host runs
([`tpu7x_multihost_results.md`](tpu7x_multihost_results.md): `1 (16)`, `1 (32)`, `2 (32)` and `4 (64)` rows,
`safetensor` only) to the P1 runs. Every cell is plain text, so it pastes into the sheet without cleanup. Clear the tab
first (select all, press Delete, then Format > Clear formatting), then do one of the following:
- **Copy the rendered text**: in the rendered view (GitHub or a markdown preview), select from the title line below
  through the last table row, copy, and paste into cell A1.
- **Copy the tab-separated text**: expand "The same title, legend and table as tab-separated text" under the table,
  copy the block, and paste into cell A1. Sheets splits each line into cells at the tabs.

<!-- SHEET TABLE START -->
TPU v7x (bodaborg-tpu7x-nap), >200B models: to_maxtext vs safetensor (safetensors_dynamic with PRs #5706, #5828, #5705)

- For each (Model, Scan Mode, Slices) configuration, to_maxtext and safetensor are on consecutive rows.
- Col 1 (HF -> GCS): to_maxtext Step 1 vs safetensor Step 1, a cold download of the .safetensors shards, measured once and shared by the configurations that reuse it. DeepSeek models read synthetic sparse files staged on each host (local stage).
- Col 2 (Convert to Orbax): to_maxtext Step 2, a CPU conversion on one host that every slice count shares. safetensor has no conversion step (—).
- Col 3 (Load into MaxText): to_maxtext Step 3 (Load Orbax) vs safetensor Step 2 (Safetensors+Convert = load_sharded_hf_state + transform_hf_state_to_mt_state). Peak host RAM is the runner host's.
- Slices (Devices): a tpu7x host has 4 chips = 8 TPU cores, 813.8 GB HBM and 1,014.5 GB RAM. 1 (8), 2 (16), 4 (32) = 1, 2, 4 × 2x2x1 single-host slices (P1 runs). 1 (16) = 1 × 2x2x2, 1 (32) = 1 × 2x2x4, 2 (32) = 2 × 2x2x2, 4 (64) = 4 × 2x2x2, 1 (64) = 1 × 2x4x4 multi-host slices (safetensor: Lance Wang's runs and the 10-30 runs; to_maxtext: Orbax loads of the checkpoint that the single-host conversion wrote, so Steps 1-2 are the ones of 1 (8)).
- Speedup: to_maxtext total / safetensor total, end-to-end / without Step 1. N/A: no total because a step failed or was not run.
- Status symbols: ⏳ In progress = currently running on the cluster; ⏳ Queued = queued in Kueue waiting for spot slice capacity.
- safetensor (with fix): the same configuration re-run with this branch's MaxText fixes for two HBM peaks (composite leaves stacked one by one; reshard slices joined in pairs), from a separate code tree. The plain safetensor rows don't include these fixes.

| Model | Scan Mode | Slices (Devices) | Pipeline | 1. HF -> GCS (s) | 2. Convert to Orbax (s) | 3. Load into MaxText (s) | Total (No Step 1) (s) | Total (End-to-End) (s) | Speedup (E2E / No Step 1) |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| qwen3-235b-a22b (470.4 GB MoE) | scanned | 1 (8) | to_maxtext | 583.32 | Does not fit one host (the source must be staged in RAM next to the converted state) | — | N/A | N/A | N/A |
| ↳ | scanned | 1 (8) | safetensor | 583.32 | — | 757.50 (Safetensors+Convert: load 674.81 + transform 82.66; peak host RAM 391.8 GB) | 757.50 | 1,340.81 | Only safetensor loads this config |
| ↳ | scanned | 1 (8) | safetensor (with fix) | 583.32 | — | 818.66 (Safetensors+Convert: load 709.45 + transform 109.18; peak host RAM 390.7 GB) | 818.66 | 1,401.98 | Only safetensor loads this config |
| qwen3-235b-a22b (470.4 GB MoE) | scanned | 1 (16) | to_maxtext | — | No checkpoint to load (the single-host conversion fails, see 1 (8)) | — | N/A | N/A | N/A |
| ↳ | scanned | 1 (16) | safetensor | 583.32 | — | 908.33 (Safetensors+Convert: load 826.12 + transform 82.18; peak host RAM 220.1 GB) | 908.33 | 1,491.65 | Only safetensor loads this config |
| qwen3-235b-a22b (470.4 GB MoE) | scanned | 2 (16) | to_maxtext | 583.32 | Does not fit one host (the source must be staged in RAM next to the converted state) | — | N/A | N/A | N/A |
| ↳ | scanned | 2 (16) | safetensor | 583.32 | — | 792.38 (Safetensors+Convert: load 707.97 + transform 83.58; peak host RAM 255.8 GB) | 792.38 | 1,375.69 | Only safetensor loads this config |
| qwen3-235b-a22b (470.4 GB MoE) | scanned | 2 (32) | to_maxtext | — | No checkpoint to load (the single-host conversion fails, see 1 (8)) | — | N/A | N/A | N/A |
| ↳ | scanned | 2 (32) | safetensor | 583.32 | — | 800.69 (Safetensors+Convert: load 713.38 + transform 86.77; peak host RAM 182.8 GB) | 800.69 | 1,384.01 | Only safetensor loads this config |
| qwen3-235b-a22b (470.4 GB MoE) | unscanned | 1 (8) | to_maxtext | 583.32 | 2,625.50 (16 parallel shard reads; single stream: 5,150.13) | 545.51 (Load Orbax) | 3,171.02 | 3,754.33 | 1.00x (baseline) |
| ↳ | unscanned | 1 (8) | safetensor | 583.32 | — | 697.13 (Safetensors+Convert: load 661.84 + transform 35.23; peak host RAM 392.7 GB) | 697.13 | 1,280.44 | 2.93x / 4.55x |
| qwen3-235b-a22b (470.4 GB MoE) | unscanned | 1 (16) | to_maxtext | 583.32 | 2,625.50 (shared with 1 (8)) | 841.12 (Load Orbax) | 3,466.62 | 4,049.94 | 1.00x (baseline) |
| ↳ | unscanned | 1 (16) | safetensor | 583.32 | — | 817.59 (Safetensors+Convert: load 775.01 + transform 42.52; peak host RAM 222.0 GB) | 817.59 | 1,400.91 | 2.89x / 4.24x |
| qwen3-235b-a22b (470.4 GB MoE) | unscanned | 2 (16) | to_maxtext | 583.32 | 2,625.50 (shared with 1 (8)) | 659.15 (Load Orbax) | 3,284.65 | 3,867.97 | 1.00x (baseline) |
| ↳ | unscanned | 2 (16) | safetensor | 583.32 | — | 749.00 (Safetensors+Convert: load 704.88 + transform 44.03; peak host RAM 254.6 GB) | 749.00 | 1,332.31 | 2.90x / 4.39x |
| qwen3-235b-a22b (470.4 GB MoE) | unscanned | 2 (32) | to_maxtext | 583.32 | 2,625.50 (shared with 1 (8)) | 682.05 (Load Orbax) | 3,307.56 | 3,890.87 | 1.00x (baseline) |
| ↳ | unscanned | 2 (32) | safetensor | 583.32 | — | 788.66 (Safetensors+Convert: load 736.01 + transform 52.54; peak host RAM 177.7 GB) | 788.66 | 1,371.98 | 2.84x / 4.19x |
| deepseek4-284b (581.9 GB MoE, synthetic sparse files) | scanned | 1 (8) | to_maxtext | 0.45 (local stage) | Host OOM-killed (peak host RAM 989.6 GB) | — | N/A | N/A | N/A |
| ↳ | scanned | 1 (8) | safetensor | 0.00 (local stage) | — | 453.27 (Safetensors+Convert: load 366.34 + transform 86.85; peak host RAM 963.4 GB) | 453.27 | 453.27 | Only safetensor loads this config |
| deepseek4-284b (581.9 GB MoE, synthetic sparse files) | scanned | 1 (16) | to_maxtext | — | No checkpoint to load (the single-host conversion fails, see 1 (8)) | — | N/A | N/A | N/A |
| ↳ | scanned | 1 (16) | safetensor | 0.00 (local stage) | — | 414.93 (Safetensors+Convert: load 327.55 + transform 87.14; peak host RAM 534.7 GB) | 414.93 | 414.93 | Only safetensor loads this config |
| deepseek4-284b (581.9 GB MoE, synthetic sparse files) | scanned | 2 (16) | to_maxtext | 0.45 (local stage) | Host OOM-killed (peak host RAM 989.6 GB) | — | N/A | N/A | N/A |
| ↳ | scanned | 2 (16) | safetensor | 0.00 (local stage) | — | 440.72 (Safetensors+Convert: load 352.94 + transform 87.63; peak host RAM 822.6 GB) | 440.72 | 440.72 | Only safetensor loads this config |
| deepseek4-284b (581.9 GB MoE, synthetic sparse files) | scanned | 2 (32) | to_maxtext | — | No checkpoint to load (the single-host conversion fails, see 1 (8)) | — | N/A | N/A | N/A |
| ↳ | scanned | 2 (32) | safetensor | 0.00 (local stage) | — | 302.91 (Safetensors+Convert: load 209.34 + transform 93.48; peak host RAM 318.4 GB) | 302.91 | 302.91 | Only safetensor loads this config |
| deepseek4-284b (581.9 GB MoE, synthetic sparse files) | unscanned | 1 (8) | to_maxtext | 0.45 (local stage) | Host OOM-killed (peak host RAM 989.6 GB) | — | N/A | N/A | N/A |
| ↳ | unscanned | 1 (8) | safetensor | 0.00 (local stage) | — | 414.09 (Safetensors+Convert: load 376.97 + transform 36.96; peak host RAM 980.3 GB) | 414.09 | 414.09 | Only safetensor loads this config |
| deepseek4-284b (581.9 GB MoE, synthetic sparse files) | unscanned | 1 (16) | to_maxtext | — | No checkpoint to load (the single-host conversion fails, see 1 (8)) | — | N/A | N/A | N/A |
| ↳ | unscanned | 1 (16) | safetensor | 0.00 (local stage) | — | 349.57 (Safetensors+Convert: load 307.45 + transform 41.90; peak host RAM 520.1 GB) | 349.57 | 349.57 | Only safetensor loads this config |
| deepseek4-284b (581.9 GB MoE, synthetic sparse files) | unscanned | 2 (16) | to_maxtext | 0.45 (local stage) | Host OOM-killed (peak host RAM 989.6 GB) | — | N/A | N/A | N/A |
| ↳ | unscanned | 2 (16) | safetensor | 0.00 (local stage) | — | 352.96 (Safetensors+Convert: load 306.30 + transform 46.47; peak host RAM 822.6 GB) | 352.96 | 352.96 | Only safetensor loads this config |
| deepseek4-284b (581.9 GB MoE, synthetic sparse files) | unscanned | 2 (32) | to_maxtext | — | No checkpoint to load (the single-host conversion fails, see 1 (8)) | — | N/A | N/A | N/A |
| ↳ | unscanned | 2 (32) | safetensor | 0.00 (local stage) | — | 258.40 (Safetensors+Convert: load 200.93 + transform 57.23; peak host RAM 318.3 GB) | 258.40 | 258.40 | Only safetensor loads this config |
| qwen3.5-397b-a17b (795.5 GB Hybrid MoE) | scanned | 1 (8) | to_maxtext | 814.53 | Does not fit one host (the source must be staged in RAM next to the converted state) | — | N/A | N/A | N/A |
| ↳ | scanned | 1 (8) | safetensor | 814.53 | — | HBM OOM in transform (load 942.10 s; the 795.5 GB checkpoint fills 98% of the 813.8 GB HBM) | N/A | N/A | N/A |
| qwen3.5-397b-a17b (795.5 GB Hybrid MoE) | scanned | 1 (16) | to_maxtext | — | No checkpoint to load (the single-host conversion fails, see 1 (8)) | — | N/A | N/A | N/A |
| ↳ | scanned | 1 (16) | safetensor | 814.53 | — | HBM OOM in transform (load 488.18 s; 11.25 GiB needed, 2.77 GiB free) | N/A | N/A | N/A |
| ↳ | scanned | 1 (16) | safetensor (with fix) | 814.53 | — | 1,475.17 (Safetensors+Convert: load 483.50 + transform 991.62; peak host RAM 63.8 GB) | 1,475.17 | 2,289.70 | Only safetensor loads this config |
| qwen3.5-397b-a17b (795.5 GB Hybrid MoE) | scanned | 2 (16) | to_maxtext | 814.53 | Does not fit one host (the source must be staged in RAM next to the converted state) | — | N/A | N/A | N/A |
| ↳ | scanned | 2 (16) | safetensor | 814.53 | — | HBM OOM in transform (load 483.84 s; 11.25 GiB needed, 2.77 GiB free; same with hbmfix) | N/A | N/A | N/A |
| ↳ | scanned | 2 (16) | safetensor (with fix) | 814.53 | — | 1,608.32 (Safetensors+Convert: load 513.30 + transform 1,093.42; peak host RAM 93.0 GB) | 1,608.32 | 2,422.85 | Only safetensor loads this config |
| qwen3.5-397b-a17b (795.5 GB Hybrid MoE) | scanned | 1 (32) | to_maxtext | — | No checkpoint to load (the single-host conversion fails, see 1 (8)) | — | N/A | N/A | N/A |
| ↳ | scanned | 1 (32) | safetensor | 814.53 | — | 2,618.13 (Safetensors+Convert: load 261.18 + transform 2,356.85; peak host RAM 88.6 GB) | 2,618.13 | 3,432.66 | Only safetensor loads this config |
| qwen3.5-397b-a17b (795.5 GB Hybrid MoE) | scanned | 2 (32) | to_maxtext | — | No checkpoint to load (the single-host conversion fails, see 1 (8)) | — | N/A | N/A | N/A |
| ↳ | scanned | 2 (32) | safetensor | 814.53 | — | 1,361.95 (Safetensors+Convert: load 248.24 + transform 1,113.67; peak host RAM 130.8 GB) | 1,361.95 | 2,176.48 | Only safetensor loads this config |
| qwen3.5-397b-a17b (795.5 GB Hybrid MoE) | scanned | 4 (32) | to_maxtext | 814.53 | Does not fit one host (the source must be staged in RAM next to the converted state) | — | N/A | N/A | N/A |
| ↳ | scanned | 4 (32) | safetensor | 814.53 | — | 1,399.68 (Safetensors+Convert: load 260.01 + transform 1,139.07; peak host RAM 127.7 GB) | 1,399.68 | 2,214.21 | Only safetensor loads this config |
| qwen3.5-397b-a17b (795.5 GB Hybrid MoE) | unscanned | 1 (8) | to_maxtext | 814.53 | 8,096.53 (with this branch's Orbax save-buffer cap, OOM-killed at the save without it; 16 parallel shard reads) | Not run (the 795.5 GB checkpoint fills 98% of the 813.8 GB HBM) | N/A | N/A | N/A |
| ↳ | unscanned | 1 (8) | safetensor | 814.53 | — | Not run (the 795.5 GB checkpoint fills 98% of the HBM) | N/A | N/A | N/A |
| qwen3.5-397b-a17b (795.5 GB Hybrid MoE) | unscanned | 1 (16) | to_maxtext | 814.53 | 8,096.53 (shared with 1 (8)) | 1,838.26 (Load Orbax) | 9,934.79 | 10,749.32 | 1.00x (baseline) |
| ↳ | unscanned | 1 (16) | safetensor | 814.53 | — | 1,150.92 (Safetensors+Convert: load 484.68 + transform 666.14; peak host RAM 48.9 GB) | 1,150.92 | 1,965.45 | 5.47x / 8.63x |
| qwen3.5-397b-a17b (795.5 GB Hybrid MoE) | unscanned | 2 (16) | to_maxtext | 814.53 | 8,096.53 (shared with 1 (8)) | 2,025.76 (Load Orbax) | 10,122.30 | 10,936.83 | 1.00x (baseline) |
| ↳ | unscanned | 2 (16) | safetensor | 814.53 | — | 1,311.85 (Safetensors+Convert: load 515.91 + transform 795.72; peak host RAM 98.7 GB) | 1,311.85 | 2,126.38 | 5.14x / 7.72x |
| ↳ | unscanned | 2 (16) | safetensor (with fix) | 814.53 | — | 1,272.19 (Safetensors+Convert: load 478.85 + transform 793.27; peak host RAM 98.9 GB) | 1,272.19 | 2,086.72 | 5.24x / 7.96x |
| qwen3.5-397b-a17b (795.5 GB Hybrid MoE) | unscanned | 1 (32) | to_maxtext | 814.53 | 8,096.53 (shared with 1 (8)) | ⏳ Queued (waiting for 1 × 2x2x4 spot slice to load the converted Orbax checkpoint) | ⏳ | ⏳ | ⏳ Queued |
| ↳ | unscanned | 1 (32) | safetensor | 814.53 | — | 954.34 (Safetensors+Convert: load 268.01 + transform 686.26; peak host RAM 88.1 GB) | 954.34 | 1,768.87 | ⏳ Queued (waiting for to_maxtext Step 3) |
| qwen3.5-397b-a17b (795.5 GB Hybrid MoE) | unscanned | 2 (32) | to_maxtext | 814.53 | 8,096.53 (shared with 1 (8)) | Orbax barrier timeout after 2,306 s (3/4 hosts reached _load_impl barrier; >20-min GCS read skew, peak host RAM 95.2 GB) | N/A | N/A | N/A |
| ↳ | unscanned | 2 (32) | safetensor | 814.53 | — | 1,018.83 (Safetensors+Convert: load 266.41 + transform 751.78; peak host RAM 130.5 GB) | 1,018.83 | 1,833.36 | Only safetensor loads this config |
| qwen3-480b-a35b (961.0 GB MoE) | scanned | 1 (8) | to_maxtext | 1,135.50 | Does not fit one host (the source must be staged in RAM next to the converted state) | — | N/A | N/A | N/A |
| ↳ | scanned | 1 (8) | safetensor | 1,046.14 | — | HBM OOM (961.0 GB > 813.8 GB HBM) | N/A | N/A | N/A |
| qwen3-480b-a35b (961.0 GB MoE) | scanned | 1 (16) | to_maxtext | — | No checkpoint to load (the single-host conversion fails, see 1 (8)) | — | N/A | N/A | N/A |
| ↳ | scanned | 1 (16) | safetensor | 1,046.14 | — | 1,298.61 (Safetensors+Convert: load 953.07 + transform 345.50; peak host RAM 380.6 GB) | 1,298.61 | 2,344.75 | Only safetensor loads this config |
| qwen3-480b-a35b (961.0 GB MoE) | scanned | 2 (16) | to_maxtext | 1,135.50 | Does not fit one host (the source must be staged in RAM next to the converted state) | — | N/A | N/A | N/A |
| ↳ | scanned | 2 (16) | safetensor | 1,046.14 | — | 1,067.86 (Safetensors+Convert: load 941.98 + transform 125.78; peak host RAM 415.7 GB) | 1,067.86 | 2,114.00 | Only safetensor loads this config |
| qwen3-480b-a35b (961.0 GB MoE) | scanned | 2 (32) | to_maxtext | — | No checkpoint to load (the single-host conversion fails, see 1 (8)) | — | N/A | N/A | N/A |
| ↳ | scanned | 2 (32) | safetensor | 1,046.14 | — | 757.70 (Safetensors+Convert: load 644.22 + transform 111.22; peak host RAM 244.9 GB) | 757.70 | 1,803.84 | Only safetensor loads this config |
| qwen3-480b-a35b (961.0 GB MoE) | unscanned | 1 (8) | to_maxtext | 1,135.50 | Does not fit one host (the converted state exceeds host RAM) | — | N/A | N/A | N/A |
| ↳ | unscanned | 1 (8) | safetensor | 1,046.14 | — | Not run (961.0 GB > 813.8 GB HBM) | N/A | N/A | N/A |
| qwen3-480b-a35b (961.0 GB MoE) | unscanned | 1 (16) | to_maxtext | — | No checkpoint to load (the single-host conversion fails, see 1 (8)) | — | N/A | N/A | N/A |
| ↳ | unscanned | 1 (16) | safetensor | 1,046.14 | — | 1,231.98 (Safetensors+Convert: load 938.51 + transform 293.40; peak host RAM 380.5 GB) | 1,231.98 | 2,278.12 | Only safetensor loads this config |
| qwen3-480b-a35b (961.0 GB MoE) | unscanned | 2 (16) | to_maxtext | 1,135.50 | Does not fit one host (the converted state exceeds host RAM) | — | N/A | N/A | N/A |
| ↳ | unscanned | 2 (16) | safetensor | 1,046.14 | — | 992.17 (Safetensors+Convert: load 907.65 + transform 84.30; peak host RAM 415.5 GB) | 992.17 | 2,038.31 | Only safetensor loads this config |
| qwen3-480b-a35b (961.0 GB MoE) | unscanned | 2 (32) | to_maxtext | — | No checkpoint to load (the single-host conversion fails, see 1 (8)) | — | N/A | N/A | N/A |
| ↳ | unscanned | 2 (32) | safetensor | 1,046.14 | — | 718.65 (Safetensors+Convert: load 642.10 + transform 76.42; peak host RAM 244.6 GB) | 718.65 | 1,764.79 | Only safetensor loads this config |
| deepseek3-671b (1,342.4 GB MoE, synthetic sparse files) | scanned | 1 (8) | to_maxtext | — | Not run (1.34 TB can't be converted on one host) | — | N/A | N/A | N/A |
| ↳ | scanned | 1 (8) | safetensor | 0.00 (local stage) | — | HBM OOM (1,342.4 GB > 813.8 GB HBM) | N/A | N/A | N/A |
| deepseek3-671b (1,342.4 GB MoE, synthetic sparse files) | scanned | 2 (16) | to_maxtext | — | Not run (1.34 TB can't be converted on one host) | — | N/A | N/A | N/A |
| ↳ | scanned | 2 (16) | safetensor | 0.00 (local stage) | — | Killed (-9) on slice 0 at 742.1 GB sampled peak host RAM; loads on 4 slices | N/A | N/A | N/A |
| deepseek3-671b (1,342.4 GB MoE, synthetic sparse files) | scanned | 1 (32) | to_maxtext | — | Not run (1.34 TB can't be converted on one host) | — | N/A | N/A | N/A |
| ↳ | scanned | 1 (32) | safetensor | 0.00 (local stage) | — | 749.86 (Safetensors+Convert: load 451.28 + transform 298.30; peak host RAM 654.9 GB) | 749.86 | 749.86 | Only safetensor loads this config |
| deepseek3-671b (1,342.4 GB MoE, synthetic sparse files) | scanned | 2 (32) | to_maxtext | — | Not run (1.34 TB can't be converted on one host) | — | N/A | N/A | N/A |
| ↳ | scanned | 2 (32) | safetensor | 0.00 (local stage) | — | 657.75 (Safetensors+Convert: load 490.93 + transform 166.72; peak host RAM 673.7 GB) | 657.75 | 657.75 | Only safetensor loads this config |
| deepseek3-671b (1,342.4 GB MoE, synthetic sparse files) | scanned | 4 (32) | to_maxtext | — | Not run (1.34 TB can't be converted on one host) | — | N/A | N/A | N/A |
| ↳ | scanned | 4 (32) | safetensor | 0.00 (local stage) | — | 649.35 (Safetensors+Convert: load 468.92 + transform 176.84; peak host RAM 689.9 GB) | 649.35 | 649.35 | Only safetensor loads this config |
| deepseek3-671b (1,342.4 GB MoE, synthetic sparse files) | unscanned | 1 (8) / 2 (16) | to_maxtext | — | Not run (1.34 TB can't be converted on one host) | — | N/A | N/A | N/A |
| ↳ | unscanned | 1 (8) / 2 (16) | safetensor | — | — | Skipped (same capacity limits as scanned) | N/A | N/A | N/A |
| deepseek3-671b (1,342.4 GB MoE, synthetic sparse files) | unscanned | 1 (32) | to_maxtext | — | Not run (1.34 TB can't be converted on one host) | — | N/A | N/A | N/A |
| ↳ | unscanned | 1 (32) | safetensor | 0.00 (local stage) | — | 672.66 (Safetensors+Convert: load 468.79 + transform 203.51; peak host RAM 655.0 GB) | 672.66 | 672.66 | Only safetensor loads this config |
| deepseek3-671b (1,342.4 GB MoE, synthetic sparse files) | unscanned | 2 (32) | to_maxtext | — | Not run (1.34 TB can't be converted on one host) | — | N/A | N/A | N/A |
| ↳ | unscanned | 2 (32) | safetensor | 0.00 (local stage) | — | 587.07 (Safetensors+Convert: load 480.07 + transform 106.76; peak host RAM 645.0 GB) | 587.07 | 587.07 | Only safetensor loads this config |
| deepseek3-671b (1,342.4 GB MoE, synthetic sparse files) | unscanned | 4 (32) | to_maxtext | — | Not run (1.34 TB can't be converted on one host) | — | N/A | N/A | N/A |
| ↳ | unscanned | 4 (32) | safetensor | 0.00 (local stage) | — | 530.96 (Safetensors+Convert: load 409.94 + transform 120.74; peak host RAM 689.9 GB) | 530.96 | 530.96 | Only safetensor loads this config |
| deepseek3.2-671b (1,367.0 GB MoE, synthetic sparse files) | scanned | 1 (32) | to_maxtext | — | Not run (1.37 TB can't be converted on one host) | — | N/A | N/A | N/A |
| ↳ | scanned | 1 (32) | safetensor | 0.00 (local stage) | — | ⏳ Queued (waiting for 1 × 2x2x4 spot slice) | ⏳ | ⏳ | ⏳ Queued |
| deepseek3.2-671b (1,367.0 GB MoE, synthetic sparse files) | scanned | 2 (32) | to_maxtext | — | Not run (1.37 TB can't be converted on one host) | — | N/A | N/A | N/A |
| ↳ | scanned | 2 (32) | safetensor | 0.00 (local stage) | — | 672.46 (Safetensors+Convert: load 493.25 + transform 178.76; peak host RAM 653.4 GB) | 672.46 | 672.46 | Only safetensor loads this config |
| deepseek3.2-671b (1,367.0 GB MoE, synthetic sparse files) | unscanned | 1 (32) | to_maxtext | — | Not run (1.37 TB can't be converted on one host) | — | N/A | N/A | N/A |
| ↳ | unscanned | 1 (32) | safetensor | 0.00 (local stage) | — | ⏳ Queued (waiting for 1 × 2x2x4 spot slice) | ⏳ | ⏳ | ⏳ Queued |
| deepseek3.2-671b (1,367.0 GB MoE, synthetic sparse files) | unscanned | 2 (32) | to_maxtext | — | Not run (1.37 TB can't be converted on one host) | — | N/A | N/A | N/A |
| ↳ | unscanned | 2 (32) | safetensor | 0.00 (local stage) | — | 569.71 (Safetensors+Convert: load 459.39 + transform 109.96; peak host RAM 690.9 GB) | 569.71 | 569.71 | Only safetensor loads this config |
| deepseek4-1600b (3,197.7 GB MoE, synthetic sparse files) | scanned | 1 (64) | to_maxtext | — | Not run (3.2 TB can't be converted on one host) | — | N/A | N/A | N/A |
| ↳ | scanned | 1 (64) | safetensor | 0.00 (local stage) | — | ⏳ Queued (waiting for 1 × 2x4x4 spot slice) | ⏳ | ⏳ | ⏳ Queued |
| deepseek4-1600b (3,197.7 GB MoE, synthetic sparse files) | scanned | 4 (64) | to_maxtext | — | Not run (3.2 TB can't be converted on one host) | — | N/A | N/A | N/A |
| ↳ | scanned | 4 (64) | safetensor | 0.00 (local stage) | — | 1,047.93 (Safetensors+Convert: load 757.75 + transform 289.55; peak host RAM 696.1 GB) | 1,047.93 | 1,047.93 | Only safetensor loads this config |
| deepseek4-1600b (3,197.7 GB MoE, synthetic sparse files) | unscanned | 1 (64) | to_maxtext | — | Not run (3.2 TB can't be converted on one host) | — | N/A | N/A | N/A |
| ↳ | unscanned | 1 (64) | safetensor | 0.00 (local stage) | — | ⏳ Queued (waiting for 1 × 2x4x4 spot slice) | ⏳ | ⏳ | ⏳ Queued |
| deepseek4-1600b (3,197.7 GB MoE, synthetic sparse files) | unscanned | 4 (64) | to_maxtext | — | Not run (3.2 TB can't be converted on one host) | — | N/A | N/A | N/A |
| ↳ | unscanned | 4 (64) | safetensor | 0.00 (local stage) | — | 1,008.62 (Safetensors+Convert: load 771.74 + transform 236.49; peak host RAM 676.0 GB) | 1,008.62 | 1,008.62 | Only safetensor loads this config |

<details>
<summary>The same title, legend and table as tab-separated text</summary>

```text
TPU v7x (bodaborg-tpu7x-nap), >200B models: to_maxtext vs safetensor (safetensors_dynamic with PRs #5706, #5828, #5705)

For each (Model, Scan Mode, Slices) configuration, to_maxtext and safetensor are on consecutive rows.
Col 1 (HF -> GCS): to_maxtext Step 1 vs safetensor Step 1, a cold download of the .safetensors shards, measured once and shared by the configurations that reuse it. DeepSeek models read synthetic sparse files staged on each host (local stage).
Col 2 (Convert to Orbax): to_maxtext Step 2, a CPU conversion on one host that every slice count shares. safetensor has no conversion step (—).
Col 3 (Load into MaxText): to_maxtext Step 3 (Load Orbax) vs safetensor Step 2 (Safetensors+Convert = load_sharded_hf_state + transform_hf_state_to_mt_state). Peak host RAM is the runner host's.
Slices (Devices): a tpu7x host has 4 chips = 8 TPU cores, 813.8 GB HBM and 1,014.5 GB RAM. 1 (8), 2 (16), 4 (32) = 1, 2, 4 × 2x2x1 single-host slices (P1 runs). 1 (16) = 1 × 2x2x2, 1 (32) = 1 × 2x2x4, 2 (32) = 2 × 2x2x2, 4 (64) = 4 × 2x2x2, 1 (64) = 1 × 2x4x4 multi-host slices (safetensor: Lance Wang's runs and the 10-30 runs; to_maxtext: Orbax loads of the checkpoint that the single-host conversion wrote, so Steps 1-2 are the ones of 1 (8)).
Speedup: to_maxtext total / safetensor total, end-to-end / without Step 1. N/A: no total because a step failed or was not run.
Status symbols: ⏳ In progress = currently running on the cluster; ⏳ Queued = queued in Kueue waiting for spot slice capacity.
safetensor (with fix): the same configuration re-run with this branch's MaxText fixes for two HBM peaks (composite leaves stacked one by one; reshard slices joined in pairs), from a separate code tree. The plain safetensor rows don't include these fixes.

Model	Scan Mode	Slices (Devices)	Pipeline	1. HF -> GCS (s)	2. Convert to Orbax (s)	3. Load into MaxText (s)	Total (No Step 1) (s)	Total (End-to-End) (s)	Speedup (E2E / No Step 1)
qwen3-235b-a22b (470.4 GB MoE)	scanned	1 (8)	to_maxtext	583.32	Does not fit one host (the source must be staged in RAM next to the converted state)	—	N/A	N/A	N/A
↳	scanned	1 (8)	safetensor	583.32	—	757.50 (Safetensors+Convert: load 674.81 + transform 82.66; peak host RAM 391.8 GB)	757.50	1,340.81	Only safetensor loads this config
↳	scanned	1 (8)	safetensor (with fix)	583.32	—	818.66 (Safetensors+Convert: load 709.45 + transform 109.18; peak host RAM 390.7 GB)	818.66	1,401.98	Only safetensor loads this config
qwen3-235b-a22b (470.4 GB MoE)	scanned	1 (16)	to_maxtext	—	No checkpoint to load (the single-host conversion fails, see 1 (8))	—	N/A	N/A	N/A
↳	scanned	1 (16)	safetensor	583.32	—	908.33 (Safetensors+Convert: load 826.12 + transform 82.18; peak host RAM 220.1 GB)	908.33	1,491.65	Only safetensor loads this config
qwen3-235b-a22b (470.4 GB MoE)	scanned	2 (16)	to_maxtext	583.32	Does not fit one host (the source must be staged in RAM next to the converted state)	—	N/A	N/A	N/A
↳	scanned	2 (16)	safetensor	583.32	—	792.38 (Safetensors+Convert: load 707.97 + transform 83.58; peak host RAM 255.8 GB)	792.38	1,375.69	Only safetensor loads this config
qwen3-235b-a22b (470.4 GB MoE)	scanned	2 (32)	to_maxtext	—	No checkpoint to load (the single-host conversion fails, see 1 (8))	—	N/A	N/A	N/A
↳	scanned	2 (32)	safetensor	583.32	—	800.69 (Safetensors+Convert: load 713.38 + transform 86.77; peak host RAM 182.8 GB)	800.69	1,384.01	Only safetensor loads this config
qwen3-235b-a22b (470.4 GB MoE)	unscanned	1 (8)	to_maxtext	583.32	2,625.50 (16 parallel shard reads; single stream: 5,150.13)	545.51 (Load Orbax)	3,171.02	3,754.33	1.00x (baseline)
↳	unscanned	1 (8)	safetensor	583.32	—	697.13 (Safetensors+Convert: load 661.84 + transform 35.23; peak host RAM 392.7 GB)	697.13	1,280.44	2.93x / 4.55x
qwen3-235b-a22b (470.4 GB MoE)	unscanned	1 (16)	to_maxtext	583.32	2,625.50 (shared with 1 (8))	841.12 (Load Orbax)	3,466.62	4,049.94	1.00x (baseline)
↳	unscanned	1 (16)	safetensor	583.32	—	817.59 (Safetensors+Convert: load 775.01 + transform 42.52; peak host RAM 222.0 GB)	817.59	1,400.91	2.89x / 4.24x
qwen3-235b-a22b (470.4 GB MoE)	unscanned	2 (16)	to_maxtext	583.32	2,625.50 (shared with 1 (8))	659.15 (Load Orbax)	3,284.65	3,867.97	1.00x (baseline)
↳	unscanned	2 (16)	safetensor	583.32	—	749.00 (Safetensors+Convert: load 704.88 + transform 44.03; peak host RAM 254.6 GB)	749.00	1,332.31	2.90x / 4.39x
qwen3-235b-a22b (470.4 GB MoE)	unscanned	2 (32)	to_maxtext	583.32	2,625.50 (shared with 1 (8))	682.05 (Load Orbax)	3,307.56	3,890.87	1.00x (baseline)
↳	unscanned	2 (32)	safetensor	583.32	—	788.66 (Safetensors+Convert: load 736.01 + transform 52.54; peak host RAM 177.7 GB)	788.66	1,371.98	2.84x / 4.19x
deepseek4-284b (581.9 GB MoE, synthetic sparse files)	scanned	1 (8)	to_maxtext	0.45 (local stage)	Host OOM-killed (peak host RAM 989.6 GB)	—	N/A	N/A	N/A
↳	scanned	1 (8)	safetensor	0.00 (local stage)	—	453.27 (Safetensors+Convert: load 366.34 + transform 86.85; peak host RAM 963.4 GB)	453.27	453.27	Only safetensor loads this config
deepseek4-284b (581.9 GB MoE, synthetic sparse files)	scanned	1 (16)	to_maxtext	—	No checkpoint to load (the single-host conversion fails, see 1 (8))	—	N/A	N/A	N/A
↳	scanned	1 (16)	safetensor	0.00 (local stage)	—	414.93 (Safetensors+Convert: load 327.55 + transform 87.14; peak host RAM 534.7 GB)	414.93	414.93	Only safetensor loads this config
deepseek4-284b (581.9 GB MoE, synthetic sparse files)	scanned	2 (16)	to_maxtext	0.45 (local stage)	Host OOM-killed (peak host RAM 989.6 GB)	—	N/A	N/A	N/A
↳	scanned	2 (16)	safetensor	0.00 (local stage)	—	440.72 (Safetensors+Convert: load 352.94 + transform 87.63; peak host RAM 822.6 GB)	440.72	440.72	Only safetensor loads this config
deepseek4-284b (581.9 GB MoE, synthetic sparse files)	scanned	2 (32)	to_maxtext	—	No checkpoint to load (the single-host conversion fails, see 1 (8))	—	N/A	N/A	N/A
↳	scanned	2 (32)	safetensor	0.00 (local stage)	—	302.91 (Safetensors+Convert: load 209.34 + transform 93.48; peak host RAM 318.4 GB)	302.91	302.91	Only safetensor loads this config
deepseek4-284b (581.9 GB MoE, synthetic sparse files)	unscanned	1 (8)	to_maxtext	0.45 (local stage)	Host OOM-killed (peak host RAM 989.6 GB)	—	N/A	N/A	N/A
↳	unscanned	1 (8)	safetensor	0.00 (local stage)	—	414.09 (Safetensors+Convert: load 376.97 + transform 36.96; peak host RAM 980.3 GB)	414.09	414.09	Only safetensor loads this config
deepseek4-284b (581.9 GB MoE, synthetic sparse files)	unscanned	1 (16)	to_maxtext	—	No checkpoint to load (the single-host conversion fails, see 1 (8))	—	N/A	N/A	N/A
↳	unscanned	1 (16)	safetensor	0.00 (local stage)	—	349.57 (Safetensors+Convert: load 307.45 + transform 41.90; peak host RAM 520.1 GB)	349.57	349.57	Only safetensor loads this config
deepseek4-284b (581.9 GB MoE, synthetic sparse files)	unscanned	2 (16)	to_maxtext	0.45 (local stage)	Host OOM-killed (peak host RAM 989.6 GB)	—	N/A	N/A	N/A
↳	unscanned	2 (16)	safetensor	0.00 (local stage)	—	352.96 (Safetensors+Convert: load 306.30 + transform 46.47; peak host RAM 822.6 GB)	352.96	352.96	Only safetensor loads this config
deepseek4-284b (581.9 GB MoE, synthetic sparse files)	unscanned	2 (32)	to_maxtext	—	No checkpoint to load (the single-host conversion fails, see 1 (8))	—	N/A	N/A	N/A
↳	unscanned	2 (32)	safetensor	0.00 (local stage)	—	258.40 (Safetensors+Convert: load 200.93 + transform 57.23; peak host RAM 318.3 GB)	258.40	258.40	Only safetensor loads this config
qwen3.5-397b-a17b (795.5 GB Hybrid MoE)	scanned	1 (8)	to_maxtext	814.53	Does not fit one host (the source must be staged in RAM next to the converted state)	—	N/A	N/A	N/A
↳	scanned	1 (8)	safetensor	814.53	—	HBM OOM in transform (load 942.10 s; the 795.5 GB checkpoint fills 98% of the 813.8 GB HBM)	N/A	N/A	N/A
qwen3.5-397b-a17b (795.5 GB Hybrid MoE)	scanned	1 (16)	to_maxtext	—	No checkpoint to load (the single-host conversion fails, see 1 (8))	—	N/A	N/A	N/A
↳	scanned	1 (16)	safetensor	814.53	—	HBM OOM in transform (load 488.18 s; 11.25 GiB needed, 2.77 GiB free)	N/A	N/A	N/A
↳	scanned	1 (16)	safetensor (with fix)	814.53	—	1,475.17 (Safetensors+Convert: load 483.50 + transform 991.62; peak host RAM 63.8 GB)	1,475.17	2,289.70	Only safetensor loads this config
qwen3.5-397b-a17b (795.5 GB Hybrid MoE)	scanned	2 (16)	to_maxtext	814.53	Does not fit one host (the source must be staged in RAM next to the converted state)	—	N/A	N/A	N/A
↳	scanned	2 (16)	safetensor	814.53	—	HBM OOM in transform (load 483.84 s; 11.25 GiB needed, 2.77 GiB free; same with hbmfix)	N/A	N/A	N/A
↳	scanned	2 (16)	safetensor (with fix)	814.53	—	1,608.32 (Safetensors+Convert: load 513.30 + transform 1,093.42; peak host RAM 93.0 GB)	1,608.32	2,422.85	Only safetensor loads this config
qwen3.5-397b-a17b (795.5 GB Hybrid MoE)	scanned	1 (32)	to_maxtext	—	No checkpoint to load (the single-host conversion fails, see 1 (8))	—	N/A	N/A	N/A
↳	scanned	1 (32)	safetensor	814.53	—	2,618.13 (Safetensors+Convert: load 261.18 + transform 2,356.85; peak host RAM 88.6 GB)	2,618.13	3,432.66	Only safetensor loads this config
qwen3.5-397b-a17b (795.5 GB Hybrid MoE)	scanned	2 (32)	to_maxtext	—	No checkpoint to load (the single-host conversion fails, see 1 (8))	—	N/A	N/A	N/A
↳	scanned	2 (32)	safetensor	814.53	—	1,361.95 (Safetensors+Convert: load 248.24 + transform 1,113.67; peak host RAM 130.8 GB)	1,361.95	2,176.48	Only safetensor loads this config
qwen3.5-397b-a17b (795.5 GB Hybrid MoE)	scanned	4 (32)	to_maxtext	814.53	Does not fit one host (the source must be staged in RAM next to the converted state)	—	N/A	N/A	N/A
↳	scanned	4 (32)	safetensor	814.53	—	1,399.68 (Safetensors+Convert: load 260.01 + transform 1,139.07; peak host RAM 127.7 GB)	1,399.68	2,214.21	Only safetensor loads this config
qwen3.5-397b-a17b (795.5 GB Hybrid MoE)	unscanned	1 (8)	to_maxtext	814.53	8,096.53 (with this branch's Orbax save-buffer cap, OOM-killed at the save without it; 16 parallel shard reads)	Not run (the 795.5 GB checkpoint fills 98% of the 813.8 GB HBM)	N/A	N/A	N/A
↳	unscanned	1 (8)	safetensor	814.53	—	Not run (the 795.5 GB checkpoint fills 98% of the HBM)	N/A	N/A	N/A
qwen3.5-397b-a17b (795.5 GB Hybrid MoE)	unscanned	1 (16)	to_maxtext	814.53	8,096.53 (shared with 1 (8))	1,838.26 (Load Orbax)	9,934.79	10,749.32	1.00x (baseline)
↳	unscanned	1 (16)	safetensor	814.53	—	1,150.92 (Safetensors+Convert: load 484.68 + transform 666.14; peak host RAM 48.9 GB)	1,150.92	1,965.45	5.47x / 8.63x
qwen3.5-397b-a17b (795.5 GB Hybrid MoE)	unscanned	2 (16)	to_maxtext	814.53	8,096.53 (shared with 1 (8))	2,025.76 (Load Orbax)	10,122.30	10,936.83	1.00x (baseline)
↳	unscanned	2 (16)	safetensor	814.53	—	1,311.85 (Safetensors+Convert: load 515.91 + transform 795.72; peak host RAM 98.7 GB)	1,311.85	2,126.38	5.14x / 7.72x
↳	unscanned	2 (16)	safetensor (with fix)	814.53	—	1,272.19 (Safetensors+Convert: load 478.85 + transform 793.27; peak host RAM 98.9 GB)	1,272.19	2,086.72	5.24x / 7.96x
qwen3.5-397b-a17b (795.5 GB Hybrid MoE)	unscanned	1 (32)	to_maxtext	814.53	8,096.53 (shared with 1 (8))	⏳ Queued (waiting for 1 × 2x2x4 spot slice to load the converted Orbax checkpoint)	⏳	⏳	⏳ Queued
↳	unscanned	1 (32)	safetensor	814.53	—	954.34 (Safetensors+Convert: load 268.01 + transform 686.26; peak host RAM 88.1 GB)	954.34	1,768.87	⏳ Queued (waiting for to_maxtext Step 3)
qwen3.5-397b-a17b (795.5 GB Hybrid MoE)	unscanned	2 (32)	to_maxtext	814.53	8,096.53 (shared with 1 (8))	Orbax barrier timeout after 2,306 s (3/4 hosts reached _load_impl barrier; >20-min GCS read skew, peak host RAM 95.2 GB)	N/A	N/A	N/A
↳	unscanned	2 (32)	safetensor	814.53	—	1,018.83 (Safetensors+Convert: load 266.41 + transform 751.78; peak host RAM 130.5 GB)	1,018.83	1,833.36	Only safetensor loads this config
qwen3-480b-a35b (961.0 GB MoE)	scanned	1 (8)	to_maxtext	1,135.50	Does not fit one host (the source must be staged in RAM next to the converted state)	—	N/A	N/A	N/A
↳	scanned	1 (8)	safetensor	1,046.14	—	HBM OOM (961.0 GB > 813.8 GB HBM)	N/A	N/A	N/A
qwen3-480b-a35b (961.0 GB MoE)	scanned	1 (16)	to_maxtext	—	No checkpoint to load (the single-host conversion fails, see 1 (8))	—	N/A	N/A	N/A
↳	scanned	1 (16)	safetensor	1,046.14	—	1,298.61 (Safetensors+Convert: load 953.07 + transform 345.50; peak host RAM 380.6 GB)	1,298.61	2,344.75	Only safetensor loads this config
qwen3-480b-a35b (961.0 GB MoE)	scanned	2 (16)	to_maxtext	1,135.50	Does not fit one host (the source must be staged in RAM next to the converted state)	—	N/A	N/A	N/A
↳	scanned	2 (16)	safetensor	1,046.14	—	1,067.86 (Safetensors+Convert: load 941.98 + transform 125.78; peak host RAM 415.7 GB)	1,067.86	2,114.00	Only safetensor loads this config
qwen3-480b-a35b (961.0 GB MoE)	scanned	2 (32)	to_maxtext	—	No checkpoint to load (the single-host conversion fails, see 1 (8))	—	N/A	N/A	N/A
↳	scanned	2 (32)	safetensor	1,046.14	—	757.70 (Safetensors+Convert: load 644.22 + transform 111.22; peak host RAM 244.9 GB)	757.70	1,803.84	Only safetensor loads this config
qwen3-480b-a35b (961.0 GB MoE)	unscanned	1 (8)	to_maxtext	1,135.50	Does not fit one host (the converted state exceeds host RAM)	—	N/A	N/A	N/A
↳	unscanned	1 (8)	safetensor	1,046.14	—	Not run (961.0 GB > 813.8 GB HBM)	N/A	N/A	N/A
qwen3-480b-a35b (961.0 GB MoE)	unscanned	1 (16)	to_maxtext	—	No checkpoint to load (the single-host conversion fails, see 1 (8))	—	N/A	N/A	N/A
↳	unscanned	1 (16)	safetensor	1,046.14	—	1,231.98 (Safetensors+Convert: load 938.51 + transform 293.40; peak host RAM 380.5 GB)	1,231.98	2,278.12	Only safetensor loads this config
qwen3-480b-a35b (961.0 GB MoE)	unscanned	2 (16)	to_maxtext	1,135.50	Does not fit one host (the converted state exceeds host RAM)	—	N/A	N/A	N/A
↳	unscanned	2 (16)	safetensor	1,046.14	—	992.17 (Safetensors+Convert: load 907.65 + transform 84.30; peak host RAM 415.5 GB)	992.17	2,038.31	Only safetensor loads this config
qwen3-480b-a35b (961.0 GB MoE)	unscanned	2 (32)	to_maxtext	—	No checkpoint to load (the single-host conversion fails, see 1 (8))	—	N/A	N/A	N/A
↳	unscanned	2 (32)	safetensor	1,046.14	—	718.65 (Safetensors+Convert: load 642.10 + transform 76.42; peak host RAM 244.6 GB)	718.65	1,764.79	Only safetensor loads this config
deepseek3-671b (1,342.4 GB MoE, synthetic sparse files)	scanned	1 (8)	to_maxtext	—	Not run (1.34 TB can't be converted on one host)	—	N/A	N/A	N/A
↳	scanned	1 (8)	safetensor	0.00 (local stage)	—	HBM OOM (1,342.4 GB > 813.8 GB HBM)	N/A	N/A	N/A
deepseek3-671b (1,342.4 GB MoE, synthetic sparse files)	scanned	2 (16)	to_maxtext	—	Not run (1.34 TB can't be converted on one host)	—	N/A	N/A	N/A
↳	scanned	2 (16)	safetensor	0.00 (local stage)	—	Killed (-9) on slice 0 at 742.1 GB sampled peak host RAM; loads on 4 slices	N/A	N/A	N/A
deepseek3-671b (1,342.4 GB MoE, synthetic sparse files)	scanned	1 (32)	to_maxtext	—	Not run (1.34 TB can't be converted on one host)	—	N/A	N/A	N/A
↳	scanned	1 (32)	safetensor	0.00 (local stage)	—	749.86 (Safetensors+Convert: load 451.28 + transform 298.30; peak host RAM 654.9 GB)	749.86	749.86	Only safetensor loads this config
deepseek3-671b (1,342.4 GB MoE, synthetic sparse files)	scanned	2 (32)	to_maxtext	—	Not run (1.34 TB can't be converted on one host)	—	N/A	N/A	N/A
↳	scanned	2 (32)	safetensor	0.00 (local stage)	—	657.75 (Safetensors+Convert: load 490.93 + transform 166.72; peak host RAM 673.7 GB)	657.75	657.75	Only safetensor loads this config
deepseek3-671b (1,342.4 GB MoE, synthetic sparse files)	scanned	4 (32)	to_maxtext	—	Not run (1.34 TB can't be converted on one host)	—	N/A	N/A	N/A
↳	scanned	4 (32)	safetensor	0.00 (local stage)	—	649.35 (Safetensors+Convert: load 468.92 + transform 176.84; peak host RAM 689.9 GB)	649.35	649.35	Only safetensor loads this config
deepseek3-671b (1,342.4 GB MoE, synthetic sparse files)	unscanned	1 (8) / 2 (16)	to_maxtext	—	Not run (1.34 TB can't be converted on one host)	—	N/A	N/A	N/A
↳	unscanned	1 (8) / 2 (16)	safetensor	—	—	Skipped (same capacity limits as scanned)	N/A	N/A	N/A
deepseek3-671b (1,342.4 GB MoE, synthetic sparse files)	unscanned	1 (32)	to_maxtext	—	Not run (1.34 TB can't be converted on one host)	—	N/A	N/A	N/A
↳	unscanned	1 (32)	safetensor	0.00 (local stage)	—	672.66 (Safetensors+Convert: load 468.79 + transform 203.51; peak host RAM 655.0 GB)	672.66	672.66	Only safetensor loads this config
deepseek3-671b (1,342.4 GB MoE, synthetic sparse files)	unscanned	2 (32)	to_maxtext	—	Not run (1.34 TB can't be converted on one host)	—	N/A	N/A	N/A
↳	unscanned	2 (32)	safetensor	0.00 (local stage)	—	587.07 (Safetensors+Convert: load 480.07 + transform 106.76; peak host RAM 645.0 GB)	587.07	587.07	Only safetensor loads this config
deepseek3-671b (1,342.4 GB MoE, synthetic sparse files)	unscanned	4 (32)	to_maxtext	—	Not run (1.34 TB can't be converted on one host)	—	N/A	N/A	N/A
↳	unscanned	4 (32)	safetensor	0.00 (local stage)	—	530.96 (Safetensors+Convert: load 409.94 + transform 120.74; peak host RAM 689.9 GB)	530.96	530.96	Only safetensor loads this config
deepseek3.2-671b (1,367.0 GB MoE, synthetic sparse files)	scanned	1 (32)	to_maxtext	—	Not run (1.37 TB can't be converted on one host)	—	N/A	N/A	N/A
↳	scanned	1 (32)	safetensor	0.00 (local stage)	—	⏳ Queued (waiting for 1 × 2x2x4 spot slice)	⏳	⏳	⏳ Queued
deepseek3.2-671b (1,367.0 GB MoE, synthetic sparse files)	scanned	2 (32)	to_maxtext	—	Not run (1.37 TB can't be converted on one host)	—	N/A	N/A	N/A
↳	scanned	2 (32)	safetensor	0.00 (local stage)	—	672.46 (Safetensors+Convert: load 493.25 + transform 178.76; peak host RAM 653.4 GB)	672.46	672.46	Only safetensor loads this config
deepseek3.2-671b (1,367.0 GB MoE, synthetic sparse files)	unscanned	1 (32)	to_maxtext	—	Not run (1.37 TB can't be converted on one host)	—	N/A	N/A	N/A
↳	unscanned	1 (32)	safetensor	0.00 (local stage)	—	⏳ Queued (waiting for 1 × 2x2x4 spot slice)	⏳	⏳	⏳ Queued
deepseek3.2-671b (1,367.0 GB MoE, synthetic sparse files)	unscanned	2 (32)	to_maxtext	—	Not run (1.37 TB can't be converted on one host)	—	N/A	N/A	N/A
↳	unscanned	2 (32)	safetensor	0.00 (local stage)	—	569.71 (Safetensors+Convert: load 459.39 + transform 109.96; peak host RAM 690.9 GB)	569.71	569.71	Only safetensor loads this config
deepseek4-1600b (3,197.7 GB MoE, synthetic sparse files)	scanned	1 (64)	to_maxtext	—	Not run (3.2 TB can't be converted on one host)	—	N/A	N/A	N/A
↳	scanned	1 (64)	safetensor	0.00 (local stage)	—	⏳ Queued (waiting for 1 × 2x4x4 spot slice)	⏳	⏳	⏳ Queued
deepseek4-1600b (3,197.7 GB MoE, synthetic sparse files)	scanned	4 (64)	to_maxtext	—	Not run (3.2 TB can't be converted on one host)	—	N/A	N/A	N/A
↳	scanned	4 (64)	safetensor	0.00 (local stage)	—	1,047.93 (Safetensors+Convert: load 757.75 + transform 289.55; peak host RAM 696.1 GB)	1,047.93	1,047.93	Only safetensor loads this config
deepseek4-1600b (3,197.7 GB MoE, synthetic sparse files)	unscanned	1 (64)	to_maxtext	—	Not run (3.2 TB can't be converted on one host)	—	N/A	N/A	N/A
↳	unscanned	1 (64)	safetensor	0.00 (local stage)	—	⏳ Queued (waiting for 1 × 2x4x4 spot slice)	⏳	⏳	⏳ Queued
deepseek4-1600b (3,197.7 GB MoE, synthetic sparse files)	unscanned	4 (64)	to_maxtext	—	Not run (3.2 TB can't be converted on one host)	—	N/A	N/A	N/A
↳	unscanned	4 (64)	safetensor	0.00 (local stage)	—	1,008.62 (Safetensors+Convert: load 771.74 + transform 236.49; peak host RAM 676.0 GB)	1,008.62	1,008.62	Only safetensor loads this config
```

</details>
<!-- SHEET TABLE END -->

The text between the `SHEET TABLE` markers is generated by
[`scripts/build_tpu7x_sheet_table.py`](../scripts/build_tpu7x_sheet_table.py) (`--suite p1`) from the P1 result JSONs and
[`tpu7x_multihost_results.csv`](tpu7x_multihost_results.csv). The models up to 120B are in the same format in
[`tpu7x_benchmark_ckpt_load_time_summary.md`](tpu7x_benchmark_ckpt_load_time_summary.md) §3.1.
- Unscanned 235B `to_maxtext` values come from the re-run with 16 parallel shard reads (`run_tpu7x_200b_pstream`). Unscanned
  397B comes from the conversion with this branch's Orbax save-buffer cap (`run_tpu7x_200b_tmfix`, issue U-8); without
  the cap, the same conversion was OOM-killed when its save started.
- `safetensor (with fix)` rows come from `run_tpu7x_200b_fix` (this branch's fixes for issues U-1 and U-6); the
  multi-host one from `run_tpu7x_multihost_fix`.
- Multi-host `to_maxtext` rows load the single-host conversion's checkpoint (`<scan>[_slices<N>]_hosts<H>` results in
  the same run dirs).
- The multi-host runs read the shard cache that the P1 runs had already filled in GCS, so their end-to-end totals add
  the model's cold Step 1 from the P1 runs.
- A failed step shows its reason in that step's column. Configurations without any record are listed under "All tests"
  and "Not run".

## All tests

**WITH S2** is `safetensors_dynamic` Step 2 (sharded load + transform). **TM S2** is the `to_maxtext` CPU conversion,
run once per scan mode on one host and shared by the 1- and 2-slice loads. **TM S3** is the Orbax load into HBM.

| Model | Config | 1A BEFORE (`origin/main`) | 1A WITH PRs: WITH S2 (load + transform) | 1B TM S2 convert | 1B TM S3 load |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `qwen3-235b-a22b` | scanned · 1 | ❌ **timeout at 1800 s**, still loading (peak host RAM 528.0 GB) | ✅ **757.5 s** (674.8 + 82.7) | ❌ doesn't fit one host: 470 GB staged + 470 GB converted state (the full-staging attempt was OOM-killed at leaf 4/15); streaming thrashes | ➖ no checkpoint |
| | unscanned · 1 | ➖ skipped after the scanned timeout | ✅ **697.1 s** (661.8 + 35.2) | ✅ **2,625.5 s** (16 parallel shard reads: 569 GB in 1,538 s; peak host RAM 671.5 GB. Single stream: 5,150.1 s) | ✅ **545.5 s** |
| | scanned · 2 | ❌ **timeout at 1800 s**, still loading (peak host RAM 300.8 GB) | ✅ **792.4 s** (708.0 + 83.6) | (shared) | ➖ no checkpoint |
| | unscanned · 2 | ➖ skipped after the scanned timeout | ✅ **749.0 s** (704.9 + 44.0) | (shared) | ✅ **659.1 s** |
| `deepseek4-284b` | scanned · 1 | ❌ **timeout at 1800 s**, still loading (peak host RAM 501.4 GB) | ✅ **453.3 s** (366.3 + 86.8) | ❌ OOM-killed at 989.6 GB host RAM | ➖ no checkpoint |
| | unscanned · 1 | ➖ skipped after the scanned timeout | ✅ **414.1 s** (377.0 + 37.0) | ❌ OOM-killed at 989.6 GB host RAM | ➖ no checkpoint |
| | scanned · 2 | ❌ **timeout at 1800 s**, still loading (peak host RAM 576.0 GB) | ✅ **440.7 s** (352.9 + 87.6) | (shared) | ➖ no checkpoint |
| | unscanned · 2 | ➖ skipped after the scanned timeout | ✅ **353.0 s** (306.3 + 46.5) | (shared) | ➖ no checkpoint |
| `qwen3.5-397b-a17b` | scanned · 1 | ⚠️ killed at 360 s | ❌ load finished (942.1 s), then TPU HBM OOM in the transform (795 GB ≈ 98% of HBM) | ❌ doesn't fit one host (the 807 GB source must be staged in RAM next to the converted state) | ➖ |
| | unscanned · 1 | ➖ | ➖ does not fit 1 slice | ❌ host OOM-killed at the start of the Orbax save, after 6,860 s of transform (~807 GB converted state); single-stream attempts hit the ~3 h workload limit | ➖ no checkpoint |
| | scanned · 2 | ⚠️ killed at 360 s (WITH load alone takes 484 s) | ❌ load finished (483.8 s), then HBM OOM splitting `wi_0`/`wi_1`. `hbmfix`: load 469.7 s, then the same 11.25 GiB allocation fails (2.77 GiB free). ✅ **with the fix** (composite leaves stacked one by one; U-1): **1,608.3 s** (513.3 + 1,093.4; peak host RAM 93.0 GB) | ➖ | ➖ |
| | unscanned · 2 | ❌ **timeout at 1800 s** while still loading (WITH load: 515.9 s; peak host RAM 301.7 GB vs 98.7 GB WITH) | ✅ **1,311.9 s** (515.9 + 795.7; peak host RAM 98.7 GB). With the fix (regression check): 1,272.4 s (478.9 + 793.3; 98.9 GB) | (shared) | ➖ no checkpoint |
| | scanned · 4 | ❌ **timeout at 1800 s**, still loading (peak host RAM 316.6 GB) | ✅ **1,399.7 s** (260.0 + 1,139.1; peak host RAM 127.7 GB) | ➖ | ➖ |
| `qwen3-480b-a35b` | scanned · 1 | ❌ `KeyError: 'qwen3-480b-a35b'` | ❌ TPU HBM OOM (961 GB > 813.8 GB) | ❌ doesn't fit one host (all 961 GB must be staged) | ➖ |
| | unscanned · 1 | ❌ `KeyError` (skipped after scanned) | ➖ does not fit 1 slice | ❌ the 960 GB converted state exceeds ~943 GB usable host RAM (fails fast) | ➖ does not fit 1 slice |
| | scanned · 2 | ❌ `KeyError` | ✅ **1,067.9 s** (942.0 + 125.8) | ➖ | ➖ |
| | unscanned · 2 | ❌ `KeyError` (skipped) | ✅ **992.2 s** (907.7 + 84.3) | (shared) | ➖ no checkpoint |
| `deepseek3-671b` | scanned · 1 | ⚠️ killed at 360 s | ❌ exceeds 1-slice HBM (1,342 GB > 813.8 GB) | ➖ | ➖ |
| | scanned · 2 | ⚠️ killed at 360 s | ❌ host OOM-killed at 742.1 GB sampled peak; needs ≥ 4 slices | ➖ | ➖ |
| | unscanned · 1 / 2 | ➖ skipped | ➖ same capacity limits | ➖ | ➖ |
| | scanned · 4 | ❌ **timeout at 1800 s**, still loading (peak host RAM 344.3 GB; a first attempt was cut off by a spot preemption) | ✅ **649.4 s** (468.9 + 176.8; peak host RAM 689.9 GB) | ➖ | ➖ |
| | unscanned · 4 | ➖ skipped after the scanned timeout | ✅ **531.0 s** (409.9 + 120.7; peak host RAM 689.9 GB) | ➖ | ➖ |

## Notes
- **Peak host RAM (WITH PRs)**:
  - 235B: 392 GB on 1 slice, 255 GB on 2 slices.
  - 284B: 963–980 GB on 1 slice, 823 GB on 2 slices (synthetic sparse files staged in local RAM).
  - 397B: 99 GB on 2 slices (unscanned).
  - 480B: 416 GB on 2 slices.
  - 671B: 690 GB per host on 4 slices.
- **BEFORE watchdog**: the original BEFORE runs used `PHASE_TIMEOUT_S=360`, shorter than the WITH-PRs load times (306–942 s), so their kills don't show that `origin/main` fails. The 1800 s re-runs (`before_*` tasks) resolve this. Conclusive BEFORE failures: `qwen3-480b-a35b` (`KeyError`), and seven runs that timed out at 1800 s while still loading: 235B scanned · 1 and · 2, 284B scanned · 1 and · 2, 397B unscanned · 2, 397B scanned · 4, 671B scanned · 4. With the PRs, the same load phases take 260–708 s. The harness skips unscanned after a scanned timeout at the same slice count.
- **DeepSeek numbers** use synthetic sparse BF16 files staged on the host (`STAGE_SPARSE_DEEPSEEK_OOM=1`). They measure load/transform mechanics and capacity, not GCS bandwidth.
- **`to_maxtext` S1** is the same HF→GCS download as `safetensors_dynamic` S1, measured once per model (480B's TM S1 was measured separately at 1,135.5 s). TM S2 conversions that re-stage the checkpoint locally now record `hf_to_local_restage_s`.
- **Why `to_maxtext` S2 fails for >200B models on one host**: `to_maxtext` keeps the whole converted state (about the checkpoint size) in host RAM before the Orbax save. A tpu7x host has 1,014.5 GB RAM, of which 986.7 GB is allocatable to pods, and a ~100 GB boot disk, so the source shards can only be staged in RAM.
  - Scanned conversions read every shard for every MoE expert, so they need the whole source staged: source plus converted state exceeds host RAM from 235B up, and streaming through a small shard cache thrashes.
  - Unscanned conversions read one layer at a time, so shards can be streamed. Only 235B fits. 397B finished its transform, then the pod was OOM-killed when the Orbax save started (~807 GB converted state plus save buffers), and 480B (960 GB) fails the RAM check up front.
  - 284B (synthetic sparse files) was OOM-killed at 989.6 GB in both scan modes.
- **`qwen3.5-397b-a17b` scanned on 2 slices** runs out of HBM while splitting the stacked `(wi_0, wi_1)` composite (11.25 GiB needed, 2.77 GiB free), both with Lance's PRs and with [`patches/hbmfix`](../patches/hbmfix.patch). This branch's fix stacks `wi_0` and `wi_1` one by one, so the composite never exists (issue U-1): scanned on 2 slices then loads in 1,608.3 s (load 513.3 + transform 1,093.4), and unscanned on 2 slices is unchanged (1,272.4 s vs 1,311.9 s). With 4 slices of HBM headroom and no fix, scanned loads in 1,399.7 s; the transform takes 1,139.1 s of that, vs 795.7 s for unscanned on 2 slices.

## Not run
- **Models on single-host slices**: `deepseek3.2-671b` (1,367 GB) and `deepseek4-1600b` (3,198 GB), the two extra
  DeepSeek capacity tests from the v5p matrix. They weren't in the tpu7x Priority 1/2 lists. Lance's multi-host runs
  loaded both with `safetensor` ([`tpu7x_multihost_results.md`](tpu7x_multihost_results.md)): `deepseek3.2-671b` on
  2 × `2x2x2` in 672.46 s scanned and 569.71 s unscanned, `deepseek4-1600b` on 4 × `2x2x2` in 1,047.93 s and
  1,008.62 s. `to_maxtext` and BEFORE weren't run for them.
- **Configurations of the 5 models**:
  - `deepseek3-671b`: unscanned on 1–2 slices (same capacity limits as scanned), and `to_maxtext` (1.34 TB can't be converted on one host).
  - `qwen3.5-397b-a17b` unscanned on 1 slice for `safetensor` (98% of HBM).
  - `qwen3-480b-a35b` unscanned on 1 slice (exceeds HBM).
  - BEFORE unscanned variants after a scanned timeout at the same slice count (harness rule).

## How to refresh
```bash
# Per-phase results (any run dir) -> status matrix
gcloud storage rsync -r gs://mesa-maxtext/mesakh_tpu7x_bench/p1_lance_prs/run_tpu7x_200b/logs /tmp/p1 --exclude='.*\.log$'
python3 checkpoint_load_benchmark/scripts/summarize_tpu7x_results.py /tmp/p1
# Queue: finished tasks / locks
gcloud storage ls gs://mesa-maxtext/mesakh_tpu7x_bench/queue/p1/done/ gs://mesa-maxtext/mesakh_tpu7x_bench/queue/p1/locks/
# The row-by-row "tpu-v7x" sheet section (needs the logs too, for failure reasons)
gcloud storage rsync -r gs://mesa-maxtext/mesakh_tpu7x_bench/p1_lance_prs/run_tpu7x_200b/logs /tmp/t7/p1
gcloud storage rsync -r gs://mesa-maxtext/mesakh_tpu7x_bench/p1_lance_prs/run_tpu7x_200b_pstream/logs /tmp/t7/p1_pstream
gcloud storage rsync -r gs://mesa-maxtext/mesakh_tpu7x_bench/p1_lance_prs/run_tpu7x_200b_tmfix/logs /tmp/t7/p1_tmfix
gcloud storage rsync -r gs://mesa-maxtext/mesakh_tpu7x_bench/p1_lance_prs/run_tpu7x_200b_fix/logs /tmp/t7/p1_fix
python3 checkpoint_load_benchmark/scripts/build_tpu7x_sheet_table.py --suite p1 /tmp/t7/p1 /tmp/t7/p1_pstream /tmp/t7/p1_tmfix \
    --cold-s1 qwen3.5-397b-a17b=814.53 \
    --multihost-csv checkpoint_load_benchmark/reports/tpu7x_multihost_results.csv \
    --variant "with fix=/tmp/t7/p1_fix" \
    --md-into checkpoint_load_benchmark/reports/tpu7x_priority1_progress.md
```
Run dirs: `p1_lance_prs/run_tpu7x_200b` (WITH + `to_maxtext`), `p1_origin_main/run_tpu7x_200b` (BEFORE, 360 s),
`p1_origin_main/run_tpu7x_200b_t1800` (BEFORE re-runs), `p1_lance_prs/run_tpu7x_200b_hbmfix` (fix variant).
