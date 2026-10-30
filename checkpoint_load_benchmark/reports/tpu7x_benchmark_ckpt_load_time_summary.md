# tpu7x Checkpoint Load Time Benchmark Summary (`to_maxtext` vs `safetensor`)

> TPU v7x (`bodaborg-tpu7x-nap`, `tpu7x-standard-4t`). Code: `src/` of branch `docs/checkpoint-load-benchmark`
> (= `origin/main` + Lance Wang's PRs #5706, #5705, #5828), plus the pod bootstrap's patches (see §2). The v5p results are in
> [`benchmark_ckpt_load_time_summary.md`](benchmark_ckpt_load_time_summary.md) and stay unchanged.

## 1. What we tested

Two ways to load HuggingFace checkpoints into MaxText. **19 models** were run, from 0.6B to 671B, in 2 scan modes
(`scanned`, `unscanned`) and on 1 slice (8 TPU cores), 2 slices (16 cores) and, for two models, 4 slices (32 cores).

### Pipelines compared
1. **`to_maxtext`** (offline, 3 steps):
   - **Step 1, HF → GCS:** download the `.safetensors` shards to GCS, then stage them on the host.
   - **Step 2, convert to Orbax:** convert on one host's CPU and write an Orbax checkpoint to GCS.
   - **Step 3, load Orbax:** restore that checkpoint into TPU HBM.
2. **`safetensor`** (`safetensors_dynamic`, 2 steps):
   - **Step 1, HF → GCS:** cache the shards in GCS.
   - **Step 2, load + convert:** each host reads its shards' byte ranges from GCS, then the weights are converted to the MaxText layout on device.

Step 1 is the same cold download for both pipelines. It is measured once per model and reused by every configuration.

### Models and hardware

| Tier | Models (BF16 checkpoint size) | Notes |
| :--- | :--- | :--- |
| Small/mid dense | `qwen3-0.6b` (1.5 GB), `gemma2-2b` (10.5 GB, FP32), `qwen3-4b` (8.0 GB), `gemma3-4b` (8.6 GB), `llama3.1-8b` (16.1 GB), `qwen3-8b` (16.4 GB), `qwen3-14b` (29.5 GB), `qwen3-32b` (65.5 GB) | |
| MoE ≤ 120B | `deepseek2-16b` (31.4 GB), `gpt-oss-20b` (41.8 GB), `qwen3-30b-a3b` (61.1 GB), `mixtral-8x7b` (93.4 GB), `gpt-oss-120b` (233.7 GB) | |
| Large dense | `llama3.1-70b` (141.1 GB) | |
| >200B (Priority 1) | `qwen3-235b-a22b` (470.2 GB), `deepseek4-284b` (581.9 GB), `qwen3.5-397b-a17b` (806.8 GB), `qwen3-480b-a35b` (960.3 GB), `deepseek3-671b` (1,342.4 GB) | DeepSeek models use synthetic sparse files (no HF download). Detailed in [`tpu7x_large_model_200b_experiment_report.md`](tpu7x_large_model_200b_experiment_report.md) |

**Hardware per slice:** 4 chips = 8 TPU cores, 813.8 GB HBM, 1,014.5 GB host RAM (986.7 GB allocatable), and a
~100 GB boot disk. Multi-slice runs go over DCN with `jax.distributed`: models ≤47B use `dcn_data_parallelism`,
larger ones `dcn_fsdp_parallelism`.

## 2. How it was run

- **Harness:** [`benchmark_ckpt_load_time.py`](../benchmark_ckpt_load_time.py). Each phase runs in a fresh process,
  and its result is written to `<run dir>/logs/<model>/<config>/<phase>_result.json` in GCS.
- **Orchestration:** pods from [`generate_tpu7x_manifests.py`](../generate_tpu7x_manifests.py). The remaining runs
  were drained by a preemption-tolerant GCS task queue ([`scripts/tpu7x_task_queue.py`](../scripts/tpu7x_task_queue.py),
  tasks in [`configs/p1_completion_tasks.json`](../configs/p1_completion_tasks.json)), because spot capacity and Kueue
  reclamation removed pods throughout the day.
- **Code on the pods** ([`bootstrap_tpu7x.py`](../bootstrap_tpu7x.py)):
  - Lance's PRs, plus single-GET HF→GCS streaming (Step 1 only).
  - Two Step 2 HBM-peak patches: free reshard/stack intermediates, and split composite leaves before the reshard.
  - **`safetensor (with fix)` rows** come from a separate code tree (`/tmp/maxtext_fix`) with this branch's two MaxText
    fixes (issues U-1 and U-6): the leaves of a stacked composite key are stacked one by one, and `_reshard` joins its
    slices two at a time. Their results are in their own run dir (`run_tpu7x_16m_fix`).
- **Tables and CSV:** generated from the result JSONs: the CSV by
  [`scripts/build_tpu7x_tables.py`](../scripts/build_tpu7x_tables.py), the row-by-row tables by
  [`scripts/build_tpu7x_sheet_table.py`](../scripts/build_tpu7x_sheet_table.py). The repeated `qwen3-14b` and
  `qwen3-32b` scanned 1-slice runs (`run_tpu7x_16m_rerun`) replace the first runs.

## 3. Results (`to_maxtext` vs `safetensor`, row by row)

The row-by-row tables are the content of two tabs of the benchmark Google Sheet. Every cell is plain text, so they
paste into the sheet without cleanup:
- **3.1 Models up to 120B** (Priority 2, 14 models), below: a tab like "tpu-v5p".
- **3.2 >200B models** (Priority 1, with Lance Wang's multi-host runs): the "tpu-v7x" tab, in
  [`tpu7x_priority1_progress.md`](tpu7x_priority1_progress.md).

To paste a table, clear the tab first (select all, press Delete, then Format > Clear formatting), then do one of the
following:
- **Copy the rendered text**: in the rendered view (GitHub or a markdown preview), select from the table's title line
  through its last row, copy, and paste into cell A1.
- **Copy the tab-separated text**: expand "The same title, legend and table as tab-separated text" under the table,
  copy the block, and paste into cell A1. Sheets splits each line into cells at the tabs.

The same values, with sub-phases, are in [`tpu7x_benchmark_ckpt_load_time_summary.csv`](tpu7x_benchmark_ckpt_load_time_summary.csv).

### 3.1 Models up to 120B (Priority 2)

<!-- SHEET TABLE START -->
TPU v7x (bodaborg-tpu7x-nap), models up to 120B: to_maxtext vs safetensor (safetensors_dynamic with PRs #5706, #5828, #5705)

- For each (Model, Scan Mode, Slices) configuration, to_maxtext and safetensor are on consecutive rows.
- Col 1 (HF -> GCS): to_maxtext Step 1 vs safetensor Step 1, a cold download of the .safetensors shards, measured once and shared by the configurations that reuse it.
- Col 2 (Convert to Orbax): to_maxtext Step 2, a CPU conversion on one host that every slice count shares. safetensor has no conversion step (—).
- Col 3 (Load into MaxText): to_maxtext Step 3 (Load Orbax) vs safetensor Step 2 (Safetensors+Convert = load_sharded_hf_state + transform_hf_state_to_mt_state). Peak host RAM is the runner host's.
- Slices (Devices): a tpu7x host has 4 chips = 8 TPU cores, 813.8 GB HBM and 1,014.5 GB RAM. 1 (8) = 1 slice (2x2x1, one host); 2 (16) = 2 slices over DCN; 1 (16) = 1 × 2x2x2, 2 (32) = 2 × 2x2x2 multi-host slices (smoke test).
- Speedup: to_maxtext total / safetensor total, end-to-end / without Step 1. N/A: no total because a step failed or was not run.
- safetensor (with fix): the same configuration re-run with this branch's MaxText fixes for two HBM peaks (composite leaves stacked one by one; reshard slices joined in pairs), from a separate code tree. The plain safetensor rows don't include these fixes.

| Model | Scan Mode | Slices (Devices) | Pipeline | 1. HF -> GCS (s) | 2. Convert to Orbax (s) | 3. Load into MaxText (s) | Total (No Step 1) (s) | Total (End-to-End) (s) | Speedup (E2E / No Step 1) |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| qwen3-0.6b (1.5 GB Dense) | scanned | 1 (8) | to_maxtext | 15.98 | 19.40 | 2.54 (Load Orbax) | 21.94 | 37.92 | 1.00x (baseline) |
| ↳ | scanned | 1 (8) | safetensor | 15.98 | — | 16.14 (Safetensors+Convert: load 4.47 + transform 11.67; peak host RAM 26.8 GB) | 16.14 | 32.12 | 1.18x / 1.36x |
| qwen3-0.6b (1.5 GB Dense) | scanned | 1 (16) | to_maxtext | — | Not run on multi-host slices | — | N/A | N/A | N/A |
| ↳ | scanned | 1 (16) | safetensor | 15.98 | — | 18.03 (Safetensors+Convert: load 8.99 + transform 9.00; peak host RAM 38.2 GB) | 18.03 | 34.01 | N/A (to_maxtext not run on this layout) |
| qwen3-0.6b (1.5 GB Dense) | scanned | 2 (16) | to_maxtext | 15.98 | 19.40 | 3.28 (Load Orbax) | 22.68 | 38.66 | 1.00x (baseline) |
| ↳ | scanned | 2 (16) | safetensor | 15.98 | — | 16.13 (Safetensors+Convert: load 7.01 + transform 9.08; peak host RAM 73.6 GB) | 16.13 | 32.11 | 1.20x / 1.41x |
| qwen3-0.6b (1.5 GB Dense) | scanned | 2 (32) | to_maxtext | 15.98 | 19.40 (shared with 1 (8)) | 2.94 (Load Orbax) | 22.35 | 38.33 | 1.00x (baseline) |
| ↳ | scanned | 2 (32) | safetensor | 15.98 | — | 19.34 (Safetensors+Convert: load 8.66 + transform 10.67; peak host RAM 78.3 GB) | 19.34 | 35.32 | 1.09x / 1.16x |
| qwen3-0.6b (1.5 GB Dense) | unscanned | 1 (8) | to_maxtext | 15.98 | 66.78 | 3.33 (Load Orbax) | 70.11 | 86.09 | 1.00x (baseline) |
| ↳ | unscanned | 1 (8) | safetensor | 15.98 | — | 6.49 (Safetensors+Convert: load 3.73 + transform 2.75; peak host RAM 26.8 GB) | 6.49 | 22.47 | 3.83x / 10.80x |
| qwen3-0.6b (1.5 GB Dense) | unscanned | 1 (16) | to_maxtext | — | Not run on multi-host slices | — | N/A | N/A | N/A |
| ↳ | unscanned | 1 (16) | safetensor | 15.98 | — | 9.61 (Safetensors+Convert: load 6.32 + transform 3.28; peak host RAM 38.0 GB) | 9.61 | 25.59 | N/A (to_maxtext not run on this layout) |
| qwen3-0.6b (1.5 GB Dense) | unscanned | 2 (16) | to_maxtext | 15.98 | 66.78 | 3.38 (Load Orbax) | 70.16 | 86.14 | 1.00x (baseline) |
| ↳ | unscanned | 2 (16) | safetensor | 15.98 | — | 9.46 (Safetensors+Convert: load 4.96 + transform 4.47; peak host RAM 73.4 GB) | 9.46 | 25.44 | 3.39x / 7.42x |
| qwen3-0.6b (1.5 GB Dense) | unscanned | 2 (32) | to_maxtext | 15.98 | 66.78 (shared with 1 (8)) | 4.03 (Load Orbax) | 70.81 | 86.79 | 1.00x (baseline) |
| ↳ | unscanned | 2 (32) | safetensor | 15.98 | — | 11.50 (Safetensors+Convert: load 5.50 + transform 5.95; peak host RAM 78.1 GB) | 11.50 | 27.48 | 3.16x / 6.16x |
| gemma2-2b (10.5 GB Dense, FP32 checkpoint) | scanned | 1 (8) | to_maxtext | 169.39 | 54.41 | 5.26 (Load Orbax) | 59.67 | 229.06 | 1.00x (baseline) |
| ↳ | scanned | 1 (8) | safetensor | 169.39 | — | 27.98 (Safetensors+Convert: load 15.15 + transform 12.82; peak host RAM 35.4 GB) | 27.98 | 197.37 | 1.16x / 2.13x |
| gemma2-2b (10.5 GB Dense, FP32 checkpoint) | scanned | 2 (16) | to_maxtext | 169.39 | 54.41 | 6.68 (Load Orbax) | 61.09 | 230.48 | 1.00x (baseline) |
| ↳ | scanned | 2 (16) | safetensor | 169.39 | — | 24.91 (Safetensors+Convert: load 10.22 + transform 14.68; peak host RAM 72.6 GB) | 24.91 | 194.30 | 1.19x / 2.45x |
| gemma2-2b (10.5 GB Dense, FP32 checkpoint) | unscanned | 1 (8) | to_maxtext | 169.39 | 91.17 | 7.83 (Load Orbax) | 99.00 | 268.39 | 1.00x (baseline) |
| ↳ | unscanned | 1 (8) | safetensor | 169.39 | — | 19.90 (Safetensors+Convert: load 13.95 + transform 5.94; peak host RAM 35.8 GB) | 19.90 | 189.29 | 1.42x / 4.98x |
| gemma2-2b (10.5 GB Dense, FP32 checkpoint) | unscanned | 2 (16) | to_maxtext | 169.39 | 91.17 | 8.46 (Load Orbax) | 99.63 | 269.02 | 1.00x (baseline) |
| ↳ | unscanned | 2 (16) | safetensor | 169.39 | — | 17.91 (Safetensors+Convert: load 9.21 + transform 8.69; peak host RAM 72.5 GB) | 17.91 | 187.30 | 1.44x / 5.56x |
| qwen3-4b (8.0 GB Dense) | scanned | 1 (8) | to_maxtext | 51.92 | 62.21 | 14.25 (Load Orbax) | 76.46 | 128.38 | 1.00x (baseline) |
| ↳ | scanned | 1 (8) | safetensor | 51.92 | — | 24.66 (Safetensors+Convert: load 12.43 + transform 12.23; peak host RAM 31.0 GB) | 24.66 | 76.58 | 1.68x / 3.10x |
| qwen3-4b (8.0 GB Dense) | scanned | 2 (16) | to_maxtext | 51.92 | 62.21 | 18.93 (Load Orbax) | 81.14 | 133.06 | 1.00x (baseline) |
| ↳ | scanned | 2 (16) | safetensor | 51.92 | — | 22.81 (Safetensors+Convert: load 9.57 + transform 13.24; peak host RAM 69.6 GB) | 22.81 | 74.74 | 1.78x / 3.56x |
| qwen3-4b (8.0 GB Dense) | unscanned | 1 (8) | to_maxtext | 51.92 | 95.90 | 9.73 (Load Orbax) | 105.63 | 157.55 | 1.00x (baseline) |
| ↳ | unscanned | 1 (8) | safetensor | 51.92 | — | 14.40 (Safetensors+Convert: load 11.94 + transform 2.44; peak host RAM 31.1 GB) | 14.40 | 66.32 | 2.38x / 7.33x |
| qwen3-4b (8.0 GB Dense) | unscanned | 2 (16) | to_maxtext | 51.92 | 95.90 | 9.85 (Load Orbax) | 105.74 | 157.66 | 1.00x (baseline) |
| ↳ | unscanned | 2 (16) | safetensor | 51.92 | — | 15.03 (Safetensors+Convert: load 9.54 + transform 5.45; peak host RAM 70.7 GB) | 15.03 | 66.95 | 2.35x / 7.04x |
| gemma3-4b (8.6 GB Dense) | scanned | 1 (8) | to_maxtext | 55.81 | 56.42 | 7.84 (Load Orbax) | 64.26 | 120.08 | 1.00x (baseline) |
| ↳ | scanned | 1 (8) | safetensor | 55.81 | — | 24.33 (Safetensors+Convert: load 13.81 + transform 10.50; peak host RAM 33.5 GB) | 24.33 | 80.14 | 1.50x / 2.64x |
| gemma3-4b (8.6 GB Dense) | scanned | 2 (16) | to_maxtext | 55.81 | 56.42 | 7.05 (Load Orbax) | 63.47 | 119.28 | 1.00x (baseline) |
| ↳ | scanned | 2 (16) | safetensor | 55.81 | — | 24.14 (Safetensors+Convert: load 11.99 + transform 12.10; peak host RAM 80.6 GB) | 24.14 | 79.95 | 1.49x / 2.63x |
| gemma3-4b (8.6 GB Dense) | unscanned | 1 (8) | to_maxtext | 55.81 | 108.93 | 9.90 (Load Orbax) | 118.83 | 174.64 | 1.00x (baseline) |
| ↳ | unscanned | 1 (8) | safetensor | 55.81 | — | 17.83 (Safetensors+Convert: load 12.65 + transform 5.16; peak host RAM 33.4 GB) | 17.83 | 73.64 | 2.37x / 6.66x |
| gemma3-4b (8.6 GB Dense) | unscanned | 2 (16) | to_maxtext | 55.81 | 108.93 | 8.13 (Load Orbax) | 117.06 | 172.87 | 1.00x (baseline) |
| ↳ | unscanned | 2 (16) | safetensor | 55.81 | — | 21.15 (Safetensors+Convert: load 13.68 + transform 7.42; peak host RAM 81.5 GB) | 21.15 | 76.96 | 2.25x / 5.53x |
| llama3.1-8b (16.1 GB Dense) | scanned | 1 (8) | to_maxtext | 62.91 | 89.58 | 22.02 (Load Orbax) | 111.60 | 174.50 | 1.00x (baseline) |
| ↳ | scanned | 1 (8) | safetensor | 62.91 | — | 51.76 (Safetensors+Convert: load 36.97 + transform 14.79; peak host RAM 33.0 GB) | 51.76 | 114.66 | 1.52x / 2.16x |
| llama3.1-8b (16.1 GB Dense) | scanned | 2 (16) | to_maxtext | 62.91 | 89.58 | 17.92 (Load Orbax) | 107.50 | 170.40 | 1.00x (baseline) |
| ↳ | scanned | 2 (16) | safetensor | 62.91 | — | 28.52 (Safetensors+Convert: load 10.89 + transform 17.61; peak host RAM 67.6 GB) | 28.52 | 91.42 | 1.86x / 3.77x |
| llama3.1-8b (16.1 GB Dense) | unscanned | 1 (8) | to_maxtext | 62.91 | 105.31 | 14.25 (Load Orbax) | 119.56 | 182.47 | 1.00x (baseline) |
| ↳ | unscanned | 1 (8) | safetensor | 62.91 | — | 36.73 (Safetensors+Convert: load 31.19 + transform 5.52; peak host RAM 37.0 GB) | 36.73 | 99.63 | 1.83x / 3.26x |
| llama3.1-8b (16.1 GB Dense) | unscanned | 2 (16) | to_maxtext | 62.91 | 105.31 | 12.51 (Load Orbax) | 117.83 | 180.73 | 1.00x (baseline) |
| ↳ | unscanned | 2 (16) | safetensor | 62.91 | — | 18.12 (Safetensors+Convert: load 10.28 + transform 7.81; peak host RAM 68.6 GB) | 18.12 | 81.03 | 2.23x / 6.50x |
| qwen3-8b (16.4 GB Dense) | scanned | 1 (8) | to_maxtext | 54.26 | 96.45 | 21.56 (Load Orbax) | 118.01 | 172.27 | 1.00x (baseline) |
| ↳ | scanned | 1 (8) | safetensor | 54.26 | — | 43.08 (Safetensors+Convert: load 21.75 + transform 21.32; peak host RAM 33.4 GB) | 43.08 | 97.33 | 1.77x / 2.74x |
| qwen3-8b (16.4 GB Dense) | scanned | 2 (16) | to_maxtext | 54.26 | 96.45 | 22.54 (Load Orbax) | 118.99 | 173.25 | 1.00x (baseline) |
| ↳ | scanned | 2 (16) | safetensor | 54.26 | — | 33.96 (Safetensors+Convert: load 10.32 + transform 23.63; peak host RAM 68.2 GB) | 33.96 | 88.22 | 1.96x / 3.50x |
| qwen3-8b (16.4 GB Dense) | unscanned | 1 (8) | to_maxtext | 54.26 | 119.91 | 13.27 (Load Orbax) | 133.18 | 187.43 | 1.00x (baseline) |
| ↳ | unscanned | 1 (8) | safetensor | 54.26 | — | 28.15 (Safetensors+Convert: load 20.31 + transform 7.83; peak host RAM 33.6 GB) | 28.15 | 82.41 | 2.27x / 4.73x |
| qwen3-8b (16.4 GB Dense) | unscanned | 2 (16) | to_maxtext | 54.26 | 119.91 | 11.37 (Load Orbax) | 131.28 | 185.54 | 1.00x (baseline) |
| ↳ | unscanned | 2 (16) | safetensor | 54.26 | — | 22.35 (Safetensors+Convert: load 11.05 + transform 11.28; peak host RAM 67.5 GB) | 22.35 | 76.60 | 2.42x / 5.87x |
| qwen3-14b (29.5 GB Dense) | scanned | 1 (8) | to_maxtext | 51.35 | 145.85 | 48.90 (Load Orbax) | 194.75 | 246.10 | 1.00x (baseline) |
| ↳ | scanned | 1 (8) | safetensor | 51.35 | — | 329.90 (Safetensors+Convert: load 40.25 + transform 289.65; peak host RAM 35.3 GB) | 329.90 | 381.25 | 0.65x / 0.59x |
| qwen3-14b (29.5 GB Dense) | scanned | 2 (16) | to_maxtext | 51.35 | 145.85 | 61.90 (Load Orbax) | 207.75 | 259.10 | 1.00x (baseline) |
| ↳ | scanned | 2 (16) | safetensor | 51.35 | — | 69.02 (Safetensors+Convert: load 18.17 + transform 50.84; peak host RAM 68.8 GB) | 69.02 | 120.37 | 2.15x / 3.01x |
| qwen3-14b (29.5 GB Dense) | unscanned | 1 (8) | to_maxtext | 51.35 | 144.74 | 19.69 (Load Orbax) | 164.42 | 215.77 | 1.00x (baseline) |
| ↳ | unscanned | 1 (8) | safetensor | 51.35 | — | 40.31 (Safetensors+Convert: load 35.53 + transform 4.76; peak host RAM 34.5 GB) | 40.31 | 91.66 | 2.35x / 4.08x |
| qwen3-14b (29.5 GB Dense) | unscanned | 2 (16) | to_maxtext | 51.35 | 144.74 | 19.05 (Load Orbax) | 163.79 | 215.13 | 1.00x (baseline) |
| ↳ | unscanned | 2 (16) | safetensor | 51.35 | — | 57.86 (Safetensors+Convert: load 17.50 + transform 40.19; peak host RAM 68.0 GB) | 57.86 | 109.21 | 1.97x / 2.83x |
| qwen3-32b (65.5 GB Dense) | scanned | 1 (8) | to_maxtext | 107.24 | 381.46 | 203.69 (Load Orbax) | 585.15 | 692.38 | 1.00x (baseline) |
| ↳ | scanned | 1 (8) | safetensor | 107.24 | — | 100.72 (Safetensors+Convert: load 82.04 + transform 18.68; peak host RAM 34.2 GB) | 100.72 | 207.96 | 3.33x / 5.81x |
| qwen3-32b (65.5 GB Dense) | scanned | 2 (16) | to_maxtext | 107.24 | 381.46 | 238.09 (Load Orbax) | 619.55 | 726.79 | 1.00x (baseline) |
| ↳ | scanned | 2 (16) | safetensor | 107.24 | — | 88.92 (Safetensors+Convert: load 35.44 + transform 53.42; peak host RAM 70.3 GB) | 88.92 | 196.16 | 3.71x / 6.97x |
| qwen3-32b (65.5 GB Dense) | unscanned | 1 (8) | to_maxtext | 107.24 | 208.04 | 36.22 (Load Orbax) | 244.26 | 351.50 | 1.00x (baseline) |
| ↳ | unscanned | 1 (8) | safetensor | 107.24 | — | 77.57 (Safetensors+Convert: load 72.96 + transform 4.58; peak host RAM 35.0 GB) | 77.57 | 184.80 | 1.90x / 3.15x |
| qwen3-32b (65.5 GB Dense) | unscanned | 2 (16) | to_maxtext | 107.24 | 208.04 | 30.37 (Load Orbax) | 238.41 | 345.64 | 1.00x (baseline) |
| ↳ | unscanned | 2 (16) | safetensor | 107.24 | — | 74.67 (Safetensors+Convert: load 33.47 + transform 41.10; peak host RAM 70.3 GB) | 74.67 | 181.90 | 1.90x / 3.19x |
| deepseek2-16b (31.4 GB MoE) | scanned | 1 (8) | to_maxtext | 137.26 | 76.65 | 94.41 (Load Orbax) | 171.06 | 308.32 | 1.00x (baseline) |
| ↳ | scanned | 1 (8) | safetensor | 137.26 | — | 75.00 (Safetensors+Convert: load 50.91 + transform 24.08; peak host RAM 38.4 GB) | 75.00 | 212.25 | 1.45x / 2.28x |
| deepseek2-16b (31.4 GB MoE) | scanned | 2 (16) | to_maxtext | 137.26 | 76.65 | 112.52 (Load Orbax) | 189.17 | 326.42 | 1.00x (baseline) |
| ↳ | scanned | 2 (16) | safetensor | 137.26 | — | 125.49 (Safetensors+Convert: load 98.62 + transform 26.85; peak host RAM 115.5 GB) | 125.49 | 262.74 | 1.24x / 1.51x |
| deepseek2-16b (31.4 GB MoE) | unscanned | 1 (8) | to_maxtext | 137.26 | 147.18 | 18.17 (Load Orbax) | 165.35 | 302.60 | 1.00x (baseline) |
| ↳ | unscanned | 1 (8) | safetensor | 137.26 | — | 58.36 (Safetensors+Convert: load 50.57 + transform 7.77; peak host RAM 42.4 GB) | 58.36 | 195.61 | 1.55x / 2.83x |
| deepseek2-16b (31.4 GB MoE) | unscanned | 2 (16) | to_maxtext | 137.26 | 147.18 | 17.59 (Load Orbax) | 164.76 | 302.02 | 1.00x (baseline) |
| ↳ | unscanned | 2 (16) | safetensor | 137.26 | — | 115.00 (Safetensors+Convert: load 102.40 + transform 12.58; peak host RAM 116.7 GB) | 115.00 | 252.25 | 1.20x / 1.43x |
| gpt-oss-20b (41.8 GB MoE) | scanned | 1 (8) | to_maxtext | 93.12 | 102.63 | 63.35 (Load Orbax) | 165.98 | 259.10 | 1.00x (baseline) |
| ↳ | scanned | 1 (8) | safetensor | 93.12 | — | 127.47 (Safetensors+Convert: load 48.79 + transform 78.68; peak host RAM 39.2 GB) | 127.47 | 220.59 | 1.17x / 1.30x |
| ↳ | scanned | 1 (8) | safetensor (with fix) | 93.12 | — | 127.90 (Safetensors+Convert: load 52.06 + transform 75.83; peak host RAM 36.2 GB) | 127.90 | 221.02 | 1.17x / 1.30x |
| gpt-oss-20b (41.8 GB MoE) | scanned | 2 (16) | to_maxtext | 93.12 | 102.63 | 58.31 (Load Orbax) | 160.94 | 254.06 | 1.00x (baseline) |
| ↳ | scanned | 2 (16) | safetensor | 93.12 | — | 111.90 (Safetensors+Convert: load 25.63 + transform 86.24; peak host RAM 67.1 GB) | 111.90 | 205.02 | 1.24x / 1.44x |
| gpt-oss-20b (41.8 GB MoE) | unscanned | 1 (8) | to_maxtext | 93.12 | 185.80 | 12.63 (Load Orbax) | 198.43 | 291.55 | 1.00x (baseline) |
| ↳ | unscanned | 1 (8) | safetensor | 93.12 | — | 78.26 (Safetensors+Convert: load 48.34 + transform 29.90; peak host RAM 37.2 GB) | 78.26 | 171.38 | 1.70x / 2.54x |
| ↳ | unscanned | 1 (8) | safetensor (with fix) | 93.12 | — | 147.78 (Safetensors+Convert: load 79.02 + transform 68.75; peak host RAM 39.8 GB) | 147.78 | 240.91 | 1.21x / 1.34x |
| gpt-oss-20b (41.8 GB MoE) | unscanned | 2 (16) | to_maxtext | 93.12 | 185.80 | 12.23 (Load Orbax) | 198.03 | 291.15 | 1.00x (baseline) |
| ↳ | unscanned | 2 (16) | safetensor | 93.12 | — | 104.55 (Safetensors+Convert: load 32.48 + transform 72.02; peak host RAM 68.3 GB) | 104.55 | 197.67 | 1.47x / 1.89x |
| qwen3-30b-a3b (61.1 GB MoE) | scanned | 1 (8) | to_maxtext | 78.82 | 151.57 | 386.06 (Load Orbax) | 537.62 | 616.45 | 1.00x (baseline) |
| ↳ | scanned | 1 (8) | safetensor | 78.82 | — | 129.52 (Safetensors+Convert: load 99.12 + transform 30.38; peak host RAM 61.4 GB) | 129.52 | 208.34 | 2.96x / 4.15x |
| qwen3-30b-a3b (61.1 GB MoE) | scanned | 2 (16) | to_maxtext | 78.82 | 151.57 | 449.33 (Load Orbax) | 600.90 | 679.72 | 1.00x (baseline) |
| ↳ | scanned | 2 (16) | safetensor | 78.82 | — | 385.50 (Safetensors+Convert: load 351.63 + transform 33.47; peak host RAM 156.1 GB) | 385.50 | 464.32 | 1.46x / 1.56x |
| qwen3-30b-a3b (61.1 GB MoE) | unscanned | 1 (8) | to_maxtext | 78.82 | 233.63 | 31.57 (Load Orbax) | 265.20 | 344.02 | 1.00x (baseline) |
| ↳ | unscanned | 1 (8) | safetensor | 78.82 | — | 110.70 (Safetensors+Convert: load 95.77 + transform 14.90; peak host RAM 61.0 GB) | 110.70 | 189.53 | 1.82x / 2.40x |
| qwen3-30b-a3b (61.1 GB MoE) | unscanned | 2 (16) | to_maxtext | 78.82 | 233.63 | 25.79 (Load Orbax) | 259.42 | 338.24 | 1.00x (baseline) |
| ↳ | unscanned | 2 (16) | safetensor | 78.82 | — | 354.16 (Safetensors+Convert: load 333.13 + transform 20.99; peak host RAM 151.1 GB) | 354.16 | 432.98 | 0.78x / 0.73x |
| mixtral-8x7b (93.4 GB MoE) | scanned | 1 (8) | to_maxtext | 239.75 | 134.07 | 533.60 (Load Orbax) | 667.67 | 907.42 | 1.00x (baseline) |
| ↳ | scanned | 1 (8) | safetensor | 239.75 | — | 126.93 (Safetensors+Convert: load 113.62 + transform 13.31; peak host RAM 39.6 GB) | 126.93 | 366.67 | 2.47x / 5.26x |
| mixtral-8x7b (93.4 GB MoE) | scanned | 2 (16) | to_maxtext | 239.75 | 134.07 | 792.49 (Load Orbax) | 926.56 | 1,166.31 | 1.00x (baseline) |
| ↳ | scanned | 2 (16) | safetensor | 239.75 | — | 63.12 (Safetensors+Convert: load 45.96 + transform 17.16; peak host RAM 74.7 GB) | 63.12 | 302.87 | 3.85x / 14.68x |
| mixtral-8x7b (93.4 GB MoE) | unscanned | 1 (8) | to_maxtext | 239.75 | 208.28 | 31.95 (Load Orbax) | 240.24 | 479.98 | 1.00x (baseline) |
| ↳ | unscanned | 1 (8) | safetensor | 239.75 | — | 115.61 (Safetensors+Convert: load 110.64 + transform 4.96; peak host RAM 39.1 GB) | 115.61 | 355.36 | 1.35x / 2.08x |
| mixtral-8x7b (93.4 GB MoE) | unscanned | 2 (16) | to_maxtext | 239.75 | 208.28 | 47.67 (Load Orbax) | 255.95 | 495.70 | 1.00x (baseline) |
| ↳ | unscanned | 2 (16) | safetensor | 239.75 | — | 56.55 (Safetensors+Convert: load 44.84 + transform 11.62; peak host RAM 69.6 GB) | 56.55 | 296.30 | 1.67x / 4.53x |
| llama3.1-70b (141.1 GB Dense) | scanned | 1 (8) | to_maxtext | 197.80 | 780.27 | 411.50 (Load Orbax) | 1,191.78 | 1,389.57 | 1.00x (baseline) |
| ↳ | scanned | 1 (8) | safetensor | 197.80 | — | 199.20 (Safetensors+Convert: load 165.08 + transform 34.12; peak host RAM 37.9 GB) | 199.20 | 397.00 | 3.50x / 5.98x |
| llama3.1-70b (141.1 GB Dense) | scanned | 2 (16) | to_maxtext | 197.80 | 780.27 | 295.79 (Load Orbax) | 1,076.06 | 1,273.86 | 1.00x (baseline) |
| ↳ | scanned | 2 (16) | safetensor | 197.80 | — | 104.98 (Safetensors+Convert: load 72.44 + transform 32.48; peak host RAM 73.7 GB) | 104.98 | 302.78 | 4.21x / 10.25x |
| llama3.1-70b (141.1 GB Dense) | unscanned | 1 (8) | to_maxtext | 171.23 | 398.09 | 67.68 (Load Orbax) | 465.77 | 636.99 | 1.00x (baseline) |
| ↳ | unscanned | 1 (8) | safetensor | 197.80 | — | 168.09 (Safetensors+Convert: load 159.02 + transform 9.05; peak host RAM 38.8 GB) | 168.09 | 365.89 | 1.74x / 2.77x |
| llama3.1-70b (141.1 GB Dense) | unscanned | 2 (16) | to_maxtext | 171.23 | 398.09 | 42.48 (Load Orbax) | 440.56 | 611.79 | 1.00x (baseline) |
| ↳ | unscanned | 2 (16) | safetensor | 197.80 | — | 113.96 (Safetensors+Convert: load 100.40 + transform 13.52; peak host RAM 73.7 GB) | 113.96 | 311.76 | 1.96x / 3.87x |
| gpt-oss-120b (233.7 GB MoE) | scanned | 1 (8) | to_maxtext | 350.33 | 581.39 | 839.25 (Load Orbax) | 1,420.65 | 1,770.97 | 1.00x (baseline) |
| ↳ | scanned | 1 (8) | safetensor | 350.33 | — | HBM OOM in transform (load 253.62 s; 200.32 GiB of HLO temporaries per device, 94.74 GiB available) | N/A | N/A | Only to_maxtext loads this config |
| ↳ | scanned | 1 (8) | safetensor (with fix) | 350.33 | — | 504.92 (Safetensors+Convert: load 260.44 + transform 244.47; peak host RAM 44.6 GB) | 504.92 | 855.24 | 2.07x / 2.81x |
| gpt-oss-120b (233.7 GB MoE) | scanned | 2 (16) | to_maxtext | 350.33 | 581.39 | 893.97 (Load Orbax) | 1,475.36 | 1,825.69 | 1.00x (baseline) |
| ↳ | scanned | 2 (16) | safetensor | 350.33 | — | HBM OOM in transform (load 126.92 s; 100.21 GiB of HLO temporaries per device, 94.74 GiB available) | N/A | N/A | Only to_maxtext loads this config |
| ↳ | scanned | 2 (16) | safetensor (with fix) | 350.33 | — | 429.98 (Safetensors+Convert: load 127.52 + transform 302.46; peak host RAM 71.4 GB) | 429.98 | 780.31 | 2.34x / 3.43x |
| gpt-oss-120b (233.7 GB MoE) | unscanned | 1 (8) | to_maxtext | 350.33 | 617.84 | 110.61 (Load Orbax) | 728.46 | 1,078.78 | 1.00x (baseline) |
| ↳ | unscanned | 1 (8) | safetensor | 350.33 | — | 446.71 (Safetensors+Convert: load 250.76 + transform 195.93; peak host RAM 306.0 GB) | 446.71 | 797.04 | 1.35x / 1.63x |
| gpt-oss-120b (233.7 GB MoE) | unscanned | 2 (16) | to_maxtext | 350.33 | 617.84 | 115.05 (Load Orbax) | 732.90 | 1,083.22 | 1.00x (baseline) |
| ↳ | unscanned | 2 (16) | safetensor | 350.33 | — | 390.69 (Safetensors+Convert: load 122.59 + transform 268.06; peak host RAM 338.5 GB) | 390.69 | 741.01 | 1.46x / 1.88x |

<details>
<summary>The same title, legend and table as tab-separated text</summary>

```text
TPU v7x (bodaborg-tpu7x-nap), models up to 120B: to_maxtext vs safetensor (safetensors_dynamic with PRs #5706, #5828, #5705)

For each (Model, Scan Mode, Slices) configuration, to_maxtext and safetensor are on consecutive rows.
Col 1 (HF -> GCS): to_maxtext Step 1 vs safetensor Step 1, a cold download of the .safetensors shards, measured once and shared by the configurations that reuse it.
Col 2 (Convert to Orbax): to_maxtext Step 2, a CPU conversion on one host that every slice count shares. safetensor has no conversion step (—).
Col 3 (Load into MaxText): to_maxtext Step 3 (Load Orbax) vs safetensor Step 2 (Safetensors+Convert = load_sharded_hf_state + transform_hf_state_to_mt_state). Peak host RAM is the runner host's.
Slices (Devices): a tpu7x host has 4 chips = 8 TPU cores, 813.8 GB HBM and 1,014.5 GB RAM. 1 (8) = 1 slice (2x2x1, one host); 2 (16) = 2 slices over DCN; 1 (16) = 1 × 2x2x2, 2 (32) = 2 × 2x2x2 multi-host slices (smoke test).
Speedup: to_maxtext total / safetensor total, end-to-end / without Step 1. N/A: no total because a step failed or was not run.
safetensor (with fix): the same configuration re-run with this branch's MaxText fixes for two HBM peaks (composite leaves stacked one by one; reshard slices joined in pairs), from a separate code tree. The plain safetensor rows don't include these fixes.

Model	Scan Mode	Slices (Devices)	Pipeline	1. HF -> GCS (s)	2. Convert to Orbax (s)	3. Load into MaxText (s)	Total (No Step 1) (s)	Total (End-to-End) (s)	Speedup (E2E / No Step 1)
qwen3-0.6b (1.5 GB Dense)	scanned	1 (8)	to_maxtext	15.98	19.40	2.54 (Load Orbax)	21.94	37.92	1.00x (baseline)
↳	scanned	1 (8)	safetensor	15.98	—	16.14 (Safetensors+Convert: load 4.47 + transform 11.67; peak host RAM 26.8 GB)	16.14	32.12	1.18x / 1.36x
qwen3-0.6b (1.5 GB Dense)	scanned	1 (16)	to_maxtext	—	Not run on multi-host slices	—	N/A	N/A	N/A
↳	scanned	1 (16)	safetensor	15.98	—	18.03 (Safetensors+Convert: load 8.99 + transform 9.00; peak host RAM 38.2 GB)	18.03	34.01	N/A (to_maxtext not run on this layout)
qwen3-0.6b (1.5 GB Dense)	scanned	2 (16)	to_maxtext	15.98	19.40	3.28 (Load Orbax)	22.68	38.66	1.00x (baseline)
↳	scanned	2 (16)	safetensor	15.98	—	16.13 (Safetensors+Convert: load 7.01 + transform 9.08; peak host RAM 73.6 GB)	16.13	32.11	1.20x / 1.41x
qwen3-0.6b (1.5 GB Dense)	scanned	2 (32)	to_maxtext	15.98	19.40 (shared with 1 (8))	2.94 (Load Orbax)	22.35	38.33	1.00x (baseline)
↳	scanned	2 (32)	safetensor	15.98	—	19.34 (Safetensors+Convert: load 8.66 + transform 10.67; peak host RAM 78.3 GB)	19.34	35.32	1.09x / 1.16x
qwen3-0.6b (1.5 GB Dense)	unscanned	1 (8)	to_maxtext	15.98	66.78	3.33 (Load Orbax)	70.11	86.09	1.00x (baseline)
↳	unscanned	1 (8)	safetensor	15.98	—	6.49 (Safetensors+Convert: load 3.73 + transform 2.75; peak host RAM 26.8 GB)	6.49	22.47	3.83x / 10.80x
qwen3-0.6b (1.5 GB Dense)	unscanned	1 (16)	to_maxtext	—	Not run on multi-host slices	—	N/A	N/A	N/A
↳	unscanned	1 (16)	safetensor	15.98	—	9.61 (Safetensors+Convert: load 6.32 + transform 3.28; peak host RAM 38.0 GB)	9.61	25.59	N/A (to_maxtext not run on this layout)
qwen3-0.6b (1.5 GB Dense)	unscanned	2 (16)	to_maxtext	15.98	66.78	3.38 (Load Orbax)	70.16	86.14	1.00x (baseline)
↳	unscanned	2 (16)	safetensor	15.98	—	9.46 (Safetensors+Convert: load 4.96 + transform 4.47; peak host RAM 73.4 GB)	9.46	25.44	3.39x / 7.42x
qwen3-0.6b (1.5 GB Dense)	unscanned	2 (32)	to_maxtext	15.98	66.78 (shared with 1 (8))	4.03 (Load Orbax)	70.81	86.79	1.00x (baseline)
↳	unscanned	2 (32)	safetensor	15.98	—	11.50 (Safetensors+Convert: load 5.50 + transform 5.95; peak host RAM 78.1 GB)	11.50	27.48	3.16x / 6.16x
gemma2-2b (10.5 GB Dense, FP32 checkpoint)	scanned	1 (8)	to_maxtext	169.39	54.41	5.26 (Load Orbax)	59.67	229.06	1.00x (baseline)
↳	scanned	1 (8)	safetensor	169.39	—	27.98 (Safetensors+Convert: load 15.15 + transform 12.82; peak host RAM 35.4 GB)	27.98	197.37	1.16x / 2.13x
gemma2-2b (10.5 GB Dense, FP32 checkpoint)	scanned	2 (16)	to_maxtext	169.39	54.41	6.68 (Load Orbax)	61.09	230.48	1.00x (baseline)
↳	scanned	2 (16)	safetensor	169.39	—	24.91 (Safetensors+Convert: load 10.22 + transform 14.68; peak host RAM 72.6 GB)	24.91	194.30	1.19x / 2.45x
gemma2-2b (10.5 GB Dense, FP32 checkpoint)	unscanned	1 (8)	to_maxtext	169.39	91.17	7.83 (Load Orbax)	99.00	268.39	1.00x (baseline)
↳	unscanned	1 (8)	safetensor	169.39	—	19.90 (Safetensors+Convert: load 13.95 + transform 5.94; peak host RAM 35.8 GB)	19.90	189.29	1.42x / 4.98x
gemma2-2b (10.5 GB Dense, FP32 checkpoint)	unscanned	2 (16)	to_maxtext	169.39	91.17	8.46 (Load Orbax)	99.63	269.02	1.00x (baseline)
↳	unscanned	2 (16)	safetensor	169.39	—	17.91 (Safetensors+Convert: load 9.21 + transform 8.69; peak host RAM 72.5 GB)	17.91	187.30	1.44x / 5.56x
qwen3-4b (8.0 GB Dense)	scanned	1 (8)	to_maxtext	51.92	62.21	14.25 (Load Orbax)	76.46	128.38	1.00x (baseline)
↳	scanned	1 (8)	safetensor	51.92	—	24.66 (Safetensors+Convert: load 12.43 + transform 12.23; peak host RAM 31.0 GB)	24.66	76.58	1.68x / 3.10x
qwen3-4b (8.0 GB Dense)	scanned	2 (16)	to_maxtext	51.92	62.21	18.93 (Load Orbax)	81.14	133.06	1.00x (baseline)
↳	scanned	2 (16)	safetensor	51.92	—	22.81 (Safetensors+Convert: load 9.57 + transform 13.24; peak host RAM 69.6 GB)	22.81	74.74	1.78x / 3.56x
qwen3-4b (8.0 GB Dense)	unscanned	1 (8)	to_maxtext	51.92	95.90	9.73 (Load Orbax)	105.63	157.55	1.00x (baseline)
↳	unscanned	1 (8)	safetensor	51.92	—	14.40 (Safetensors+Convert: load 11.94 + transform 2.44; peak host RAM 31.1 GB)	14.40	66.32	2.38x / 7.33x
qwen3-4b (8.0 GB Dense)	unscanned	2 (16)	to_maxtext	51.92	95.90	9.85 (Load Orbax)	105.74	157.66	1.00x (baseline)
↳	unscanned	2 (16)	safetensor	51.92	—	15.03 (Safetensors+Convert: load 9.54 + transform 5.45; peak host RAM 70.7 GB)	15.03	66.95	2.35x / 7.04x
gemma3-4b (8.6 GB Dense)	scanned	1 (8)	to_maxtext	55.81	56.42	7.84 (Load Orbax)	64.26	120.08	1.00x (baseline)
↳	scanned	1 (8)	safetensor	55.81	—	24.33 (Safetensors+Convert: load 13.81 + transform 10.50; peak host RAM 33.5 GB)	24.33	80.14	1.50x / 2.64x
gemma3-4b (8.6 GB Dense)	scanned	2 (16)	to_maxtext	55.81	56.42	7.05 (Load Orbax)	63.47	119.28	1.00x (baseline)
↳	scanned	2 (16)	safetensor	55.81	—	24.14 (Safetensors+Convert: load 11.99 + transform 12.10; peak host RAM 80.6 GB)	24.14	79.95	1.49x / 2.63x
gemma3-4b (8.6 GB Dense)	unscanned	1 (8)	to_maxtext	55.81	108.93	9.90 (Load Orbax)	118.83	174.64	1.00x (baseline)
↳	unscanned	1 (8)	safetensor	55.81	—	17.83 (Safetensors+Convert: load 12.65 + transform 5.16; peak host RAM 33.4 GB)	17.83	73.64	2.37x / 6.66x
gemma3-4b (8.6 GB Dense)	unscanned	2 (16)	to_maxtext	55.81	108.93	8.13 (Load Orbax)	117.06	172.87	1.00x (baseline)
↳	unscanned	2 (16)	safetensor	55.81	—	21.15 (Safetensors+Convert: load 13.68 + transform 7.42; peak host RAM 81.5 GB)	21.15	76.96	2.25x / 5.53x
llama3.1-8b (16.1 GB Dense)	scanned	1 (8)	to_maxtext	62.91	89.58	22.02 (Load Orbax)	111.60	174.50	1.00x (baseline)
↳	scanned	1 (8)	safetensor	62.91	—	51.76 (Safetensors+Convert: load 36.97 + transform 14.79; peak host RAM 33.0 GB)	51.76	114.66	1.52x / 2.16x
llama3.1-8b (16.1 GB Dense)	scanned	2 (16)	to_maxtext	62.91	89.58	17.92 (Load Orbax)	107.50	170.40	1.00x (baseline)
↳	scanned	2 (16)	safetensor	62.91	—	28.52 (Safetensors+Convert: load 10.89 + transform 17.61; peak host RAM 67.6 GB)	28.52	91.42	1.86x / 3.77x
llama3.1-8b (16.1 GB Dense)	unscanned	1 (8)	to_maxtext	62.91	105.31	14.25 (Load Orbax)	119.56	182.47	1.00x (baseline)
↳	unscanned	1 (8)	safetensor	62.91	—	36.73 (Safetensors+Convert: load 31.19 + transform 5.52; peak host RAM 37.0 GB)	36.73	99.63	1.83x / 3.26x
llama3.1-8b (16.1 GB Dense)	unscanned	2 (16)	to_maxtext	62.91	105.31	12.51 (Load Orbax)	117.83	180.73	1.00x (baseline)
↳	unscanned	2 (16)	safetensor	62.91	—	18.12 (Safetensors+Convert: load 10.28 + transform 7.81; peak host RAM 68.6 GB)	18.12	81.03	2.23x / 6.50x
qwen3-8b (16.4 GB Dense)	scanned	1 (8)	to_maxtext	54.26	96.45	21.56 (Load Orbax)	118.01	172.27	1.00x (baseline)
↳	scanned	1 (8)	safetensor	54.26	—	43.08 (Safetensors+Convert: load 21.75 + transform 21.32; peak host RAM 33.4 GB)	43.08	97.33	1.77x / 2.74x
qwen3-8b (16.4 GB Dense)	scanned	2 (16)	to_maxtext	54.26	96.45	22.54 (Load Orbax)	118.99	173.25	1.00x (baseline)
↳	scanned	2 (16)	safetensor	54.26	—	33.96 (Safetensors+Convert: load 10.32 + transform 23.63; peak host RAM 68.2 GB)	33.96	88.22	1.96x / 3.50x
qwen3-8b (16.4 GB Dense)	unscanned	1 (8)	to_maxtext	54.26	119.91	13.27 (Load Orbax)	133.18	187.43	1.00x (baseline)
↳	unscanned	1 (8)	safetensor	54.26	—	28.15 (Safetensors+Convert: load 20.31 + transform 7.83; peak host RAM 33.6 GB)	28.15	82.41	2.27x / 4.73x
qwen3-8b (16.4 GB Dense)	unscanned	2 (16)	to_maxtext	54.26	119.91	11.37 (Load Orbax)	131.28	185.54	1.00x (baseline)
↳	unscanned	2 (16)	safetensor	54.26	—	22.35 (Safetensors+Convert: load 11.05 + transform 11.28; peak host RAM 67.5 GB)	22.35	76.60	2.42x / 5.87x
qwen3-14b (29.5 GB Dense)	scanned	1 (8)	to_maxtext	51.35	145.85	48.90 (Load Orbax)	194.75	246.10	1.00x (baseline)
↳	scanned	1 (8)	safetensor	51.35	—	329.90 (Safetensors+Convert: load 40.25 + transform 289.65; peak host RAM 35.3 GB)	329.90	381.25	0.65x / 0.59x
qwen3-14b (29.5 GB Dense)	scanned	2 (16)	to_maxtext	51.35	145.85	61.90 (Load Orbax)	207.75	259.10	1.00x (baseline)
↳	scanned	2 (16)	safetensor	51.35	—	69.02 (Safetensors+Convert: load 18.17 + transform 50.84; peak host RAM 68.8 GB)	69.02	120.37	2.15x / 3.01x
qwen3-14b (29.5 GB Dense)	unscanned	1 (8)	to_maxtext	51.35	144.74	19.69 (Load Orbax)	164.42	215.77	1.00x (baseline)
↳	unscanned	1 (8)	safetensor	51.35	—	40.31 (Safetensors+Convert: load 35.53 + transform 4.76; peak host RAM 34.5 GB)	40.31	91.66	2.35x / 4.08x
qwen3-14b (29.5 GB Dense)	unscanned	2 (16)	to_maxtext	51.35	144.74	19.05 (Load Orbax)	163.79	215.13	1.00x (baseline)
↳	unscanned	2 (16)	safetensor	51.35	—	57.86 (Safetensors+Convert: load 17.50 + transform 40.19; peak host RAM 68.0 GB)	57.86	109.21	1.97x / 2.83x
qwen3-32b (65.5 GB Dense)	scanned	1 (8)	to_maxtext	107.24	381.46	203.69 (Load Orbax)	585.15	692.38	1.00x (baseline)
↳	scanned	1 (8)	safetensor	107.24	—	100.72 (Safetensors+Convert: load 82.04 + transform 18.68; peak host RAM 34.2 GB)	100.72	207.96	3.33x / 5.81x
qwen3-32b (65.5 GB Dense)	scanned	2 (16)	to_maxtext	107.24	381.46	238.09 (Load Orbax)	619.55	726.79	1.00x (baseline)
↳	scanned	2 (16)	safetensor	107.24	—	88.92 (Safetensors+Convert: load 35.44 + transform 53.42; peak host RAM 70.3 GB)	88.92	196.16	3.71x / 6.97x
qwen3-32b (65.5 GB Dense)	unscanned	1 (8)	to_maxtext	107.24	208.04	36.22 (Load Orbax)	244.26	351.50	1.00x (baseline)
↳	unscanned	1 (8)	safetensor	107.24	—	77.57 (Safetensors+Convert: load 72.96 + transform 4.58; peak host RAM 35.0 GB)	77.57	184.80	1.90x / 3.15x
qwen3-32b (65.5 GB Dense)	unscanned	2 (16)	to_maxtext	107.24	208.04	30.37 (Load Orbax)	238.41	345.64	1.00x (baseline)
↳	unscanned	2 (16)	safetensor	107.24	—	74.67 (Safetensors+Convert: load 33.47 + transform 41.10; peak host RAM 70.3 GB)	74.67	181.90	1.90x / 3.19x
deepseek2-16b (31.4 GB MoE)	scanned	1 (8)	to_maxtext	137.26	76.65	94.41 (Load Orbax)	171.06	308.32	1.00x (baseline)
↳	scanned	1 (8)	safetensor	137.26	—	75.00 (Safetensors+Convert: load 50.91 + transform 24.08; peak host RAM 38.4 GB)	75.00	212.25	1.45x / 2.28x
deepseek2-16b (31.4 GB MoE)	scanned	2 (16)	to_maxtext	137.26	76.65	112.52 (Load Orbax)	189.17	326.42	1.00x (baseline)
↳	scanned	2 (16)	safetensor	137.26	—	125.49 (Safetensors+Convert: load 98.62 + transform 26.85; peak host RAM 115.5 GB)	125.49	262.74	1.24x / 1.51x
deepseek2-16b (31.4 GB MoE)	unscanned	1 (8)	to_maxtext	137.26	147.18	18.17 (Load Orbax)	165.35	302.60	1.00x (baseline)
↳	unscanned	1 (8)	safetensor	137.26	—	58.36 (Safetensors+Convert: load 50.57 + transform 7.77; peak host RAM 42.4 GB)	58.36	195.61	1.55x / 2.83x
deepseek2-16b (31.4 GB MoE)	unscanned	2 (16)	to_maxtext	137.26	147.18	17.59 (Load Orbax)	164.76	302.02	1.00x (baseline)
↳	unscanned	2 (16)	safetensor	137.26	—	115.00 (Safetensors+Convert: load 102.40 + transform 12.58; peak host RAM 116.7 GB)	115.00	252.25	1.20x / 1.43x
gpt-oss-20b (41.8 GB MoE)	scanned	1 (8)	to_maxtext	93.12	102.63	63.35 (Load Orbax)	165.98	259.10	1.00x (baseline)
↳	scanned	1 (8)	safetensor	93.12	—	127.47 (Safetensors+Convert: load 48.79 + transform 78.68; peak host RAM 39.2 GB)	127.47	220.59	1.17x / 1.30x
↳	scanned	1 (8)	safetensor (with fix)	93.12	—	127.90 (Safetensors+Convert: load 52.06 + transform 75.83; peak host RAM 36.2 GB)	127.90	221.02	1.17x / 1.30x
gpt-oss-20b (41.8 GB MoE)	scanned	2 (16)	to_maxtext	93.12	102.63	58.31 (Load Orbax)	160.94	254.06	1.00x (baseline)
↳	scanned	2 (16)	safetensor	93.12	—	111.90 (Safetensors+Convert: load 25.63 + transform 86.24; peak host RAM 67.1 GB)	111.90	205.02	1.24x / 1.44x
gpt-oss-20b (41.8 GB MoE)	unscanned	1 (8)	to_maxtext	93.12	185.80	12.63 (Load Orbax)	198.43	291.55	1.00x (baseline)
↳	unscanned	1 (8)	safetensor	93.12	—	78.26 (Safetensors+Convert: load 48.34 + transform 29.90; peak host RAM 37.2 GB)	78.26	171.38	1.70x / 2.54x
↳	unscanned	1 (8)	safetensor (with fix)	93.12	—	147.78 (Safetensors+Convert: load 79.02 + transform 68.75; peak host RAM 39.8 GB)	147.78	240.91	1.21x / 1.34x
gpt-oss-20b (41.8 GB MoE)	unscanned	2 (16)	to_maxtext	93.12	185.80	12.23 (Load Orbax)	198.03	291.15	1.00x (baseline)
↳	unscanned	2 (16)	safetensor	93.12	—	104.55 (Safetensors+Convert: load 32.48 + transform 72.02; peak host RAM 68.3 GB)	104.55	197.67	1.47x / 1.89x
qwen3-30b-a3b (61.1 GB MoE)	scanned	1 (8)	to_maxtext	78.82	151.57	386.06 (Load Orbax)	537.62	616.45	1.00x (baseline)
↳	scanned	1 (8)	safetensor	78.82	—	129.52 (Safetensors+Convert: load 99.12 + transform 30.38; peak host RAM 61.4 GB)	129.52	208.34	2.96x / 4.15x
qwen3-30b-a3b (61.1 GB MoE)	scanned	2 (16)	to_maxtext	78.82	151.57	449.33 (Load Orbax)	600.90	679.72	1.00x (baseline)
↳	scanned	2 (16)	safetensor	78.82	—	385.50 (Safetensors+Convert: load 351.63 + transform 33.47; peak host RAM 156.1 GB)	385.50	464.32	1.46x / 1.56x
qwen3-30b-a3b (61.1 GB MoE)	unscanned	1 (8)	to_maxtext	78.82	233.63	31.57 (Load Orbax)	265.20	344.02	1.00x (baseline)
↳	unscanned	1 (8)	safetensor	78.82	—	110.70 (Safetensors+Convert: load 95.77 + transform 14.90; peak host RAM 61.0 GB)	110.70	189.53	1.82x / 2.40x
qwen3-30b-a3b (61.1 GB MoE)	unscanned	2 (16)	to_maxtext	78.82	233.63	25.79 (Load Orbax)	259.42	338.24	1.00x (baseline)
↳	unscanned	2 (16)	safetensor	78.82	—	354.16 (Safetensors+Convert: load 333.13 + transform 20.99; peak host RAM 151.1 GB)	354.16	432.98	0.78x / 0.73x
mixtral-8x7b (93.4 GB MoE)	scanned	1 (8)	to_maxtext	239.75	134.07	533.60 (Load Orbax)	667.67	907.42	1.00x (baseline)
↳	scanned	1 (8)	safetensor	239.75	—	126.93 (Safetensors+Convert: load 113.62 + transform 13.31; peak host RAM 39.6 GB)	126.93	366.67	2.47x / 5.26x
mixtral-8x7b (93.4 GB MoE)	scanned	2 (16)	to_maxtext	239.75	134.07	792.49 (Load Orbax)	926.56	1,166.31	1.00x (baseline)
↳	scanned	2 (16)	safetensor	239.75	—	63.12 (Safetensors+Convert: load 45.96 + transform 17.16; peak host RAM 74.7 GB)	63.12	302.87	3.85x / 14.68x
mixtral-8x7b (93.4 GB MoE)	unscanned	1 (8)	to_maxtext	239.75	208.28	31.95 (Load Orbax)	240.24	479.98	1.00x (baseline)
↳	unscanned	1 (8)	safetensor	239.75	—	115.61 (Safetensors+Convert: load 110.64 + transform 4.96; peak host RAM 39.1 GB)	115.61	355.36	1.35x / 2.08x
mixtral-8x7b (93.4 GB MoE)	unscanned	2 (16)	to_maxtext	239.75	208.28	47.67 (Load Orbax)	255.95	495.70	1.00x (baseline)
↳	unscanned	2 (16)	safetensor	239.75	—	56.55 (Safetensors+Convert: load 44.84 + transform 11.62; peak host RAM 69.6 GB)	56.55	296.30	1.67x / 4.53x
llama3.1-70b (141.1 GB Dense)	scanned	1 (8)	to_maxtext	197.80	780.27	411.50 (Load Orbax)	1,191.78	1,389.57	1.00x (baseline)
↳	scanned	1 (8)	safetensor	197.80	—	199.20 (Safetensors+Convert: load 165.08 + transform 34.12; peak host RAM 37.9 GB)	199.20	397.00	3.50x / 5.98x
llama3.1-70b (141.1 GB Dense)	scanned	2 (16)	to_maxtext	197.80	780.27	295.79 (Load Orbax)	1,076.06	1,273.86	1.00x (baseline)
↳	scanned	2 (16)	safetensor	197.80	—	104.98 (Safetensors+Convert: load 72.44 + transform 32.48; peak host RAM 73.7 GB)	104.98	302.78	4.21x / 10.25x
llama3.1-70b (141.1 GB Dense)	unscanned	1 (8)	to_maxtext	171.23	398.09	67.68 (Load Orbax)	465.77	636.99	1.00x (baseline)
↳	unscanned	1 (8)	safetensor	197.80	—	168.09 (Safetensors+Convert: load 159.02 + transform 9.05; peak host RAM 38.8 GB)	168.09	365.89	1.74x / 2.77x
llama3.1-70b (141.1 GB Dense)	unscanned	2 (16)	to_maxtext	171.23	398.09	42.48 (Load Orbax)	440.56	611.79	1.00x (baseline)
↳	unscanned	2 (16)	safetensor	197.80	—	113.96 (Safetensors+Convert: load 100.40 + transform 13.52; peak host RAM 73.7 GB)	113.96	311.76	1.96x / 3.87x
gpt-oss-120b (233.7 GB MoE)	scanned	1 (8)	to_maxtext	350.33	581.39	839.25 (Load Orbax)	1,420.65	1,770.97	1.00x (baseline)
↳	scanned	1 (8)	safetensor	350.33	—	HBM OOM in transform (load 253.62 s; 200.32 GiB of HLO temporaries per device, 94.74 GiB available)	N/A	N/A	Only to_maxtext loads this config
↳	scanned	1 (8)	safetensor (with fix)	350.33	—	504.92 (Safetensors+Convert: load 260.44 + transform 244.47; peak host RAM 44.6 GB)	504.92	855.24	2.07x / 2.81x
gpt-oss-120b (233.7 GB MoE)	scanned	2 (16)	to_maxtext	350.33	581.39	893.97 (Load Orbax)	1,475.36	1,825.69	1.00x (baseline)
↳	scanned	2 (16)	safetensor	350.33	—	HBM OOM in transform (load 126.92 s; 100.21 GiB of HLO temporaries per device, 94.74 GiB available)	N/A	N/A	Only to_maxtext loads this config
↳	scanned	2 (16)	safetensor (with fix)	350.33	—	429.98 (Safetensors+Convert: load 127.52 + transform 302.46; peak host RAM 71.4 GB)	429.98	780.31	2.34x / 3.43x
gpt-oss-120b (233.7 GB MoE)	unscanned	1 (8)	to_maxtext	350.33	617.84	110.61 (Load Orbax)	728.46	1,078.78	1.00x (baseline)
↳	unscanned	1 (8)	safetensor	350.33	—	446.71 (Safetensors+Convert: load 250.76 + transform 195.93; peak host RAM 306.0 GB)	446.71	797.04	1.35x / 1.63x
gpt-oss-120b (233.7 GB MoE)	unscanned	2 (16)	to_maxtext	350.33	617.84	115.05 (Load Orbax)	732.90	1,083.22	1.00x (baseline)
↳	unscanned	2 (16)	safetensor	350.33	—	390.69 (Safetensors+Convert: load 122.59 + transform 268.06; peak host RAM 338.5 GB)	390.69	741.01	1.46x / 1.88x
```

</details>
<!-- SHEET TABLE END -->

The text between the `SHEET TABLE` markers is generated by
[`scripts/build_tpu7x_sheet_table.py`](../scripts/build_tpu7x_sheet_table.py) (`--suite p2`, plus
`--variant "with fix=<run_tpu7x_16m_fix logs>"` for the `safetensor (with fix)` rows) from the result JSONs.
A failed step shows its reason in that step's column.

### 3.2 >200B models (Priority 1)

The >200B table, with Lance Wang's multi-host runs, is in the same format in the row-by-row section of
[`tpu7x_priority1_progress.md`](tpu7x_priority1_progress.md).

## 4. Key takeaways

1. **`safetensor` is faster end-to-end in 54 of the 56 configurations where both pipelines load the model:**
   median 1.80×, up to 4.21× (`llama3.1-70b` scanned · 2 slices). `to_maxtext` needs an extra CPU conversion
   before its Orbax load: 19 s (0.6B) to 780 s (70B), and 2,626 s for unscanned 235B.
2. **`to_maxtext` Orbax loads are fast for unscanned checkpoints but slow for scanned MoE checkpoints:**
   - unscanned: 3.3–115 s
   - scanned: `mixtral-8x7b` 534 s / 793 s, `qwen3-30b-a3b` 386 s / 449 s, `gpt-oss-120b` 839 s / 894 s (1 / 2 slices)

   The same pattern appeared on v5p.
3. **Above 200B, only `safetensor` works in most configurations** (details in the large-model report):
   - `to_maxtext` keeps the whole converted state in host RAM.
   - Scanned conversions also need the source staged in RAM.
   - So 235B scanned, 284B, 397B, and 480B can't be converted on one tpu7x host. Unscanned 397B finished its transform, then was OOM-killed when the Orbax save started.
   - Unscanned 235B converts by streaming shards from GCS, in 2,626 s with 16 parallel range reads (5,150 s single-stream).
   - `safetensor` loads them directly: 235B and 284B on 1 slice, 480B and unscanned 397B on 2, 671B on 4.
4. **Two configurations favor `to_maxtext`** (see the issues report):
   - `qwen3-14b` scanned · 1 slice (0.65×): its `safetensor` transform takes ~290 s, reproduced in a re-run.
   - `qwen3-30b-a3b` unscanned · 2 slices (0.78×): its 2-slice `safetensor` load takes 333 s, vs 96 s on 1 slice.
5. **`gpt-oss-120b` scanned needs this branch's fix for `safetensor`.** With Lance's PRs, the transform runs out of
   HBM: one jitted concatenate of up to 128 reshard slices asks for 200 GB of temporaries per device on 1 slice
   (100 GB on 2). Joining the slices two at a time (issue U-6) fixes it: 505.1 s on 1 slice and 430.2 s on 2, so
   `safetensor` is 2.07× and 2.34× faster end-to-end than `to_maxtext` there too (the `safetensor (with fix)` rows).
   Unscanned `safetensor` works without the fix: 446.7 s and 390.7 s.
6. **Lance's PRs are required on tpu7x for >200B models.** Without them, `origin/main` either has no mapping (480B)
   or doesn't finish loading within 1800 s.

## 5. Not run

| What | Why |
| :--- | :--- |
| `deepseek3.2-671b` (1,367 GB), `deepseek4-1600b` (3,198 GB) on single-host slices | Extra DeepSeek capacity tests from the v5p matrix that weren't in the tpu7x lists. Lance's multi-host runs loaded both with `safetensor` (scanned / unscanned): `deepseek3.2-671b` in 672.46 / 569.71 s on 2 × `2x2x2`, `deepseek4-1600b` in 1,047.93 / 1,008.62 s on 4 × `2x2x2` ([`tpu7x_multihost_results.md`](tpu7x_multihost_results.md)). |
| `deepseek3-671b` unscanned on 1–2 slices; `deepseek3-671b` `to_maxtext` | Same capacity limits as scanned; 1.34 TB can't be converted on one host. |
| `qwen3.5-397b-a17b` and `qwen3-480b-a35b` unscanned on 1 slice (`safetensor`) | 98% of / more than one slice's HBM. |

## 6. Issues found and fixed

See [`tpu7x_benchmark_ckpt_load_time_issues.md`](tpu7x_benchmark_ckpt_load_time_issues.md):
- **T-1 … T-17:** harness, infrastructure and capacity issues.
- **U-1 … U-8:** MaxText issues and capacity limits. Two MaxText bugs are fixed on this branch and validated on
  tpu7x: the `qwen3.5-397b-a17b` composite-split HBM peak (U-1) and the `gpt-oss-120b` reshard temporaries (U-6).
