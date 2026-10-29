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
- **Tables and CSV:** generated from the result JSONs by
  [`scripts/build_tpu7x_tables.py`](../scripts/build_tpu7x_tables.py). The repeated `qwen3-14b` and `qwen3-32b` scanned
  1-slice runs (`run_tpu7x_16m_rerun`) replace the first runs.

## 3. Results (`to_maxtext` vs `safetensor`, row by row)

How to read the table:
- **Col 2:** the `to_maxtext` conversion. It runs once per scan mode and is shared by the 1- and 2-slice loads.
- **Col 3:** the `to_maxtext` Orbax load, or the `safetensor` load + convert.
- **Speedup:** `to_maxtext` total ÷ `safetensor` total, end-to-end / without Step 1.
- A failed step shows its reason in that step's column.

The same values, with sub-phases, are in [`tpu7x_benchmark_ckpt_load_time_summary.csv`](tpu7x_benchmark_ckpt_load_time_summary.csv).

<!-- TABLE START -->
| Model | Scan Mode | Slices (Devices) | Pipeline | 1. `HF -> GCS` (s) | 2. `Convert to Orbax` (s) | 3. `Load into MaxText` (s) | **Total (No Step 1) (s)** | **Total (End-to-End) (s)** | **Speedup (E2E / No Step 1)** |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`qwen3-0.6b`** | `scanned` | `1 (8)` | `to_maxtext` | 15.98 | 19.40 | 2.54 | 21.94 | 37.92 | 1.00x (baseline) |
| ↳ | `scanned` | `1 (8)` | **`safetensor`** | **15.98** | **—** | **16.14** | **16.14** | **32.12** | **1.18x / 1.36x** |
| **`qwen3-0.6b`** | `scanned` | `2 (16)` | `to_maxtext` | 15.98 | 19.40 | 3.28 | 22.68 | 38.66 | 1.00x (baseline) |
| ↳ | `scanned` | `2 (16)` | **`safetensor`** | **15.98** | **—** | **16.13** | **16.13** | **32.11** | **1.20x / 1.41x** |
| **`qwen3-0.6b`** | `unscanned` | `1 (8)` | `to_maxtext` | 15.98 | 66.78 | 3.33 | 70.11 | 86.09 | 1.00x (baseline) |
| ↳ | `unscanned` | `1 (8)` | **`safetensor`** | **15.98** | **—** | **6.49** | **6.49** | **22.47** | **3.83x / 10.80x** |
| **`qwen3-0.6b`** | `unscanned` | `2 (16)` | `to_maxtext` | 15.98 | 66.78 | 3.38 | 70.16 | 86.14 | 1.00x (baseline) |
| ↳ | `unscanned` | `2 (16)` | **`safetensor`** | **15.98** | **—** | **9.46** | **9.46** | **25.44** | **3.39x / 7.42x** |
| **`gemma2-2b`** | `scanned` | `1 (8)` | `to_maxtext` | 169.39 | 54.41 | 5.26 | 59.67 | 229.06 | 1.00x (baseline) |
| ↳ | `scanned` | `1 (8)` | **`safetensor`** | **169.39** | **—** | **27.98** | **27.98** | **197.37** | **1.16x / 2.13x** |
| **`gemma2-2b`** | `scanned` | `2 (16)` | `to_maxtext` | 169.39 | 54.41 | 6.68 | 61.09 | 230.48 | 1.00x (baseline) |
| ↳ | `scanned` | `2 (16)` | **`safetensor`** | **169.39** | **—** | **24.91** | **24.91** | **194.30** | **1.19x / 2.45x** |
| **`gemma2-2b`** | `unscanned` | `1 (8)` | `to_maxtext` | 169.39 | 91.17 | 7.83 | 99.00 | 268.39 | 1.00x (baseline) |
| ↳ | `unscanned` | `1 (8)` | **`safetensor`** | **169.39** | **—** | **19.90** | **19.90** | **189.29** | **1.42x / 4.98x** |
| **`gemma2-2b`** | `unscanned` | `2 (16)` | `to_maxtext` | 169.39 | 91.17 | 8.46 | 99.63 | 269.02 | 1.00x (baseline) |
| ↳ | `unscanned` | `2 (16)` | **`safetensor`** | **169.39** | **—** | **17.91** | **17.91** | **187.30** | **1.44x / 5.56x** |
| **`qwen3-4b`** | `scanned` | `1 (8)` | `to_maxtext` | 51.92 | 62.21 | 14.25 | 76.46 | 128.38 | 1.00x (baseline) |
| ↳ | `scanned` | `1 (8)` | **`safetensor`** | **51.92** | **—** | **24.66** | **24.66** | **76.58** | **1.68x / 3.10x** |
| **`qwen3-4b`** | `scanned` | `2 (16)` | `to_maxtext` | 51.92 | 62.21 | 18.93 | 81.14 | 133.06 | 1.00x (baseline) |
| ↳ | `scanned` | `2 (16)` | **`safetensor`** | **51.92** | **—** | **22.81** | **22.81** | **74.74** | **1.78x / 3.56x** |
| **`qwen3-4b`** | `unscanned` | `1 (8)` | `to_maxtext` | 51.92 | 95.90 | 9.73 | 105.63 | 157.55 | 1.00x (baseline) |
| ↳ | `unscanned` | `1 (8)` | **`safetensor`** | **51.92** | **—** | **14.40** | **14.40** | **66.32** | **2.38x / 7.33x** |
| **`qwen3-4b`** | `unscanned` | `2 (16)` | `to_maxtext` | 51.92 | 95.90 | 9.85 | 105.74 | 157.66 | 1.00x (baseline) |
| ↳ | `unscanned` | `2 (16)` | **`safetensor`** | **51.92** | **—** | **15.03** | **15.03** | **66.95** | **2.35x / 7.04x** |
| **`gemma3-4b`** | `scanned` | `1 (8)` | `to_maxtext` | 55.81 | 56.42 | 7.84 | 64.26 | 120.08 | 1.00x (baseline) |
| ↳ | `scanned` | `1 (8)` | **`safetensor`** | **55.81** | **—** | **24.33** | **24.33** | **80.14** | **1.50x / 2.64x** |
| **`gemma3-4b`** | `scanned` | `2 (16)` | `to_maxtext` | 55.81 | 56.42 | 7.05 | 63.47 | 119.28 | 1.00x (baseline) |
| ↳ | `scanned` | `2 (16)` | **`safetensor`** | **55.81** | **—** | **24.14** | **24.14** | **79.95** | **1.49x / 2.63x** |
| **`gemma3-4b`** | `unscanned` | `1 (8)` | `to_maxtext` | 55.81 | 108.93 | 9.90 | 118.83 | 174.64 | 1.00x (baseline) |
| ↳ | `unscanned` | `1 (8)` | **`safetensor`** | **55.81** | **—** | **17.83** | **17.83** | **73.64** | **2.37x / 6.66x** |
| **`gemma3-4b`** | `unscanned` | `2 (16)` | `to_maxtext` | 55.81 | 108.93 | 8.13 | 117.06 | 172.87 | 1.00x (baseline) |
| ↳ | `unscanned` | `2 (16)` | **`safetensor`** | **55.81** | **—** | **21.15** | **21.15** | **76.96** | **2.25x / 5.53x** |
| **`llama3.1-8b`** | `scanned` | `1 (8)` | `to_maxtext` | 62.91 | 89.58 | 22.02 | 111.60 | 174.50 | 1.00x (baseline) |
| ↳ | `scanned` | `1 (8)` | **`safetensor`** | **62.91** | **—** | **51.76** | **51.76** | **114.66** | **1.52x / 2.16x** |
| **`llama3.1-8b`** | `scanned` | `2 (16)` | `to_maxtext` | 62.91 | 89.58 | 17.92 | 107.50 | 170.40 | 1.00x (baseline) |
| ↳ | `scanned` | `2 (16)` | **`safetensor`** | **62.91** | **—** | **28.52** | **28.52** | **91.42** | **1.86x / 3.77x** |
| **`llama3.1-8b`** | `unscanned` | `1 (8)` | `to_maxtext` | 62.91 | 105.31 | 14.25 | 119.56 | 182.47 | 1.00x (baseline) |
| ↳ | `unscanned` | `1 (8)` | **`safetensor`** | **62.91** | **—** | **36.73** | **36.73** | **99.63** | **1.83x / 3.26x** |
| **`llama3.1-8b`** | `unscanned` | `2 (16)` | `to_maxtext` | 62.91 | 105.31 | 12.51 | 117.83 | 180.73 | 1.00x (baseline) |
| ↳ | `unscanned` | `2 (16)` | **`safetensor`** | **62.91** | **—** | **18.12** | **18.12** | **81.03** | **2.23x / 6.50x** |
| **`qwen3-8b`** | `scanned` | `1 (8)` | `to_maxtext` | 54.26 | 96.45 | 21.56 | 118.01 | 172.27 | 1.00x (baseline) |
| ↳ | `scanned` | `1 (8)` | **`safetensor`** | **54.26** | **—** | **43.08** | **43.08** | **97.33** | **1.77x / 2.74x** |
| **`qwen3-8b`** | `scanned` | `2 (16)` | `to_maxtext` | 54.26 | 96.45 | 22.54 | 118.99 | 173.25 | 1.00x (baseline) |
| ↳ | `scanned` | `2 (16)` | **`safetensor`** | **54.26** | **—** | **33.96** | **33.96** | **88.22** | **1.96x / 3.50x** |
| **`qwen3-8b`** | `unscanned` | `1 (8)` | `to_maxtext` | 54.26 | 119.91 | 13.27 | 133.18 | 187.43 | 1.00x (baseline) |
| ↳ | `unscanned` | `1 (8)` | **`safetensor`** | **54.26** | **—** | **28.15** | **28.15** | **82.41** | **2.27x / 4.73x** |
| **`qwen3-8b`** | `unscanned` | `2 (16)` | `to_maxtext` | 54.26 | 119.91 | 11.37 | 131.28 | 185.54 | 1.00x (baseline) |
| ↳ | `unscanned` | `2 (16)` | **`safetensor`** | **54.26** | **—** | **22.35** | **22.35** | **76.60** | **2.42x / 5.87x** |
| **`qwen3-14b`** | `scanned` | `1 (8)` | `to_maxtext` | 51.35 | 145.85 | 48.90 | 194.75 | 246.10 | 1.00x (baseline) |
| ↳ | `scanned` | `1 (8)` | **`safetensor`** | **51.35** | **—** | **329.90** | **329.90** | **381.25** | **0.65x / 0.59x** |
| **`qwen3-14b`** | `scanned` | `2 (16)` | `to_maxtext` | 51.35 | 145.85 | 61.90 | 207.75 | 259.10 | 1.00x (baseline) |
| ↳ | `scanned` | `2 (16)` | **`safetensor`** | **51.35** | **—** | **69.02** | **69.02** | **120.37** | **2.15x / 3.01x** |
| **`qwen3-14b`** | `unscanned` | `1 (8)` | `to_maxtext` | 51.35 | 144.74 | 19.69 | 164.42 | 215.77 | 1.00x (baseline) |
| ↳ | `unscanned` | `1 (8)` | **`safetensor`** | **51.35** | **—** | **40.31** | **40.31** | **91.66** | **2.35x / 4.08x** |
| **`qwen3-14b`** | `unscanned` | `2 (16)` | `to_maxtext` | 51.35 | 144.74 | 19.05 | 163.79 | 215.13 | 1.00x (baseline) |
| ↳ | `unscanned` | `2 (16)` | **`safetensor`** | **51.35** | **—** | **57.86** | **57.86** | **109.21** | **1.97x / 2.83x** |
| **`qwen3-32b`** | `scanned` | `1 (8)` | `to_maxtext` | 107.24 | 381.46 | 203.69 | 585.15 | 692.38 | 1.00x (baseline) |
| ↳ | `scanned` | `1 (8)` | **`safetensor`** | **107.24** | **—** | **100.72** | **100.72** | **207.96** | **3.33x / 5.81x** |
| **`qwen3-32b`** | `scanned` | `2 (16)` | `to_maxtext` | 107.24 | 381.46 | 238.09 | 619.55 | 726.79 | 1.00x (baseline) |
| ↳ | `scanned` | `2 (16)` | **`safetensor`** | **107.24** | **—** | **88.92** | **88.92** | **196.16** | **3.71x / 6.97x** |
| **`qwen3-32b`** | `unscanned` | `1 (8)` | `to_maxtext` | 107.24 | 208.04 | 36.22 | 244.26 | 351.50 | 1.00x (baseline) |
| ↳ | `unscanned` | `1 (8)` | **`safetensor`** | **107.24** | **—** | **77.57** | **77.57** | **184.80** | **1.90x / 3.15x** |
| **`qwen3-32b`** | `unscanned` | `2 (16)` | `to_maxtext` | 107.24 | 208.04 | 30.37 | 238.41 | 345.64 | 1.00x (baseline) |
| ↳ | `unscanned` | `2 (16)` | **`safetensor`** | **107.24** | **—** | **74.67** | **74.67** | **181.90** | **1.90x / 3.19x** |
| **`deepseek2-16b`** | `scanned` | `1 (8)` | `to_maxtext` | 137.26 | 76.65 | 94.41 | 171.06 | 308.32 | 1.00x (baseline) |
| ↳ | `scanned` | `1 (8)` | **`safetensor`** | **137.26** | **—** | **75.00** | **75.00** | **212.25** | **1.45x / 2.28x** |
| **`deepseek2-16b`** | `scanned` | `2 (16)` | `to_maxtext` | 137.26 | 76.65 | 112.52 | 189.17 | 326.42 | 1.00x (baseline) |
| ↳ | `scanned` | `2 (16)` | **`safetensor`** | **137.26** | **—** | **125.49** | **125.49** | **262.74** | **1.24x / 1.51x** |
| **`deepseek2-16b`** | `unscanned` | `1 (8)` | `to_maxtext` | 137.26 | 147.18 | 18.17 | 165.35 | 302.60 | 1.00x (baseline) |
| ↳ | `unscanned` | `1 (8)` | **`safetensor`** | **137.26** | **—** | **58.36** | **58.36** | **195.61** | **1.55x / 2.83x** |
| **`deepseek2-16b`** | `unscanned` | `2 (16)` | `to_maxtext` | 137.26 | 147.18 | 17.59 | 164.76 | 302.02 | 1.00x (baseline) |
| ↳ | `unscanned` | `2 (16)` | **`safetensor`** | **137.26** | **—** | **115.00** | **115.00** | **252.25** | **1.20x / 1.43x** |
| **`gpt-oss-20b`** | `scanned` | `1 (8)` | `to_maxtext` | 93.12 | 102.63 | 63.35 | 165.98 | 259.10 | 1.00x (baseline) |
| ↳ | `scanned` | `1 (8)` | **`safetensor`** | **93.12** | **—** | **127.47** | **127.47** | **220.59** | **1.17x / 1.30x** |
| **`gpt-oss-20b`** | `scanned` | `2 (16)` | `to_maxtext` | 93.12 | 102.63 | 58.31 | 160.94 | 254.06 | 1.00x (baseline) |
| ↳ | `scanned` | `2 (16)` | **`safetensor`** | **93.12** | **—** | **111.90** | **111.90** | **205.02** | **1.24x / 1.44x** |
| **`gpt-oss-20b`** | `unscanned` | `1 (8)` | `to_maxtext` | 93.12 | 185.80 | 12.63 | 198.43 | 291.55 | 1.00x (baseline) |
| ↳ | `unscanned` | `1 (8)` | **`safetensor`** | **93.12** | **—** | **78.26** | **78.26** | **171.38** | **1.70x / 2.54x** |
| **`gpt-oss-20b`** | `unscanned` | `2 (16)` | `to_maxtext` | 93.12 | 185.80 | 12.23 | 198.03 | 291.15 | 1.00x (baseline) |
| ↳ | `unscanned` | `2 (16)` | **`safetensor`** | **93.12** | **—** | **104.55** | **104.55** | **197.67** | **1.47x / 1.89x** |
| **`qwen3-30b-a3b`** | `scanned` | `1 (8)` | `to_maxtext` | 78.82 | 151.57 | 386.06 | 537.62 | 616.45 | 1.00x (baseline) |
| ↳ | `scanned` | `1 (8)` | **`safetensor`** | **78.82** | **—** | **129.52** | **129.52** | **208.34** | **2.96x / 4.15x** |
| **`qwen3-30b-a3b`** | `scanned` | `2 (16)` | `to_maxtext` | 78.82 | 151.57 | 449.33 | 600.90 | 679.72 | 1.00x (baseline) |
| ↳ | `scanned` | `2 (16)` | **`safetensor`** | **78.82** | **—** | **385.50** | **385.50** | **464.32** | **1.46x / 1.56x** |
| **`qwen3-30b-a3b`** | `unscanned` | `1 (8)` | `to_maxtext` | 78.82 | 233.63 | 31.57 | 265.20 | 344.02 | 1.00x (baseline) |
| ↳ | `unscanned` | `1 (8)` | **`safetensor`** | **78.82** | **—** | **110.70** | **110.70** | **189.53** | **1.82x / 2.40x** |
| **`qwen3-30b-a3b`** | `unscanned` | `2 (16)` | `to_maxtext` | 78.82 | 233.63 | 25.79 | 259.42 | 338.24 | 1.00x (baseline) |
| ↳ | `unscanned` | `2 (16)` | **`safetensor`** | **78.82** | **—** | **354.16** | **354.16** | **432.98** | **0.78x / 0.73x** |
| **`mixtral-8x7b`** | `scanned` | `1 (8)` | `to_maxtext` | 239.75 | 134.07 | 533.60 | 667.67 | 907.42 | 1.00x (baseline) |
| ↳ | `scanned` | `1 (8)` | **`safetensor`** | **239.75** | **—** | **126.93** | **126.93** | **366.67** | **2.47x / 5.26x** |
| **`mixtral-8x7b`** | `scanned` | `2 (16)` | `to_maxtext` | 239.75 | 134.07 | 792.49 | 926.56 | 1,166.31 | 1.00x (baseline) |
| ↳ | `scanned` | `2 (16)` | **`safetensor`** | **239.75** | **—** | **63.12** | **63.12** | **302.87** | **3.85x / 14.68x** |
| **`mixtral-8x7b`** | `unscanned` | `1 (8)` | `to_maxtext` | 239.75 | 208.28 | 31.95 | 240.24 | 479.98 | 1.00x (baseline) |
| ↳ | `unscanned` | `1 (8)` | **`safetensor`** | **239.75** | **—** | **115.61** | **115.61** | **355.36** | **1.35x / 2.08x** |
| **`mixtral-8x7b`** | `unscanned` | `2 (16)` | `to_maxtext` | 239.75 | 208.28 | 47.67 | 255.95 | 495.70 | 1.00x (baseline) |
| ↳ | `unscanned` | `2 (16)` | **`safetensor`** | **239.75** | **—** | **56.55** | **56.55** | **296.30** | **1.67x / 4.53x** |
| **`llama3.1-70b`** | `scanned` | `1 (8)` | `to_maxtext` | 197.80 | 780.27 | 411.50 | 1,191.78 | 1,389.57 | 1.00x (baseline) |
| ↳ | `scanned` | `1 (8)` | **`safetensor`** | **197.80** | **—** | **199.20** | **199.20** | **397.00** | **3.50x / 5.98x** |
| **`llama3.1-70b`** | `scanned` | `2 (16)` | `to_maxtext` | 197.80 | 780.27 | 295.79 | 1,076.06 | 1,273.86 | 1.00x (baseline) |
| ↳ | `scanned` | `2 (16)` | **`safetensor`** | **197.80** | **—** | **104.98** | **104.98** | **302.78** | **4.21x / 10.25x** |
| **`llama3.1-70b`** | `unscanned` | `1 (8)` | `to_maxtext` | 171.23 | 398.09 | 67.68 | 465.77 | 636.99 | 1.00x (baseline) |
| ↳ | `unscanned` | `1 (8)` | **`safetensor`** | **197.80** | **—** | **168.09** | **168.09** | **365.89** | **1.74x / 2.77x** |
| **`llama3.1-70b`** | `unscanned` | `2 (16)` | `to_maxtext` | 171.23 | 398.09 | 42.48 | 440.56 | 611.79 | 1.00x (baseline) |
| ↳ | `unscanned` | `2 (16)` | **`safetensor`** | **197.80** | **—** | **113.96** | **113.96** | **311.76** | **1.96x / 3.87x** |
| **`gpt-oss-120b`** | `scanned` | `1 (8)` | `to_maxtext` | 350.33 | 581.39 | 839.25 | 1,420.65 | 1,770.97 | 1.00x (baseline) |
| ↳ | `scanned` | `1 (8)` | **`safetensor`** | **350.33** | **—** | **TPU HBM OOM** | — | — | — |
| **`gpt-oss-120b`** | `scanned` | `2 (16)` | `to_maxtext` | 350.33 | 581.39 | 893.97 | 1,475.36 | 1,825.69 | 1.00x (baseline) |
| ↳ | `scanned` | `2 (16)` | **`safetensor`** | **350.33** | **—** | **TPU HBM OOM** | — | — | — |
| **`gpt-oss-120b`** | `unscanned` | `1 (8)` | `to_maxtext` | 350.33 | 617.84 | 110.61 | 728.46 | 1,078.78 | 1.00x (baseline) |
| ↳ | `unscanned` | `1 (8)` | **`safetensor`** | **350.33** | **—** | **446.71** | **446.71** | **797.04** | **1.35x / 1.63x** |
| **`gpt-oss-120b`** | `unscanned` | `2 (16)` | `to_maxtext` | 350.33 | 617.84 | 115.05 | 732.90 | 1,083.22 | 1.00x (baseline) |
| ↳ | `unscanned` | `2 (16)` | **`safetensor`** | **350.33** | **—** | **390.69** | **390.69** | **741.01** | **1.46x / 1.88x** |
| **`qwen3-235b-a22b`** | `scanned` | `1 (8)` | `to_maxtext` | 583.32 | **source must be staged in RAM next to the converted state; exceeds host RAM** | — | N/A | N/A | N/A |
| ↳ | `scanned` | `1 (8)` | **`safetensor`** | **583.32** | **—** | **757.50** | **757.50** | **1,340.81** | **Only `safetensor` loads this config** |
| **`qwen3-235b-a22b`** | `scanned` | `2 (16)` | `to_maxtext` | 583.32 | **source must be staged in RAM next to the converted state; exceeds host RAM** | — | N/A | N/A | N/A |
| ↳ | `scanned` | `2 (16)` | **`safetensor`** | **583.32** | **—** | **792.38** | **792.38** | **1,375.69** | **Only `safetensor` loads this config** |
| **`qwen3-235b-a22b`** | `unscanned` | `1 (8)` | `to_maxtext` | 583.32 | 2,625.50 | 545.51 | 3,171.02 | 3,754.33 | 1.00x (baseline) |
| ↳ | `unscanned` | `1 (8)` | **`safetensor`** | **583.32** | **—** | **697.13** | **697.13** | **1,280.44** | **2.93x / 4.55x** |
| **`qwen3-235b-a22b`** | `unscanned` | `2 (16)` | `to_maxtext` | 583.32 | 2,625.50 | 659.15 | 3,284.65 | 3,867.97 | 1.00x (baseline) |
| ↳ | `unscanned` | `2 (16)` | **`safetensor`** | **583.32** | **—** | **749.00** | **749.00** | **1,332.31** | **2.90x / 4.39x** |
| **`deepseek4-284b`** | `scanned` | `1 (8)` | `to_maxtext` | 0.45 | **host OOM-killed, peak host RAM 989.6 GB** | — | N/A | N/A | N/A |
| ↳ | `scanned` | `1 (8)` | **`safetensor`** | **0.00** | **—** | **453.27** | **453.27** | **453.27** | **Only `safetensor` loads this config** |
| **`deepseek4-284b`** | `scanned` | `2 (16)` | `to_maxtext` | 0.45 | **host OOM-killed, peak host RAM 989.6 GB** | — | N/A | N/A | N/A |
| ↳ | `scanned` | `2 (16)` | **`safetensor`** | **0.00** | **—** | **440.72** | **440.72** | **440.72** | **Only `safetensor` loads this config** |
| **`deepseek4-284b`** | `unscanned` | `1 (8)` | `to_maxtext` | 0.45 | **host OOM-killed, peak host RAM 989.6 GB** | — | N/A | N/A | N/A |
| ↳ | `unscanned` | `1 (8)` | **`safetensor`** | **0.00** | **—** | **414.09** | **414.09** | **414.09** | **Only `safetensor` loads this config** |
| **`deepseek4-284b`** | `unscanned` | `2 (16)` | `to_maxtext` | 0.45 | **host OOM-killed, peak host RAM 989.6 GB** | — | N/A | N/A | N/A |
| ↳ | `unscanned` | `2 (16)` | **`safetensor`** | **0.00** | **—** | **352.96** | **352.96** | **352.96** | **Only `safetensor` loads this config** |
| **`qwen3.5-397b-a17b`** | `scanned` | `1 (8)` | `to_maxtext` | 814.53 | **source must be staged in RAM next to the converted state; exceeds host RAM** | — | N/A | N/A | N/A |
| ↳ | `scanned` | `1 (8)` | **`safetensor`** | **814.53** | **—** | **TPU HBM OOM** | — | — | — |
| **`qwen3.5-397b-a17b`** | `scanned` | `2 (16)` | `to_maxtext` | 814.53 | **source must be staged in RAM next to the converted state; exceeds host RAM** | — | N/A | N/A | N/A |
| ↳ | `scanned` | `2 (16)` | **`safetensor`** | **814.53** | **—** | **killed (-9) on slice 0, peak host RAM 161.2 GB** | — | — | — |
| **`qwen3.5-397b-a17b`** | `scanned` | `4 (32)` | `to_maxtext` | 814.53 | **source must be staged in RAM next to the converted state; exceeds host RAM** | — | N/A | N/A | N/A |
| ↳ | `scanned` | `4 (32)` | **`safetensor`** | **814.53** | **—** | **1,399.68** | **1,399.68** | **2,214.21** | **Only `safetensor` loads this config** |
| **`qwen3.5-397b-a17b`** | `unscanned` | `1 (8)` | `to_maxtext` | 814.53 | **host OOM-killed** | — | N/A | N/A | N/A |
| ↳ | `unscanned` | `1 (8)` | **`safetensor`** | **814.53** | **—** | **not run** | — | — | — |
| **`qwen3.5-397b-a17b`** | `unscanned` | `2 (16)` | `to_maxtext` | 814.53 | **host OOM-killed** | — | N/A | N/A | N/A |
| ↳ | `unscanned` | `2 (16)` | **`safetensor`** | **814.53** | **—** | **1,311.85** | **1,311.85** | **2,126.38** | **Only `safetensor` loads this config** |
| **`qwen3-480b-a35b`** | `scanned` | `1 (8)` | `to_maxtext` | 1,135.50 | **source must be staged in RAM next to the converted state; exceeds host RAM** | — | N/A | N/A | N/A |
| ↳ | `scanned` | `1 (8)` | **`safetensor`** | **1,046.14** | **—** | **TPU HBM OOM** | — | — | — |
| **`qwen3-480b-a35b`** | `scanned` | `2 (16)` | `to_maxtext` | 1,135.50 | **source must be staged in RAM next to the converted state; exceeds host RAM** | — | N/A | N/A | N/A |
| ↳ | `scanned` | `2 (16)` | **`safetensor`** | **1,046.14** | **—** | **1,067.86** | **1,067.86** | **2,114.00** | **Only `safetensor` loads this config** |
| **`qwen3-480b-a35b`** | `unscanned` | `1 (8)` | `to_maxtext` | 1,135.50 | **converted state exceeds host RAM** | — | N/A | N/A | N/A |
| ↳ | `unscanned` | `1 (8)` | **`safetensor`** | **1,046.14** | **—** | **not run** | — | — | — |
| **`qwen3-480b-a35b`** | `unscanned` | `2 (16)` | `to_maxtext` | 1,135.50 | **converted state exceeds host RAM** | — | N/A | N/A | N/A |
| ↳ | `unscanned` | `2 (16)` | **`safetensor`** | **1,046.14** | **—** | **992.17** | **992.17** | **2,038.31** | **Only `safetensor` loads this config** |
| **`deepseek3-671b`** | `scanned` | `1 (8)` | `to_maxtext` | 0.00 | **not run** | — | N/A | N/A | N/A |
| ↳ | `scanned` | `1 (8)` | **`safetensor`** | **0.00** | **—** | **TPU HBM OOM** | — | — | — |
| **`deepseek3-671b`** | `scanned` | `2 (16)` | `to_maxtext` | 0.00 | **not run** | — | N/A | N/A | N/A |
| ↳ | `scanned` | `2 (16)` | **`safetensor`** | **0.00** | **—** | **killed (-9) on slice 0, peak host RAM 742.1 GB** | — | — | — |
| **`deepseek3-671b`** | `scanned` | `4 (32)` | `to_maxtext` | 0.00 | **not run** | — | N/A | N/A | N/A |
| ↳ | `scanned` | `4 (32)` | **`safetensor`** | **0.00** | **—** | **649.35** | **649.35** | **649.35** | **Only `safetensor` loads this config** |
| **`deepseek3-671b`** | `unscanned` | `4 (32)` | `to_maxtext` | 0.00 | **not run** | — | N/A | N/A | N/A |
| ↳ | `unscanned` | `4 (32)` | **`safetensor`** | **0.00** | **—** | **530.96** | **530.96** | **530.96** | **Only `safetensor` loads this config** |
<!-- TABLE END -->

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
5. **`gpt-oss-120b` scanned loads only with `to_maxtext`.** The `safetensor` transform runs out of HBM in a jitted
   expert transform: 200 GB of temporaries per device on 1 slice, 100 GB on 2. Unscanned `safetensor` works: 446.7 s
   on 1 slice and 390.7 s on 2.
6. **Lance's PRs are required on tpu7x for >200B models.** Without them, `origin/main` either has no mapping (480B)
   or doesn't finish loading within 1800 s.

## 5. Not run

| What | Why |
| :--- | :--- |
| `deepseek3.2-671b` (1,367 GB), `deepseek4-1600b` (3,198 GB) | Extra DeepSeek capacity tests from the v5p matrix that weren't in the tpu7x lists. They would need about 4 and 8 slices. |
| `deepseek3-671b` unscanned on 1–2 slices; `deepseek3-671b` `to_maxtext` | Same capacity limits as scanned; 1.34 TB can't be converted on one host. |
| `qwen3.5-397b-a17b` and `qwen3-480b-a35b` unscanned on 1 slice (`safetensor`) | 98% of / more than one slice's HBM. |

## 6. Issues found and fixed

See [`tpu7x_benchmark_ckpt_load_time_issues.md`](tpu7x_benchmark_ckpt_load_time_issues.md):
- **T-1 … T-15:** harness, infrastructure and capacity issues, all resolved.
- **U-1 … U-7:** MaxText issues and capacity limits. Two are open MaxText bugs: the `qwen3.5-397b-a17b` composite-split HBM peak and the `gpt-oss-120b` jitted transform temporaries.
