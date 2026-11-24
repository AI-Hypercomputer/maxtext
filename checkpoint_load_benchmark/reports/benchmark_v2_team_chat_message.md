**Checkpoint Load Benchmark v2 — Update**

We completed the Benchmark v2 matrix comparing `safetensors_dynamic` vs `to_maxtext` across TPU `v7x`, `v6e`, and `v5e` (352 planned configs across 21 models, `0.6B–1600B`). Full results and two new plain-English verdict columns (`Cols AC & AD`) are in the [Benchmark v2 Sheet](https://docs.google.com/spreadsheets/d/1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I/edit?resourcekey=0-0-7jmiYhnTwPWKEHfo5tbw&gid=100#gid=100):

**1. Direct findings (two scenarios — no ratio math needed):**
- **Starting from HuggingFace weights (no Orbax checkpoint yet — `Summary Col AC`):**
  `safetensors_dynamic` is faster in **118 / 121** paired `OK` configs (**~5.0x faster** median without HF download; **~1.8x faster** end-to-end including cold HF download) *because it loads HF safetensors directly into TPU HBM and skips the single-host CPU Orbax conversion step (`to_maxtext` Step 2)*. It also loads **235B–480B** Qwen3 models where single-host CPU `to_maxtext` Step 2 hits `OOM-HOST`.
- **Checkpoint is ALREADY in Orbax format (load-only: `to_maxtext` Step 3 vs `safetensors_dynamic` Step 2 — `Summary Col AD`):**
  - Overall across 121 paired `OK` configs, loading pre-converted Orbax is **~2x faster** (wins **96 / 121** pairs).
  - **`unscanned` (`scan_layers=False`):** Pre-converted Orbax wins **61 / 63** pairs (**~2.4x faster** median; exception: `qwen3.5-397b-a17b` on `v7x`, where `safetensors_dynamic` is `1.5x–1.8x` faster).
  - **`scanned` (`scan_layers=True`):** Pre-converted Orbax is faster on smaller models (`<30B`, **35 / 58** pairs), while `safetensors_dynamic` is **2x–10.4x faster** on larger scanned models (`30B–120B`, **23 / 58** pairs: `mixtral-8x7b`, `qwen3-30b-a3b`, `llama3.1-70b`, `gpt-oss-120b`) because scanned Orbax stacked-chunk reads are throttled by `checkpoint_storage_concurrent_gb`.

**2. Issues found during experiments & where the fixes live:**
1. **Draft PR [#6353](https://github.com/AI-Hypercomputer/maxtext/pull/6353)** — Fixes `safetensors_dynamic` **TPU HBM peak OOM** on large scanned MoE models (`qwen3.5-397b` and `gpt-oss-120b` on TPU7x) by stacking composite layer/expert leaves incrementally and joining reshard slices in pairs instead of holding all inputs in HBM simultaneously.
2. **Draft PR [#6354](https://github.com/AI-Hypercomputer/maxtext/pull/6354)** — Fixes `to_maxtext` **host RAM OOM** during Orbax checkpoint save (`qwen3.5-397b`) by passing `checkpoint_storage_concurrent_gb` into the `to_maxtext` Orbax save path (which previously ignored the concurrency cap).
3. **PR [#6319](https://github.com/AI-Hypercomputer/maxtext/pull/6319) (Lance)** — Fixes `safetensors_dynamic` **host RAM OOM on 64-host `v5e`** for unfused per-expert MoE checkpoints (Qwen3 MoE + real-weight DeepSeek) by reading `gs://` safetensors through `gcsfs` instead of `tf.io.gfile` 64 MiB read-ahead.
4. **Benchmark harness fix (`docs/checkpoint-load-benchmark` @ `f2bdfe636`)** — Fixed a harness env override (`STAGE_SPARSE_DEEPSEEK_OOM=1`) that caused the initial `v2|` DeepSeek `284B–1600B` rows to read local `synthetic_sparse` files instead of real GCS BF16 checkpoints. Those rows are now explicitly marked `synthetic_sparse` on the sheet, and real-weight GCS BF16 re-runs are in progress (`284B`, `671B`, and `3.2-671B` already verified `OK` on `v7x` and `v5e + PR #6319`; remaining configs finishing up).

Both draft PRs ([#6353](https://github.com/AI-Hypercomputer/maxtext/pull/6353) and [#6354](https://github.com/AI-Hypercomputer/maxtext/pull/6354)) are single-commit off `main` with unit tests passing locally — reviews welcome!
