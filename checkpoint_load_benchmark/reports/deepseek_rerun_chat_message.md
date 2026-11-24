Below are the issues found during experiments & where the fixes live:

**MaxText PR Fixes**
1. **Draft PR [#6353](https://github.com/AI-Hypercomputer/maxtext/pull/6353)** — Fixes `safetensors_dynamic` **TPU HBM peak OOM** on large scanned MoE models (`qwen3.5-397b` and `gpt-oss-120b` on TPU7x) by stacking composite layer/expert leaves incrementally and joining reshard slices in pairs instead of holding all inputs in HBM simultaneously.
2. **Draft PR [#6354](https://github.com/AI-Hypercomputer/maxtext/pull/6354)** — Fixes `to_maxtext` **host RAM OOM** during Orbax checkpoint save (`qwen3.5-397b`) by passing `checkpoint_storage_concurrent_gb` into the `to_maxtext` Orbax save path (which previously ignored the concurrency cap).
3. **PR [#6319](https://github.com/AI-Hypercomputer/maxtext/pull/6319) (Lance's fix)** — Fixes `safetensors_dynamic` **host RAM OOM on 64-host `v5e`** for unfused per-expert MoE checkpoints (Qwen3 MoE + real-weight DeepSeek) by reading `gs://` safetensors through `gcsfs` instead of `tf.io.gfile` 64 MiB read-ahead.

---

**Benchmark Harness Issue & Re-Run: DeepSeek Real-Weight (`safetensors_dynamic`)**

**Why the bug happened (`284B`, `671B`, `3.2-671B`, `1600B`):**
- HuggingFace publishes these 4 DeepSeek models **only in quantized `FP8` / `FP4` (+ scale) format** (`DeepSeek-V3`, `DeepSeek-V3.2`, `DeepSeek-V4-Flash`, `DeepSeek-V4-Pro`), **not in `BF16`**, while MaxText's loaders require `BF16` weights (neither loader dequantizes `FP8`/`FP4` on the fly).
- In **Benchmark v1** (before we had BF16 checkpoints), the harness used a fallback flag (`STAGE_SPARSE_DEEPSEEK_OOM=1`) to generate local `synthetic_sparse` BF16 files for HBM capacity checks.
- In **Benchmark v2**, we **did** prepare the real dequantized BF16 checkpoints in regional GCS (`hf_cache_bf16/`, Step 1b), **but two harness bugs caused Step 2 to still load local synthetic files and label them `real`**:
  1. `manifests_v2.py` still enabled `STAGE_SPARSE_DEEPSEEK_OOM=1` by default, and `benchmark_ckpt_load_time.py` checked that flag *before* `BENCH_SAFETENSORS_GCS_DIR`.
  2. `harness_v2.py` hardcoded `"checkpoint_weights_kind": "real"` in result JSONs instead of recording the actual source.

**What `synthetic_sparse` vs `real` actually means:**
- **Both versions load `.safetensors` files** (neither initializes random weights in JAX), and **`Step 1` (`HF → GCS`) + `Step 1b` (`FP8/FP4 → BF16` in GCS) were already 100% real**.
- **`synthetic_sparse` (initial `v2|` DeepSeek rows):** In Step 2, each TPU host created all-zero Linux sparse `.safetensors` files (`os.truncate`, exact tensor shapes & headers) on its **own local `/tmp` disk** and read from `/tmp` — **0 bytes read over GCS**. That made load times artificially ~2x faster than network I/O and bypassed `tf.io.gfile` GCS reads (hiding `v5e` `OOM-HOST`), though TPU HBM capacity stayed exact.
- **`real` (`v2real` re-runs):** Streams the **real trained `582 GB–3.2 TB` `BF16` `.safetensors` files over the network from regional GCS (`gs://.../hf_cache_bf16/`)**, measuring true production `GCS → TPU HBM` load time and real GCS client host RAM (confirming `v5e` baseline hits `OOM-HOST` and PR [#6319](https://github.com/AI-Hypercomputer/maxtext/pull/6319) resolves it cleanly up to `3.2 TB`).

**What we implemented to fix it (`docs/checkpoint-load-benchmark` @ `f2bdfe636`):**
1. **Fixed the harness & queue generator (`f2bdfe636`):**
   - Disabled `STAGE_SPARSE_DEEPSEEK_OOM=1` by default in `manifests_v2.py` (`clean_sparse_env=False`).
   - Updated `_should_stage_sparse_bf16()` in `benchmark_ckpt_load_time.py` so an explicit `BENCH_SAFETENSORS_GCS_DIR` **always** takes precedence over synthetic sparse staging.
   - Replaced the hardcoded `"real"` label in `harness_v2.py` so every phase JSON records the true `checkpoint_weights_kind` (`real` vs `synthetic_sparse`) and exact `checkpoint_load_source_uri` (`gs://.../hf_cache_bf16/<model>`).
2. **Updated the [Benchmark v2 Sheet](https://docs.google.com/spreadsheets/d/1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I/edit?resourcekey=0-0-7jmiYhnTwPWKEHfo5tbw&gid=100#gid=100):** Relabelled the initial `v2|` DeepSeek rows as `synthetic_sparse` and documented the exact `local /tmp sparse .safetensors` vs `regional GCS real .safetensors` difference in `README`, `Legend`, and `Columns`.
3. **Launched real-weight GCS BF16 (`hf_cache_bf16`) re-runs** across `v5e` (100% done), `v7x`, and `v6e` using the fixed bundle (`f2bdfe636`), and will add the verified `v2real` rows to the sheet once all queued slices finish.
