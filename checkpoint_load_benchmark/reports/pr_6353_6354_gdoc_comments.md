# Google Doc Comments — PR #6353 & PR #6354

## PR #6353 (`safetensors_dynamic` TPU HBM OOM)

**[PR #6353](https://github.com/AI-Hypercomputer/maxtext/pull/6353) — Fix `safetensors_dynamic` TPU HBM peak OOM on scanned MoE models**
- **Run where found:** TPU7x `safetensors_dynamic` runs on scanned `qwen3.5-397b-a17b` (16 devices; [before OOM](https://paste.googleplex.com/6755399436565968)) and scanned `gpt-oss-120b` (8 devices; [before OOM](https://paste.googleplex.com/6755399436565980)).
- **Issue:** Final weights fit in TPU HBM, but post-load transforms hit HBM OOM because:
  1. All layers of fused `(wi_0, wi_1)` expert weights were stacked together (~22.5 GiB/device) before slicing.
  2. `_reshard` joined up to 128 slices in one `jnp.concatenate` (200.3 GiB/device XLA temporaries).
- **Fix:** Split composite weights per layer *before* stacking, concatenate reshard slices in pairs, and free intermediate buffers immediately.
- **Result:** Both models load cleanly on TPU7x — `qwen3.5-397b-a17b` in **1,456.5s** ([after OK](https://paste.googleplex.com/6755399437013104)) and `gpt-oss-120b` in **537.8s** ([after OK](https://paste.googleplex.com/6755399436980822)).

---

## PR #6354 (`to_maxtext` Host RAM OOM During Orbax Save)

**[PR #6354](https://github.com/AI-Hypercomputer/maxtext/pull/6354) — Respect `checkpoint_storage_concurrent_gb` in `to_maxtext` Orbax save**
- **Run where found:** Single-host TPU7x CPU conversion (`to_maxtext` Step 2) of unscanned `qwen3.5-397b-a17b` (~807 GB state on a ~1 TB RAM host; [before OOM](https://paste.googleplex.com/6755399436565971)).
- **Issue:** `save_weights_to_checkpoint` ignored `config.checkpoint_storage_concurrent_gb` and always used Orbax's 96 GB write-buffer default on top of the ~807 GB in-memory model, causing a Host RAM OOM right as saving started.
- **Fix:** Pass `config.checkpoint_storage_concurrent_gb` into `create_orbax_checkpoint_manager` when `config` is provided (default `96` is unchanged).
- **Result:** With `checkpoint_storage_concurrent_gb=40`, `qwen3.5-397b-a17b` completes in **7,877.3s** at **885.0 GB** peak Host RAM ([after OK](https://paste.googleplex.com/6755399437015561)), and `llama3.1-70b` peak Host RAM drops from **403.6 GB → 354.6 GB** ([after OK](https://paste.googleplex.com/6755399436958380)).
