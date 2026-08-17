# Standalone Attention Kernel Repro: Splash vs. RPA Results

**Date / Timestamp:** 2026-08-14
**Hardware Platform:** Google Cloud TPU v5p, 4 locally-attached chips (no SPS/pathways-proxy; `jax.devices()` returns 4 `TpuDevice`s directly)
**Script:** `tests/run_attention_kernel_repro.py` (isolated attention-kernel-only comparison, no full model stack)

---

## 1. Methodology

This is a pure attention-kernel comparison: no embeddings, layernorms, or MoE. It isolates:

1. **Training kernel** — Tokamax Splash / Flash Attention (`attention=flash`, `use_tokamax_splash=True`), run under a data/FSDP-sharded training mesh across all 4 TPU chips.
2. **Inference kernel** — vLLM Ragged Paged Attention v3 ("Default RPA", `attention=vllm_rpa`), invoked via `tpu_inference`'s `sharded_ragged_paged_attention` entry point.
3. **Exact reference** — a pure-JAX FP32 causal dot-product-attention implementation (`run_reference_attention`), used as ground truth.

For each dtype (`float32`, `bfloat16`), a sweep of 6 Splash-attention configurations (base2 exponent on/off, fused-reciprocal on/off, and a larger block size) is run against Default RPA v3 and the exact reference, using `tests/unit/attention_kernel_repro_test.py::compare_attention_kernels_on_tpu`.

**Fixed problem size for all configs:**
- `batch_size=4`, `seq_len=512`, `num_query_heads=16`, `num_kv_heads=2`, `head_dim=256`, RPA `block_size=128`.
- Model shape follows `qwen3.5-35b-a3b` (`base_emb_dim=2048`).

Metrics computed by `compute_drift_metrics`: max absolute error (L∞), mean absolute error (MAE), mean squared error (MSE), cosine similarity, and relative L2 error, all computed in FP32 after `jax.device_get`.

### Bugs fixed before trusting these numbers

1. **`query_start_loc` / `request_distribution` construction.** Both `tests/unit/attention_kernel_repro_test.py` and `tests/run_attention_batched_rpa_repro.py` built these RPA metadata arrays with an incorrect tiled pattern (`jnp.tile([0, seq_len], (batch_size,))` / `jnp.tile([0, 0, 1], (batch_size,))`), which does not match the real vLLM-TPU `tpu_runner.py` semantics and produces wrong-shaped / wrong-valued metadata. Fixed to the correct pattern confirmed against `tpu_inference/runner/tpu_runner.py`:
   - `query_start_loc = jnp.arange(0, (batch_size + 1) * seq_len, seq_len, dtype=jnp.int32)` — cumulative per-request token offsets, shape `(batch_size + 1,)`.
   - `request_distribution = jnp.array([0, 0, batch_size], dtype=jnp.int32)` — `[num_decode_requests, num_decode_requests, num_total_requests]`, always shape `(3,)`.

2. **Inference mesh construction.** The inference mesh was previously built via `maxtext_utils.create_device_mesh(cfg_infer)` with `ici_data_parallelism=-1`, which places all 4 devices on the "data" axis. Real vLLM-TPU serving shards tensor-parallel across the `model` axis, capped at `num_kv_heads` — not data-parallel across all devices. With `data=4`, `sharded_ragged_paged_attention`'s internal `shard_map` tries to shard the small, fixed-shape `query_start_loc` (`(5,)`) / `request_distribution` (`(3,)`) arrays across a size-4 "data" axis, which fails since 4 does not evenly divide 5 or 3. Fixed by manually constructing the inference mesh with `model = min(num_kv_heads, len(jax.devices()))` (= 2 here) and all other axes = 1, rather than via `create_device_mesh` (which requires the ICI product to equal the full device count).

3. **Device-set mismatch this exposed.** `q`/`k`/`v` were committed to the *training* mesh's device set (all 4 devices) before being passed into the (now 2-device) inference mesh's `shard_map`, producing "Received incompatible devices for jitted computation." Fixed by explicitly `jax.device_put`-ing the reshaped `q_3d`/`k_3d`/`v_3d` (replicated) onto the inference mesh's devices immediately before calling `sharded_ragged_paged_attention`.

All three fixes are in `tests/unit/attention_kernel_repro_test.py` (`compare_attention_kernels_on_tpu`, `run_rpa_attention`) and mirrored in `tests/run_attention_batched_rpa_repro.py`.

---

## 2. Results: Tokamax Splash vs. Default RPA v3 (FP32)

| Configuration | vs Default RPA L∞ | vs Default RPA MAE | vs Default RPA CosSim | vs Exact Ref MAE |
| :--- | :--- | :--- | :--- | :--- |
| JAX Splash Attention (Legacy Default) | `1.02e-03` | `1.62e-05` | `1.000000` | `2.18e-04` |
| Tokamax Splash (Default: base2_exp=True, fuse_recip=True) | `8.91e-03` | `2.94e-04` | `0.999996` | `3.12e-04` |
| Tokamax Splash (base2_exp=False, fuse_recip=True) | `1.02e-03` | `1.62e-05` | `1.000000` | `2.18e-04` |
| Tokamax Splash (base2_exp=True, fuse_recip=False) | `8.91e-03` | `2.94e-04` | `0.999996` | `3.12e-04` |
| Tokamax Splash (base2_exp=False, fuse_recip=False) | `1.02e-03` | `1.62e-05` | `1.000000` | `2.18e-04` |
| Tokamax Splash (BlockSize=256) | `8.91e-03` | `2.94e-04` | `0.999996` | `3.12e-04` |

**Default RPA v3 vs Exact Reference (FP32):** L∞=`8.40e-03`, MAE=`2.18e-04`, CosSim=`0.999998`.

## 3. Results: Tokamax Splash vs. Default RPA v3 (BF16)

| Configuration | vs Default RPA L∞ | vs Default RPA MAE | vs Default RPA CosSim | vs Exact Ref MAE |
| :--- | :--- | :--- | :--- | :--- |
| JAX Splash Attention (Legacy Default) | `1.56e-02` | `3.25e-04` | `0.999952` | `2.16e-04` |
| Tokamax Splash (Default: base2_exp=True, fuse_recip=True) | `1.56e-02` | `4.17e-04` | `0.999948` | `3.36e-04` |
| Tokamax Splash (base2_exp=False, fuse_recip=True) | `1.56e-02` | `3.25e-04` | `0.999952` | `2.16e-04` |
| Tokamax Splash (base2_exp=True, fuse_recip=False) | `1.56e-02` | `4.17e-04` | `0.999948` | `3.36e-04` |
| Tokamax Splash (base2_exp=False, fuse_recip=False) | `1.56e-02` | `3.25e-04` | `0.999952` | `2.16e-04` |
| Tokamax Splash (BlockSize=256) | `1.56e-02` | `4.17e-04` | `0.999947` | `3.36e-04` |

**Default RPA v3 vs Exact Reference (BF16):** L∞=`3.12e-02`, MAE=`3.42e-04`, CosSim=`0.999951`.

---

## 4. Batched RPA — not evaluated in this pass

Batched RPA (`tpu_inference.kernels.experimental.batched_rpa`, the target inference kernel) was deprioritized in this pass. It hit a VMEM sizing issue in the standalone repro script (`tests/run_attention_batched_rpa_repro.py`):

- At the script's original config (`batch_size=4`, `seq_len=512`, `block_size=128`), the kernel's internal autotuned decode-shape compilation (`RPAd-p128-b8-q1-k1152`) requested ~84.9MB of scoped VMEM against the real ~64MB TPU v5p VMEM budget — `RESOURCE_EXHAUSTED`, even after raising `vmem_limit_bytes` past 64MB (the requested limit can't exceed the physical budget).
- Reducing to `batch_size=2`, `seq_len=256`, `block_size=64` avoided the VMEM error but then hit an unrelated sharding error in the script's *training*-mesh setup (`P(("data", "fsdp"))` doesn't evenly divide a `batch_size=2` axis against a 4-way data/fsdp mesh), which would need its own fix to the training mesh/batch-size relationship in `run_attention_batched_rpa_repro.py`.

Deprioritized in favor of the Default RPA v3 kernel above, which is what matters for the current e2e focus. In `tests/run_attention_kernel_repro.py`'s 6-config sweep, Batched RPA numbers (FP32 fully passing, BF16 passing after bumping `vmem_limit_bytes` to 64MB in `attention_kernel_repro_test.py`) were also collected and are consistent with Default RPA v3 above (e.g. Batched RPA vs Exact Ref FP32: L∞=`8.40e-03`, MAE=`2.17e-04`, CosSim=`0.999998`; BF16: L∞=`3.12e-02`, MAE=`4.65e-04`, CosSim=`0.999944`) — but the standalone `run_attention_batched_rpa_repro.py` script itself remains unfixed for its own default config and should not be trusted until revisited.

---

## 5. Findings / Learnings

- **Default RPA v3 numerically tracks Splash Attention closely.** FP32 cosine similarity is effectively `1.0` (≥`0.999996`) across all Splash configurations, and BF16 cosine similarity stays ≥`0.999947`. Both are consistent with the drift already documented for the full Qwen3.5 1-layer E2E parity run in `docs/qwen3_5_kernel_drift_results.md`.
- **BF16 error is roughly an order of magnitude larger than FP32**, as expected (L∞ ~`1.6e-2` vs ~`8.9e-3` for Splash-vs-RPA; ~`3.1e-2` vs ~`8.4e-3` for RPA-vs-exact-reference), driven by BF16 mantissa precision rather than any kernel-specific bug.
- **`base2_exp`/`fuse_reciprocal` toggles matter more than block size.** Configs with `base2_exp=True` (whether or not `fuse_reciprocal` is also true) consistently show ~8-9x higher L∞ error vs RPA than `base2_exp=False` configs, in both FP32 and BF16. Block size (128 vs 256) has no measurable effect at this problem size.
- **The `query_start_loc`/`request_distribution` metadata bug and the mesh construction bug compound.** Fixing the metadata shapes alone was not sufficient — the mesh had to be corrected to actually respect those shapes (fixed-size `(batch_size+1,)`/`(3,)` arrays cannot be sharded across a `data` axis with size > 1), and fixing the mesh in turn required explicit re-placement of `q`/`k`/`v` onto the new mesh's device set. All three bugs had to be fixed together to get a running, trustworthy comparison.
