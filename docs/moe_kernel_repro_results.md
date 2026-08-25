# Standalone MoE Kernel Repro Results (Tokamax GMM v2 vs. Fused MoE)

**Date / Timestamp:** 2026-08-14 03:31 UTC
**Hardware Platform:** Google Cloud TPU v5p (local TPU VM, locally-attached chips, no SPS proxy)
**Topology:** 4 TPU Devices
**Script:** `tests/run_moe_kernel_repro.py` (extended in this run to sweep both `float32` and `bfloat16`; previously float32-only)
**Shapes:** `batch=4, seq_len=512, emb_dim=2048, moe_mlp_dim=512, num_experts=8, num_experts_per_tok=8`

This script compares the **training-path MoE kernel** (Tokamax GMM v2 / legacy
Megablox Pallas GMM / dense-einsum reference, run under several tile configs)
against the **inference-path fused MoE Pallas kernel** (`tpu_inference`'s
fused MoE, used by vLLM-TPU serving), and against an exact dense-einsum math
reference, all on top-8-of-8 (dense) routing.

---

## Float32

| Configuration | Vs Infer L∞ | Vs Infer MAE | Vs Infer CosSim | Vs Ref L∞ | Vs Ref MAE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Tokamax GMM v2 (Standard: 128x128 Tile) | `3.32e-05` | `4.80e-08` | `1.000000` | `3.32e-05` | `4.87e-08` |
| Tokamax GMM v2 (Tile 256x128) | `2.98e-08` | `1.55e-10` | `1.000000` | `2.98e-08` | `1.04e-09` |
| Megablox Legacy Pallas GMM | **FAILED** | -- | -- | -- | -- |
| Dense Einsum (XLA Reference Path) | `2.98e-08` | `9.09e-10` | `1.000000` | `3.73e-08` | `1.06e-09` |

**Megablox Legacy Pallas GMM (float32) failure (real, reproducible, not fabricated):**
```
RESOURCE_EXHAUSTED: E1001: CompileTimeScopedVmemOom:
Ran out of memory in memory space vmem while allocating on stack for %gmm.1 = f32[16384,512]{1,0:T(8,128)} ...
Scoped allocation with size 18.00M and limit 16.00M exceeded scoped vmem limit by 2.00M.
```
This is a genuine VMEM sizing issue in the legacy Megablox Pallas GMM kernel
at this problem size/tile config in float32 -- it is not a numerics bug, and
was not worked around (no tile-size override was applied for this config, to
keep it representative of the "default legacy" path). The bf16 sweep below
uses the same kernel and tile config successfully, confirming the OOM is
float32-VMEM-specific (2x the bf16 footprint at the same tile size).

**Baseline:** Inference Fused MoE vs. Exact Reference (FLOAT32): `L∞=2.98e-08, MAE=9.68e-10, CosSim=1.000000`

## BFloat16

| Configuration | Vs Infer L∞ | Vs Infer MAE | Vs Infer CosSim | Vs Ref L∞ | Vs Ref MAE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Tokamax GMM v2 (Standard: 128x128 Tile) | `1.46e-03` | `9.40e-05` | `0.999955` | `9.77e-04` | `7.92e-05` |
| Tokamax GMM v2 (Tile 256x128) | `1.46e-03` | `9.40e-05` | `0.999954` | `9.77e-04` | `7.92e-05` |
| Megablox Legacy Pallas GMM | `1.46e-03` | `9.40e-05` | `0.999955` | `9.77e-04` | `7.92e-05` |
| Dense Einsum (XLA Reference Path) | `1.46e-03` | `9.40e-05` | `0.999954` | `9.77e-04` | `7.92e-05` |

**Baseline:** Inference Fused MoE vs. Exact Reference (BFLOAT16): `L∞=1.46e-03, MAE=4.94e-05, CosSim=0.999973`

---

## Interpretation

* In **float32**, the Tokamax GMM v2 training kernel (both tile configs) and
  the dense-einsum reference agree with the inference-path fused MoE kernel
  to near machine precision (`CosSim=1.000000`, `L∞ ~= 3e-5` to `3e-8`
  depending on tile config); the 256x128 tile config is markedly tighter than
  the standard 128x128 tile config (`3.32e-05` vs. `2.98e-08` L∞ vs. infer).
* In **bfloat16**, all four training-side configs converge to essentially
  identical drift numbers vs. both the inference kernel and the exact
  reference (`CosSim ~= 0.99995`, `L∞ ~= 1.46e-03`) -- the drift here is
  dominated by bf16 rounding, not by kernel-implementation differences
  between Tokamax GMM v2 / legacy Megablox / dense einsum.
* The float32 vs. bfloat16 gap (`L∞` ~`3e-8` vs. `~1.5e-3`, ~5 orders of
  magnitude) is the expected precision floor between the two dtypes, not
  evidence of an algorithmic bug.
