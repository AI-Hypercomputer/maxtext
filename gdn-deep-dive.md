# Gated Delta Net: what it computes, and where the time goes

A stage-level breakdown of GatedDeltaNet as MaxText implements it, measured
rather than modelled, plus the time and MFU split against the other components
of the same layer stack.

The motivation is a specific failure. On `qwen3.5-35b-a3b` the step runs at 8.4%
MFU and the whole delta-rule path collapses into one opaque `shard_map`
custom-call, so an xplane profile reports a single 5.69 s block and cannot say
which stage inside it is expensive. This benchmark runs each stage as its own
jitted function at the per-device shapes the real run uses, so every stage gets
its own time and its own arithmetic intensity.

## Measurement setup and what it does not cover

| | |
|---|---|
| script | `scripts/gdn_component_bench.py` |
| device | 1x **TPU v4**, bf16 peak 275 TF/s |
| geometry | `qwen3.5-35b-a3b`: emb 2048, 40 layers, cycle 4 (30 GDN + 10 full attention), 256 experts top-8, moe hidden 512 |
| GDN | 16 key heads, 32 value heads, `d_k = d_v = 128`, conv kernel 4, chunk 64 |
| batch | per-device batch 1, the shape the real runs use |
| estimator | median of 20 jitted calls after 3 warmups |
| FLOPs | conventional `2*M*N*K`; causal attention counted at half the dense score matrix |

**Caveats, stated up front.** This is v4, not Ironwood, so absolute TF/s does not
transfer; the ratios are the point. It is forward only, single device, so there
is no backward pass and no collective. It exercises the **pure-JAX reference
path** (`jax_chunk_gated_delta_rule` in `models/qwen3.py`), not the fused
custom-call production uses, which is exactly what makes the stages visible.

## What GDN actually computes

Four stages per layer. Only the first and last are ordinary dense algebra.

**A. Input projections.** `in_proj_qkvz` maps `emb -> 2*key_dim + 2*value_dim`
(2048 -> 12288 here, carrying q, k, v and the output gate z), and `in_proj_ba`
maps `emb -> 2*num_value_heads` for the two scalar gates.

**B. Depthwise causal conv1d** of width 4 over the concatenated q, k, v stream
(`conv_dim = 2*key_dim + value_dim` = 8192), followed by silu. Depthwise, so one
multiply-add per element per tap and effectively no arithmetic intensity.

**C. The chunked gated delta rule.** This is the whole game. Sequence is cut into
chunks of `C`, and within each chunk:

    beta  = sigmoid(b)
    g     = -exp(A_log) * softplus(a + dt_bias)      per-head forget gate
    S     = (k * beta) @ k^T  * exp(g_i - g_j), strictly lower triangular
    A     = (I + S)^-1                                unit lower triangular solve
    u     = A @ (v * beta)                            WY factor
    w     = A @ (k * beta * exp(g_cumsum))            WY factor

then a **sequential scan over chunks** carrying the recurrent state
`h` of shape `[B, H, d_k, d_v]`:

    v_new = u - w @ h
    o     = (q * exp(g)) @ h  +  tril(q @ k^T * exp(g_i - g_j)) @ v_new
    h     = h * exp(g_last)  +  (k * exp(g_last - g))^T @ v_new

`A = (I+S)^-1` is the WY representation of a product of `C` rank-1 delta
updates. It is what lets the chunk be applied as matmuls instead of `C`
sequential rank-1 corrections. Every matmul in this stage runs at
`jax.lax.Precision.HIGHEST`, and `S`, `A` and `h` are float32 by construction.

**D. Output.** Gated RMSNorm against silu(z), then `out_proj` back to `emb`.

## Stage breakdown, seq 8192, chunk 64

| component | stage | ms | GFLOP | TF/s | MFU |
|---|---|---:|---:|---:|---:|
| GDN | **in_proj_qkvz** | 1.709 | 412.32 | 241.3 | **87.7%** |
| GDN | in_proj_ba | 0.149 | 2.15 | 14.4 | 5.2% |
| GDN | conv1d depthwise + silu | 0.836 | 0.54 | 0.6 | 0.2% |
| GDN | gates (sigmoid, softplus, exp) | 0.197 | 0 | 0 | 0% |
| GDN | `S = k_beta @ k^T` | 0.555 | 4.29 | 7.7 | 2.8% |
| GDN | cumsum + `exp(g_diff)` mask | 0.218 | 0 | 0 | 0% |
| GDN | **`solve_triangular` (A)** | **5.826** | 0.36 | 0.1 | **0.02%** |
| GDN | WY `u = A @ v_beta` | 0.648 | 4.29 | 6.6 | 2.4% |
| GDN | WY `w = A @ k_beta_g` | 0.646 | 4.29 | 6.6 | 2.4% |
| GDN | **inter-chunk scan** (5 matmuls) | **4.219** | 34.36 | 8.1 | 3.0% |
| GDN | **out_proj** | 0.671 | 137.44 | 204.9 | **74.5%** |
| Attention | scores + softmax + AV, causal | 7.713 | 549.76 | 71.3 | 25.9% |
| MoE | expert GEMMs, 256 rows/expert | 2.340 | 412.32 | 176.2 | 64.1% |
| MoE | router + top-k | 0.773 | 8.59 | 11.1 | 4.0% |

### The one-sentence version

**The two dense projections are 16% of GDN's time and carry 92% of its FLOPs.
The delta-rule core is 84% of the time and carries 8% of the FLOPs.**

`in_proj_qkvz` and `out_proj` run at 88% and 75% of peak. They are not the
problem and there is nothing to win there. Everything between them runs at
0.02% to 3% of peak.

`solve_triangular` alone is **37% of the GDN layer at 0.02% MFU**. It is a
sequential unit-lower-triangular inverse of a 64x64 matrix, done `B * NC * H`
times, which at this shape is 8192 independent tiny solves. It is the single
worst op in the model by a wide margin.

The inter-chunk scan is second at 27%. It is five HIGHEST-precision matmuls per
chunk with a carried dependency, so `NC` iterations cannot overlap.

## Component split across the stack

30 GDN layers, 10 full-attention layers, MoE in all 40. Forward only.

| component | ms | share of time | GFLOP | share of FLOPs |
|---|---:|---:|---:|---:|
| **GDN** | 470.2 | **70.0%** | 18001 | 44.6% |
| Attention | 77.1 | 11.5% | 5498 | 13.6% |
| MoE | 124.5 | 18.5% | 16836 | 41.7% |
| total | 671.8 | | 40335 | |

Forward **60.0 TF/s = 21.8% MFU**.

GDN takes 70% of the time for 45% of the work. MoE takes 18.5% of the time for
42% of the work. The MoE is not what is wrong with this model, which is
consistent with the earlier finding that MoE and attention kernel levers moved
nothing and Tokamax actually regressed.

### The size of the prize

If the delta-rule core (stages B and C) were free and only the projections
remained, the GDN layer would drop from 15.674 ms to 2.529 ms and the stack
would go:

| | ms | TF/s | MFU |
|---|---:|---:|---:|
| as measured | 671.8 | 60.0 | 21.8% |
| delta-rule core free (bound) | 277.5 | 145.3 | **52.8%** |

So a perfect fused delta-rule kernel is worth roughly **2.4x** on the forward
pass of this geometry. That is the ceiling any kernel work is chasing, and it
explains why the fused KDA kernel mattered so much on OLMo 3.5: the measured
1.26x from `TOKAMAX_KDA_DENSE_PAIRS` and the 303 ms
`_fused_dhu_wy_intra_cumsum_pallas_` that xla-shell named as 27% of step are the
same WY and cumsum stages seen here, collapsed into one Pallas kernel.

## chunk_size is a real but weak and sequence-dependent lever

`gdn_chunk_size` is the one knob exposed in config. It does not remove work, it
moves it between two low-MFU sequential ops:

- `solve_triangular` total cost is `NC * C^3/3 = T * C^2/3`, so it grows as **C squared**.
- the inter-chunk scan runs `NC = T/C` sequential iterations, so it shrinks as **1/C**.

Measured, GDN layer time in ms:

| chunk | seq 4096 solve | seq 4096 scan | **seq 4096 GDN total** | seq 8192 solve | seq 8192 scan | **seq 8192 GDN total** |
|---|---:|---:|---:|---:|---:|---:|
| 16 | 0.884 | 6.840 | 11.078 | | | |
| **32** | 1.436 | 3.582 | **8.264** | 2.882 | 6.956 | **15.393** |
| 64 (shipped) | 5.471 | 2.292 | 11.160 | 5.826 | 4.219 | 15.674 |
| 128 | 12.448 | 1.530 | 17.450 | | | |
| 256 | 12.952 | 1.373 | 18.124 | | | |

At **seq 4096, chunk 32 is 1.35x faster than the shipped 64** and takes the full
forward from 40.4 to 49.5 TF/s, **+22%**. At **seq 8192 the same change is worth
only 1.02x**, because doubling the sequence doubles `NC` and the scan term grows
back to cancel the solve saving.

Two conclusions. The optimum is **sequence-length dependent**, so a single
default is wrong for a model trained at more than one context length. And the
lever is fundamentally weak: it only trades one near-zero-MFU sequential op for
another, and the best achievable GDN layer time across the whole sweep (8.264 ms
at seq 4096) is still 3.3x the 2.529 ms the projections alone would take.

Chunking is a blocking strategy, not an approximation, so the result is
mathematically the same at any `C`. Only float32 accumulation order changes.

## Why this shape is hostile to a TPU

1. **The MXU never fills.** The delta-rule matmuls are `[C, d] x [d, C]` with
   `C = 64` and `d = 128`, one per head per chunk. A 256x256 MXU tile is never
   more than a quarter occupied on the M dimension at `C = 64`, and batching is
   across `B * NC * H` independent tiny problems rather than along a single big
   contraction.
2. **`solve_triangular` is not a matmul at all.** It is a sequential forward
   substitution over `C` rows. There is no MXU work in it.
3. **Everything is HIGHEST precision.** Every matmul in the core is explicitly
   `Precision.HIGHEST` and `S`, `A`, `h` are float32. On TPU, HIGHEST bf16 is a
   three-pass or six-pass decomposition, so these ops pay 3x to 6x the passes of
   a bf16 matmul for operands that are already tiny.
4. **The scan carries a dependency.** `h` of shape `[B, H, d_k, d_v]` flows
   chunk to chunk, so `NC` iterations serialise and there is nothing to overlap
   them with inside the layer.
5. **State size scales as `1/num_heads`, the opposite of softmax attention.**
   Raising `head_dim` grows the per-head state `d_k x d_v` but shrinks the head
   count, so total state is flat. This is why the head_dim lever that buys
   21-46% on softmax-attention models does nothing here, and it is the answer we
   gave on the Ling 3 MAX question.

## What to do about it, in priority order

1. **Fuse the core.** The bound above says 2.4x on the forward. Everything else
   is a rounding error next to it. This is what the tokamax KDA kernel does for
   OLMo 3.5, and the qwen3.5 path has no equivalent fused training kernel, which
   is precisely why it sits at 8.4%.
2. **Drop HIGHEST precision** in the core where the gradient tolerates it. On
   OLMo 3.5 the bf16 KDA flags measured 1.06x, but only after the dense-pairs
   change removed the bottleneck they were hiding behind, so this is
   order-dependent and must be measured after the fusion, not before.
3. **Tune `gdn_chunk_size` per context length**, expecting 1.35x at 4k and
   almost nothing at 8k. Cheap, safe, and its value has now been quantified
   rather than assumed.
4. **Do not spend effort on MoE or attention kernels for this model.** They are
   18.5% and 11.5% of the time and already run at 64% and 26% of peak.

## Reproducing

    python3 scripts/gdn_component_bench.py --seq 8192 --chunk 64 --peak-tflops 275

`--seq`, `--batch`, `--chunk`, `--dtype` and `--peak-tflops` are all settable.
Geometry lives in the `GEOM` dict at the top of the script.

Related: `olmo35-ironwood-runs.md` for the OLMo 3.5 / KDA measurements,
`olmo35-ironwood-plan.md` for the Ironwood programme.
