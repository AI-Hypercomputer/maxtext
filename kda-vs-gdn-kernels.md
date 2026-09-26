# KDA vs GDN kernels on Ironwood: roofline and measured

Measured 2026-09-26 on one tpu7x device (flex 2x2x2 node, jax 0.11.1, image
`agagik-olmoe3:kdaj24`). Shape is the OLMo 3.5 tiny per-device KDA layer at
pdb 3: B 3, T 8192, H 8, dk 128, dv 256, chunk 64. Script:
`scripts/kda_vs_gdn_bench.py`. Times are the median of 10 calls, one layer.

KDA and GDN run the same chunked delta rule. The only difference is the decay.
GDN has one scalar per head and token. KDA has one per head, token and key
channel. For GDN the intra-chunk decay is a C x C factor applied after the
K K^T matmul. For KDA it sits inside the contraction
(sum_d k_id k_jd exp(G_id - G_jd)). It only becomes a matmul if the decay is
folded into the operands, which overflows fp32 unless the chunk is split into
sub-blocks.

## Result

| implementation | fwd ms | fwd+bwd ms | % of roofline | vs best |
|---|---|---|---|---|
| **KDA, tokamax Pallas kernel (shipped: bf16, dense pairs)** | **2.71** | **7.50** | **6.4%** | **1.00x** |
| KDA, tokamax kernel, fp32 kernel math | 2.90 | 8.37 | 5.7% | 1.12x |
| KDA, tokamax kernel, fp32 q/k/v in (pre-l3 model) | 5.36 | 13.71 | 3.5% | 1.83x |
| GDN, tokamax KDA kernel, gate broadcast over dk | 2.65 | 7.42 | 5.9% | 0.99x |
| GDN, MaxText `jax_chunk_gated_delta_rule` (what qwen3.5 trains with) | 4.62 | 16.00 | 2.7% | 2.13x |
| GDN, MaxText `naive_jax_chunk_gated_delta_rule` | 10.40 | 38.97 | 1.1% | 5.20x |
| KDA, olmoe3 `_delta_rule_chunked`, fp32 state | 17.05 | 47.11 | 1.0% | 6.28x |
| KDA, olmoe3 `_delta_rule_chunked`, bf16 state | 11.58 | 41.88 | 1.1% | 5.58x |

"% of roofline" is the fwd+bwd roofline time divided by the measured time.
Outputs agree within bf16 noise: tokamax GDN vs MaxText GDN max abs 9.8e-4, and
tokamax KDA vs olmoe3 jnp KDA max abs 1.1e-3.

**No implementation reaches 10% of roofline.** The tokamax kernel is the best
for both KDA and GDN. **GDN has no fused kernel in MaxText**: its pure-JAX path
is 2.1x slower than the KDA kernel even though GDN is the simpler math. Running
GDN through the KDA kernel with the gate broadcast over dk gives the same answer
in 7.42 ms. That is 2.2x faster than what qwen3.5 uses today. It matters there
because qwen3.5-35b is GDN-bound at 8.4% MFU.

## Roofline

Chunked delta rule in WY form, per device, per layer. MXU work per chunk:
K K^T and Q K^T (2 C^2 dk each), the WY solve applied to k and v
(2 C^2 dk + 2 C^2 dv), the intra-chunk output (2 C^2 dv), and three
C x dk x dv state products. The bwd is taken as 2x the fwd. Minimum bytes is
one bf16 read of q, k, v, gate and beta, one write of o, plus one fp32 dk x dv
state per chunk for the bwd.

| | KDA | GDN |
|---|---|---|
| MXU FLOP, fwd / fwd+bwd | 62.0 / 186.0 G | 62.0 / 186.0 G |
| min HBM bytes, fwd+bwd | 1.76 GB | 1.61 GB |
| arithmetic intensity | 106 FLOP/B | 115 FLOP/B |
| compute floor (1153.5 TF/s) | 0.161 ms | 0.161 ms |
| **HBM floor (3.69 TB/s)** | **0.478 ms** | **0.437 ms** |

Both are HBM-bound at the roofline: the intensity is about 110 FLOP/B against a
ridge of 313. The matmul work is identical. KDA's only roofline penalty is the
dk-wide gate, which is 9% more bytes. **KDA vs GDN at the roofline is 1.09x, but
measured it is 0.47x to 1.0x depending on which kernel exists.** The
implementation decides the speed, not the algorithm.

## Where the tokamax KDA time goes

xplane profile, fwd+bwd, shipped settings (7.06 ms in the profile).

| kernel | ms | share |
|---|---|---|
| bwd `_fused_dhu_wy_intra_cumsum_pallas` | 2.59 | 37% |
| **XLA glue outside the kernels** | **1.70** | **24%** |
| of which l2norm bwd for dq, dk (2 fusions) | 1.19 | 17% |
| of which l2norm fwd reduce, transposes | 0.44 | 6% |
| fwd `pallas_kda_fwd_intra_fused` | 1.19 | 17% |
| fwd `chunk_kda_fwd_h_o_varlen` | 0.62 | 9% |
| converts | 0.31 | 4% |
| bwd `fused_recompute_w_u_vnew_from_h_pallas` | 0.25 | 4% |
| bwd `chunk_kda_bwd_dAv_kernel` | 0.21 | 3% |

The l2norm bwd is two elementwise fusions over [8, 3, 8192, 128]. That should
cost about 30 us of HBM traffic. It takes 1.19 ms, which points to a layout
problem (the rstd operand carries a {0,2,1} tiled layout), not to real work.

### Kernel knobs

| change (tokamax KDA, bf16) | fwd+bwd ms | vs shipped |
|---|---|---|
| shipped: chunk 64, sub-block BC 4 | 7.50 | |
| chunk 128 | 7.76 | +3% |
| BC 8 | 6.56 | **−13%** |
| BC 16 (upstream value) | 6.18 | **−18%** |
| BC 32 | 6.02 | −20% |
| BC 64 (one block, GDN-only proxy) | 5.83 | −22% |
| fp32 q/k/v inputs | 13.71 | +83% |
| batch 1 / 3 / 6 | 2.57 / 7.50 / 15.21 | linear in B |

The kda8 patch cut BC from 16 to 4 to stop an fp32 exp2 overflow (NaN at
reference init). On our path the gate is activated outside the kernel (decay
floor > 0), so the fwd centers the exponent mid-block, but the bwd
`compute_intra_backward` does not, and its exponent spans the whole sub-block.
The binding constraint is the bwd: **BC x floor x log2(e) < 127**, where floor
is the per-token log-decay floor (`tokamax_kda_log_decay_floor`, 20 nats today).
The microbench stays finite at any BC only because its random gates are mild
(about 0.13 nats per token).

| BC | exponent at floor 20 | exponent at floor 11 | floor needed | in-model (o-series) |
|---|---|---|---|---|
| 4 | 115 | | | 121.9 TF/s control |
| 8 | 231 | 127 | ≤ 10 nats | NaN at step 1, floor 20 |
| 16 | 462 | 253 | ≤ 5 nats | NaN at step 1, floor 11 |

A first version of this table used BC/2 (the fwd centering) and called BC 8
safe at floor 20; the o-series disproved it. Lowering the floor from 20 to 11
leaves loss unchanged (o4: 10.835 vs 10.836 at step 19), since e^-11 per token
already wipes the state. So **BC 8 needs floor 10** and BC 16 needs floor 5,
which starts to clip real decays and needs a loss check. GDN has no per-channel
exponent, so a GDN-specialized kernel needs no sub-blocks at all. BC 64 on the
broadcast gate is its proxy: 5.76 ms, 2.8x faster than MaxText GDN today.

With BC 64 there is still 5.8 ms against a 0.48 ms roofline. The rest is the
kernels' own structure (a sequential loop over 128 chunks of 64-row matmuls)
plus the glue above. Closing that is kernel work, not a flag.

## Why the pure-JAX paths are slow

MaxText GDN (15.1 ms profiled) is launch-bound in `lax.scan`. The bwd scan body
runs 1024 small fp32 64x64 dot fusions at HIGHEST precision (4.28 ms). About
2000 small loop fusions add up to 7.1 ms. The triangular_solve custom call takes
1.58 ms, and the dynamic-update-slice that stacks scan outputs takes 2.57 ms.
The olmoe3 jnp KDA is worse (40 ms): the exact per-pair decay is a three-operand
einsum over [C, C, dk], which runs on the VPU and not the MXU (31 ms of loop
fusions).

## What it means for OLMo 3.5 tiny

In-model, the f1 profile (pdb 3, before the conv-cast fix) puts KDA at 224 ms
of a 1098 ms step (20%), or 16 ms per layer. That matches the fp32-input
microbench (13.7 ms) plus glue. Going from fp32 to bf16 inputs predicts
6.2 ms x 14 layers = 87 ms saved per step. The l3 conv-cast arm measured
100 ms (0.968 to 0.868 s). **The microbench explains the l3 win.**

At bf16 inputs, KDA is now about 14 x 7.5 = 105 ms, or roughly 12% of the
0.87 s step. That is an estimate; there is no post-l3 in-model profile yet. The
MoE grouped matmuls are larger (about 180 ms in the f1 profile).

| KDA lever | per-layer saving | step saving (est.) | risk |
|---|---|---|---|
| BC 8 + floor 10 | 0.94 ms | ~13 ms, ~1.5% | floor 11 measured loss-neutral |
| BC 16 + floor 5 | 1.32 ms | ~18 ms, ~2% | floor 5 clips decays, needs a loss check |
| fix the l2norm bwd layout or fuse it in-kernel | up to 1.1 ms | ~16 ms, ~2% | kernel change |

For qwen3.5 the order flips. GDN is the bottleneck there, and routing it
through the tokamax kernel is a 2.2x layer win available today, or 2.8x with a
GDN-specialized sub-block.

## Reproduce

The flex 2x2x2 nodes are two-host slices. One pod on one node runs as a single
host with `TPU_SKIP_MDS_QUERY=true TPU_HOST_BOUNDS=1,1,1
TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 TPU_WORKER_ID=0 TPU_WORKER_HOSTNAMES=localhost`
(8 devices). Fetch `gs://agagik-us/olmo35/src.tgz`, set `PYTHONPATH=/wt/src`,
then:

```
python3 scripts/kda_vs_gdn_bench.py --tag fp32
TOKAMAX_KDA_BF16_FWD=1 TOKAMAX_KDA_BF16_BWD=1 TOKAMAX_KDA_DENSE_PAIRS=1 \
  python3 scripts/kda_vs_gdn_bench.py --tag bf16 --impls kda_tokamax,gdn_tokamax
```

The BC sweep edits `BC = 4` in the installed `pallas_mosaic_tpu_fwd_kernel.py`
(line 643) and `BC = min(4, BT)` in `pallas_mosaic_tpu_bwd_kernel.py`
(line 894).
