# OLMo 3.5 tiny on Ironwood: the path to 20% MFU

Goal: 20% MFU for `olmo35-tiny` on tpu7x 4x4x4 (128 devices) with **no quality
change**. Every lever here either computes the same function (bit-identical, or
the same math with a different summation order) or is a kernel replacement that
returns identical outputs. Nothing changes the model, the data, the routing
decision or the precision of a matmul.

Method: an analytic roofline per component (`scripts/olmo35_roofline.py`), a
measured per-component profile of the current base (xla_shell over the dd1
xplane), perfsim at the same operating point (`scripts/olmo35_perfsim.py`), and
measured before/after runs with controls. Each component gets an explicit kernel
verdict: **effective** (at or near what a good TPU kernel achieves) or **not
effective** (with the gap and what a good kernel would reach). Where no good
kernel exists yet, the projection says so and states the assumed efficiency.

## Target

| quantity | value |
|---|---|
| model FLOPs per device per step (MaxText count) | 106.1 TFLOP |
| peak bf16 per device | 1153.5 TF/s |
| HBM per device | 3.69 TB/s |
| original config, `a2_p2s8k` (pdb 2, stock flags) | 78.2 TF/s, **6.78%** |
| current base (dd1/dd2, pdb 3) | 144.3 TF/s, **12.51%**, step **0.735 s** |
| 20% MFU | 230.7 TF/s, step **0.460 s** |
| gap | **275 ms** per step (37% of the step) |

Geometry per device: 24576 tokens (pdb 3 x seq 8192), d=1024, 16 layers (14 KDA,
2 attention at 7 and 15), layer 0 dense (mlp 8192), 15 MoE layers with 512 experts
top-16, latent 512, expert hidden 1024, vocab 100352. Routed rows per device:
393216, about 768 per expert.

## Three views of the same step

| view | step s | MFU | note |
|---|---|---|---|
| measured, current base | 0.735 | 12.5% | median of steps 10-19 |
| roofline, current kernels unfused | 0.156 | 59% | per-component max(MXU, HBM) |
| roofline, fused routed-expert kernel | 0.115 | 80% | SwiGLU in the gmm epilogue |
| perfsim, same operating point | 0.338 | 34.8% | remat=full, counts 135 TF/device |

perfsim is **2.17x optimistic** here (0.735 / 0.338). Its FLOP count includes the
full-remat recompute, so its MFU is not directly comparable; the step ratio is the
usable number. The roofline says the model itself is not the limit: the step is
4.7x its roofline, and the whole gap to 20% is kernel and scheduling efficiency.

## Roofline by component

From `scripts/olmo35_roofline.py`, per device, fwd+bwd.

| component | roofline ms | bound |
|---|---|---|
| MoE routed GEMMs, 9 per layer, unfused | 63.8 | HBM (0.473 vs 0.357 MXU per GEMM) |
| MoE routed, fused SwiGLU kernel | 48.3 | MXU |
| SwiGLU, unfused | 26.2 | HBM |
| KDA projections | 14.1 | MXU |
| LM head | 13.1 | MXU |
| MoE router, latent, shared expert | 9.0 | MXU |
| norms | 8.7 | HBM |
| dispatch and combine | 7.0 | HBM |
| KDA core (chunked delta rule) | 6.7 | HBM (186 GF, 1.76 GB per layer) |
| attention (2 layers) | 3.6 | MXU |
| dense MLP (layer 0) | 3.2 | MXU |
| optimizer | 0.7 | HBM |
| top-k and sorts | 0.2 | HBM |
| **total, unfused** | **156** | |
| **total, fused** | **115** | |

## Measured bottlenecks and kernel verdicts

dd1 profile (`o35n280354`, arm dd1_prof, same config as the base), components
attributed by joining xla_shell `ops()` rows to `records()` on stripped op name
and tf_op, dominant source line wins. Profiled device ops total 728 ms of the
735 ms step; remat recompute is 66.6 ms of that and is folded into its component.

| component | measured ms | roofline ms | efficiency | kernel verdict |
|---|---|---|---|---|
| KDA, total | **208** | 6.7 core + share of proj | ~3% | **not effective** |
| . tokamax varlen wrapper glue (olmoe3.py:289) | 88 | | | not effective: pad to 10240, gathers, relayout |
| . conv, gates, o_norm glue | 50 | | | not effective: unfused elementwise |
| . Pallas chunk kernels | 67.6 | 6.7 | 10% | not effective |
| MoE routed GEMMs (gmm_v2 fwd+dlhs 113.4, tgmm 53.3) | **167** | 63.8 | 38% | **not effective**: 999-1195 us/call vs 473 roof |
| collectives, exposed | **58.2** | ~0 hidden | | **not effective** scheduling: expert all-gather 23.7, expert-grad all-reduce 16.4 |
| other loop fusions | 42.4 | | | not effective (elementwise, unattributed) |
| dense matmuls (projections, shared, router) | 40.7 | ~27 | 63-72% | **effective** |
| LM head (GEMM + loss) | 37.8 | 13.1 | 35% | GEMM tiles **effective** (67-90%); loss loop fusions (11 ms) not |
| MoE other (router, latent, shared glue) | 33.8 | 9.0 | 27% | not effective |
| top-k and sorts | **31.7** | 0.2 | <1% | **not effective**: `lax.top_k` 1163 us/call vs 55 us roof (5%), sorts 14.2 |
| splash attention | 26.5 | 3.6 | 14% | **not effective**: fwd 3.9, dq 4.2, dkv 4.9 ms per call |
| SwiGLU | 22.3 | 26.2 | >100% | **effective** (XLA fuses the fwd into the wo gmm lhs) |
| dispatch and combine | 17.2 | 7.0 | 41% | not effective |
| copies | 14.1 | 0 | | not effective (layout) |
| EMo threshold (bisection) | 10.6 | ~0.5 | 5% | not effective |
| norms | 10.0 | 8.7 | 87% | **effective** |

What "good kernel" means in the projections below, from kernels that already
reach it on this chip: matmul-class 75% of roofline (the dense projections are at
63-72%), HBM-streaming 60%, chunked linear-attention 30% (no public kernel
reaches this yet; tokamax KDA is at 6.4% single-device), splash 50%, exposed
comm about 10 ms.

### Why each unfit kernel is slow

**KDA (208 ms, the largest item).** Three separate taxes. The tokamax varlen
wrapper runs even though synthetic data has one segment per row: it pads 8192 to
8192 + 63 x 32 = 10240 rows and adds gather/scatter and relayout (88 ms). The
conv, gate and output-norm math around the kernel is unfused elementwise (50 ms).
The chunk kernels themselves run at 10% of the HBM roofline (67.6 ms). Per layer
that is 14.9 ms against 7.5 ms for the same kernel in the single-device bench
without segments. The ee-series measures the varlen tax directly (below).

**Routed-expert GEMMs (167 ms).** gmm_v2 at tm=1024 with ~768 rows per expert
computes about 896 (tile, group) pairs against 384 ideal, so the MXU does ~2.3x
padded work. Smaller tiles lost to per-grid-step overhead (u and v series). A
group-aligned kernel (256-row sub-blocks that never straddle an expert) would run
at about 0.55 ms/call, 86% of roofline.

**top-k (17.4 ms) and sorts (14.2 ms).** XLA lowers `lax.top_k` over 512 experts
to a full sort. The Pallas kernel in `src/maxtext/kernels/topk.py` does 16 rounds
of max-and-mask in VMEM on order-preserving int keys, experts on sublanes so each
max is an elementwise reduction across vregs. Exact against `lax.top_k` (random,
coarse ties, mostly -inf, signed zeros, bf16). Local v4, [3, 8192, 512], k=16:

| impl | us | vs lax.top_k |
|---|---|---|
| `lax.top_k` | 3381 | 1.0x |
| Pallas, float keys, bt=256 | 1074 | 3.1x |
| Pallas, int keys, bt=256 | 679 | **5.0x** |
| Pallas, int keys, bt=1024 | 670 | 5.0x |

Real v7x (tpu7x-cluster-flex, 2x2x1, one device), same shape, exact in both dtypes:

| input | `lax.top_k` us | Pallas best us (bt) | speedup | HBM roofline us | kernel efficiency |
|---|---|---|---|---|---|
| f32 | 1480 | 391 (512) | **3.8x** | 14.1 | 3.6% |
| bf16 | 356 | 364 (512) | 1.0x | 7.2 | 2.0% |

The in-model `top_k` runs 1163 us/call, which matches the f32 path, so the
expected saving is about 15 x (1163 - 394) = 11.5 ms per step. **Verdict: faster
than XLA, still not an effective kernel.** It is VPU-bound: 16 rounds, each with a
512-deep max and a 512-deep min over every token column, about 1.2M vreg ops per
call. A good kernel would stream the logits once (HBM-bound, ~15-25 us), for
example a per-token threshold found from a coarse histogram followed by one
compaction pass; that would also absorb the EMo threshold (10.6 ms) and the
routing sorts (14.2 ms) into the same pass.

**Fused routed-expert kernel, measured at this shape.** The prototype in
`scripts/latent_moe_fusion_bench.py` (wi_0 and wi_1 GEMMs, SwiGLU and the wo GEMM
in one Pallas kernel, hidden activation kept in VMEM, custom VJP with a dx kernel
and a dW kernel) was run on a real v7x device (tpu7x-cluster-flex 2x2x1, image
kdaj24) at the OLMo 3.5 per-device shape: 393216 rows (768 per expert), 512
experts, latent 512, hidden 1024, bf16, balanced groups.

| variant, one MoE layer | fwd ms | fwd+bwd ms | vs XLA | fwd+bwd roofline ms | efficiency |
|---|---|---|---|---|---|
| XLA `ragged_dot` x3 plus GLU | 6.03 | 17.6 | 1.00x | 3.22 | 18% |
| prefused wi, concat hoisted (best unfused) | 5.05 | 14.4 | 1.22x | 3.22 | 22% |
| fused, tm=256 tn=256 tn_dw=256 | 5.77 | 17.3 | 1.02x | 3.22 | 19% |
| fused, tm=256 tn=512 tn_dw=512 | 3.86 | 11.1 | 1.58x | 3.22 | 29% |
| **fused, tm=256 tn=1024 tn_dw=1024** | **2.55** | **7.54** | **2.32x** | 3.22 | **43%** |

The full-width hidden block (tn = 1024) is what makes it win: every narrower block
re-reads x and re-accumulates the output. At 43% of the MXU roofline this is
**close to an effective kernel**, and it is the measured basis for waterfall steps
3 and 4. In the model the routed GEMMs plus SwiGLU cost 189 ms, 12.6 ms per layer;
15 layers at 7.54 ms is 113 ms, a **~76 ms** saving with balanced groups. With
dropless raggedness the bench's own model costs tm=256 about 14% (1.84x vs 2.14x
at cv 0.05 on the OLMoE3 shape), so ~8.8 ms per layer and ~57 ms saved. Not wired
into MaxText yet: the prototype assumes each expert's rows start on a 256-row
boundary, so integration needs group-aligned padding in the dispatch.

Quality: the result is not bit-identical to the shipping path. It differs by
4.7e-3 relative (fwd) because the shipping path rounds the hidden activation to
bf16 in HBM between kernels, while the fused kernel keeps it in f32 in VMEM, so
it is closer to an f32 reference, not further. It is the same math at higher
intermediate precision.

**Collectives (58 ms exposed).** The expert-weight all-gather
(quantizations.py:120) and the expert-grad all-reduce (moe.py:3068) are not
overlapped with the layer's compute. Overlap does not change any value.

**Splash (26.5 ms).** 2 layers only, 14% of an upper-bound roofline. Block sizes
are untuned for seq 8192 at this head count.

### Profile with the varlen tax removed (gg8, `packing=False`)

Same attribution as dd1, 613 ms of profiled ops (step 0.617 s), remat 64.3 ms.
Only KDA moves; this isolates what a reset-aware kernel buys and what is left.

| component | dd1 ms (packed) | gg8 ms (unpacked) | delta | verdict |
|---|---|---|---|---|
| KDA glue | 140.4 | 50.3 | -90.1 | not effective: conv, gates, l2norm, o_norm unfused |
| KDA Pallas kernels | 67.6 | 51.3 | -16.3 | **not effective**: 3.7 ms/layer vs 0.48 roofline, 13% |
| MoE routed GEMMs (gmm + tgmm) | 166.7 | 166.4 | | not effective, 38% |
| MoE other | 33.8 | 58.9 | +25.1 | attribution shift from the varlen glue, same ops |
| collectives, exposed | 58.2 | 53.1 | | scheduling |
| all others | 261 | 233 | | unchanged kernels |

xla_shell `analyze_profile` and `roadmap --all` on gg8:

| quantity | value |
|---|---|
| TensorCore lane | 540 ms: compute 326, VPU 187, relayout 27 |
| SparseCore lane (collectives) | 325 ms, 257 hidden, 68 exposed |
| best-overlap ceiling (schedule only) | 540 ms |
| roadmap floor, all levers | **325 ms**, binder SparseCore comm |

| roadmap step | lever | step ms | gain |
|---|---|---|---|
| 0 | as profiled | 614 | |
| 1 | schedule the exposed comm | 540 | 74 |
| 2 | kernels, bounded by comm slack | 325 | 216 |
| 3 | relayout reduction | 325 | 0 |
| 4 | host-offload remat | 325 | 0 |

Non-matmul TensorCore work (VPU plus relayout) is 35% of the step. The floor
matters for planning: with this FSDP=32 x DP=4 sharding the step cannot go below
325 ms (32.6% MFU) however good the kernels get; past that, only less comm
volume helps. 20% needs 460 ms, so the target sits inside the kernel budget and
the comm floor does not bind it.

## Quality neutrality, per lever

| lever | math change | why quality is unchanged |
|---|---|---|
| Pallas top-k (`moe_topk_pallas`) | none | same indices, same order; values and grads bit-identical (CPU test) |
| `packing=False` on this benchmark | none | synthetic rows are one segment each; the varlen path computes the same result |
| reset-aware KDA kernel | none | same chunked recurrence with resets at segment starts instead of padding |
| group-aligned gmm | none | same GEMMs, tiles aligned to expert groups |
| fused SwiGLU epilogue | none | same ops inside the kernel |
| collective overlap | none | scheduling only |
| splash block sizes | summation order | fp32 accumulate, reorder only |
| EMo and top-k by bisection (bb, cc) | summation order | same expert set; combine sum order changes (grads within 1e-3 of leaf max) |

Levers deliberately **excluded** because they change the function: fewer experts
or smaller top-k, fp8 or int8 matmuls, bf16 KDA state, larger EMo pools, capacity
factors that drop tokens, `use_random_routing`.

## Waterfall to 20%

Savings are measured component time minus the good-kernel time. They are not
all additive with step time because some work overlaps; the measured runs are
the check.

| step | lever | kernel status | save ms | step s | MFU |
|---|---|---|---|---|---|
| 0 | current base | | | 0.735 | 12.5% |
| 1 | Pallas top-k | **built and measured**, gg-series +1.6% | 11 | **0.722 (measured)** | **12.7%** |
| 2 | KDA: reset-aware kernel at 30% + fused glue | varlen part **measured** (116 ms, gg nopack), rest assumed | ~173 | 0.550 | 16.7% |
| 3 | fused routed-expert kernel (wi+SwiGLU+wo, custom VJP) | **prototype measured** 2.32x at this shape, ragged-adjusted; not wired | ~57 to 76 | 0.474 to 0.455 | 19.4% to 20.2% |
| 4 | group alignment and tile tuning on top of 3 (43% to ~60% of roofline) | assumed buildable | ~25 | 0.449 to 0.430 | **20.5% to 21.4%** |
| 5 | collective overlap | flags and scheduling | ~48 | 0.401 to 0.382 | 22.9% to 24.1% |
| 6 | top-k/sort/EMo fused, dispatch, LM-head loss, splash, copies | assumed buildable | ~90 | 0.311 to 0.292 | 29.6% to 31.5% |

KDA plus the MoE expert kernel are the two levers that matter: together they
carry 255 to 275 of the 275 ms. The MoE half now rests on a measured prototype;
the KDA half is still an assumed kernel. Everything else is second order. With
all good kernels the step reaches about 0.29 to 0.31 s, 30% MFU, still 2.6x its
roofline.

## Measured runs

| series | test | ctrl TF/s | test TF/s | delta | outcome |
|---|---|---|---|---|---|
| bb | EMo threshold by bisection | 141.5 / 141.8 | 144.2 / 144.3 | +1.9% | kept (base) |
| cc | router top-k by bisection | 144.0 / 144.4 | 140.8 / 141.4 | -2.1% | dropped |
| dd | profile of the base | 144.2 | 144.6 (prof) | | profile source |
| gg | Pallas top-k (`moe_topk_pallas`) | 144.6 / 144.5 | 146.9 / 147.0 | **+1.6%** | kept, loss identical |
| gg | `packing=False` (varlen KDA tax, diagnostic) | 144.6 / 144.5 | 171.8 / 171.6 | **+18.8%** | ceiling of a reset-aware KDA kernel |
| gg | original config, pdb 2, stock flags | | 80.1 | | reference: base is 1.81x the original |
| hh | splash blocks 1024 / 2048 (gpt-oss recipe) | 147.1 / 146.9 | 149.0 | **+1.3%** | reorder only (loss 10.832 vs 10.834); ii confirms |
| hh | full recipe splash set | | failed | | `SEQ` layout name, rerun as `SEQ_MINOR` in ii |

Capacity, 2026-09-28: the ee-series has been queued on nap since 04:10 UTC. From
04:31 on, every resubmit is suspended with `insufficient unused quota for
google.com/tpu in flavor tpu7x-flavor, 52 more needed` (other users hold the
4x4x4 quota); the flex spot 4x4x4 and dws 2x4x4 routes are pending for capacity.
The first supervisor hit its 3-hour limit at 07:12 without admission. At 07:37 the
ee and ff arms were merged into one gg-series (original, base x2, Pallas top-k x2,
`packing=False` x2, `packing=False` with profile) under an 8-hour supervisor.

## Reproduce

```
python3 scripts/olmo35_roofline.py
PYTHONPATH=/home/agagik_google_com/olmo35/perfsim:/home/agagik_google_com/olmo35/perfsim-deps \
  python scripts/olmo35_perfsim.py --seq 8192 --batch-tokens 3145728
JAX_PLATFORMS=cpu PYTHONPATH=src pytest tests/unit/olmoe3_test.py -q -k topk
```
