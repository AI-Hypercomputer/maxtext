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

## Status at the end of the 2026-09-28 loop

| quantity | value |
|---|---|
| original config | 78.0 to 79.6 TF/s, 6.8 to 6.9%, 0.89 to 0.91 s |
| optimized base | 151.0 TF/s, **13.09%**, 0.702 s, 1.93x the original |
| quality-neutral wins this study | Pallas top-k +1.6%, splash blocks +1.8%, fused splash bwd +0.6% |
| flag and mesh levers tested neutral or worse | scheduler concurrency, LHS rerun, SC all-reduce offload, pipeliner, `SEQ_MINOR`, DP 2 and DP 8 |
| binder | TensorCore lane 626 ms of 699 (kk3); exposed comm 65 ms, dependency-bound |
| comm floor (roadmap, all kernels good) | 372 ms, 37.9% MFU |
| what 20% needs | reset-aware KDA kernel (~173 ms) plus the fused routed-expert kernel wired with group-aligned dispatch (~57 to 76 ms) |

20% is **not reached by flags or mesh**; every such lever is now used up on this
base. It stays inside the kernel budget (460 ms target vs 372 ms comm floor), and
the two kernels that carry it are the KDA varlen path (208 ms, about 10% of
roofline, not effective) and the routed-expert GEMMs (165 ms, 39% of roofline,
not effective; a prototype reaches 2.32x at this shape on v7x).

## Target

| quantity | value |
|---|---|
| model FLOPs per device per step (MaxText count) | 106.1 TFLOP |
| peak bf16 per device | 1153.5 TF/s |
| HBM per device | 3.69 TB/s |
| original config, `a2_p2s8k` (pdb 2, stock flags) | 78.2 TF/s, **6.78%** |
| study start base (dd1/dd2, pdb 3) | 144.3 TF/s, 12.51%, step 0.735 s |
| current base (kk to mm controls, pdb 3) | 151.0 TF/s, **13.09%**, step **0.702 s** |
| 20% MFU | 230.7 TF/s, step **0.460 s** |
| gap from the current base | **242 ms** per step (34% of the step) |

Geometry per device: 24576 tokens (pdb 3 x seq 8192), d=1024, 16 layers (14 KDA,
2 attention at 7 and 15), layer 0 dense (mlp 8192), 15 MoE layers with 512 experts
top-16, latent 512, expert hidden 1024, vocab 100352. Routed rows per device:
393216, about 768 per expert.

## Three views of the same step

| view | step s | MFU | note |
|---|---|---|---|
| measured, study start base | 0.735 | 12.5% | median of steps 10-19 |
| measured, current base | 0.702 | 13.1% | kk to mm controls, nap and flex agree within 0.6% |
| roofline, current kernels unfused | 0.156 | 59% | per-component max(MXU, HBM) |
| roofline, fused routed-expert kernel | 0.115 | 80% | SwiGLU in the gmm epilogue |
| perfsim, same operating point | 0.338 | 34.8% | remat=full, counts 135 TF/device |

perfsim is **2.17x optimistic** at the study start (0.735 / 0.338) and 2.08x on
the current base (0.702 / 0.338; the operating point is unchanged, so perfsim was not rerun). Its FLOP count includes the
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

### Profile of the 0.704 s base and of the original (kk3, kk4)

kk-series (`o35n281137`): kk3 profiles the optimized base (0.699 s step, 695 ms
of profiled ops), kk4 the original config (pdb 2, stock flags, 0.900 s step).
Same attribution as dd1. The original moves 16384 tokens per step, the base 24576.

| component | dd1 ms | kk3 ms (base) | kk4 ms (original) | kernel verdict on kk3 |
|---|---|---|---|---|
| KDA glue | 140.4 | 140.3 | 110.9 | not effective: varlen pack, unfused elementwise |
| KDA Pallas kernels | 67.6 | 67.5 | 108.3 | **not effective**, 10% of roofline |
| MoE routed GEMMs (gmm + tgmm) | 166.7 | 164.9 | 134.0 | **not effective**, 39% of 63.8 |
| MoE other | 33.8 | 56.6 | 34.2 | not effective |
| collectives, exposed | 58.2 | 51.8 | **274.9** | scheduling |
| LM head + loss | 37.8 | 37.8 | 25.0 | GEMM effective, loss not |
| dense matmuls | 40.7 | 40.7 | 34.2 | **effective** |
| MoE top-k + sort | 31.7 | **14.4** | 69.6 | top-k now a 4.4 ms Pallas custom call; the sorts remain |
| attention (splash) | 26.5 | **9.3** | 20.2 | **close to effective**: 39% of the 3.6 ms roofline, was 14% |
| dispatch + combine | 17.2 | 17.1 | 15.5 | not effective |
| copies / relayout | 14.1 | 14.2 | 42.8 | layout |
| EMo bisection | 10.6 | 10.6 | | not effective |

The top-k and splash levers show up where expected: together -34.5 ms of device
time, which matches the measured 0.735 s to 0.704 s. Splash went from 14% to
39% of roofline with the 1024 / 2048 blocks and the fused backward, and is now
the only non-matmul kernel near the 50% good-kernel bar. The Pallas top-k is
faster than XLA's sort but still VPU-bound, **not effective**.

xla_shell on both:

| quantity | kk3 (base) | kk4 (original) |
|---|---|---|
| step | 699 ms | 900 ms |
| TensorCore lane | 626 ms: compute 365, VPU 213, relayout 49 | 607 ms: compute 322, VPU 218, relayout 67 |
| SparseCore lane (collectives) | 372 ms, 307 hidden, **65 exposed** | 539 ms, 250 hidden, **289 exposed** |
| comm hidden | 82% | 46% |
| best-overlap ceiling | 626 ms (1.12x headroom) | 607 ms (1.48x headroom) |
| roadmap floor | **372 ms**, binder SparseCore comm | |

| roadmap step (kk3) | lever | step ms | gain |
|---|---|---|---|
| 0 | as profiled | 699 | |
| 1 | schedule the exposed comm | 626 | 72 |
| 2 | kernels, bounded by comm slack | 372 | 254 |
| 3 | relayout | 372 | 0 |
| 4 | host-offload remat | 372 | 0 |

The hill climb's main win over the original is scheduling: exposed comm fell
from 289 ms to 65 ms per step while the TensorCore lane stayed near 610 to 630 ms.
The base is now TC-lane-bound with 1.12x of schedule headroom left, so what
remains is kernel work, which is what the flag sweeps (jj) also showed. `roadmap
--collective` places the remaining 97 ms of exposed comm on expert-grad
all-reduces (2 to 4 ms each, near the end of the step) and the gmm-adjacent
expert all-gathers (1 to 2 ms each). The ll-series tested SparseCore all-reduce
offload and the recipe's pipeliner flags against exactly those: both neutral. The comm floor
rose from 325 ms (gg8, unpacked) to 372 ms (37.9% MFU); 20% at 460 ms is still
inside the kernel budget.

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
| 1b | splash blocks + fused bwd | **measured**, hh and ii +2.4% | 18 | **0.704 (measured)** | **13.1%** |
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
| ii | splash blocks, same-run control | 147.1 (no blocks) | 149.7 / 149.8 | **+1.8%** | kept |
| ii | fused splash bwd + tokamax splash | 149.7 / 149.8 | 150.2 / 151.0 | **+0.6%** | kept, loss unchanged |
| ii | `SEQ_MINOR` layouts | 150.6 (fused) | 150.5 | 0 | dropped |
| jj | all-gather / reduce-scatter concurrency 4 | 150.4 / 151.1 | 151.0 / 150.9 | 0 | neutral |
| jj | `--xla_latency_hiding_scheduler_rerun=2` | 150.4 / 151.1 | 150.9 | 0 | neutral; flag levers used up |
| kk | profile of the base and the original | 150.7 / 151.1 (flex) | 151.2 (prof) | | orig 78.3 / 78.8; profile source |
| ll | SparseCore all-reduce offload | 150.7 / 151.1 | 151.0 / 151.4 (flex) | 0 | neutral |
| ll | pipeliner + experimental scheduler features | 150.7 / 151.1 | 150.4 / 151.3 (flex) | 0 | neutral; exposed comm is dependency-bound |
| mm | mesh DP 2 x FSDP 64 | 151.1 / 151.0 | 149.4 / 149.9 (flex) | -1.1% | dropped |
| mm | mesh DP 8 x FSDP 16 | 151.1 / 151.0 | 135.3 / 135.7 (flex) | -10.4% | dropped; DP 4 x FSDP 32 stays |

Capacity, 2026-09-28: the ee-series has been queued on nap since 04:10 UTC. From
04:31 on, every resubmit is suspended with `insufficient unused quota for
google.com/tpu in flavor tpu7x-flavor, 52 more needed` (other users hold the
4x4x4 quota); the flex spot 4x4x4 and dws 2x4x4 routes are pending for capacity.
The first supervisor hit its 3-hour limit at 07:12 without admission. At 07:37 the
ee and ff arms were merged into one gg-series (original, base x2, Pallas top-k x2,
`packing=False` x2, `packing=False` with profile) under an 8-hour supervisor.

## Profiles

Each capture covers steps 5 to 7 on device 0 (`profiler=xplane
skip_first_n_steps_for_profiler=5 profiler_steps=3`). xla_shell and attribution
outputs are in `gs://agagik-us/olmo35/4x4x4/analysis/`, alongside the two attribution scripts
(`srcsurvey.py` builds the records pickle, `components.py` groups it).

| profile | config | step | xplane | analysis |
|---|---|---|---|---|
| dd1 | study start base, 144.6 TF/s | 0.735 s | `gs://agagik-us/olmo35/4x4x4/o35n280354-olmo35-tiny-dd1_prof/tensorboard/plugins/profile/2026_09_28_04_03_29/gke-tpu-8fcc56a0-sr15.xplane.pb` | tables in this doc |
| gg8 | base, `packing=False` (diagnostic) | 0.617 s | `gs://agagik-us/olmo35/4x4x4/o35n280808-olmo35-tiny-gg8_nopack_prof/tensorboard/plugins/profile/2026_09_28_08_38_36/gke-tpu-42351936-1w5v.xplane.pb` | `gg8_analyze.txt`, `gg8_roadmap.txt`, `gg8_components.txt` |
| **kk3** | **current base**, 151.2 TF/s | 0.701 s | `gs://agagik-us/olmo35/4x4x4/o35n281137-olmo35-tiny-kk3_prof/tensorboard/plugins/profile/2026_09_28_11_48_50/gke-tpu-c19ec041-mw2h.xplane.pb` | `kk3_analyze.txt`, `kk3_roadmap.txt`, `kk3_collective.txt`, `kk3_components.txt` |
| **kk4** | **original config**, 78.5 TF/s | 0.901 s | `gs://agagik-us/olmo35/4x4x4/o35n281137-olmo35-tiny-kk4_orig_prof/tensorboard/plugins/profile/2026_09_28_11_51_39/gke-tpu-c19ec041-mw2h.xplane.pb` | `kk4_analyze.txt`, `kk4_roadmap.txt`, `kk4_components.txt` |

All captures of this model, in GCS (full paths: `gcloud storage ls "gs://agagik-us/olmo35/**.xplane.pb"`):

| capture | what | location |
|---|---|---|
| sps8 iw | 8 devices on SPS, first best config (2026-09-21) | `gs://agagik-us/olmo35/profiles/sps8-20260921-iw/` |
| e2, e4 | 128 devices, pdb 2 and pdb 3, early flags | `gs://agagik-us/olmo35/4x4x4/o35n242234-olmo35-tiny-e2_p2_prof/`, `...-e4_p3_prof/` |
| f1 | scheduler flags (nap and flex spot) | `gs://agagik-us/olmo35/4x4x4/o35n251513-olmo35-tiny-f1_p3_sched_prof/`, `gs://agagik-us/olmo35/profiles/flexspot-o35s251513-f1_p3_sched_prof/` |
| h1 | pdb 3, sched, DP 4 | `gs://agagik-us/olmo35/4x4x4/o35n251712-olmo35-tiny-h1_ctrl_prof/` |
| r1 | p2 base (nap and flex spot) | `gs://agagik-us/olmo35/4x4x4/o35n270031-olmo35-tiny-r1_prof/`, `gs://agagik-us/olmo35/profiles/flexspot-o35s270031-r1_prof/` |
| s4 | lean routing | `gs://agagik-us/olmo35/4x4x4/o35n270116-olmo35-tiny-s4_lean_prof/` |
| x1 | l2out base | `gs://agagik-us/olmo35/4x4x4/o35n270751-olmo35-tiny-x1_prof/` |
| z1 | trhs base (two captures, retry) | `gs://agagik-us/olmo35/4x4x4/o35n270912-olmo35-tiny-z1_prof/` |
| dd1, gg8, kk3, kk4 | this study, table above | as above |
| KDA vs GDN | single device, tokamax KDA, jnp KDA, MaxText GDN | `gs://agagik-us/olmo35/profiles/kda-vs-gdn-single-device-20260926/` |

Text reports for e2, e4, f1 and h1 are under `gs://agagik-us/olmo35/4x4x4/analysis/<run>-<arm>/`.
The c1 capture was lost with its pod and does not exist.

The fused routed-expert numbers (2.32x) and the Pallas top-k numbers come from
single-device microbenchmarks on tpu7x-cluster-flex (`scripts/latent_moe_fusion_bench.py`,
wall-clock timing, no xplane), not from these captures.

```
PYTHONPATH=/home/agagik_google_com/olmo35/xla-shell python -m xla_shell -c "read_xplane kk3.xplane.pb; analyze_profile"
PYTHONPATH=/home/agagik_google_com/olmo35/xla-shell python -m xla_shell -c "read_xplane kk3.xplane.pb; roadmap --all"
PYTHONPATH=/home/agagik_google_com/olmo35/xla-shell python srcsurvey.py kk3.xplane.pb kk3recs.pkl
PYTHONPATH=/home/agagik_google_com/olmo35/xla-shell python components.py kk3.xplane.pb kk3recs.pkl
```

## Reproduce

```
python3 scripts/olmo35_roofline.py
PYTHONPATH=/home/agagik_google_com/olmo35/perfsim:/home/agagik_google_com/olmo35/perfsim-deps \
  python scripts/olmo35_perfsim.py --seq 8192 --batch-tokens 3145728
JAX_PLATFORMS=cpu PYTHONPATH=src pytest tests/unit/olmoe3_test.py -q -k topk
```
