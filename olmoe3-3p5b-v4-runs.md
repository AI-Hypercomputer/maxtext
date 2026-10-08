# olmoe3-3p5b on TPU v4: run record and hill climb

Model `src/maxtext/configs/models/olmoe3-3p5b.yml`: 62.9B total, 3.48B active, 30
layers (24 KDA, 6 full attention), 512 experts top-16, latent 768.

| hardware | value |
|---|---|
| cluster | `v4-128-bodaborg-us-central2-b` (project `cloud-tpu-multipod-dev`), 3 Ready v4-128 slices, Kueue quota 256 chips |
| slice | 4x4x4, 64 chips, 16 hosts, 4 JAX devices per host (one megacore device per chip) |
| per chip | 275 TF/s bf16, 32 GiB HBM (30.75 GB usable), 1.2 TB/s, 16 MiB VMEM |
| MFU | TF/s per device / 275 (the launcher's TSV column divides by the v7x peak; ignore it) |

Launch: `scripts/olmoe3_v4_launch.sh <letter> <arms file>` (from the j-series on), which wraps
`scripts/olmo35_xpk_4x8x8.sh` with `ACCEL=tpu-v4-podslice CLUSTER=v4-128-bodaborg-us-central2-b
PROJECT=cloud-tpu-multipod-dev REGION=us-central2 PLACEMENT_POLICY= RESERVATION= OUT=/tmp/out
MODELS=olmoe3-3p5b LIBTPU_ARGS=<v4 flags>` and archives source, arms and JobSet to
`gs://agagik-us/olmo35/v4/runs/<run>/`. Pods cannot write GCS, so `scripts/olmoe3_v4_harvest.sh <run>` copies
the per-arm logs, results and profiles off the leader pod while `HOLD_S` keeps it up. Current bests and how to
rerun them: `olmoe3-3p5b-v4-best-configs.md`.

v4 constraints: tokamax KDA and tokamax GMM do not run on v4 (`supported_on` needs generation >= 6), so KDA
is MaxText's pure-JAX `_delta_rule_chunked` and the MoE is megablox or XLA `ragged_dot`. libtpu flags:
`--xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_bf16_emission_mode=NATIVE_EMISSION
--xla_tpu_scoped_vmem_limit_kib=16384`.

## Local v4-8 checks (this VM, 4 chips)

| check | result |
|---|---|
| 5-layer, 64-expert cut, megablox, default tiles | runs |
| same, megablox, zero-scratch tiles (128 x 768 x 1792) | runs, 0.398 s/step, 44.2 TF/s/device |
| same, XLA `ragged_dot`, default tiles | compile VMEM OOM, 16.05 of 16 MiB |
| same, XLA `ragged_dot`, zero-scratch tiles | compile VMEM OOM, 22.16 MiB: the tiling tag only reaches the fwd; the autodiff bwd ragged_dots need their own (a per-pass custom VJP) |

### KDA core, one v4 device, B 1, T 4096, H 8, dk 256, dv 512, chunk 64

`scripts/kda_vs_gdn_bench.py --impls kda_jnp,kda_sub,kda_jnp_bf16,kda_sub_bf16 --batch 1 --seq 4096 --heads 8 --dk 256 --dv 512`

| implementation | fwd ms | fwd+bwd ms | vs pairwise |
|---|---|---|---|
| `_delta_rule_chunked` (pairwise, fp32 state) | 4.97 | 28.21 | |
| **`_delta_rule_chunked_subblock`** (fp32 state) | 6.28 | **15.80** | **1.78x** |
| pairwise, bf16 state | 4.87 | 27.87 | |
| sub-block, bf16 state | 6.20 | 14.80 | 1.91x |

`kda_chunked_impl=subblock` (default `pairwise`). Below-diagonal 16-row sub-blocks factor the per-channel
decay through the sub-block's first row, so both exponents are <= 0 and the product is a matmul that cannot
overflow; only the diagonal sub-blocks keep the exact per-pair tensor (4x smaller). (I + M)^-1 is block
doubling from 16 x 16 with the analytic VJP dM = tril(-X^T dX X^T, -1). u, w, the scores and the readout
are computed for all chunks outside the scan, which keeps two matmuls per chunk on the sequential path; only
the intra-chunk matrix build is rematerialized. Matches `_delta_rule_scan` to 7.4e-6 fwd and 2.4e-5 grads,
with resets on and off chunk edges and per-step decay to about -50 nats (`OLMoE3SubblockDeltaRuleTest`).
At 24 KDA layers the fwd+bwd saving is about 300 ms per step.

## perfsim projection

`scripts/olmoe3_v4_perfsim.py` (perfsim has no v4 preset; the chip is v5p with v4's numbers).

| chips | pdb | seq | EP | remat | step ms | tok/s/chip |
|---|---|---|---|---|---|---|
| 64 | 1 | 4096 | 4 | full | 1382.8 | 2962 |

perfsim ran 2.1x to 2.3x optimistic on v7x for this family.

## v-series and w-series: all out of memory (o3v4a071724, o3v4w071737)

| arm | layout | pdb | seq | HLO temporaries (30.75 GB usable) |
|---|---|---|---|---|
| v0_full | FSDP 64 | 1 | 4096 | 31.19 GB |
| v1_tiles | FSDP 64, zero-scratch tiles | 1 | 4096 | 31.19 GB |
| v2_custom | v1 + custom remat, per-layer remat | 1 | 4096 | 31.11 GB |
| v3_p2 | v1 + custom remat | 2 | 4096 | 31.34 GB |
| v4_s8k | v1 + custom remat | 1 | 8192 | 31.34 GB |
| w0_ep4 | FSDP 16 x EP 4, rbf 1.25 | 1 | 4096 | 32.47 GB |
| w1_ep8 | FSDP 8 x EP 8 | 1 | 4096 | mesh error: 8 is not a product of the 4x4x4 axes |
| w2_ep4_sub | w0 + sub-block KDA | 1 | 4096 | 35.12 GB |

Temporaries barely move with batch or sequence, so activations are not what fills HBM.

## Memory fit with AOT (v4-128 topology, compiled on this VM)

`/tmp/aot_v4/aot.sh` mirrors the pod command line with `compile_topology=v4-128`; it reproduces the slice OOM
(31.22 GB local vs 31.19 GB on hardware). XLA's compile check covers temporaries only (30.75 GB); the chip
also holds the arguments, so the target is args + temps under 32 GiB (34.4 GB). Megacore: one device per chip,
64 devices on a v4-128.

| config (pdb 1, seq 4096) | args GB | temps GB | args + temps |
|---|---|---|---|
| FSDP 64 (`shard_exp_on_fsdp`) | | 31.22 | compile OOM |
| FSDP 16 x EP 4 | | 32.49 | compile OOM |
| EP 4, optimizer state on host | 3.95 | 28.87 | 32.81 |
| EP 4, bf16 weights | 5.92 | 29.07 | 34.99 |
| FSDP 4 x EP 16 | 11.84 | 26.00 | 37.84 |
| FSDP 1 x EP 64 | 11.84 | 25.49 | 37.33 |
| **FSDP 16 x EP 4, per-layer remat** | 11.84 | 16.60 | **28.44** |
| **FSDP 4 x EP 16, per-layer remat** | 11.84 | 15.80 | **27.64** |
| **FSDP 1 x EP 64, per-layer remat** | 11.84 | 15.58 | **27.42** |
| FSDP 4 x EP 16, per-layer remat, seq 8192 | 11.84 | 22.89 | 34.73 (too tight) |

What filled HBM, from the XLA buffer assignment (`XLA_FLAGS=--xla_dump_to=...`, memory-usage report):

| buffers live at peak | size | cause |
|---|---|---|
| `bf16[128,768,1792]` and `[128,1792,768]` (EP 4) | about 10 GB | each MoE layer's local experts all-gathered over FSDP, for all 5 layers of a scanned cycle plus the next cycle's prefetch |
| `f32[8,5,768,1792]`, `f32[5,8,...]` (EP 64) | about 12 GiB | fp32 expert weight grads for a whole cycle, in two layouts |
| `bf16[81920,768]`, `bf16[81920,1792]` | about 7.5 GiB | MoE activations of all 5 layers in the cycle at once |

`shard_exp_on_fsdp=True` shards the expert dim on FSDP but still gathers the whole 4.2 GB bank inside the MoE,
which is why FSDP 64 cannot fit. The scan unit is a 5-layer mixer cycle, and the stock remat recomputes the
whole cycle in the bwd; `olmoe3_per_layer_remat=True` recomputes one layer at a time and removes about 10 GB.

## perfsim by layout (64 chips, seq 4096, full remat)

| EP x FSDP | pdb 1 step ms | pdb 1 tok/s/chip | pdb 2 tok/s/chip |
|---|---|---|---|
| 4 x 16 | 1382.8 | 2962 | 3032 |
| 16 x 4 | 1395.8 | 2934 | 3010 |
| 64 x 1 | 2784.4 | 1471 | 1497 |

At EP 64 perfsim exposes about 1.4 s of all-to-all (dispatch bwd 556, combine fwd 516, dispatch fwd 324 ms).
At EP 16 the step is the expert GEMMs (gate/up dW 231, gate/up dX 127, down dW 116 ms, fwd 79 + 39,
recompute 79) and the optimizer (67 ms).

## x-series: first fitting runs, FSDP x EP (o3v4x071908, one v4-128)

All arms: pdb 1, seq 4096, megablox, `olmoe3_per_layer_remat=True`, full remat, rbf 1.25, 20 steps synthetic.
64 devices confirmed (`total_weights` 262,144 = 64 x 4096), so megacore is on.

| arm | change | step s | TF/s/dev | MFU | loss @19 |
|---|---|---|---|---|---|
| x0_ep4 | FSDP 16 x EP 4, zero-scratch tiles | 3.545 | 23.7 | 8.6% | 7.884 |
| x3_ep8 | FSDP 8 x EP 8 (`allow_split_physical_axes`) | 3.401 | 24.7 | 9.0% | 7.906 |
| x1_ep16 | FSDP 4 x EP 16 | 3.018 | 27.9 | 10.1% | 7.900 |
| x2_ep64 | FSDP 1 x EP 64 | 2.850 | 29.5 | 10.7% | 7.804 |
| x5_ep16_notile | x1 with default megablox tiles | 3.133 | 26.8 | 9.7% | 7.881 |
| **x4_ep16_sub** | **x1 + `kda_chunked_impl=subblock`** | **2.767** | **30.4** | **11.1%** | 7.875 |
| x6_prof | x4, profiled | 2.766 | 30.4 | 11.1% | 7.875 |

Zero-scratch tiles: -115 ms (3.7%). Sub-block KDA: -251 ms (8.3%), loss unchanged. More expert parallelism
wins monotonically: EP 64 avoids gathering expert weights, and its all-to-all costs less than perfsim assumes.

| layout | measured step | perfsim step | measured / perfsim |
|---|---|---|---|
| EP 4 | 3.545 s | 1.383 s | 2.56x |
| EP 16 | 3.018 s | 1.396 s | 2.16x |
| EP 64 | 2.850 s | 2.784 s | 1.02x |

perfsim gets the EP trend backwards on v4: its all-to-all model (v4 ICI taken as half of v5p per link) is too
pessimistic, and it under-counts the FSDP expert-weight gathers.

### Profile of x6 (EP 16 + sub-block), xla-shell

Profile archived at `gs://agagik-us/olmo35/v4/o3v4x071908-x6_prof.xplane.pb`. The profiler writes only on
JAX process 0, which was pod `slice-job-0-14`, not `0-0`.

| xla-shell view | value |
|---|---|
| step | 2.76 s |
| TensorCore lane | 2.35 s: matmul 1.08 s, VPU 954 ms, relayout 315 ms |
| SparseCore lane | 0 (v4 has no collective offload) |
| best-overlap ceiling | 2.35 s, 1.18x headroom |
| roadmap floor | 414 ms of comm; kernels are the whole budget |

| component | ms | share | source |
|---|---|---|---|
| expert GEMMs (megablox Pallas in shard_map) | 523 | 18.9% | `kernels/megablox/backend.py` |
| ragged all-to-all (TensorCore DMA on v4) | 413 | 14.9% | `layers/moe.py:2731` |
| dense projection matmuls | ~520 | ~19% | `layers/linears.py:160` |
| MoE permute gather + scatter | 175 | 6.3% | `moe.py:196`, `moe.py:1405` |
| FSDP all-gathers | 215 | 7.8% | per-layer remat site |
| all-reduce, reduce-scatter | 160 | 5.8% | |
| EMo threshold sort | 56 | 2.0% | `olmoe3.py:817` (bisection exists: `emo_threshold_by_bisection`) |
| grad and param norms | 74 | 2.7% | `utils/max_utils.py:115` |

## y-series: EP 64 base (o3v4y071949, one v4-128)

Base: x-series flags + EP 64 x FSDP 1 + sub-block KDA, pdb 1, seq 4096. Each row adds to y0 unless noted.

| arm | change | step s | TF/s/dev | MFU | vs y0 |
|---|---|---|---|---|---|
| y0_ep64_sub | EP 64 + sub-block | 2.618 | 32.1 | 11.7% | |
| y1_bisect | + `emo_threshold_by_bisection` | 2.592 | 32.4 | 11.8% | -26 ms |
| **y2_custom** | y1 + `remat_policy=custom moe_routing=device` | **2.562** | **32.8** | **11.9%** | **-56 ms** |
| y3_bf16state | y1 + `gdn_state_dtype=bfloat16` | 2.566 | 32.8 | 11.9% | -52 ms (numerics change) |
| y4_ep32 | y1 at EP 32 x FSDP 2 | 2.700 | 31.1 | 11.3% | +82 ms |
| y5_prof | y1, profiled | 2.596 | 32.4 | 11.8% | |

### Profile of y5 (EP 64), xla-shell

`gs://agagik-us/olmo35/v4/o3v4y071949-y5_prof.xplane.pb`. Step 2.59 s; TensorCore 2.43 s (matmul 1.05 s, VPU
1.08 s, relayout 302 ms); best-overlap headroom 1.07x. VPU plus relayout is 53% of the step.

| component | EP 64 (y5) | EP 16 (x6) |
|---|---|---|
| ragged all-to-all (`moe.py:2731`) | 544 ms (21%) | 413 ms |
| expert GEMMs (megablox) | 493 ms | 523 ms |
| dense projections (`linears.py:160`) | ~520 ms | ~520 ms |
| weight all-gathers | 92 ms | 215 ms |
| EMo threshold sort | 0 (bisection) | 56 ms |

The ragged all-to-all moves about 100 MB per device per call in about 4.7 ms (~21 GB/s per chip), well under
what a 4x4x4 torus should sustain, and xla-shell books it as TensorCore data formatting. libtpu has TPU knobs
for it (`xla_tpu_enable_async_ragged_all_to_all`, `xla_tpu_ragged_all_to_all_max_rdma_size_kib`,
`xla_tpu_enable_ragged_all_to_all_pipelined_local_copy`,
`xla_tpu_debug_disable_ragged_all_to_all_vmem_bounce_buffer`); they are the next series.

AOT memory at EP 64 for the next series: pdb 2 with full remat 35.18 GB (no); pdb 2 with
`decoder_layer_input=offload` 28.67 GB; seq 8192 with offload 28.88 GB.

## z-series: batch, sequence, offload, tiles (o3v4z072025, one v4-128)

Base y2 (EP 64, sub-block, bisection, custom remat with `moe_routing=device`), pdb 1, seq 4096 unless noted.

| arm | change | step s | TF/s/dev | MFU | per 4,096 tokens |
|---|---|---|---|---|---|
| y2 (base) | | 2.562 | 32.8 | 11.9% | 2.562 s |
| z2_off | + `decoder_layer_input=offload` | 2.577 | 32.6 | 11.9% | 2.577 s |
| z0_p2off | pdb 2 + offload | 5.558 | 30.3 | 11.0% | 2.779 s |
| z1_s8koff | seq 8192 + offload | 5.857 | 29.1 | 10.6% | 2.929 s |
| **z3_mt512** | megablox `*_batch_seq` 128 -> 512 | **2.525** | **33.3** | **12.1%** | 2.525 s |
| z4_mt1024 | `*_batch_seq` 1024 | compile VMEM OOM, 18.23 of 16 MiB | | | |

pdb 2 and seq 8192 lose per token: the step is TensorCore-bound per token with no fixed cost to amortize,
and attention grows quadratically. Host offload of the layer input costs 15 ms. With 8 experts per device
(about 8K rows each), the 128-row tile chosen for EP 4's 128 small groups is too small; 512 is the largest
that fits v4's 16 MiB VMEM.

## Hill climb so far (one v4-128, pdb 1, seq 4096)

| step | change | step s | TF/s/dev | MFU |
|---|---|---|---|---|
| 1 | FSDP 16 x EP 4, per-layer remat, zero-scratch tiles (first fit) | 3.545 | 23.7 | 8.6% |
| 2 | EP 16 | 3.018 | 27.9 | 10.1% |
| 3 | EP 64 | 2.850 | 29.5 | 10.7% |
| 4 | + sub-block KDA | 2.618 | 32.1 | 11.7% |
| 5 | + EMo bisection | 2.592 | 32.4 | 11.8% |
| 6 | + custom remat (`moe_routing=device`) | 2.562 | 32.8 | 11.9% |
| 7 | + megablox 512-row tiles | **2.525** | **33.3** | **12.1%** |

## a-series: ragged all-to-all knobs and buffer (o3v4b072055, one v4-128)

Base z3 (EP 64, sub-block, bisection, custom remat, megablox 512-row tiles), pdb 1, seq 4096.

| arm | change | step s | TF/s/dev | MFU | loss @19 |
|---|---|---|---|---|---|
| a0_ctrl | | 2.524 | 33.3 | 12.1% | 7.821 |
| a1_async | `--xla_tpu_enable_async_ragged_all_to_all=true` | 2.521 | 33.4 | 12.1% | 7.838 |
| a2_rdma4m | `--xla_tpu_ragged_all_to_all_max_rdma_size_kib=4096` | 2.625 | 32.0 | 11.6% | 7.821 |
| a3_pipe | `--xla_tpu_enable_ragged_all_to_all_pipelined_local_copy=true` | 2.522 | 33.3 | 12.1% | 7.821 |
| a4_nobounce | `--xla_tpu_debug_disable_ragged_all_to_all_vmem_bounce_buffer=true` | 2.526 | 33.3 | 12.1% | 7.821 |
| **a5_rbf11** | `ragged_buffer_factor` 1.25 -> 1.125 | **2.447** | **34.4** | **12.5%** | 7.839 |

The libtpu knobs are neutral or worse. The smaller buffer is 77 ms faster (less all-to-all and GMM work) but
drops the overflow tokens of experts hotter than 1.125x; the 0.018 loss difference is within the run-to-run
spread here (a1 changes no math and moved by the same amount). Real packed data needs the
`log_required_ragged_buffer_factor` probe before 1.125 is used.

## Local v4 KDA micro-optimizations (one device, B 1, T 4096, H 8, dk 256, dv 512)

| sub-block variant | fp32 state fwd / fwd+bwd ms | bf16 state fwd / fwd+bwd ms |
|---|---|---|
| first version | 6.28 / 15.80 | 6.20 / 14.80 |
| q and k stacked (one matmul, one pass over the pair tensor), cast before chunking | 3.95 / 17.59 | 3.64 / 14.53 |
| diagonal split, off-diagonal stacked | 4.08 / 16.01 | 3.92 / 15.23 |
| off-diagonal split, diagonal stacked | 3.69 / 17.88 | 3.44 / 17.45 |
| scan `unroll` 2 / 4 / 8 (bf16) | | 15.40 / 14.90 / 15.03 |
| **stacked + cumsum as a lower-triangular matmul** (kept) | | **3.19 / 14.16** |

`jnp.cumsum` lowers to an O(C^2) reduce_window on TPU; the HIGHEST-precision triangular matmul is exact to
fp32 roundoff. The local profile puts the rest in the scan's 64 sequential steps of 64-row matmuls (about 4 ms)
and the pair-tensor gradient reductions (about 2 ms).

## b-series: KDA micro-opts, bf16 state, fused projection, buffer, bf16 weights, EP 16 (o3v4c072139)

Base a0 plus the KDA micro-optimizations above (q/k stacked, cumsum as matmul), EP 64, buffer 1.25 unless noted.

| arm | change | step s | TF/s/dev | MFU |
|---|---|---|---|---|
| b0_ctrl | new KDA code, fp32 state | 2.470 | 34.0 | 12.4% |
| b1_bf16state | + `gdn_state_dtype=bfloat16` | 2.455 | 34.2 | 12.4% |
| b2_fused | + `kda_fused_input_proj=True` | 2.463 | 34.1 | 12.4% |
| b3_both | bf16 state + fused projection | 2.445 | 34.4 | 12.5% |
| **b4_both_rbf11** | b3 + buffer 1.125 | **2.364** | **35.6** | **12.9%** |
| b5_bf16w | b3 + `weight_dtype=bfloat16` | 2.454 | 34.3 | 12.5% |
| b6_ep16 | b3 at EP 16 x FSDP 4 | 2.654 | 31.7 | 11.5% |
| b7_prof | b3, profiled | 2.442 | 34.4 | 12.5% |

The KDA micro-optimizations are worth 54 ms in the model. bf16 weights do not help in this layout: at EP 64
each device's expert weights are local, so the fp32 -> bf16 cast is cheap; fp32 weights also avoid pulling
Adam state to bf16. EP 64 stays ahead of EP 16 with megablox.

`moe_x_sorted` (new checkpoint name on the expert-local tokens after dispatch) does not fit on 64 chips: saving
it raises temporaries 17.71 -> 23.39 GB (35.2 GB with args); host offload 35.5 GB; with the layer input offloaded
and buffer 1.125 it lands at 34.36 GB, the physical limit. It becomes usable at 128 chips. Saving the KDA
q/k/v projections (now tagged `query_proj`/`key_proj`/`value_proj`) fits: 32.42 GB, or 31.76 GB with
`out_proj` and `context` as well.

Profile of b7 (2.44 s; recompute 669 ms of it): ragged all-to-all 534 ms, megablox 455 ms (about 50% of its
roofline), dense projections and their gathers about 614 ms (about 20% of roofline), copies 162 ms, MoE gather
139 ms, vocab-tiling all-reduce 54 ms. Profile archived at `gs://agagik-us/olmo35/v4/o3v4c072139-b7_prof.xplane.pb`.

## d-series: saving projections, direct gather (o3v4d072221)

Base b4 (EP 64, sub-block, bisection, custom remat, 512-row tiles, bf16 state, fused projection, buffer 1.125).

| arm | change | step s | TF/s/dev |
|---|---|---|---|
| d0_ctrl | | 2.367 | 35.5 |
| d1_qkv | + `query_proj/key_proj/value_proj=device` (KDA projections now named) | 2.373 | 35.4 |
| d2_qkvo | + `out_proj` and `context` too | 2.370 | 35.5 |
| d3_dgather | + `moe_use_direct_token_gather` | 2.362 | 35.6 |

Neither lever moves the step. With the MFU accounting of the reference report (91.14 TFLOPs per step per
chip), 2.364 s is 38.6 TF/s, 14.0% MFU; the report's best 64-chip run is 2.275 s (14.6%).

## Question 1: KDA on TPU v4, kernel headroom and co-design (measured)

Per layer, one v4 device, B 1, T 4096 unless noted, `scripts/kda_vs_gdn_bench.py`.

| measurement | value |
|---|---|
| KDA core, sub-block, bf16 state: fwd / fwd+bwd | 3.17 / 14.15 ms |
| roofline per layer: 100 GFLOP fwd+bwd (0.36 ms at 275 TF/s), ~0.86 GB (0.71 ms at 1.2 TB/s) | 0.71 ms, so the core runs at ~5% of roofline |
| in-model share | about 416 ms of the 2.44 s step (17%): 24 layers x (fwd + remat fwd + bwd) |
| microbench vs model check | pairwise -> sub-block predicted -266 ms, measured -232 to -251 ms in the model |
| seq 8192: sub-block vs pairwise | 29.04 vs 55.56 ms fwd+bwd (linear in T) |

Where the core time goes (local profile): the scan's 64 sequential chunk steps of 64-row matmuls in the bwd
(about 4 ms), gradient reductions over the diagonal per-pair decay tensor (about 2 ms), scan slicing and
stacking copies (about 1.6 ms), and the remaining glue.

Co-design options, measured at equal total width (H x dk = 2048, H x dv = 4096):

| option | fwd+bwd ms | vs today |
|---|---|---|
| today: KDA, 8 heads x (256, 512), chunk 64 | 14.15 | |
| chunk 32 | 17.46 | +23% |
| chunk 128 | 13.81 | -2% |
| 16 heads x (128, 256) | 13.96 | -1% |
| 4 heads x (512, 1024) | 16.14 | +14% |
| **GDN (scalar decay), MaxText's unoptimized pure-JAX path** | **12.80** | **-10%** |
| GDN, optimized (reference report, v4) | 5.97 | -58% |

Answer. Kernel headroom is large on paper (5% of roofline) and moderate in practice: the remaining cost is the
sequential chunk scan and the per-channel decay. A Pallas KDA kernel that keeps the [256, 512] state in VMEM
across chunks is the main kernel lever; at the tokamax kernel's efficiency on v7x (about 6% of roofline at its
shape, but without the XLA glue) it would be worth roughly 2-3x on the core, about 140-280 ms per step. The
biggest co-design lever is the decay granularity: scalar per-head decay (GDN) removes the per-channel per-pair
tensor and is about 2x faster at equal width (5.97 vs 14.15 ms measured across the two reports), about 200 ms
per step. Chunk size and head shape are within a few percent at C = 64-128 and dk = 128-256; larger per-head
state (dk 512) costs 14%.

## Question 2: LatentMoE on TPU v4, roofline and the fused kernel (measured)

Expert block = gate GEMM, up GEMM, GLU, down GEMM. EP 64 per-device shape: 65,536 rows (4,096 tokens x
top-16), 8 local experts, 8,192 rows per expert, latent 768, hidden 1792, bf16. One v4 device.

| variant | fwd ms | fwd TF/s | fwd+bwd ms | fwd+bwd TF/s |
|---|---|---|---|---|
| compute floor (275 TF/s): 0.541 TFLOP fwd, 1.62 fwd+bwd | 1.97 | | 5.90 | |
| XLA `ragged_dot` x2 + GLU | 5.92 | 91 | 14.88 | 109 |
| **megablox x3, tm 512 (what the model runs)** | 5.15 | 105 | **12.27** | **132 (48%)** |
| fused kernel prototype (`scripts/latent_moe_fusion_bench.py`), best tm 512 tn 896 | **4.64** | **117** | | |
| fused prototype fwd+bwd, tm 256 tn 896 (tm 512 bwd does not fit 16 MiB VMEM) | | | 17.78 | 0.84x of XLA |

Byte roofline at this shape (`scripts/latent_moe_roofline.py --num-experts 8`): the unfused block is at 323
FLOP/byte (329 at seq 8192), already past v4's ridge of 229. The block is compute-bound, so the fusion's byte
cut (2.3x with the GLU in the epilogue, 6.3x with full fusion) does not turn into time; only better MXU
utilization does. At EP 4 (128 rows per expert) the block sits below the ridge, which is why a fused kernel
helped much more in the reference report's EP 4 study.

In the model (b7 profile, 2.44 s step) the MoE block costs about 1.13 s: ragged all-to-all 534 ms, expert GEMMs
455 ms (consistent with the microbench: 29 x (5.15 + 12.27) = 505 ms), MoE gather 139 ms. The block's own
roofline is about 460 ms: GEMMs 229 ms (fwd + remat fwd + bwd at 275 TF/s) plus all-to-all about 190 ms (6
all-to-alls per layer of 100 MB each over a 4x4x4 torus at an assumed ~90 GB/s per chip) plus gathers.

Answer. With megablox already at 48% of the GEMM compute floor, a fused LatentMoE kernel on v4 at EP 64 is worth
little: the measured forward gain is 1.11x over megablox (about -30 ms per step through fwd and remat fwd), and
the prototype backward is slower than XLA because its weight-gradient kernel is VMEM-bound at 16 MiB. A fused
backward reaching ~70% MXU would be worth about -110 ms per step at most. The larger MoE lever on v4 is the
ragged all-to-all (534 ms, about 3x its ICI floor), then avoiding its recompute (`moe_x_sorted`, which needs
128 chips to fit).

## e-series: sequence length 8192, as in OLMo 3 (o3v4e072258, one v4-128)

Best 4k config plus `decoder_layer_input=offload` (needed to fit). AOT at 8k: buffer 1.25 without offload is a
compile OOM; buffer 1.25 with offload 37.07 GB; **buffer 1.125 with offload 28.92 GB** (the only fit).

| arm | change | step s | TF/s/dev | MFU | tok/s/chip |
|---|---|---|---|---|---|
| **e0_s8k** | best config at seq 8192 | **5.391** | **31.6** | **11.5%** | **1,520** |
| e2_s8k_pairwise | e0 with the pairwise KDA | 6.038 | 28.3 | 10.3% | 1,357 |
| e3_s8k_prof | e0, profiled | 5.393 | 31.6 | 11.5% | 1,519 |
| (4k best, d0) | for reference | 2.367 | 35.5 | 12.9% | 1,731 |

Sub-block KDA at 8k: -647 ms (10.7%); the microbench predicted -705 ms. Per token, 8k is 12% slower than 4k.

8k profile (`gs://agagik-us/olmo35/v4/o3v4e072258-e3_s8k_prof.xplane.pb`, 5.39 s, TensorCore 96%):

| component | 8k ms | share | 4k ms (d6) | 8k / 4k |
|---|---|---|---|---|
| MoE ragged all-to-all | 1,109 | 20.6% | 504 | 2.2x |
| dense projections | 1,052 | 19.5% | 494 | 2.1x |
| **MoE routing, gather, scatter** | **972** | **18.1%** | 175 | **5.6x** |
| MoE expert GEMMs (megablox) | 889 | 16.5% | 459 | 1.9x |
| KDA core, norms, conv, elementwise | 602 | 11.2% | 276 | 2.2x |
| copies / relayout | 390 | 7.2% | 199 | 2.0x |
| collectives | 179 | 3.3% | 127 | 1.4x |
| full attention (splash) | 94 | 1.7% | 21 | 4.5x (quadratic) |

The outlier is the row gather in `_sort_activations` (`moe.py:196`): 882 ms at 8k vs 131 ms at 4k. A local v4
microbench shows a throughput cliff on the gather's source size:

| source rows x 768 bf16 | gather ms | GB/s |
|---|---|---|
| 36,864 (57 MB) | 0.259 | 437 |
| 73,728 (113 MB) | 0.363 | 625 |
| 110,592 (170 MB) | 1.909 | 178 |
| 147,456 (226 MB) | 2.510 | 181 |

The cliff sits where the source stops fitting v4's 128 MiB CMEM. At seq 4096 the post-dispatch buffer is about
74K rows (fits); at 8192 it is about 147K. Chunking the index array does not help, and gathering from CMEM-sized
source blocks with masks is 3-17x slower. The fix is to split the dispatch along tokens so each chunk's buffer
stays under about 80K rows. MaxText's `num_moe_token_chunks` does this only on the ring-of-experts path; using
it on the all-to-all path needs the EMo document pools to be computed before the split, or routing changes.
That is worth about -600 ms at 8k (11%).

## f-series: KDA chunk 128 (o3v4f072320)

| arm | change | step s | TF/s/dev | MFU | loss @19 |
|---|---|---|---|---|---|
| **f1_s4k_c128** | 4k best + `gdn_chunk_size=128` (needs `override_model_config=True`) | **2.304** | **36.5** | **13.3%** | 7.780 |
| f0_s8k_c128 | 8k best + chunk 128 | 5.297 | 32.2 | 11.7% | 8.490 |
| f2_s8k_rep | e0 repeated | 5.390 | 31.7 | 11.5% | 8.499 |

Chunk 128 is exact (only the blocking changes) and is worth -63 ms at 4k and -94 ms at 8k, more than the
single-layer microbench's -2% (the in-model remat forward and the scan steps both halve). Repeats agree to 1 ms.

## Summary: hill climb on one v4-128 (64 chips), olmoe3-3p5b, pdb 1

| step | change | seq 4096 step s | TF/s/dev | MFU |
|---|---|---|---|---|
| 1 | first fit: FSDP 16 x EP 4, per-layer remat, zero-scratch tiles | 3.545 | 23.7 | 8.6% |
| 2 | EP 64 x FSDP 1 | 2.850 | 29.5 | 10.7% |
| 3 | sub-block KDA | 2.618 | 32.1 | 11.7% |
| 4 | EMo bisection, custom remat (save routing) | 2.562 | 32.8 | 11.9% |
| 5 | megablox 512-row tiles | 2.525 | 33.3 | 12.1% |
| 6 | KDA micro-opts (stacked q/k, cumsum as matmul) | 2.470 | 34.0 | 12.4% |
| 7 | bf16 KDA state, fused KDA input projection | 2.445 | 34.4 | 12.5% |
| 8 | ragged buffer 1.125 (validate on real data) | 2.364 | 35.6 | 12.9% |
| 9 | KDA chunk 128 | **2.304** | **36.5** | **13.3%** |

MFU here uses MaxText's logged TF/s. With the reference report's 91.14 TFLOPs per step per chip, the final
2.304 s is 39.6 TF/s, 14.4% (report's best on the same slice size: 2.275 s, 14.6%). At seq 8192 the same
config runs 5.297 s, 32.2 TF/s, 11.7%, 1,547 tok/s/chip.

Measured and not adopted: pdb 2 and seq 8192 at fixed config (worse per token), bf16 weights (no gain at EP 64),
EP 4 / 8 / 16 / 32 (all slower than 64), libtpu ragged all-to-all knobs (neutral or worse), megablox 1024-row
tiles (VMEM OOM), saving projections or the MoE tokens (neutral, or does not fit on 64 chips), direct token
gather (neutral), scan unroll (worse).

Recommended v4-128 flags (on top of `olmoe3-3p5b.yml`):

    use_tokamax_kda=False use_tokamax_gmm=False use_gmm_v2=False megablox=True sparse_matmul=True
    ici_expert_parallelism=64 ici_fsdp_parallelism=1 shard_exp_on_fsdp=False capacity_factor=-1
    ragged_buffer_factor=1.125   # 1.25 is the conservative setting (+81 ms) until real-data routing is probed
    olmoe3_per_layer_remat=True remat_policy=custom moe_routing=device
    kda_chunked_impl=subblock gdn_state_dtype=bfloat16 kda_fused_input_proj=True
    override_model_config=True gdn_chunk_size=128
    emo_threshold_by_bisection=True moe_lean_routing=True moe_topk_pallas=True kda_conv_in_compute_dtype=True
    num_vocab_tiling=4 wi/wo_tile_*: batch_seq 512, embed 768, mlp 1792 (drhs mlp 896)
    seq 8192: add decoder_layer_input=offload
    LIBTPU_INIT_ARGS: --xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_bf16_emission_mode=NATIVE_EMISSION
                      --xla_tpu_scoped_vmem_limit_kib=16384

Next levers, by measured size: ragged all-to-all (504 ms at 4k, about 3x its ICI floor); the 8k CMEM gather
cliff (about -600 ms at 8k); a Pallas KDA kernel with VMEM-resident state (about -140 to -280 ms); KDA -> GDN as
co-design (about -200 ms); dense projections at about 20% of roofline (494 ms); `moe_x_sorted` at 128 chips
(skips about 176 ms of all-to-all recompute).

## g-series: confirm the best, safe buffer, chunk 256, final profile (o3v4g080004)

Base f1: EP 64, sub-block KDA, chunk 128, bf16 state, fused projection, buffer 1.125, seq 4096, pdb 1.

| arm | change | step s | TF/s/dev | MFU | loss @19 |
|---|---|---|---|---|---|
| **g0_c128** | best config, repeated | **2.304** | **36.5** | **13.3%** | 7.780 |
| g1_c128_rbf125 | buffer 1.25 (production-safe until real-data routing is probed) | 2.386 | 35.2 | 12.8% | 7.822 |
| g2_c256 | KDA chunk 256 | 2.420 | 34.7 | 12.6% | 7.810 |
| g3_prof | g0, profiled | 2.304 | 36.5 | 13.3% | 7.780 |

The best reproduces to the millisecond. Buffer 1.25 costs 82 ms. Chunk 256 is 116 ms slower than 128, so 128
is the optimum on v4 (64 and 256 both lose). The loss at step 19 is consistently about 0.03-0.04 higher with
buffer 1.25 than 1.125 (also d5 vs d0); with 20 steps and one seed this is not a quality signal either way.

Final profile (`gs://agagik-us/olmo35/v4/o3v4g080004-g3_prof.xplane.pb`): step 2.30 s, TensorCore 2.14 s
(matmul 979 ms, VPU 889 ms, relayout 277 ms), best-overlap headroom 1.07x; VPU plus relayout is 51%.

| component | ms | share | of which recompute | vs chunk 64 (d6) |
|---|---|---|---|---|
| MoE ragged all-to-all | 501 | 21.8% | 176 | -3 |
| dense projections | 473 | 20.6% | 74 | -21 |
| MoE expert GEMMs (megablox) | 461 | 20.0% | 120 | +2 |
| KDA core, norms, conv, elementwise | 260 | 11.3% | 80 | -15 |
| MoE routing, gather, scatter | 174 | 7.5% | 62 | -1 |
| copies / relayout | 165 | 7.2% | 33 | -34 |
| weight and gradient collectives | 129 | 5.6% | 57 | +2 |
| LM head + loss | 56 | 2.4% | 0 | +3 |
| grad / param norms | 56 | 2.4% | 3 | 0 |
| full attention (splash) | 25 | 1.1% | 0 | +4 |

Chunk 128's gain is fewer scan steps: fewer copies and relayouts, and less KDA glue. Recompute is 605 ms of the
step; the largest recomputed item is the dispatch all-to-all (176 ms), which `moe_x_sorted=device` removes once
memory allows (128 chips).

## h-series: seq 8192 hill climb (o3v4h080239)

Base f0 (8k best: EP 64, sub-block, chunk 128, buffer 1.125, `decoder_layer_input=offload`, fp32 weights).

| arm | change | step s | TF/s/dev | MFU | loss @19 |
|---|---|---|---|---|---|
| h0_s8k | base, repeated | 5.297 | 32.2 | 11.7% | 8.490 |
| h1_async | + `--xla_tpu_overlap_compute_collective_tc --xla_enable_async_all_gather --xla_enable_async_collective_permute` | 5.341 | 31.9 | 11.6% | |
| **h2_vt8** | + `num_vocab_tiling=8 vocab_tiling_ag_once=True` | **5.270** | **32.4** | **11.8%** | |
| h3_vmem64m | + `--xla_tpu_scoped_vmem_limit_kib=65536` | 5.295 | 32.2 | 11.7% | |
| h4_bf16w | + `weight_dtype=bfloat16` | 5.300 | 32.2 | 11.7% | **10.324** |
| h5_bf16w_xs | + bf16 weights + `moe_x_sorted=device` (dispatch and combine tensors saved) | **4.703** | **36.3** | **13.2%** | **10.322** |
| h6_prof | h2, profiled | 5.271 | 32.4 | 11.8% | |

Saving the MoE dispatch and combine tensors removes the recomputed all-to-alls, wo GEMM and post-dispatch gather
(including the CMEM-cliff gather): -594 ms at 8k. bf16 weights only free the memory for it, and they **break
training**: loss 10.32 at step 19 vs 8.49 with fp32 weights, because bf16 master weights and bf16 Adam state
(with `mu_dtype` unset) round away the early updates. The speed of h5 is real; its config is not usable.
The async-collective and VMEM flags from the reference report are neutral or worse here.
Profile: `gs://agagik-us/olmo35/v4/o3v4h080239-h6_prof.xplane.pb`.

## TPU_MEGACORE=MEGACORE_DENSE: +1 GiB HBM

MaxText's v4 recipes (`src/maxtext/configs/tpu/v4/{22b,52b}.sh`, 56-59% MFU on dense models) run with
`TPU_MEGACORE=MEGACORE_DENSE`, the megacore mode that turns BarnaCore off for dense workloads. Measured on the
local v4 (`jax.devices()[0].memory_stats()['bytes_limit']`):

| setting | usable HBM per chip |
|---|---|
| default | 33,014,413,312 B (30.75 GiB) |
| `TPU_MEGACORE=MEGACORE_DENSE` inside `LIBTPU_INIT_ARGS` (as in the recipe scripts) | 30.75 GiB, no effect |
| **`TPU_MEGACORE=MEGACORE_DENSE` as an environment variable** | **34,088,155,136 B (31.75 GiB)** |
| `--use_barna_core_for_offloading=false`, `--barna_core_max_hbm_fraction_for_embeddings=0`, `--xla_tpu_user_reserved_hbm_bytes=0` | 30.75 GiB |

AOT compiles with `compile_topology=v4-128` do not see it (the topology description keeps 30.75 G), so configs
that need the extra GiB have to be tried on hardware. Pass it per arm in the launcher's env field.

## Quality checks for the precision levers

| lever | numerics (vs fp32 HIGHEST scan, local v4, 2 heads of 256 x 512) | loss @19 in the model |
|---|---|---|
| bf16 KDA state (`gdn_state_dtype=bfloat16`) | output 2.09e-2 vs 2.08e-2 with fp32 state, at both 4k and 8k; gradients equal to 3 digits | 7.784 vs 7.797 (b1 vs b0) |
| fused KDA projection | same math (loss 7e-8 in the unit test) | 7.826 vs 7.797 (run-to-run spread) |
| bf16 weights | | 10.32 vs 8.49 at 8k: does not train |

Both KDA state dtypes sit at the same ~2% error against the exact scan on TPU. That floor comes from
`matmul_precision=default` (f32 matmuls run as one bf16 pass), not from the state, and it does not grow from 4k
to 8k. A 1,000-step loss comparison would confirm bf16 state over a real horizon; nothing so far points to a cost.

## i-series: split MoE saves, MEGACORE_DENSE, memory-rule test (o3v4i080335)

New knobs: `moe_x_sorted` now tags only the dispatch side (expert-local tokens after the dispatch all-to-all and
local sort, plus their routing metadata); `moe_combine` tags the combine side (the return all-to-all output that
`unpermute`'s bwd reads, and the unpermuted layer output); the a2a group sizes are tagged `moe_routing`.

| arm | change | step s | TF/s/dev | MFU | vs best | loss @19 |
|---|---|---|---|---|---|---|
| i0_s8k_comb | 8k best (h2) + `moe_combine=device` | 5.088 | 33.5 | 12.2% | -182 ms | 8.478 |
| i1_s8k_xs | 8k + `moe_x_sorted=device` | 5.165 | 33.0 | 12.0% | -105 ms | 8.462 |
| **i2_s4k_both** | 4k best (g0) + both saves | **2.079** | **40.4** | **14.7%** | **-225 ms** | 7.783 |
| i3_s4k_comb | 4k + `moe_combine=device` | 2.173 | 38.7 | 14.1% | -131 ms | 7.792 |
| **i4_s8k_comb_dense** | i0 + env `TPU_MEGACORE=MEGACORE_DENSE` | **4.950** | **34.5** | **12.5%** | **-320 ms** | 8.491 |
| i5_s8k_ep16_dense | 8k at EP 16 x FSDP 4 + dense (fits only with dense) | 5.297 | 32.2 | 11.7% | +27 ms | |
| i6_s4k_dense | 4k best + dense | 2.388 | 35.2 | 12.8% | +84 ms | |

Memory rule: i2 runs with args + temps = 39.4 GB on a 34.4 GB chip, and i0 / i1 / i4 run at 36-37 GB. The
earlier rule (args + temps must fit the chip) was too strict; XLA's compile-time check on temporaries is the
binding constraint. Several configs rejected on that rule (for example `moe_x_sorted` at 4k in the b-series) would
have run.

MEGACORE_DENSE is not a uniform win: -138 ms at 8k (i4 vs i0), +84 ms at 4k (i6 vs g0). Both MoE saves together
do not fit at 8k (AOT 33.3-33.6 G even with vocab tiling 16 or without the fused projection; `vocab_tiling_ag_once`
costs memory here).

On the reference report's accounting (91.14 TFLOPs per step), i2 is 43.8 TF/s, 15.9% MFU; on 184.75 TFLOPs at
8k, i4 is 37.3 TF/s, 13.6%.

## Reproducibility: what is archived for each series

From the j-series on, every series is launched with `scripts/olmoe3_v4_launch.sh <letter> <arms file>` and
harvested with `scripts/olmoe3_v4_harvest.sh <run>`. Both write to `gs://agagik-us/olmo35/v4/runs/<run>/`:

| file | content |
|---|---|
| `src.tgz` | the exact source tree the pods ran (first on `PYTHONPATH`; the image only supplies dependencies) |
| `arms.txt` | the arms: `name\|pdb\|seq\|MaxText flags\|env`, separated by `;` (also in `benchmarks/olmoe3_v4/`) |
| `jobset.yaml` | the rendered JobSet: image, libtpu flags, and the full train command per arm |
| `hc/results.tsv` | one row per arm, median of the last 10 of 20 steps |
| `hc/olmoe3-3p5b-<arm>.log` | the full MaxText log of each arm |
| `hc/*.xplane.pb` | profiles (arms with `profiler=xplane`), from the JAX process-0 pod |
| `pod.log` | the leader pod's stdout |

The earlier series (w through i) were backfilled with `jobset.yaml`, `pod.log`, `arms.txt` and their profiles.
Their exact source is not archived; `gs://agagik-us/olmo35/v4/runs/src-through-j-series.tgz` is the tree as of
the j-series, which contains every flag those series used (new code paths are all off by default).

Profiles are analyzed with xla-shell plus two scripts in this tree:

    PYTHONPATH=<xla-shell> python -m xla_shell -c "read_xplane X.xplane.pb; analyze_profile; roadmap --all"
    PYTHONPATH=<xla-shell> python scripts/olmoe3_v4_srcsurvey.py X.xplane.pb Xrecs.pkl
    PYTHONPATH=<xla-shell> python scripts/olmoe3_v4_components.py X.xplane.pb Xrecs.pkl "MoE ragged all-to-all"

## j-series: confirm the i-series bests, safe buffer, profiles (o3v4j080423)

| arm | change | step s | TF/s/dev | loss @19 |
|---|---|---|---|---|
| j0_s4k_both | i2 repeated (4k best) | 2.078 | 40.5 | 7.783 |
| j1_s4k_both_dense | j0 + `TPU_MEGACORE=MEGACORE_DENSE` | 2.140 | 39.3 | 7.788 |
| j2_s4k_both_rbf125 | j0 with `ragged_buffer_factor=1.25` | 2.179 | 38.6 | 7.817 |
| j3_s8k_comb_dense | i4 repeated (8k best) | 4.948 | 34.5 | 8.491 |
| j4_s8k_comb_dense_rbf125 | j3 with buffer 1.25 | 5.161 | 33.1 | 8.493 |
| j5_s8k_xs_dense | 8k, `moe_x_sorted` instead of `moe_combine` | 5.067 | 33.7 | 8.467 |
| j6_prof4k / j7_prof8k | profiles of j0 / j3 | 2.080 / 4.952 | | |

Both bests repeat to 1-2 ms. The safe buffer 1.25 costs +101 ms at 4k and +213 ms at 8k.

## Overnight hill climb (2026-10-08)

The levers below came from the j6/j7 profiles and from the reference report. Each series ran on its own v4-128
slice (two at a time); arms files are in `benchmarks/olmoe3_v4/v4_arms_<letter>.txt`.

### Lever 1: megablox runs on one of v4's two TensorCores (k-series, o3v4k080459)

v4 exposes each chip as one megacore device with two TensorCores. A Pallas kernel only spreads over both cores
along grid axes marked `parallel`. Megablox marks only the n-tile axis parallel (m tiles revisit outputs at group
boundaries, so they are `arbitrary`). Our tiles put the whole n dimension in one tile (mlp 1792, embed 768), so
every expert GEMM ran on one core. Halving the n tile fixes it:

| GMM | tile (m, k, n) before | after |
|---|---|---|
| wi fwd | 512, 768, 1792 | 512, 768, **896** |
| wi dlhs | 512, 1792, 768 | 512, 1792, **384** |
| wi drhs | 512, 768, 896 | unchanged (already 2 n-tiles) |
| wo fwd | 512, 1792, 768 | 512, 1792, **384** |
| wo dlhs | 512, 768, 1792 | 512, 768, **896** |
| wo drhs | 512, 896, 768 | 512, 896, **384** |

Flags: `benchmarks/olmoe3_v4/tiles_megacore.txt`. Local bench (`scripts/megablox_megacore_bench.py`, one v4
device, 8 experts x 9,216 rows, the EP 64 shape):

| tiles | fwd ms | fwd + bwd ms | fwd + bwd TF/s |
|---|---|---|---|
| one n-tile, m 512 (before) | 5.760 | 13.716 | 133 |
| one n-tile, m 256 | 5.907 | 14.146 | 129 |
| two n-tiles, m 512 (adopted) | 3.546 | **9.524** | **192** |
| two n-tiles, m 1024 fwd | 3.477 | 9.909 | 184 |
| n 256 (7 tiles, odd split) | 4.851 | 12.543 | 146 |
| m 2048 | VMEM OOM (22.6 of 16 MiB) | | |

| arm | change | step s | TF/s/dev | loss @19 |
|---|---|---|---|---|
| **k0_s4k_mc** | j0 + megacore tiles | **1.877** | **44.8** | 7.783 |
| **k1_s8k_mc_dense** | j3 + megacore tiles | **4.559** | **37.4** | 8.491 |
| k2_s4k_mc_prefuse | k0 + `prefuse_moe_weights=True` | 1.933 | 43.5 | 7.052 |
| k3_prof4k_mc / k4_prof8k_mc | profiles of k0 / k1 | 1.877 / 4.556 | | |

-201 ms at 4k and -389 ms at 8k, loss unchanged. In the profile megablox fell from 425 to 252 ms at 4k (now about
83% of MXU peak over the 11 GEMM passes per layer). Prefused gate/up weights are slower and change the
initialization (different loss), so they are out. This also settles the fused LatentMoE question for v4: the
reference report's fused and hybrid kernels reach 137-152 TF/s fwd+bwd against an XLA `ragged_dot` baseline of
82-89 TF/s, while megablox with megacore tiles is at 192 TF/s on the same shape class.

### Lever 2: expert-major all-to-all (l- and m-series): correct, faster on 4 chips, slower on 64

`moe_a2a_expert_major=True` sends one ragged all-to-all slice per (destination shard, local expert) instead of
one per destination, so tokens arrive grouped by local expert and the local-permute row gathers disappear on both
sides (at 8k these hit the CMEM cliff). Params in `RoutedMoE.get_expert_major_all_to_all_params`; tested against
a simulated all-to-all in `moe_test.py` and on 4 v4 devices, including buffer truncation. A paired custom VJP
(`_ragged_a2a_paired`) sends the cotangent back with locally known offsets, skipping the generic transpose's
all-to-all of the offsets; it matches the generic transpose bit for bit.

| measurement | sender-major + local permute | expert-major |
|---|---|---|
| local, 4 devices, 65K rows/device, fwd+bwd | 9.20 ms | 6.13 ms |
| local, 4 devices, 131K rows/device, fwd+bwd | 20.53 ms | 8.81 ms |
| l0 / m2: 64 chips, 4k step | 1.877 s (k0) | 2.044 s / 2.002 s (paired) |
| l1 / m3: 64 chips, 8k step | 4.559 s (k1) | 4.714 s / 4.589 s (paired) |
| profile, ragged all-to-all, 4k | 314 ms | 514 ms |
| profile, ragged all-to-all, 8k | 838 ms | 1,380 ms |
| profile, local-permute gathers, 4k / 8k | 52 / 389 ms | 0 / 0 |

With 64 peers the all-to-all carries 512 slices per device instead of 64, and the TPU ragged all-to-all cost
grows with the slice count far more than the gathers it removes. It stays off. It may pay at small expert
parallelism (the 4-device result) or with a different collective implementation.

### Lever 3: dense routing ops (m-series, o3v4m080536)

Three small routing ops lowered to slow gather/scatter kernels (j6 profile: 33 + 28 + 18 ms at 4k):

| op | before | after (under `moe_lean_routing`) | local fwd+bwd, 4096 tokens |
|---|---|---|---|
| top-k weights | `take_along_axis` (gather; scatter in bwd) | compare-select-sum over experts | 0.517 -> 0.289 ms |
| group sizes | `jnp.bincount` (scatter) | compare-and-sum | 0.454 -> 0.191 ms |
| local expert ids | `jnp.repeat` (gather) | count group ends per row | 0.183 -> 0.155 ms |

All three are exact (`scripts/moe_routing_dense_bench.py` asserts equality; unit tests pass).

| arm | change | step s | TF/s/dev |
|---|---|---|---|
| **m0_s4k_dense_route** | k0 + dense routing | **1.833** | **45.9** |
| **m1_s8k_dense_route** | k1 + dense routing | **4.446** | **38.4** |
| m2_s4k_em_paired | m0 + expert-major with paired VJP | 2.002 | 42.0 |
| m3_s8k_em_paired | m1 + expert-major with paired VJP | 4.589 | 37.2 |
| m4_prof4k / m5_prof8k | profiles of m0 / m1 | 1.835 / | |

-44 ms at 4k, -113 ms at 8k. On the report's accounting m0 is 49.7 TF/s, **18.1% MFU** (report best on the same
slice size: 1.995 s, 16.6%), and m1 is 41.6 TF/s, 15.1%.

The loss at step 19 moves by up to 0.012 between these arms even where the change is exact. Step 0 matches to the
printed digits and the curves part from step 2: the gradient sums reassociate (for example the dense top-k
weights' gradient fuses with the router's other gradient terms), and on synthetic data early training amplifies
that. Exactness of each change is pinned by unit tests, not by the step-19 loss.

### Lever 4: token-chunked dispatch at 8k (n-series, o3v4n080602)

`moe_a2a_token_chunks=N` runs the all-to-all MoE path over N sequence chunks. Routing runs once over the full
sequence first (EMo pools are per document), and each chunk reuses its slice as forced experts; the
load-balance loss also comes from the full routing. OLMoE3's `get_topk` skips the EMo mask for forced experts,
which already lie inside their pools. Test: `test_a2a_token_chunks_match_unchunked` (a document spans both
chunks). The point is buffer size: per chunk every permute gather has a seq-4096-sized source, under v4's
128 MiB CMEM.

| arm | change | step s | TF/s/dev | loss @19 |
|---|---|---|---|---|
| **n0_s8k_tc2** | m1 + 2 token chunks | **3.923** | **43.5** | 8.506 |
| n1_s8k_tc4 | m1 + 4 token chunks | 3.925 | 43.5 | 8.507 |
| n2_s4k_tc2 | m0 + 2 token chunks | 1.852 | 45.4 | 7.743 |
| n3_prof8k_tc2 | profile of n0 | 3.917 | | |

-523 ms at 8k (the biggest single 8k lever); 4 chunks add nothing over 2. At 4k the buffers already fit CMEM, so
chunking only adds per-call overhead (+19 ms): use it at 8k only.

### Lever 5: sort-free local permute (o-series, o3v4o080611)

After the sender-major dispatch the received rows are blocks (source shard, local expert); `local_permute`
reorders them to (local expert, source) with a `jnp.repeat` gather and an argsort, and the combine argsorts
again for the inverse. `_block_transpose_indices` builds the same gather indices and their inverse from
compare-and-sum passes over the 512 block ends (2x faster at 74K rows locally,
`scripts/moe_routing_dense_bench.py` style check in `test_block_transpose_indices_match_local_permute`, exact
with and without buffer truncation). The rows then move with `_permute_rows`, whose bwd uses the known inverse
instead of argsorting. On under `moe_lean_routing`.

| arm | change | step s | TF/s/dev |
|---|---|---|---|
| **o0_s4k_sortfree** | m0 + sort-free local permute | **1.800** | **46.7** |
| o1_s4k_sortfree_wy | o0 + `kda_wy=device` (save KDA inverse and scores) | 1.817 | 46.3 |
| **o2_s8k_tc2_sortfree** | n0 + sort-free local permute | **3.867** | **44.1** |

-33 ms at 4k, -56 ms at 8k. `kda_wy` (new remat name on the KDA inverse and masked scores, 34 MB per layer,
fits: AOT temps 27.71 GB) is +17 ms: the remat forward it saves is cheaper than the extra saved traffic.

### Compiler flags at 4k (p-series, o3v4p080612; base o0, 1.800 s)

| arm | libtpu flags | step s | vs o0 |
|---|---|---|---|
| p0 | `--xla_enable_async_all_gather=true` | 1.881 | +81 ms |
| p1 | p0 + `--xla_tpu_overlap_compute_collective_tc=true` | 1.882 | +82 ms |
| p2 | `--xla_enable_async_all_reduce=true` | 1.804 | +4 ms (neutral) |
| p3 | `--xla_tpu_enable_windowed_einsum_for_all_gather=true --xla_tpu_enable_windowed_einsum_for_reduce_scatter=true` | 1.801 | neutral |
| p4 | `--xla_tpu_megacore_fusion_allow_ags=false` | 1.802 | neutral |

The 4k profile shows the non-expert weight all-gathers (FSDP over the 64-way expert axis, about 60 ms) as
synchronous ops; making them async moves them but costs more than it hides.

### Lever 6: Pallas KDA state scan with a hand-written backward (q-series, o3v4q080629)

The chunk-state recurrence was an XLA while loop at ~27 us per chunk for ~5 us of work.
`src/maxtext/kernels/kda_scan.py` runs it as one Pallas kernel per (batch x head) with the state resident in
VMEM (grid `(B*H parallel, chunks arbitrary)`, so the two megacore TensorCores split the heads), plus a reverse
kernel for the backward that carries only dL/dS and leaves the per-chunk gradients to batched einsums
(`kda_pallas_scan=True`). Forward is bit-identical to the bf16 scan; gradients sit 0.49-0.77% from an f32
reference against autodiff's 0.54-0.81% (`test_pallas_state_scan_matches_lax_scan`).

| measurement | XLA scan | Pallas kernel |
|---|---|---|
| local, one device, T 4096: fwd / fwd+bwd | 0.747 / 3.037 ms | 0.504 / 1.167 ms |
| local, one device, T 8192: fwd / fwd+bwd | 1.653 / 6.158 ms | 0.769 / 2.589 ms |
| local, one-layer model fwd+bwd (4 devices) | 329 ms | 174 ms |
| q0 / q2 (sub-block 8): 64 chips, 4k | 1.800 s (o0) | 1.851 / 1.849 s |
| q1 / q3 (sub-block 8): 64 chips, 8k | 3.867 s (o2) | 3.950 / 3.937 s |

Slower in the full model. Per source line (m4 vs q4 profiles), the scan itself drops from 56 to 23 ms per step,
but the ops that feed it grow by about 36 ms: the WY `rhs` concat (4 -> 22 ms), the `uw` einsum (6 -> 11), the
q/k/v casts (5 -> 9) and the q L2-norm (1 -> 6). The custom call pins row-major operand layouts, so XLA
materializes and relayouts tensors it used to fuse into the loop's inputs. Making it pay needs the producers
inside the kernel (read `uw` and `k_c`, `g` directly and build `w`, `k_carry` in VMEM) or the readout fused in.
Off for now; the kernel and its test stay in the tree. Sub-block 8 is worth 2 ms.

### r-series: paired all-to-all, vocab all-gather once, no host offload at 8k (o3v4r080639)

| arm | change | step s | vs base |
|---|---|---|---|
| r0_s4k_paired | o0 + paired VJP on the sender-major all-to-alls (bwd sends with locally known offsets) | 1.800 | 0 |
| r1_s8k_tc2_paired | o2 + paired VJP | 3.873 | +6 ms |
| r2_s4k_paired_vtago | r0 + `vocab_tiling_ag_once=True` | 1.830 | +30 ms |
| r3_s8k_tc2_nooffload | r1 with `decoder_layer_input=device` and no `moe_combine` (both do not fit) | 4.082 | +215 ms |

The paired VJP is exact (bit-identical gradients on 4 devices, with and without buffer truncation) and stays
on under `moe_lean_routing`, but it buys nothing: the offsets all-to-all it removes was not on the critical path.
At 8k, saving the combine output and offloading the layer inputs beats keeping the inputs on device.

### Why 8k scaled worse than 2x: the dispatch all-to-all is recomputed

In the r5 profile (8k, 3.87 s) the dispatch all-to-all costs 515 ms against 168 ms at 4k (3.1x for 2x tokens),
while the combine all-to-all scales 2x (339 vs 168 ms). At 8k only `moe_combine` is saved (`moe_x_sorted` does
not also fit), so the bwd reruns the dispatch all-to-all: three dispatch calls per layer instead of two.

### Lever 7: async ragged all-to-all over token chunks (s- and v-series, o3v4s080705, o3v4v080729)

`--xla_tpu_enable_async_ragged_all_to_all=true` was neutral in the a-series (one dispatch per layer: nothing
independent to overlap). With `moe_a2a_token_chunks=2` the two chunks are independent, so one chunk's
all-to-all overlaps the other chunk's expert GEMMs.

| arm | change | step s | TF/s/dev | vs base |
|---|---|---|---|---|
| s0_s8k_tc2_nodense | r1 without `TPU_MEGACORE=MEGACORE_DENSE` | 3.926 | 43.5 | +53 ms |
| s1_s4k_dense | r0 with MEGACORE_DENSE | 1.881 | 44.7 | +81 ms |
| **s2_s8k_tc2_async_ra2a** | r1 + async ragged all-to-all | **3.516** | **48.5** | **-357 ms** |
| u0_s4k_vt2 | r0 with `num_vocab_tiling=2` | 1.781 | 47.2 | -19 ms |
| **v0_s4k_tc2_async** | u0 + 2 token chunks + async ragged all-to-all | **1.674** | **50.2** | **-107 ms** |
| v1_s4k_async | u0 + async ragged all-to-all, no chunks | 1.798 | 46.7 | +17 ms |

Chunks alone cost +19 ms at 4k (n2) and async alone +17 ms (v1); together they are -107 ms. On the report's
accounting v0 is 54.4 TF/s, 19.8% MFU, and s2 is 52.5 TF/s, 19.1%.

### u-series: small knobs at 4k (o3v4u080715, base r0 1.800 s)

| arm | change | step s |
|---|---|---|
| **u0_s4k_vt2** | `num_vocab_tiling=2` | **1.781** |
| u1_s4k_vt8 | `num_vocab_tiling=8` | 1.835 |
| u2_s4k_sb8 | `kda_sub_block=8` | 1.809 |
| u3_s4k_c256 | `gdn_chunk_size=256` | 1.934 |
| u4_s8k_tc2_vt16 | 8k (r1) with `num_vocab_tiling=16` | 3.878 |
| v2_s4k_tc4_async | v0 with 4 chunks | 1.682 | 50.0 | +8 ms vs v0 |
| **v3_s8k_tc4_async** | s2 with 4 chunks | **3.379** | **50.5** | **-137 ms vs s2** |

At 8k four chunks beat two once the all-to-all is async (more, smaller exchanges to overlap); at 4k two suffice.
The v-series profiles (v4, v5) and w3 were lost: I deleted those JobSets before their last arms finished. The
harvest script now warns when the leader log has no `END` line. The y-series profiles replace them.

### w- and x-series: 8k offload variants, expert-major with overlap (o3v4w080737, o3v4x080744)

| arm | change | step s | TF/s/dev | loss @19 |
|---|---|---|---|---|
| w0_s8k_tc2a_xs_off | s2 + `moe_x_sorted=offload` (host) | 4.155 | 41.1 | 8.495 |
| w1_s8k_tc2a_xs_dev_comb_off | s2 with `moe_x_sorted=device moe_combine=offload` | 4.007 | 42.6 | 8.499 |
| w2_s4k_tc2a_vt1 | v0 with `num_vocab_tiling=1` | 1.672 | 50.3 | 7.745 |
| **x0_s4k_tc2a_em** | v0 + `moe_a2a_expert_major=True` | **1.656** | **50.7** | 7.757 |
| **x1_s8k_tc2a_em** | s2 + expert-major | **3.348** | **51.0** | 8.515 |
| x2_s4k_tc2a_pipe | v0 + `--xla_tpu_enable_ragged_all_to_all_pipelined_local_copy=true` | | | 7.789 |

Host offload of the MoE buffers is far too slow (6.6 GB per step over PCIe). The expert-major all-to-all, a
loss on its own (l/m-series: its 512 slices per device slow the exchange), wins once the exchange overlaps
compute: its extra all-to-all time hides behind the other chunk's GEMMs, and the local-permute gathers it removes
do not. -18 ms at 4k, -168 ms at 8k.

### y-, z- and a-series: more chunks, expert-major at 4 chunks, memory knobs (o3v4y080801, o3v4z080801, o3v4a080830)

| arm | change | step s | TF/s/dev | vs base |
|---|---|---|---|---|
| **y0_s8k_tc4a_em** | v3 (8k, 4 chunks, async) + expert-major | **3.305** | **51.6** | **-74 ms vs v3** |
| y1_s8k_tc8a | s2 with 8 chunks | 3.468 | 49.2 | +89 ms vs v3 |
| z0_s8k_tc4a_em_xs | y0 with `moe_x_sorted` instead of `moe_combine` | 3.402 | 50.1 | +97 ms |
| z1_s8k_tc4a_em_nodense | y0 without MEGACORE_DENSE | 3.826 | 44.6 | +521 ms |
| z2_s8k_tc4a_em_vt16 | y0 with vocab tiling 16 | 3.302 | 51.7 | -3 ms (noise) |
| z3_s4k_tc2a_em_comb_only | x0 without `moe_x_sorted` | 1.723 | 48.8 | +67 ms |
| a0_s4k_best_dense | x0 + MEGACORE_DENSE | 1.704 | 49.4 | +48 ms |

On the report's accounting y0 is 55.9 TF/s, 20.3% MFU. MEGACORE_DENSE is worth 521 ms at 8k: the extra GiB is
what lets XLA keep the async exchanges' buffers without rematerializing.

## Overnight summary (2026-10-08)

| step | change | 4k s | 8k s | 4k MFU (report) | 8k MFU (report) |
|---|---|---|---|---|---|
| start | i/j-series best | 2.078 | 4.948 | 15.9% | 13.6% |
| k | megablox tiles split for both megacore TensorCores | 1.877 | 4.559 | 17.6% | 14.7% |
| m | dense routing ops (no gather/scatter for top-k weights, group sizes) | 1.833 | 4.446 | 18.1% | 15.1% |
| n | token-chunked dispatch, routed once (8k) | | 3.923 | | 17.1% |
| o | sort-free local permute | 1.800 | 3.867 | 18.4% | 17.4% |
| u | vocab tiling 2 (4k) | 1.781 | | 18.6% | |
| s, v | async ragged all-to-all over the chunks (4k: 2 chunks, 8k: 4) | 1.674 | 3.379 | 19.8% | 19.9% |
| x, y | expert-major all-to-all, hidden by the overlap | **1.656** | **3.305** | **20.0%** | **20.3%** |

-422 ms (-20%) at 4k and -1.64 s (-33%) at 8k overnight; the 8k step now costs the same per token as 4k. The
code changes are all exact (unit tests in `tests/unit/moe_test.py` and `tests/unit/olmoe3_test.py`); the
step-19 losses stay within the reassociation spread (4k 7.757-7.795, 8k 8.455-8.515).

## Where the step goes at the bests (y4, y5 profiles)

Profiles: `gs://agagik-us/olmo35/v4/runs/o3v4y080801/hc/olmoe3-3p5b-y4_prof4k_tc2a_em.xplane.pb` (4k) and `...y5_prof8k_tc4a_em.xplane.pb` (8k); `scripts/olmoe3_v4_components.py` buckets ops by source line.

| component | 4k ms (y4, 1.66 s) | share | 8k ms (y5, 3.31 s) |
|---|---|---|---|
| MoE expert GEMMs (megablox) | 274.8 | 16.6% | 550.6 |
| KDA elementwise, norms, conv, gates | 215.8 | 13.0% | 490.7 |
| collectives | 177.0 | 10.7% | 180.3 |
| MoE ragged all-to-all | 174.3 | 10.5% | 403.1 |
| KDA matmuls (intra-chunk, inverse, scan) | 164.7 | 10.0% | 343.6 |
| dense matmuls (projections) | 141.8 | 8.6% | 287.0 |
| copies / relayout | 119.4 | 7.2% | 202.0 |
| other (loop fusion) | 107.5 | 6.5% | 149.5 |
| MoE other (router, EMo, combine, masks) | 96.6 | 5.8% | 208.5 |
| LM head + loss | 84.9 | 5.1% | 196.9 |
| MoE routing, gather, scatter | 28.4 | 1.7% | 75.9 |
| full attention (splash) | 25.1 | 1.5% | 94.6 |
| norms, residuals | 19.9 | 1.2% | 63.0 |
| other (broadcast) | 12.8 | 0.8% | 23.7 |
| MoE top-k + sorts | 8.2 | 0.5% | 17.0 |
| total | 1654.6 |  | 3300.6 |

xla-shell `analyze_profile` on y4: TensorCore lane 1.41 s of 1.66 s (85%): matmul 669 ms, VPU 504 ms, relayout
233 ms; best-overlap ceiling 1.41 s (1.18x). Matmul work is 40% of the step; the rest is VPU and data movement
a FLOP count does not see.

What is left, by size at 4k:

| item | ms | what would move it |
|---|---|---|
| KDA (intra-chunk pair tensor and diag einsum 55 + 47 ms, scan 56, glue) | 380 | a Pallas intra-chunk kernel that keeps the per-pair tensor in VMEM (about 1 GB per layer of HBM traffic today); the scan kernel only pays if its producers move inside it (q-series); GDN scalar decay as co-design (2.1x at the core) |
| megablox expert GEMMs | 275 | at about 83% of MXU peak over 11 GEMM passes per layer; the SwiGLU elementwise (48 ms, `moe.py:1368`) is the only fat, and fusing it means a fused backward too (h0/h1 are needed there) |
| non-expert weight collectives | 177 | all-gathers of the 64-way sharded projections (55 ms at `linears.py:453`) and the fused KDA projection's gradient, which XLA all-reduces in full (`kda_fused_proj_reduce_scatter`, d-series) |
| ragged all-to-all (exposed part) | 174 | the rest is hidden behind the chunks; more chunks stop paying at 4 (8k) and 2 (4k) |
| dense projections | 142 | at about 1/3 of MXU peak; small K/N per projection |
| copies and relayouts | 119 | memory-space copies of the 113 MB MoE buffers (`bf16[73728,2,384]`, CMEM prefetch/evict) and the RMSNorm reshape |
| loop fusions | 108 | synthetic batch generator (~30 ms, gone with real data), gradient/param norms for metrics (29 ms), scan stacking |
| LM head + loss | 85 | vocab tiling 2 trades 17 ms here for less elsewhere |

## Last checks on the bests (b-, c-, d-series: o3v4b080846, o3v4c080906, o3v4d080916)

| arm | change | step s |
|---|---|---|
| best_4k / best_4k_prof | `best_arms.txt` relaunched from a fresh snapshot | 1.656 / 1.657 |
| best_8k / best_8k_prof | same | 3.308 / 3.301 |
| c0 | best 4k + `--xla_enable_async_all_gather=true` | 1.704 |
| c1 / c2 | best 4k + `--xla_tpu_async_ragged_all_to_all_max_rdma_size_kib=256` / `1024` | 1.660 / 1.659 |
| c3 | best 8k + RDMA size 1024 | 3.308 |
| d0 / d1 / d2 | bests + `kda_fused_proj_reduce_scatter=True` (d2 also vocab 4 at 8k) | 1.659 / 3.304 / 3.309 |
| a1 (a-series) | best 8k with `num_vocab_tiling=4` | 3.295 |

The bests reproduce to 3-7 ms. The remaining flags are neutral; vocab tiling 4 at 8k (-10 ms) is within that
spread and is left as an option.

## Afternoon: MEGACORE_DENSE at 4k, and 20% on MaxText's own FLOP count (2026-10-08)

Target: 20% of 275 TF/s on the TF/s MaxText logs (stricter than the report's 91.14 / 184.75 TF per step): the
4k step must reach 1.527 s and the 8k step 3.10 s. Only loss-neutral levers.

### Why MEGACORE_DENSE lost at 4k (e-series o3v4e081412, profile e0 vs b4)

| arm | change (base: best 4k, 1.656 s) | step s |
|---|---|---|
| e0_s4k_dense_prof | + MEGACORE_DENSE | 1.709 |
| e1_s4k_dense_vt1 | + MEGACORE_DENSE, vocab tiling 1 | 1.681 |
| e2_s4k_dense_wy | + MEGACORE_DENSE, `kda_wy=device` | 1.705 |
| e3_s4k_dense_tc4 | + MEGACORE_DENSE, 4 token chunks | 1.706 |

Profile diff (e0 vs b4): collectives +25 ms, KDA elementwise +16, loop fusions +11. With the extra GiB, XLA turns
the non-expert weight all-gathers async on its own (`all-gather-done` 41 ms where b4 has a 28 ms synchronous
`all-gather`), the same loss the async all-gather flag showed (p0, c0). Forcing them back with
`--xla_enable_async_all_gather=false` is what makes MEGACORE_DENSE usable at 4k.

### f- and g-series: unrolled layers, synthetic batch, norms, sync all-gather (o3v4f081417, o3v4g081437)

| arm | change | step s | vs base |
|---|---|---|---|
| f0_s4k_synreuse | best 4k + `synthetic_data_reuse_batch=True` | 1.653 | -3 ms |
| f1_s8k_synreuse | best 8k + same | 3.296 | -9 ms |
| **f2_s4k_noscan** | best 4k + `scan_layers=False` | **1.621** | **-35 ms** |
| **f3_s8k_noscan** | best 8k + `scan_layers=False` | **3.215** | **-90 ms** |
| f4_s4k_nofusedproj | best 4k with `kda_fused_input_proj=False` | 1.689 | +33 ms |
| g0_s4k_cur | best 4k on the current tree (one all-gather for all chunks' group sizes) | 1.651 | -5 ms |
| g1_s4k_dense_syncag | g0 + MEGACORE_DENSE + sync all-gather | 1.665 | +14 ms |
| g2_s4k_dense_syncag_vt1 | g1 + vocab tiling 1 | 1.647 | -4 ms |
| g3_s8k_syncag | best 8k + sync all-gather | 3.709 | +404 ms |
| g4_s4k_normgrad | g0 + `norm_metrics=grad` | 1.631 | -20 ms |
| g5_s8k_normgrad | best 8k + `norm_metrics=grad` | 3.303 | -2 ms |

New loss-neutral knobs: `synthetic_data_reuse_batch` (the synthetic batch is the same array every step; reuse it
instead of resharding it), `norm_metrics=grad` (log the raw grad norm that gradient clipping already computes,
derive the clipped norm from it, skip the param norm: training unchanged, two fewer full passes over the
parameters), and one all-gather of every chunk's per-expert counts in the chunked expert-major path. Unrolled
layers drop the scan's per-cycle parameter slicing and gradient stacking (`nnx_decoders.py:1247`, about 50 ms at
4k). At 8k the weight all-gathers must stay async (g3: +404 ms).

### h-series: unrolled + MEGACORE_DENSE at 4k (o3v4h081502)

| arm | change | step s | TF/s/dev | MFU (logged) |
|---|---|---|---|---|
| **h0_s4k_ns_dense_vt1** | f2 + MEGACORE_DENSE + sync all-gather + vocab tiling 1 | **1.562** | **53.8** | **19.6%** |
| h1_s4k_ns_dense | f2 + MEGACORE_DENSE + sync all-gather (vocab 2) | 1.618 | 52.0 | 18.9% |

With the layers unrolled, MEGACORE_DENSE (plus sync all-gathers) and vocab tiling 1 together are worth -59 ms.
| h2_s4k_ns_vt1 | f2 + vocab tiling 1, no MEGACORE_DENSE | 1.562 | 53.8 | 19.6% |
| h3_s4k_ns_dense_vt1_ng | h0 + `norm_metrics=grad` | 1.561 | 53.9 | 19.6% |

h0 and h2 tie: with sync all-gathers MEGACORE_DENSE is neutral at 4k, and it stays on so both configs share it.

### Unrolled layers free most of the memory

AOT (`compile_topology=v4-128`) temporaries with `scan_layers=False`: 11.45 GB at 4k (27.5 GB scanned) and 13.69 GB
at 8k (24.7 GB scanned). Scanning keeps every cycle's stacked residuals and gradient buffers live at once; unrolled,
XLA frees them layer by layer. That headroom pays for the saves that did not fit before (with every KDA save on,
4k is still only 18.0 GB).

### i-, j-, k-series: 8k unrolled, attention kernels, saves (o3v4i081522, o3v4j081553, o3v4k081604)

| arm | change (8k unless noted) | step s | TF/s/dev | MFU (logged) |
|---|---|---|---|---|
| i0_s8k_ns_vt4 | f3 + vocab tiling 4 | 3.213 | 53.1 | 19.3% |
| i1_s8k_ns_vt4_ng | i0 + `norm_metrics=grad` + synthetic reuse | 3.205 | 53.2 | 19.3% |
| i2_s8k_ns_tc2 | f3 with 2 chunks | 3.352 | 50.9 | |
| i3_s8k_ns_vt2 | f3 + vocab tiling 2 | 3.213 | 53.1 | |
| j0_s8k_ns_dkvmc | i0 + `sa_bwd_dkv_megacore=True` | 3.220 | 53.0 | |
| j1_s8k_ns_fusedbwd | i0 + `sa_use_fused_bwd_kernel=True` | 3.214 | 53.1 | |
| j2_s8k_ns_sa1024 | i0 + splash blocks 1024 | VMEM OOM (16.31 of 16 MiB) | | |
| j3 (4k combo) | aborted at TPU client start (infra), not rerun | | | |
| k0_s8k_ns_xs | i0 + `moe_x_sorted=device` | 3.127 | 54.6 | 19.9% |
| k1_s8k_ns_xs_dli | k0 + `decoder_layer_input=device` (no host offload) | 3.106 | 54.9 | 20.0% |
| k2_s8k_ns_xs_dli_ctx | k1 + `context=device` | HBM OOM (36.7 G temps) | | |
| **k3_s8k_ns_xs_dli_wy** | k1 + `kda_wy=device` | **3.103** | **55.0** | **20.0%** |

With both MoE saves the bwd no longer reruns the dispatch all-to-all, and the layer inputs no longer cross PCIe.

### l- and m-series: 4k saves (o3v4l081655, o3v4m081706; base h0 1.562 s)

| arm | change | step s | TF/s/dev | MFU (logged) |
|---|---|---|---|---|
| l0_s4k_ns_ctx | h0 + `context=device` (KDA chunked inputs and WY products, attention output) | 1.542 | 54.5 | 19.8% |
| l1_s4k_ns_wy | h0 + `kda_wy=device` | 1.562 | 53.8 | |
| **m0_s4k_ns_mlpwi0** | h0 + `moe_mlpwi_0=device` (routed gate GMM output) | **1.523** | **55.2** | **20.1%** |
| l2_s4k_ns_proj | h0 + `query_proj key_proj value_proj=device` | 1.560 | 53.9 | |
| l3_s4k_ns_all | h0 + `context`, `kda_wy` and the projections | 1.543 | 54.5 | |
| m1_s4k_ns_mlpwi01 | m0 + `moe_mlpwi_1=device` | HBM OOM (34.2 G temps) | | |
| **o0_s4k_m0_ng_sr** | m0 + `norm_metrics=grad` + synthetic reuse | **1.520** | **55.3** | **20.1%** |

Saving the routed gate GMM output (`moe_mlpwi_0`, 264 MB per layer) skips one of the two expert GEMMs the bwd
reran and its share of the SwiGLU recompute: -39 ms. Both GMM outputs, or `moe_mlpwi_0` with `context`
(41.9 G), do not fit. Of the KDA saves only `context` pays (-20 ms) and not on top of `moe_mlpwi_0`.

n-series (o3v4n081726): n0 8k k3 + `moe_mlpwi_0` needs 31.86 G (0.11 G over); n2 4k m0 + `context` needs
41.9 G. n1 and n3 aborted in `make_tpu_client` right after a failed arm (as j3 and m2 did). The launcher now
retries an arm once, after a 60 s pause, when it aborts there, and pauses 30 s after any failed arm.

### o-, q-, r-series: confirmations and profiles of the 20% configs (o3v4o081736, o3v4q081755, o3v4r081817)

| arm | change | step s | TF/s/dev | MFU (logged) |
|---|---|---|---|---|
| o1_s4k_m0_prof | m0 profiled | 1.523 | 55.2 | 20.1% |
| o2_s8k_k3_prof | k3 profiled | 3.106 | 54.9 | 20.0% |
| q0_s4k_o0_tc4 | o0 with 4 chunks | 1.539 | 54.6 | |
| q1_s4k_o0_noem | o0 without expert-major | 1.566 | 53.7 | |
| q2_s4k_o0_mlpwi1 | o0 saving the up GMM instead of the gate GMM | 1.526 | 55.1 | |
| **r0_s8k_k3_ng_sr** | k3 + `norm_metrics=grad` + synthetic reuse | **3.095** | **55.1** | **20.0%** |

p-series (o3v4p081754): 8k k1 + `moe_mlpwi_0` without `kda_wy` needs 41.5 G (with `kda_wy` it was 31.86 G: XLA's
schedule, not the saved bytes, decides), and the arms after it aborted in TPU start-up even with the retry.

## Where the 20% steps go (o1, o2 profiles) and what is next

| component | 4k ms (o1, 1.52 s) | 8k ms (o2, 3.11 s) |
|---|---|---|
| MoE expert GEMMs (megablox) | 254 | 565 |
| KDA elementwise, norms, conv, gates | 237 | 507 |
| collectives | 187 | 145 |
| KDA matmuls | 151 | 326 |
| MoE ragged all-to-all (exposed) | 143 | 334 |
| dense matmuls | 136 | 346 |
| MoE other (SwiGLU, router, EMo) | 84 | 200 |
| copies / relayout | 83 | 164 |
| LM head + loss | 76 | 162 |
| loop fusions | 73 | 93 |
| full attention (splash) | 25 | 94 |

xla-shell: TensorCore lane 1.26 s of 1.52 s at 4k (1.20x best-overlap headroom), 2.78 s of 3.10 s at 8k (1.12x).
At 4k the collectives are now the third item: the non-expert weight all-gathers (59 ms), the fused KDA
projection's kernel all-gather (40 ms) and its gradient all-reduce (36 ms, a full all-reduce where a
reduce-scatter would move half), and the embedding table gathers (19 + 19 ms).

Next loss-neutral levers, by expected size:

| lever | where | expected |
|---|---|---|
| fused KDA projection gradient as reduce-scatter (`kda_fused_proj_reduce_scatter`, t-series), or gather the six kernels without the concat | 4k collectives | up to -36 ms |
| Pallas intra-chunk KDA kernel (per-pair decay tensor stays in VMEM; today about 1 GB per layer of HBM traffic) | KDA, both | -50 ms (4k), -100 ms (8k) |
| Pallas state scan with its producers and the readout inside (the standalone kernel lost to relayouts, q/t-series) | KDA scan | -30 ms (4k), -60 ms (8k) |
| more saves where memory allows (8k: `moe_mlpwi_0` is 0.11 G over with `kda_wy`) | 8k | about -40 ms if a schedule fits |
| splash attention block tuning below 1024 (1024 runs out of VMEM) | 8k attention | -10 to -20 ms |

### s-, t-, u-series: safe buffer, projection reduce-scatter, GMM save at 8k (o3v4s081838, o3v4t081848, o3v4u081918)

| arm | change | step s | TF/s/dev |
|---|---|---|---|
| s0_s4k_o0_rbf125 | best 4k with `ragged_buffer_factor=1.25` | 1.581 | 53.2 |
| s1_s8k_k3_rbf125 | k3 (+ metric/synthetic knobs) with buffer 1.25 | 3.223 | 52.9 |
| t0_s4k_o0_projrs | best 4k + `kda_fused_proj_reduce_scatter=True` | 1.522 | 55.3 |
| t1_s8k_r0_projrs | best 8k + same | 3.098 | 55.1 |
| r1_s8k_k3_rep | k3 repeated | 3.113 | 54.8 |
| r2_s8k_k3_vt8 | r0 with vocab tiling 8 | 3.099 | 55.1 |
| u0 / u1 | best 8k + `moe_mlpwi_0`, LM head gathered per tile (vocab 4 / 8) | HBM OOM (41.2 / 41.3 G) | |

The 8k config repeats at 3.095-3.113 s (55.1-54.8 TF/s, 20.0-19.9%). The safe buffer costs 61 ms at 4k and
128 ms at 8k. The GMM save that fits at 4k does not at 8k: XLA's temporaries jump between 31.9 and 41 G with small
config changes, so it is not a lever there without schedule control.

## f32 KDA state and conv on the bests (o3v4w082116)

The bests with the KDA recurrent state and depthwise conv in float32 (`gdn_state_dtype=float32
kda_conv_in_compute_dtype=False`), the strictly loss-neutral setting (`benchmarks/olmoe3_v4/v4_arms_ours_f32.txt`).

| arm | config | step s | TF/s/dev (log) | MFU (parameter-shape count) | loss @19 |
|---|---|---|---|---|---|
| q3_s4k_ours_se | best 4k (+ `shared_experts=1`, no effect on this model) | 1.521 | 55.3 | 20.4% | 7.534 |
| q0_s4k_ours_f32 | best 4k, f32 KDA state and conv | 1.558 | 53.9 | 19.9% | 7.525 |
| q2_s4k_ours_f32_noemo | q0 with EMo off | 1.560 | 53.9 | 19.9% | 6.528 |
| q1_s8k_ours_f32 | best 8k, f32 KDA state and conv | 3.167 | 53.9 | 19.8% | 8.289 |

f32 state costs 37 ms at 4k and 72 ms at 8k. EMo costs nothing measurable (q0 vs q2; the loss differs because
the routing differs).
