# OLMo 3.5 on Ironwood: run record, profiles and xla-shell output

Every measured Ironwood number for the OLMo 3.5 partner family, one row per
configuration, with the profile and the xla-shell analysis that goes with it.
The plan and the reasoning live in `olmo35-ironwood-plan.md`; this file is the
evidence.

## Conventions

| item | value |
|---|---|
| MFU | `TF/s/device / 1153.5`. A tpu7x chip is 2307 TFLOP/s bf16 and carries **two** devices |
| step time | median of the **last 10** of 20 steps, so compile and warmup are excluded |
| data | `dataset_type=synthetic`, checkpointing off |
| precision | `dtype=bfloat16 weight_dtype=float32` |
| model | `olmo35-tiny` unless stated: d 1024, 16 layers, 512 experts top-16, latent 512, full attention every 8th layer |
| always on | `sparse_matmul`, `use_tokamax_kda`, `use_gmm_v2`, `shard_exp_on_fsdp`, `num_vocab_tiling=8`, `remat_policy=full`, `TOKAMAX_KDA_DENSE_PAIRS=1`, `TOKAMAX_KDA_BF16_FWD/BWD=1` |
| XLA flags | the Ironwood sparse-core offload set from `run_olmo3_7b_stage1.sh`, via `LIBTPU_INIT_ARGS` |

The always-on set is the winning stack from the 8-device sweep. Anything an arm
passes in its `extra` column overrides it, because MaxText takes the last
occurrence of a repeated key.

## Reference anchors

`tpu-recipes/training/ironwood`, per chip against the 2307 TFLOP/s peak. These
are the yardstick, and they say the expert layout is what decides the class.

| recipe | TF/s/chip | MFU | expert layout |
|---|---|---|---|
| llama3.1-70b | 1207 | 52.3% | dense |
| qwen3-235b-a22b | 630 | 27.3% | MoE, wide experts |
| deepseek3-671b | 608 | 26.3% | MoE, wide experts |
| **gpt-oss-120b** | **330** | **14.3%** | **MoE, many narrow experts** |
| OLMoE3-3p5b, ours | 195 | 8.45% | 512 experts top-16 |

OLMo 3.5 is 512 experts at top-16 with expert hidden = d_model, which is the
gpt-oss class, not the deepseek class.

Measured on the same stack for calibration, 8 devices, pdb=1:

| model | TF/s/dev | MFU |
|---|---|---|
| olmo3-7b dense | 277 | 24.0% |
| olmo3-7b dense, `remat=custom` | 318 | 27.6% |
| olmo3-32b dense | 349 | 30.3% |

## Run log

### 8 devices (2x2x1, single host). The lever stack

Vehicle was `bodaborg-tpu7x-spot-sps` in `cloud-tpu-multipod-dev`, plain pod
create, no Kueue. Each row is one factor changed against the row above.

| step | change | TF/s/dev | MFU | factor |
|---|---|---|---|---|
| 0 | baseline, seq 8192, pdb 1 | 43.1 | 3.7% | |
| 1 | seq 8192 -> 16384 | 56.0 | 4.9% | 1.30x |
| 2 | `use_gmm_v2=True` after the tile-clamp fix, plus the sparse-core XLA flags | 61.0 | 5.3% | 1.09x |
| 3 | `shard_exp_on_fsdp=True` | 63.5 | 5.5% | 1.04x |
| 4 | `TOKAMAX_KDA_DENSE_PAIRS=1` | 80.0 | 6.9% | **1.26x** |
| 5 | `TOKAMAX_KDA_BF16_FWD/BWD=1` | **81.5** | **7.1%** | 1.06x |

Cumulative **1.89x**. Measured neutral and dropped: `use_custom_sort_vjp`,
splash attention, host offload, `remat=custom`, tokamax GMM v1.

**bf16 KDA is a lever-order trap.** It is neutral on its own and worth 1.06x only
after `dense_pairs` removes the backward bottleneck it was hiding behind. A
one-factor-at-a-time sweep that tests it first would have discarded it.

Dispatch comparison at the default tiles that previously aborted:

| path | TF/s/dev |
|---|---|
| megablox | 43.3 |
| tokamax GMM v1 | 44.5 |
| **gmm_v2, after the clamp fix** | **45.9** |

### 128 devices (4x4x4, 16 nodes, 64 chips): run `o35nap240641`

`bodaborg-tpu7x-nap`, namespace `default`, 2026-09-24. First real 4x4x4.

| arm | pdb | seq | TF/s/dev | MFU | step s | outcome |
|---|---|---|---|---|---|---|
| a1_p1s8k | 1 | 8192 | 54.2 | 4.70% | 0.652 | ok |
| a2_p2s8k | 2 | 8192 | **78.2** | **6.78%** | 0.903 | **best** |
| a3_p4s8k | 4 | 8192 | | | | OOM, temporaries **104.02G vs 94.74G** |
| a4_p2s16k | 2 | 16384 | | | | OOM, **101.51G vs 94.74G** |
| b1_rematcustom | 4 | 8192 | 75.0 | 6.50% | 1.885 | fits only via offload |
| b2_ep4 | 4 | 8192 | | | | config error, unsharded params 24.17% |
| b3_noshardexp | 4 | 8192 | | | | OOM, 105.14G |
| b4_megablox | 4 | 8192 | | | | OOM |
| z_profile | 4 | 8192 | | | | OOM, **no capture** |

Raw logs: `gs://agagik-us/olmo35/4x4x4/o35nap240641/hc/`.

#### Why the estimator is a median and not a mean

The per-step trace for `a2_p2s8k` is not smooth, and reading a single step would
give an answer that is wrong by 2x in either direction:

| step | s | TF/s/dev | | step | s | TF/s/dev |
|---|---|---|---|---|---|---|
| 0 | 0.851 | 83.0 | | 10 | 0.906 | 78.0 |
| 1 | **0.379** | **186.3** | | 11 | **1.383** | **51.1** |
| 2 | **3.547** | **19.9** | | 12 | **0.433** | **163.3** |
| 3 | 0.902 | 78.4 | | 13 | 0.898 | 78.7 |
| 4..9 | 0.901-0.910 | 77.7-78.5 | | 14..19 | 0.901-0.906 | 78.0-78.4 |

Steady state is **0.90 s / 78.2 TF/s** and it is extremely stable, 77.7 to 78.7
across twelve steps. The outliers come in pairs, one fast then one slow, which is
the signature of asynchronous dispatch: the fast step measures dispatch while the
following step absorbs the blocked wait. A median over the last 10 rejects both
(it returns 78.22 here, against a mean of 82.0 that the 163.3 outlier drags up).

The same pattern appeared on the flex cluster at the identical configuration
(step 0 at 86.0, step 1 at 215.5), which is useful independent confirmation that
the two clusters agree and that the early fast readings are an artifact, not a
difference in hardware.

#### Where the memory goes

MaxText's own compile-time estimate, same run:

| pdb | estimated temp size |
|---|---|
| 1 | 52.5 GB |
| 2 | 79.8 GB |

That is **27.3 GB per unit of pdb** with a ~25 GB batch-independent floor, so
temporaries are activation-dominated and scale linearly with batch. Extrapolating
puts pdb=4 near 134 GB estimated against the 104.02 GB the runtime actually
demanded, so the compiler over-estimates (it says so in the log) but the slope is
the real constraint. Cutting ~10 GB is enough to reach pdb=4, which is why
`capacity_factor` is the first thing the c-series tries: it is `-1.0` (dropless)
in every run above, so nothing bounds the per-expert row count.

What it establishes:

1. **HBM is the binder, not compute.** pdb=4 is short by 9.3 GiB. Every pdb=4 arm
   died the same way.
2. **128 devices does not beat 8.** 78.2 against 81.5. Scaling this geometry out
   is flat to slightly negative, which retires the "7.1% is slice-limited"
   hypothesis.
3. **perfsim is 4.3x optimistic at 128 devices**, not the 1.7x borrowed from
   OLMoE3. It predicted 29.0% at pdb=2 against 6.78% measured. Direction
   transferred, magnitude did not: pdb 1 -> 2 was +44% measured against +100%
   predicted. Use perfsim for ranking only on this family.
4. **Offload buys memory, not throughput.** b1 is the only pdb=4 arm that ran and
   it still loses to pdb=2.

`b2_ep4` was a harness bug, not a hardware result: `ici_expert_parallelism=4`
with `ici_fsdp_parallelism=-1` leaves 24.17% of parameters unsharded and MaxText
asserts. `fsdp x ep` must equal the device count.

### 128 devices (4x4x4): run `o35fle241836`, c-series

`tpu7x-cluster-flex` pool `tpu7x-full-pod-spot`, no Kueue, spot, 2026-09-24. Ran 6
of 9 arms (c7 was mid-compile) before another user's Pathways job took the pool.

| arm | pdb | extra | TF/s/dev | MFU | step s | outcome |
|---|---|---|---|---|---|---|
| c1_prof_p2 | 2 | `profiler=xplane` | **80.1** | **6.95%** | 0.882 | ok, reproduces 78.2; **capture lost** with the pod |
| c2_p3s8k | 3 | | | | | **fits in HBM**, loss NaN at step 1 (see below) |
| c3_ep4fix | 4 | `ici_fsdp_parallelism=32 ici_expert_parallelism=4` | | | | still 24.17% unsharded |
| c4_p4_cf125 | 4 | `capacity_factor=1.25` | | | | OOM at the **identical** 104.02G |
| c5_p4_optoff | 4 | `optimizer_memory_host_offload=True` | | | | abort: `MegaScaleCollective only works in multi slice` |
| c6_p2_megablox | 2 | `megablox=True` | 75.4 | 6.54% | 0.938 | 6% slower than gmm_v2 |

What it settles:

1. **pdb=3 fits.** The HBM wall is between 3 and 4, so pdb=3 is the next
   throughput step, blocked only by the NaN.
2. **`capacity_factor` does not touch the temporaries.** pdb=4 OOMs at 104.02G
   with or without it, so the expert intermediates are not what fills HBM. c8/c9
   retired.
3. **Optimizer host offload is multi-slice only** on this stack. Dead lever for a
   single 4x4x4.
4. **Expert parallelism needs `shard_exp_on_fsdp=False`.** With it True the expert
   weights are `P('fsdp', None, None)` and never touch the `expert` axis, whatever
   the mesh. The old validator says so outright (`shard_exp_on_fsdp requires
   ici_expert_parallelism = 1`). Fixed arm is e5.
5. **gmm_v2 holds at 128 devices**, 1.06x over megablox, same as at 8.

Raw logs: `/tmp/olmo35_results/flexspot4x4x4-o35fle241836-logs/`.

### The pdb=3 NaN is a KDA decay overflow, not a batch bug

The tokamax Mosaic KDA kernel folds cumulative decay into its matmul operands
inside each 4-token sub-block. MaxText called it with `use_gate_in_kernel=True`,
which turns off the kernel's `safe_gate` centering, so the fold reference sits at
the sub-block start and the exponent spans 3 steps. fp32 `exp` overflows at 88,
so any step decaying past ~30 nats returns NaN.

Reproduced on CPU with the real kernel in Pallas interpret mode (v7x VMEM
limits), A=16, constant gate:

| gate | per-step log-decay | 3-step span | kernel |
|---|---|---|---|
| 0.0 | 11.1 | 33 | finite |
| 1.5 | 27.2 | 82 | finite |
| 2.0 | 34.0 | 102 | **NaN** |

It is independent of batch: B=1, 2 and 3 NaN at the same heads (the ones with the
largest `A`). At init the gates are small and every step decays ~11 nats, which
is why step 0 is always clean; one optimizer step can push a gate past the
threshold, which is what c2 hit. **pdb=2 carries the same latent risk in a longer
run.**

**Fix**: `tokamax_kda_log_decay_floor` (default 20). MaxText activates the gate
itself, floors the per-step log-decay at -20, and passes it with
`use_gate_in_kernel=False`, which also turns `safe_gate` on. A step decaying
e^-20 or e^-50 is a full state reset either way, so the output moves by under
e^-20 per step.

| gate scale | as shipped | floor 20 |
|---|---|---|
| 0.1 | 1.8e-5 vs XLA ref | 1.9e-5 |
| 1.0 | **NaN** | 2.0e-5 |
| 4.0 | **NaN** | 2.4e-5 |

With the production `DENSE_PAIRS` and bf16 flags the same table reads 2.5e-3
across the board, the documented bf16 level. New unit test
`test_matches_unfused_at_large_decay` sets `dt_bias=3` (up to ~48 nats per step);
it fails at 2.4e-2 without the floor and passes at 4.5e-4 with it.

### 128 devices (4x4x4): run `o35n242234`, e-series

`bodaborg-tpu7x-nap`, 2026-09-24/25, all six arms with the KDA decay floor.

| arm | pdb | extra | TF/s/dev | MFU | step s | outcome |
|---|---|---|---|---|---|---|
| e1_p3 | 3 | | **87.2** | **7.56%** | 1.215 | **best**, loss finite and falling |
| e2_p2_prof | 2 | `profiler=xplane` | 79.5 | 6.89% | 0.889 | capture landed |
| e3_p2_floor0 | 2 | `tokamax_kda_log_decay_floor=0` | 79.9 | 6.93% | 0.884 | floor costs nothing |
| e4_p3_prof | 3 | `profiler=xplane` | 87.2 | 7.56% | 1.216 | capture landed |
| e5_p2_ep4 | 2 | `ici_fsdp_parallelism=32 ici_expert_parallelism=4 shard_exp_on_fsdp=False` | | | | OOM, 108.16G vs 94.74G |
| e6_p2_noshardexp | 2 | `shard_exp_on_fsdp=False` | 80.1 | 6.94% | 0.882 | neutral |

What it settles:

1. **pdb=3 is the new best, +9% over pdb=2.** The decay floor removed the only
   thing that blocked it. Estimated temporaries 90.9 GB, so pdb=3 is the last
   batch that fits under `remat_policy=full`.
2. **The floor is free.** 79.9 without it against 79.5 and 80.1 with it.
3. **EP4 costs memory, not just time.** It needs 13.4G more than pdb=2 FSDP-only
   holds, so it is off the table unless something else frees HBM.
4. **`shard_exp_on_fsdp` does not matter** at pdb=2 on this mesh.

Raw logs: `/tmp/olmo35_results/nap4x4x4-o35n242234-logs/`.

### 128 devices (4x4x4): run `o35s251513`, f-series, targeting the exposed comm

The 128-device profiles (below) show 328 ms of SparseCore comm exposed at pdb=3,
batch-independent. The f-series attacks it at pdb=3. "sched" is the 17
scheduler and SparseCore flags the gpt-oss Ironwood recipe carries and we never
ported (latency-hiding layer scheduler, concurrent SC offloading, SC collective
aggregator, reduce-scatter v2, nd/3d collective offload, two concurrent async
all-gathers and reduce-scatters, shared-memory limit 150). Per-arm XLA flags are
passed as `+--flag` tokens in the fifth arm field.

| arm | pdb | change | question |
|---|---|---|---|
| f1_p3_sched_prof | 3 | sched, captured | does the scheduler hide the comm |
| f2_p3_dp2 | 3 | `ici_data_parallelism=2 ici_fsdp_parallelism=64` | recipe mesh, DP on the intra-chip pair |
| f3_p3_sched_dp2 | 3 | both | do they compose |
| f4_p3_sched_1sc | 3 | sched + single-SC all-gather | recipe-exact against our dual-SC default |
| f5_p4_sched_offload | 4 | sched + `remat_policy=custom` offload | does hidden comm turn pdb=4 into a win |
| f6_p3_ctrl | 3 | none | same-run control |

Results, `tpu7x-cluster-flex` spot, 2026-09-25. The pod was preempted during f6,
so the control row is missing; e1/e4 (87.2) are the baseline, same code and
flags. Replicate running on nap as `o35n251513`.

| arm | TF/s/dev | MFU | step s | vs 87.2 | outcome |
|---|---|---|---|---|---|
| f1_p3_sched_prof | 96.0 | 8.32% | 1.105 | 1.10x | capture truncated by the preemption |
| f2_p3_dp2 | 100.7 | 8.73% | 1.052 | 1.15x | |
| f3_p3_sched_dp2 | **106.9** | **9.27%** | 0.992 | **1.23x** | **best**, loss matches (10.83 at step 18) |
| f4_p3_sched_1sc | 95.8 | 8.30% | 1.107 | 1.10x | single-SC gather = dual-SC, drop it |
| f5_p4_sched_offload | | | | | OOM, 102.88G vs 94.74G |

What it settles:

1. **The recipe scheduler flags are worth 1.10x** on their own and compose with
   DP=2 to 1.23x. The xla-shell roadmap called the 1.39x scheduling ceiling; this
   is most of the first lever.
2. **DP=2 on the intra-chip pair is worth 1.15x.** It doubles the parameter and
   optimizer footprint (argument size 1.1 to 2.2 GB) but shortens the FSDP ring
   to 64 chips and puts the gradient all-reduce on the fast link.
3. **pdb=4 is still out,** 8G short even with offload.
4. **Replicated on nap** (`o35n251513`): every arm within 0.1 TF/s of flex, and
   the same-run control f6 is 87.5, so f3 is **1.22x** against its own control.

xla-shell on the nap f1 capture (sched flags, no DP), against e4 (no sched):

| | e4 | f1 |
|---|---|---|
| step | 1210 ms | 1100 ms |
| TensorCore | 875 ms | 897 ms |
| SparseCore comm | 532 ms | 495 ms |
| comm exposed | 328 ms | **199 ms** |
| scheduling ceiling | 1.39x | 1.23x |

The flags hide 129 ms of comm and leave the TensorCore lane alone. The remaining
199 ms is still the first lever; the kernel list is unchanged.

### 128 devices (4x4x4): run `o35n251620`, g-series, stacked on f3

`bodaborg-tpu7x-nap`, 2026-09-25. All at pdb=3 with the sched flags.

| arm | change on top of f3 | TF/s/dev | MFU | step s | outcome |
|---|---|---|---|---|---|
| g1_ctrl | none | 106.4 | 9.23% | 0.996 | reproduces f3 |
| g2_dp4 | `ici_data_parallelism=4 ici_fsdp_parallelism=32` | **107.4** | **9.31%** | 0.986 | **best**, +1%, temp 88.2 GB, args 4.4 GB |
| g3_gradbf16 | `grad_dtype=bfloat16` | 106.7 | 9.25% | 0.994 | noise |
| g4_zero1 | `shard_optimizer_over_data=True` | | | | rejected: ZeRO-1 cannot combine with FSDP |
| g5_dp4_zero1 | DP=4 + ZeRO-1 | | | | same |
| g6_async4 | 4 concurrent async all-gathers and reduce-scatters | 106.8 | 9.25% | 0.992 | noise |
| g7_pinsc | `moe_pin_sparse_core_all_gathers=True` | | | | `UNIMPLEMENTED: all_to_all not supported on the SparseCore` |

What it settles: **the comm-side levers are exhausted.** DP past 2, halving
gradient volume and more collective concurrency all land within 1%. The binder
is now the TensorCore lane, so the next series goes after recompute.

### 128 devices (4x4x4): run `o35n251712`, h-series, trading HBM for recompute

Base is g2 (pdb=3, sched, DP=4). In OLMo 3.5 only the MoE GEMMs and the two
full-attention layers carry checkpoint names; the 14 KDA layers carry none, so
the MoE outputs are the only real save/offload targets.

| arm | pdb | change | TF/s/dev | MFU | step s | vs control |
|---|---|---|---|---|---|---|
| h1_ctrl_prof | 3 | none, captured | 107.3 | 9.30% | 0.988 | |
| h2_p3_off_mlpwo | 3 | `moe_mlpwo=offload` | 100.1 | 8.67% | 1.059 | 0.93x |
| h3_p3_off_moe | 3 | all three MoE outputs offloaded | 56.2 | 4.87% | 1.885 | 0.52x |
| h4_p2_dev_mlpwo | 2 | `moe_mlpwo=device` | 99.0 | 8.59% | 0.714 | 1.03x vs h7 |
| h5_p2_dev_moe | 2 | all three on device | 99.8 | 8.65% | 0.708 | 1.04x vs h7 |
| h6_p3_vocab4 | 3 | `num_vocab_tiling=4` | **108.0** | **9.36%** | 0.982 | 1.01x |
| h7_p2_ctrl | 2 | none | 96.4 | 8.35% | 0.733 | |

What it settles:

1. **Host offload is a loss here.** 12.9 GB per step of MoE activations over the
   host link does not hide; offloading all three halves throughput.
2. **Saving MoE outputs on device is worth 1.04x at pdb=2** but does not beat
   pdb=3 with full remat (99.8 against 107.3). Batch beats recompute savings.
3. `num_vocab_tiling=4` is a marginal +0.6%, retested in the i-series.

### 128 devices (4x4x4): run `o35n251752`, i-series, MoE routing path

Base g2 (pdb=3, sched, DP=4).

| arm | change | TF/s/dev | MFU | step s | outcome |
|---|---|---|---|---|---|
| i1_ctrl | none | 107.2 | 9.29% | 0.989 | |
| i2_raggedsort | `use_ragged_sort=True` | | | | rejected: needs EP > 1 |
| i3_directgather | `moe_use_direct_token_gather=True` | 104.6 | 9.07% | 1.014 | 0.98x |
| i4_mosaicgather | `use_gather_mosaic_kernel=True` | 107.4 | 9.31% | 0.987 | neutral |
| i5_rs_dg | ragged sort + direct gather | | | | rejected, as i2 |
| i6_nocustomsort | `use_custom_sort_vjp=False` | 98.3 | 8.52% | 1.078 | 0.92x, keep it on |
| i7_vocab4 | `num_vocab_tiling=4` | **108.0** | **9.36%** | 0.982 | reproduces h6, **adopted** |

None of the gather or sort kernels moves the routing cost. The xla-shell
"offloadable to SparseCore" tag on `argsort` is a label heuristic, and libtpu
has no flag for it. The argsort outputs do carry the checkpoint name
`moe_routing` (about 1.5 MB per layer), so the j-series saves them to stop full
remat from re-running the sort.

### j-series, save the routing so remat skips the sort (o35n251823)

Base i7 (pdb=3, sched, DP=4, `num_vocab_tiling=4`), all `remat_policy=custom`
except the control. Loss is 10.813 in every arm.

| arm | pdb | saved | TF/s/dev | MFU | step s |
|---|---|---|---|---|---|
| j1_ctrl | 3 | none (full remat) | 108.1 | 9.37% | 0.981 |
| **j2_route** | 3 | `moe_routing` | **109.6** | **9.50%** | 0.968 |
| j3_route_logits | 3 | + `moe_router_logits` | 109.4 | 9.48% | 0.969 |
| j4_route_disp | 3 | + `moe_dispatch` | 109.7 | 9.51% | 0.967 |
| j5_route_dispoff | 3 | routing + logits on device, dispatch offloaded | 95.2 | 8.25% | 1.114 |
| j6_p2_allmoe | 2 | every MoE name on device | 101.8 | 8.83% | |

Saving `moe_routing` (about 25 MB) is the whole win, +1.4%. Saving the logits or
the dispatch on top is within noise, and offloading the dispatch costs 13%, the
same host-transfer penalty as the h-series. Saving everything at pdb=2 does not
beat pdb=3 with full MoE remat. The k-series control (k1, same flags as j2)
replicated at 109.9, so j2 is the new base.

### Memory at pdb=3 (local AOT)

`train_compile` with `compile_topology=tpu7x-128` and the j2 flags, peak from the
buffer live ranges (`/tmp/aot_p4/peak.py`). The peak is 88.9 GiB of 94.7, reached at
the forward to backward boundary, so it is the saved residual set plus the gathered
weights of the first backward layers.

| live at peak | GiB | what it is |
|---|---|---|
| MoE rows `bf16[393216,1024]` | 20.3 | 27 buffers, gmm_v2 wi_0/wi_1 outputs and activations (24576 tokens x top-16 rows) |
| MoE rows `bf16[393216,512]` | 7.9 | 21 buffers, dispatched input and wo output |
| gathered expert weights, bf16 | 20.0 | `[512,512,1024]` and `[512,1024,512]` full-layer copies after the FSDP all-gather, plus weight gradients before reduce-scatter |
| KDA residuals, f32 | about 20 | chunk states `f32[8,3,160,128,256]` (6.6 GiB across the 14 KDA layers) plus q/k/v/gate in f32 |
| other activations | about 16 | layer inputs, attention, norms |
| f32 params and optimizer | 4.2 | |

pdb=4 needs about a third more activation memory, roughly 25 GiB more, so no
single lever unlocks it. The three blocks that could are the MoE rows (the
remat already drops them per layer, what is left is the working set of the
layers in flight), the full-layer gathered expert banks, which expert
parallelism would avoid by moving tokens instead of weights, and the f32 KDA
residuals, which a bf16 residual path would halve.

### k-series, layout and precision bounds (o35n251859)

Base j2 (pdb=3, sched, DP=4, vocab 4, `remat_policy=custom moe_routing=device`).

| arm | pdb | change | TF/s/dev | MFU | step s | note |
|---|---|---|---|---|---|---|
| k1_ctrl | 3 | none | 109.9 | 9.52% | 0.965 | replicates j2 |
| k2_noscan | 3 | `scan_layers=False` | 99.4 | 8.62% | 1.067 | |
| k3_wbf16_bound | 3 | `weight_dtype=bfloat16` | 122.2 | 10.59% | 0.867 | bound only, loss 11.357 vs 10.813 at step 19 |
| k4_dp8 | 3 | DP=8, FSDP=16 | 101.9 | 8.83% | 1.040 | |
| k5_tp2_p3 | 3 | DP=4, FSDP=16, TP=2 | | | | OOM, 110.89G vs 94.74G |
| k6_tp2_p4 | 4 | same as k5 | | | | OOM expected, k5 already over |

k3 is the finding. Pure bf16 weights stall learning, so it is not usable, but
it prices fp32 weights at 11% of the step: the FSDP gathers and reduce-scatters
move fp32, and the relayout lane casts them. `cast_params_to_compute_dtype`
keeps the fp32 master and optimizer and casts large weights to bf16 once per
step on the sharded copy, so the gathers inside the scan move bf16 and the
router stays fp32. That is the legitimate version of k3. TP=2 adds memory
instead of freeing it, and DP=8 loses 7%, so DP=4 x FSDP=32 stays.

### AOT memory for the two new flags

| config | pdb | temporaries | peak live | notes |
|---|---|---|---|---|
| j2 | 3 | 90.25 GiB | 88.9 GiB | |
| j2 | 4 | 105.98G | | OOM by 11.2G |
| j2 + `kda_conv_in_compute_dtype` | 3 | 90.87 GiB | 88.9 GiB | KDA residuals and chunk states go bf16 (6.6 to 3.05 GiB); the scheduler spends the saving on more MoE rows in flight (20.3 to 27.8 GiB) |
| j2 + `cast_params_to_compute_dtype` | 3 | 90.75 GiB | | gathers and reduce-scatters already bf16 in j2, so the collectives are unchanged |
| j2 + both flags | 4 | 97.10G | | OOM by 2.4G, down from 11.2G |
| j2 + both flags, `num_vocab_tiling=8` | 4 | 95.08G | | OOM by 0.34G |
| j2 + both flags, scheduler memory limit 100 | 4 | 95.31G | | OOM by 0.57G |
| j2 + both flags, vocab 8, limit 100 | 4 | 95.74G | | OOM by 1.0G, the two trims do not stack |

pdb=4 sits within half a gigabyte of fitting once KDA runs in bf16. The trims
left are small, so pdb=4 is a question of shaving the MoE row working set, not of
flags.

XLA already hoists the bf16 convert above the FSDP all-gather, and gradients are
reduce-scattered in bf16, so fp32 weights cost no extra collective bytes. The
likelier source of k3's 11% is KDA: with bf16 weights the conv weight is bf16 too.
The fp32 conv weight promoted q/k/v to fp32, so the fused KDA kernel ran and
saved residuals in fp32. With the flag the whole KDA path is bf16, which needs a
loss check on hardware before it counts.

### m-series, scheduler and VMEM limits (o35n252023)

Needs no new source, so it ran while the l-series (the two new flags, arms in
`/tmp/olmo35_l_arms.txt`) waits for the source upload. Base j2 at pdb=3.

| arm | change | TF/s/dev | MFU | step s |
|---|---|---|---|---|
| m1_ctrl | none | 109.7 | 9.51% | 0.966 |
| m2_lim100 | `xla_tpu_scheduler_percent_shared_memory_limit=100` | 107.1 | 9.28% | 0.990 |
| m3_lim200 | same, 200 | 109.8 | 9.52% | 0.966 |
| m4_vt8 | `num_vocab_tiling=8` | 108.8 | 9.44% | 0.974 |
| m5_vmem96 | `xla_tpu_scoped_vmem_limit_kib=98304` | 109.8 | 9.52% | 0.966 |
| m6_vmem32 | `xla_tpu_scoped_vmem_limit_kib=32768` | 109.9 | 9.52% | 0.965 |

Flag tuning is exhausted at pdb=3. Scheduler limit 150 and above and any scoped
VMEM limit give the same step. Tightening the limit to 100 or doubling the vocab
tiles costs 1% to 2.4%, which also prices the two pdb=4 memory trims. The
control has now landed at 109.6 to 109.9 across three series. The remaining
levers are the bf16 KDA path (l3/l4) and pdb=4 (l5/l6), both waiting on the
new source.

### l-series, bf16 KDA conv and param cast (o35n261815)

Base j2 at pdb=3 unless noted.

| arm | pdb | change | TF/s/dev | MFU | step s | loss @19 |
|---|---|---|---|---|---|---|
| l1_ctrl | 3 | none | 109.4 | 9.48% | 0.969 | 10.823 |
| l2_cast | 3 | `cast_params_to_compute_dtype` | 109.7 | 9.51% | 0.966 | |
| **l3_conv** | 3 | `kda_conv_in_compute_dtype` | **122.3** | **10.60%** | 0.867 | 10.836 |
| l4_both | 3 | both | 122.2 | 10.60% | 0.867 | |
| l5_both_p4_vt8 | 4 | both, vocab 8 | | | | OOM, 95.08G, as the AOT said |
| l6_both_p4_vt8_lim100 | 4 | both, vocab 8, limit 100 | | | | OOM |

l3 matches the k3 bound to the decimal, so all of k3's 11.8% was KDA running in
fp32, not the weights. The param cast is neutral, as the AOT predicted. The bf16
KDA path moves the loss by +0.005, +0.008, +0.011, +0.013 at steps 4, 9, 14, 19:
small but growing, so it needs a longer run before it can be the default. OLMo-core
trains under bf16 autocast, which also feeds the KDA kernel bf16.

### OLMoE3 remat: the first cycle was never rematerialized

The pdb=4 peak held 43 GiB of MoE rows. The live ranges show why. OLMo 3.5 has
`first_num_dense_layers=1`, so `_init_scanned_olmoe3` builds the first 8-layer
cycle as an unrolled `layers_0` and scans the remaining cycle.
`_apply_olmoe3_scanned_blocks` called `layers_0` with no `jax.checkpoint`, so
every activation of its 7 MoE layers was saved from the forward to the backward.
The scanned cycle was rematerialized as one 8-layer block, so its 7 MoE layers
were also live together during its backward.

`olmoe3_per_layer_remat` gives each layer its own `jax.checkpoint`
(`prevent_cse=True`, since the layers are unrolled inside the block) and skips
the block-level remat. This is the Qwen3-Next and Gemma4 pattern.

| config (AOT) | pdb | temporaries | peak live | MoE rows at peak |
|---|---|---|---|---|
| conv + cast, vocab 8 | 4 | 95.08G (OOM) | 92.5 GiB | 43 GiB |
| conv + per-layer remat | 4 | 70.63 GiB | 62.7 GiB | 3 GiB |

The biggest item at the peak is now the gathered expert weights (31 GiB).

### In flight: n-series, per-layer remat and batch (o35n261924)

Base j2 plus `kda_conv_in_compute_dtype` unless noted. plr = `olmoe3_per_layer_remat`.

| arm | pdb | change | TF/s/dev | MFU |
|---|---|---|---|---|
| n1_conv | 3 | control (l3) | | |
| n2_conv_plr | 3 | + plr | | |
| n3_conv_plr_p4 | 4 | + plr | | |
| n4_conv_plr_p5 | 5 | + plr | | |
| n5_conv_plr_p6 | 6 | + plr | | |
| n6_plr_p4_fp32kda | 4 | plr, fp32 KDA | | |
| n7_conv_plr_p4_disp | 4 | + plr, `moe_dispatch=device` | | |

### KDA kernel, single device

Details in `kda-vs-gdn-kernels.md`. The tokamax KDA layer takes 7.50 ms fwd+bwd
at pdb 3 against a 0.48 ms HBM roofline (6.4%). fp32 q/k/v inputs cost 1.83x,
which predicts 87 ms of the 100 ms l3 win. The kda8 overflow patch (sub-block
BC 16 to 4) costs 18%. BC 8 is safe at the 20-nat decay floor and saves 13%, so
the launcher now takes a per-arm `KDA_BC=<n>` token.

## Profiles and xla-shell output

Captures are pulled and analysed with `scripts/olmo35_profile_report.sh`, which
locates the `.xplane.pb` under
`<base_output_directory>/<run>-<model>-<arm>/tensorboard/plugins/profile/` and
runs `analyze_profile` plus every `roadmap` view, writing each to a text file.

    scripts/olmo35_profile_report.sh gs://cloud-pathways-staging/agagik/olmo35-out o35fle241730 c1_prof_p2

### 8 devices, best configuration

The only analysis captured so far. Ran against the step-5 configuration above.

| lane | time |
|---|---|
| TensorCore | 906 ms |
| SparseCore | 372 ms |
| host DMA | 50 ms |

Verdict: **TensorCore-bound**, not comm-bound, once the flag set is applied. The
best-overlap moving floor is **372 ms** against a 1.10 s step, so most of the
remaining headroom was software.

`roadmap --kernels`, top entries:

| kernel | time | share of step |
|---|---|---|
| `_fused_dhu_wy_intra_cumsum_pallas_` | **303 ms** | **27%** |
| next three, combined | 51 + 44 + 28 ms | 11% |

The top kernel is the KDA intra-chunk factoring and it is **5.9x the next
kernel**. That single reading is what motivated `TOKAMAX_KDA_DENSE_PAIRS` and the
bf16 KDA flags, which together delivered the 1.34x in steps 4 and 5. It is also
why the latent-MoE fusion prototype and `tokamax_gmm_tile_m` were **not**
pursued: the roadmap puts those kernels at 51+44+28 ms, so even a perfect fix
there is worth far less than the KDA work.

### 128 devices

`e2_p2_prof` and `e4_p3_prof` from `o35n242234`, analysed with
`scripts/olmo35_profile_report.sh gs://agagik-us/olmo35/4x4x4 o35n242234 <arm>`.
Reports in `/tmp/olmo35_profiles/o35n242234-<arm>/`.

| lane | pdb=2 | pdb=3 |
|---|---|---|
| step | 887 ms | 1210 ms |
| TensorCore | 608 ms | 875 ms |
| of which matmul | 323 ms | 458 ms |
| of which vpu | 218 ms | 335 ms |
| of which relayout | 67 ms | 82 ms |
| SparseCore comm | 518 ms | 532 ms |
| comm exposed | 276 ms | 328 ms |
| Host-DMA | 17 ms | 63 ms |

`roadmap --all`, pdb=3:

| # | lever | step | gain | binder |
|---|---|---|---|---|
| 0 | as profiled | 1210 ms | | imperfect overlap |
| 1 | schedule exposed comm | 875 ms | 337 ms | TensorCore |
| 2 | kernels, bounded by slack | 532 ms | 343 ms | TensorCore |
| 3 | relayout | 532 ms | 0 | SparseCore comm |

Top kernels at pdb=3 (`roadmap --kernels`): KDA
`_fused_dhu_wy_intra_cumsum_pallas` 75 ms, `gmm_v2` k=512 69 ms, generic
fusions 67 ms, `gmm_v2` k=1024 58 ms, KDA `shard_map` 53 ms, `tgmm_v2` 54 ms.

What it says:

1. **Comm is back on the critical path at 128 devices,** answering the open
   question. At 8 devices tuning had moved the binder to the TensorCore; here
   27% of the pdb=3 step is exposed collective.
2. **Comm is batch-independent.** 518 ms at pdb=2, 532 ms at pdb=3. It is the
   FSDP weight all-gather and gradient reduce-scatter, whose volume depends on
   parameters, not tokens. This is why pdb 1 to 2 to 3 keeps paying: the fixed
   cost is amortised over more work.
3. **Scheduling alone is worth 1.39x** (1210 to 875 ms, ~121 TF/s, 10.5% MFU) if
   every collective can be hidden. `roadmap --collective` finds all 375 ms
   hideable, spread over many ~4 ms reduce-scatters, so the fix is compiler
   scheduling, not one bad op.
4. **Past that the floor is SparseCore comm at 532 ms,** so going further needs
   less comm volume (DP on the intra-chip pair, lower-precision gathers), not
   faster kernels. KDA is 128 ms of kernel time at pdb=3, no longer dominant.
5. **Non-matmul TensorCore work is 34% of the step,** which is what any
   FLOP-based estimate misses and part of why perfsim is 4.3x optimistic here.

### 128 devices, best configuration (h1: pdb=3, sched, DP=4)

`o35n251712-h1_ctrl_prof`, report in `/tmp/olmo35_profiles/o35n251712-h1/`.

| lane | e4 (no sched, DP=1) | h1 |
|---|---|---|
| step | 1210 ms | 986 ms |
| TensorCore | 875 ms | 920 ms (93%) |
| of which matmul | 458 ms | 479 ms |
| of which vpu | 335 ms | 351 ms |
| of which relayout | 82 ms | 91 ms |
| SparseCore comm | 532 ms | 379 ms |
| comm exposed | 328 ms | **59 ms** |

**Comm is solved; the step is now TensorCore-bound** with 1.07x of scheduling
headroom left. The remaining levers are on the TensorCore lane:

| item | share | note |
|---|---|---|
| matmul | 479 ms | KDA fused kernel 75 ms, gmm_v2 129 ms, tgmm_v2 62 ms, KDA shard_map 53 ms |
| VPU: MoE routing `argsort` | ~22% of VPU | xla-shell tags it offloadable to SparseCore |
| VPU: `top_k`, `silu`/`ffn_act` mul, KDA bwd mul | most of the rest | inherent elementwise |
| relayout: fp32 to bf16 weight casts | largest relayout scopes | expert weights `[512,512,1024]` reformatted each step |
| VPU: `pad` | ~2% of VPU | gmm tile padding |

The i-series goes after the routing sort and gather.

## Infrastructure notes that cost real time

| thing | note |
|---|---|
| Kueue TAS on nap | admits only against a **physically existing free topology domain**. Cohort said `nominal=307 used=57 free=250` while refusing a 64-chip ask with "62 more needed" |
| NAP deadlock on nap | NAP builds the pool shape a *Pending* pod asks for, but Kueue suspends the pod first, so NAP only ever saw other people's 2x2x1 requests. No bypass: `manageJobsWithoutQueueName: true` with every framework, and `create provisioningrequests` denied |
| multi-host JobSet on a scaled-to-zero pool | the webhook creates **only the leader pod** and logs `FailedCreate: leader pod not yet scheduled`. One Pending pod for a 16-node request is the normal pre-provision state |
| pod identity, read | shared-capacity pods read `gs://agagik-us`; multipod-dev pods were expected to need `gs://cloud-pathways-staging`, but in fact read `gs://agagik-us` fine. The launcher tries both |
| **pod identity, write** | **multipod-dev pods cannot write ANY bucket.** Both `gs://agagik-us` and `gs://cloud-pathways-staging` return `GcsApiError('')` on upload while reads succeed. This is not a 403 you see in the training log: it **hangs the MaxText summary writer**, so the run sits at step 1 forever with the python process at 349% CPU looking healthy. It also silently breaks the per-arm results upload. Diagnosed by `kubectl exec ... gcloud storage cp` from inside a live pod. Fix: those routes set a **local** `base_output_directory` and the supervisor copies `/tmp/hc` out of the leader pod over the API |
| SPS | does **not** propagate `LIBTPU_INIT_ARGS` (measured 318 vs 318 with the flags on and off), so XLA-flag tuning needs a non-SPS vehicle |
| spot 4x4x4 | `GCE out of resources` in us-central1-c for two days, then cleared without warning. Worth leaving a supervised job queued rather than polling by hand |
