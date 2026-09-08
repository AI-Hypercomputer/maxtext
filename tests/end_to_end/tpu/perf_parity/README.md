# Trainer parity: `MaxTextTrainingEngine` vs tunix `PeftTrainer v2`

Consolidated conclusions from two studies run on a v7-8 (Ironwood) host, 2026-09-02 and
2026-09-03. **Both trainers drive the same MaxText model in every arm**, so what varies is
the trainer, the mesh and the sharding mode — never the implementation of the network.

Two models, chosen because they stress opposite things:

| | `qwen3.5-35b-a3b` | `qwen3-0.6b` |
| --- | --- | --- |
| Kind | sparse MoE, 256 routed + 1 shared expert, 40 layers | dense, 28 layers |
| Step time | ~0.7–2.3 s — device-bound, host cost invisible | ~50–470 ms — host and collective cost visible |
| What it tests | mesh choice (TP vs EP), MoE kernels, GA capability | sharding mode, optimizer-state sharding, host overhead |
| Detailed report | [`RESULTS-qwen35-35b-ep-20260902.md`](RESULTS-qwen35-35b-ep-20260902.md), [`RESULTS-qwen35-35b-20260902.md`](RESULTS-qwen35-35b-20260902.md) | [`RESULTS-qwen3-0p6b-zero1-20260903.md`](RESULTS-qwen3-0p6b-zero1-20260903.md) |

Every command needed to reproduce every number is in **[Appendix A](#appendix-a-reproducing)**.

## Conclusions

1. **On raw throughput the two trainers are at parity, and the gap is a fixed dispatch, not a
   scaling one.** On the 35b model they land within 0.5% of each other at every mesh
   (670.3 vs 666.6 ms at ep=8; 2313.8 vs 2303.1 at tp=2), both above 99% device utilization.
   `PeftTrainer`'s lead is its single fused `jit__train_step` against the engine's split
   `fwd_bwd` + `update` — a constant ~16 ms, which is 0.7% of a 2.3 s step and a third of a
   61 ms one.

2. **Under explicit sharding the engine wins decisively, and the win is the engine's, not the
   mesh mode's.** This is the largest trainer-attributable difference measured. Setting
   `shard_mode=explicit` on qwen3-0.6b at GA=8 makes the engine **1.50x faster** (371.7 →
   247.4 ms) and `PeftTrainer` **1.31x slower** (356.8 → 465.7 ms) — the same flag, the same
   mesh, opposite signs. Same-mesh, same-mode, GA=8: **247.4 ms against 465.7 ms, 1.88x.**
   Under Explicit axes the engine's cotangents come out `unreduced`, so the data-parallel
   all-reduce moves out of every micro-batch into `update()`; tunix's step has no such path
   and simply pays the collectives literally where GSPMD had been optimizing them.

3. **The decisive difference is capability, not speed: `PeftTrainer v2` cannot do gradient
   accumulation on the 35b model at the tunix revision MaxText pins.** GA=2, 4 and 8 each OOM
   with a byte-identical 161.94 G against 94.74 G available. The engine has no such wall and
   its per-micro-batch cost *improves* with depth. Tunix `44a35eeaf` fixes it with one line
   (`GradientAccumulator.reset()` using `v[...] * 0` instead of `jnp.zeros_like`, which loses
   sharding inside a traced function and materialises the full 129 G tree). **Recommend
   bumping `src/dependencies/extra_deps/post_train_github_deps.txt` past that commit.**

4. **Mesh choice dominates trainer choice by an order of magnitude.** Moving the 35b model
   from `fsdp=4 x tp=2` to `ep=8 --ring-of-experts --ragged-sort` is worth **3.45x** on both
   trainers alike. Nothing trainer-side in this study is worth more than 1.9x. Tune the mesh
   before arguing about trainers.

5. **For memory on small models, FSDP beats Zero-1 and it is not close.** 2.04 G against
   8.70 G at qwen3-0.6b/GA=8. Zero-1 shards the two Adam moments; FSDP shards the parameters
   and gradients too, and at 0.6 B those are most of the residual. Zero-1's advantage is that
   it adds no collectives to forward and backward — not that it saves more bytes. The engine
   forbids combining them.

6. **Neither trainer currently offers both speed and memory.** At qwen3-0.6b/GA=8 the engine's
   explicit path is 1.36x faster than `PeftTrainer` on FSDP but uses 6.2x the HBM — 4.3x even
   with Zero-1 on top. That is the open gap this comparison leaves.

## Environment

| | |
| --- | --- |
| Host | 8 x TPU7x (v7-8 Ironwood, 4 chips / 8 JAX devices), single process |
| JAX | 0.11.1 |
| Shape | micro-batch 8 x seq 1024, f32 compute and weights, `remat_policy=none` |
| Wall clock | 23 steps, median of the last 19, `--no-trace` |
| Device time | `XLA Modules` line, per-execution, from a separate `--steps 6` traced run |
| Peak HBM | `peak_bytes_in_use` off the TPU allocator, max over the 8 devices, `--no-trace` runs |
| Revisions | 35b: PR #5060 at `1b75c2479`. 0.6b: `18f4a2332` (PR #5060 scripts + PR #5099 engine) |

`remat_policy=none`, `scan_layers` matched, and `dtype=float32` on both arms throughout —
tunix's `ModelConfig` defaults to f32 and its `RematConfig.NONE` does not rematerialize, so
leaving MaxText's defaults in place would have MaxText winning on numerics rather than on
implementation. `attention: autoselected` is deliberately *not* equalized; the chosen kernel
is part of what is compared, and it is logged.

---

# 1. `qwen3.5-35b-a3b` — MoE, mesh-bound

40 layers, 256 routed + 1 shared expert, emb 2048, 16 query / 2 KV heads, head_dim 256,
vocab 248320. Scanned. `optax.sgd(1e-5)`, constant schedule, no clipping.

`--tp 8` is illegal on this model: `_validate_kv_head_sharding` requires
`num_kv_heads % tp == 0` and there are 2 KV heads, so `fsdp=4 x tp=2` is the widest legal
tensor-parallel shape on 8 devices.

## 1.1 Trainer vs trainer, same mesh

| Mesh | GA | Engine | `PeftTrainer v2` | Δ |
| --- | --- | --- | --- | --- |
| `ep=8` + ring + ragged | 1 | 670.3 ms | **666.6 ms** | Peft 0.55% |
| `ep=8` + ring + ragged | 8 | 5343.0 ms | **5278.0 ms** ¹ | Peft 1.2% |
| `fsdp=4 x tp=2` | 1 | 2313.8 ms | **2303.1 ms** | Peft 0.46% |
| `fsdp=4 x tp=2` | 8 | **18405.3 ms** | OOM ² | engine only |

¹ tunix `07dbe293` (head). OOM at the pinned `c4ec573` — see §1.3.
² OOM at the pinned revision; not run at head.

Device utilization is 99.2–99.8% on every row. At 2.3 s of device work per step every
host-side difference between the trainers hides; the 0.5% that remains is the engine's second
dispatch (`jit__update_kernel`, 16.0 ms at ep=8 and 16.46 ms at tp=2).

Per-execution device cost, ep=8, GA=1:

| Arm | Modules | Total |
| --- | --- | --- |
| engine | `jit_first_kernel` 648.9 + `jit__update_kernel` 16.0 | **664.9 ms** |
| `PeftTrainer v2` | `jit__train_step` 663.0 | **663.0 ms** |

## 1.2 The mesh is worth 3.45x, and the two trainers track each other exactly

Every row `--scan --seq 1024 --ga 1 --no-trace`, median of 19 steps.

| Mesh and flags | Engine | `PeftTrainer v2` | vs tp=2 | vs row above |
| --- | --- | --- | --- | --- |
| `--tp 2` (fsdp=4 x tp=2) | 2313.8 | 2303.1 | 1.00x | — |
| `--tp 2 --ep 4 --ring-of-experts --ragged-sort` | 1726.9 | 1704.3 | 1.34x | — |
| `--ep 8` (no MoE flags) | 1063.8 | 1059.7 | **2.17x** | 2.17x |
| `--ep 8 --ragged-sort` | 1047.6 | 1043.4 | 2.21x | 1.02x |
| `--ep 8 --ring-of-experts` | 720.1 | 717.4 | 3.21x | 1.48x |
| `--ep 8 --ring-of-experts --ragged-sort` | **670.3** | **666.6** | **3.45x** | 1.07x |

* **The mesh is the larger half.** A bare `ep=8` with no kernel flags is already 2.17x. TP
  splits a 2048-wide embedding and 2 KV heads and pays all-reduces for it; EP splits 256
  experts, which is what this model actually has a lot of.
* **Ragged sort is a multiplier on ring-of-experts, not independently useful.** 1.02x alone,
  1.07x on top of the ring. `layers/moe.py` selects the `ring_ragged_sort` kernels only under
  `use_ragged_sort and use_ring_of_experts`.
* **Splitting the mesh across both axes is the worst of both.** `--tp 2 --ep 4` with both
  flags on (1726.9 ms) is slower than a bare `--ep 8` with none (1063.8 ms).

Wall and device speedups agree to two decimals (3.45x / 3.47x), which is what says the gain is
work removed rather than overhead rearranged.

## 1.3 Gradient accumulation: a capability difference

At `ep=8` + ring + ragged, median ms/step:

| GA | Engine | per micro | `PeftTrainer`, tunix `c4ec573` (pinned) | `PeftTrainer`, tunix `07dbe293` (head) | per micro |
| --- | --- | --- | --- | --- | --- |
| 1 | 670.3 | 670.3 | 666.6 | 667.3 | 667.3 |
| 2 | 1338.8 | 669.4 | **OOM** | 1339.6 | 669.8 |
| 4 | 2672.3 | 668.1 | **OOM** | 2651.8 | 663.0 |
| 8 | 5343.0 | 667.9 | **OOM** | 5278.0 | **659.8** |

```
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Ran out of memory on HBM, the total memory
required for HLO temporaries (161.94G) exceeds available HBM (94.74G).
HLO module: jit__update_step.
```

The figure is **byte-identical at GA=2, 4 and 8**, which rules out the obvious reading that N
micro-batches are live at once. The allocation is parameter-shaped, not depth-shaped:
`PeftTrainer._is_single_microstep()` is true only at `gradient_accumulation_steps == 1`, and
only that path skips allocating the accumulator.

**The cause is one line, tested directly.** Reverting only tunix `44a35eeaf` on top of head —
leaving the other 208 commits in place — brings the failure back byte for byte. `reset()` runs
*inside* the traced `_update_step`, and there `jnp.zeros_like` does not carry its operand's
sharding, so XLA materialises the full unsharded tree:

```
129.12 G  full parameter tree, unsharded, from the traced zeros_like
+ 16.14 G  the sharded accumulator itself
+ 16.14 G  one more sharded parameter-tree copy
= 161.40 G   measured: 161.41 G
```

Once fixed, the two trainers are at parity at every depth and `PeftTrainer` pulls 1.2% ahead
by GA=8, because it amortizes its update slightly better (667.3 → 659.8 per micro against the
engine's 670.3 → 667.9).

---

# 2. `qwen3-0.6b` — dense, sharding-bound

28 layers, emb 1024, 16 query / 8 KV heads, head_dim 128, mlp 3072, vocab 151936, tied
embeddings; ~0.6 B params ≈ 2.4 G in f32. Unscanned. `adamw` (b1=0.9, b2=0.95, eps=1e-8,
wd=0.1, lr 1e-5 constant, no clipping), matched term for term across both trainers — base.yml's
constants, not optax's defaults.

`adamw` and `--dp 8` rather than the `sgd`/`--fsdp 8` the 35b arms use, because Zero-1 shards
*parameter-shaped optimizer state* over the `data` axis: it is vacuous under SGD and mutually
exclusive with FSDP.

## 2.1 Trainer vs trainer, all six arms

Median ms/step and peak HBM per device (of 101.72 G).

| Arm | `shard_mode` | GA=1 ms | GA=1 HBM | GA=8 ms | GA=8 HBM |
| --- | --- | --- | --- | --- | --- |
| engine, dp=8 | auto | 61.0 | 9.95 G | 371.7 | 12.73 G |
| engine, dp=8 | **explicit** | 62.1 | 9.96 G | **247.4** | 12.74 G |
| engine, dp=8 + **Zero-1** | explicit | 66.1 | **7.76 G** | 250.6 | **8.70 G** |
| `PeftTrainer`, dp=8 | auto | 56.0 | 10.00 G | 356.8 | 12.35 G |
| `PeftTrainer`, dp=8 | **explicit** | 64.0 | 10.47 G | 465.7 | 12.39 G |
| `PeftTrainer`, fsdp=8 | auto | **46.4** | **1.74 G** | 337.4 | **2.04 G** |

**The same-mode comparison is the one that matters**, and it flips sign with the mode:

| dp=8, GA=8 | Engine | `PeftTrainer v2` | Winner |
| --- | --- | --- | --- |
| `shard_mode=auto` | 371.7 ms | 356.8 ms | Peft, 1.04x |
| `shard_mode=explicit` | **247.4 ms** | 465.7 ms | **engine, 1.88x** |

## 2.2 Where the time goes

Per-execution device cost, `--steps 6` traces. A GA=8 step is one `first_kernel`, seven
`accum_kernel` and one `_update_kernel` on the engine; eight `_fwd_bwd_step` and one
`_update_step` on `PeftTrainer`.

| Arm, GA=8 | per micro-batch | update | Step total |
| --- | --- | --- | --- |
| engine, auto | 43.57 ms | 5.95 | **354.19** |
| engine, explicit | **25.03 ms** | 27.81 | **226.19** |
| engine, explicit + Zero-1 | 25.07 ms | 30.71 | **229.51** |
| `PeftTrainer`, auto | 42.73 ms | 6.82 | **348.66** |
| `PeftTrainer`, explicit | 56.04 ms | 6.89 | **455.21** |
| `PeftTrainer`, fsdp=8 | 40.79 ms | 1.69 | **328.01** |

**The all-reduce is moved, not removed, and the same ~20 ms shows up three times.** Going
`auto` → `explicit` on the engine at GA=8: the first kernel drops 20.08 ms, each of seven
accumulates drops 18.54 ms, and the update rises 21.86 ms. That is one all-reduce of the 2.4 G
f32 gradient tree over 8 devices, which this host does in about 20 ms — paid eight times under
`auto`, once under `explicit`. Extrapolated from these kernels it is a 1.78 ms loss at GA=1
(measured at 4.63 ms in the GA=1 trace, which runs slower throughout) and **pays from GA=2 on**.

**`PeftTrainer` gets no such benefit; Explicit costs it 31%.** Its per-micro-batch step goes
42.73 → 56.04 ms while its update is unchanged (6.82 → 6.89). Nothing moves out of the
micro-batch — the collectives stay where they were and get more expensive, because Explicit
axes stop GSPMD from rearranging them.

**Zero-1 costs +2.9 ms, entirely in `update()`, and this is provable rather than inferred.**
The Zero-1 arm's `jit_first_kernel` and `jit_accum_kernel` carry *identical HLO program hashes*
to the explicit control's (`1445966014043162594` and `614701509375139023`), at both GA depths.
Only `jit__update_kernel` differs (`8707565451094719151`), by +2.90 ms at GA=8 and +2.85 ms at
GA=1 — the price of replacing the gradient all-reduce with a reduce-scatter plus an all-gather
of the updated parameters.

**FSDP does not supply a deferral either.** It cuts the per-micro cost only 42.73 → 40.79 ms:
sharding the parameters swaps the gradient all-reduce for a parameter all-gather plus a
gradient reduce-scatter, about the same traffic, still once per micro-batch. The engine's
explicit path runs the same micro-batch in 25.03 ms, **1.63x faster**, by not running the
collective there at all.

## 2.3 Where the memory goes

At ~0.6 B params in f32: parameters 2.4 G, Adam `mu` + `nu` 4.8 G, gradients 2.4 G.

| | GA=1 | GA=8 | What is sharded |
| --- | --- | --- | --- |
| engine, dp=8 auto | 9.95 G | 12.73 G | nothing — params, moments and grads all replicated |
| engine, + Zero-1 | 7.76 G | 8.70 G | the two moments, over `data` (4.8 G → 0.6 G) |
| *saving* | *2.20 G (22%)* | ***4.04 G (32%)*** | |
| `PeftTrainer`, fsdp=8 | 1.74 G | 2.04 G | params, grads **and** moments, over `fsdp` |

Zero-1's predicted saving is 4.8 − 0.6 = **4.2 G**; it delivers 4.04 G of that at GA=8 but only
2.20 G at GA=1, because with no accumulator live the high-water mark is set partly by the
activation peak instead. **Zero-1's saving is capped by whatever else is at the peak**, which
is why it looks better the deeper the accumulation goes.

FSDP's 2.04 G is not a better-tuned version of the same idea — it shards a strictly larger set
of tensors.

---

# 3. Which to use

| Situation | Use | Why |
| --- | --- | --- |
| Large MoE, GA=1 | either | Parity to 0.5%; pick on ergonomics. Spend the effort on `--ep 8 --ring-of-experts --ragged-sort` instead — 3.45x. |
| Large MoE, GA>1, tunix at MaxText's pin | **engine** | `PeftTrainer` OOMs outright. Capability, not speed. |
| Large MoE, GA>1, tunix past `44a35eeaf` | either | Parity; `PeftTrainer` 1.2% ahead at GA=8. |
| Small dense, GA=1 | `PeftTrainer` | 46.4 ms on FSDP against the engine's 61.0. The engine's ~19 ms/step of host-side NNX graph work is a third of a step this size. |
| Small dense, GA≥2, throughput first | **engine + `--explicit`** | 247.4 ms against 337.4 (Peft/FSDP) and 465.7 (Peft/explicit). The deferred all-reduce is engine-only. |
| Small dense, memory first | `PeftTrainer` + FSDP | 2.04 G against Zero-1's 8.70 G. |
| Need optimizer-state sharding without FSDP | **engine + `--zero1`** | 1.3% for 32% of HBM at GA=8. Requires `--explicit`, which brings the deferral with it. |

Do not read the small-model host overhead as a general engine defect: on the 35b model the
same NNX graph work is 3.1 ms against a 714 ms step, 0.4%.

---

# 4. Traces

All 24 `.xplane.pb` files, full paths. `-engine` is `MaxTextTrainingEngine`; `-maxtext` is the
`PeftTrainer v2` arm (the MaxText *model* under tunix's trainer).

### qwen3-0.6b, Zero-1 and explicit sharding (2026-09-03)

```
gs://chengnuojin-xprof/zero1-qwen3-0p6b-20260903/qwen3-0.6b-engine-dp8-adamw/plugins/profile/run/t1v-n-c9d27794-w-0.xplane.pb
gs://chengnuojin-xprof/zero1-qwen3-0p6b-20260903/qwen3-0.6b-engine-dp8-adamw-explicit/plugins/profile/run/t1v-n-c9d27794-w-0.xplane.pb
gs://chengnuojin-xprof/zero1-qwen3-0p6b-20260903/qwen3-0.6b-engine-dp8-adamw-zero1/plugins/profile/run/t1v-n-c9d27794-w-0.xplane.pb
gs://chengnuojin-xprof/zero1-qwen3-0p6b-20260903/qwen3-0.6b-engine-ga8-dp8-adamw/plugins/profile/run/t1v-n-c9d27794-w-0.xplane.pb
gs://chengnuojin-xprof/zero1-qwen3-0p6b-20260903/qwen3-0.6b-engine-ga8-dp8-adamw-explicit/plugins/profile/run/t1v-n-c9d27794-w-0.xplane.pb
gs://chengnuojin-xprof/zero1-qwen3-0p6b-20260903/qwen3-0.6b-engine-ga8-dp8-adamw-zero1/plugins/profile/run/t1v-n-c9d27794-w-0.xplane.pb
gs://chengnuojin-xprof/zero1-qwen3-0p6b-20260903/qwen3-0.6b-maxtext-dp8-adamw/plugins/profile/run/t1v-n-c9d27794-w-0.xplane.pb
gs://chengnuojin-xprof/zero1-qwen3-0p6b-20260903/qwen3-0.6b-maxtext-ga8-dp8-adamw/plugins/profile/run/t1v-n-c9d27794-w-0.xplane.pb
gs://chengnuojin-xprof/zero1-qwen3-0p6b-20260903/qwen3-0.6b-maxtext-ga8-dp8-adamw-explicit/plugins/profile/run/t1v-n-c9d27794-w-0.xplane.pb
gs://chengnuojin-xprof/zero1-qwen3-0p6b-20260903/qwen3-0.6b-maxtext-adamw/plugins/profile/run/t1v-n-c9d27794-w-0.xplane.pb
gs://chengnuojin-xprof/zero1-qwen3-0p6b-20260903/qwen3-0.6b-maxtext-ga8-adamw/plugins/profile/run/t1v-n-c9d27794-w-0.xplane.pb
```

**The last two paths are the fsdp=8 arm despite not saying so.** `RunSpec.tag()` only annotates
non-default mesh fills and `--fsdp 8` *is* the default on 8 devices, so the FSDP arm gets no
mesh suffix while the `--dp 8` arms get `-dp8`. Cross-check on device time if in doubt: the
fsdp arm's GA=1 `jit__train_step` is 41.07 ms, the dp arm's 50.83 ms.

### qwen3.5-35b-a3b, expert parallelism (2026-09-02)

```
gs://chengnuojin-xprof/ep-parity-qwen35-35b-20260902/qwen3.5-35b-a3b-engine-scan-fsdp1ep8-roe-rsort/plugins/profile/2026_09_02_16_34_54/t1v-n-c9d27794-w-0.xplane.pb
gs://chengnuojin-xprof/ep-parity-qwen35-35b-20260902/qwen3.5-35b-a3b-maxtext-scan-fsdp1ep8-roe-rsort/plugins/profile/2026_09_02_16_37_15/t1v-n-c9d27794-w-0.xplane.pb
gs://chengnuojin-xprof/ep-parity-qwen35-35b-20260902/qwen3.5-35b-a3b-engine-scan-fsdp4tp2/plugins/profile/2026_09_02_16_39_03/t1v-n-c9d27794-w-0.xplane.pb
gs://chengnuojin-xprof/ep-parity-qwen35-35b-20260902/qwen3.5-35b-a3b-maxtext-scan-fsdp4tp2/plugins/profile/2026_09_02_16_40_50/t1v-n-c9d27794-w-0.xplane.pb
```

### qwen3.5-35b-a3b, tensor parallelism and the PR #5060 A/B (2026-09-02)

```
gs://chengnuojin-xprof/pr5060-qwen35-35b-20260902/head/qwen3.5-35b-a3b-engine-scan-fsdp4tp2/plugins/profile/2026_09_02_01_25_39/t1v-n-c9d27794-w-0.xplane.pb
gs://chengnuojin-xprof/pr5060-qwen35-35b-20260902/head/qwen3.5-35b-a3b-maxtext-scan-fsdp4tp2/plugins/profile/2026_09_02_01_27_21/t1v-n-c9d27794-w-0.xplane.pb
gs://chengnuojin-xprof/pr5060-qwen35-35b-20260902/head/qwen3.5-35b-a3b-engine-scan-ga8-fsdp4tp2/plugins/profile/2026_09_02_01_30_53/t1v-n-c9d27794-w-0.xplane.pb
gs://chengnuojin-xprof/pr5060-qwen35-35b-20260902/head/qwen3.5-35b-a3b-engine-scan/plugins/profile/2026_09_02_01_32_50/t1v-n-c9d27794-w-0.xplane.pb
gs://chengnuojin-xprof/pr5060-qwen35-35b-20260902/pre-pr/qwen3.5-35b-a3b-engine-scan/plugins/profile/2026_09_02_03_59_58/t1v-n-c9d27794-w-0.xplane.pb
```

### qwen3.5-35b-a3b, engine GA anatomy (2026-09-02)

```
gs://chengnuojin-xprof/engine-ga-anatomy-20260902/ga2-roe-rsort-engine/plugins/profile/2026_09_02_22_59_47/t1v-n-c9d27794-w-0.xplane.pb
gs://chengnuojin-xprof/engine-ga-anatomy-20260902/ga2-roe-rsort-peft/plugins/profile/2026_09_02_23_05_04/t1v-n-c9d27794-w-0.xplane.pb
gs://chengnuojin-xprof/engine-ga-anatomy-20260902/ga1-roe-engine/plugins/profile/2026_09_02_22_19_55/t1v-n-c9d27794-w-0.xplane.pb
gs://chengnuojin-xprof/engine-ga-anatomy-20260902/ga1-roe-peft/plugins/profile/2026_09_02_22_18_23/t1v-n-c9d27794-w-0.xplane.pb
```

Read any of them with:

```bash
python xplane_device_summary.py --steps 3 <path>.xplane.pb   # per-module device time
python xplane_host_summary.py             <path>.xplane.pb   # host-side dispatch
```

---

# Appendix A: Reproducing

## A.0 Prerequisites

A TPU VM with **8 devices** (these numbers are from a v7-8; other 8-device generations will
give different absolute times but the same comparisons). Then:

```bash
# 1. Fetch PR #5060.
git clone https://github.com/AI-Hypercomputer/maxtext.git && cd maxtext
git fetch origin pull/5060/head:pr5060 && git checkout pr5060

# 2. Install MaxText (see docs/install_maxtext.md) plus the post-training extras --
#    the PeftTrainer arms need tunix, which lives in the post-train extra deps.
install_tpu_post_train_extra_deps

# 3. Everything below runs from this directory. It matters: the arms import
#    perf_parity_common as a sibling module, and a working directory containing a
#    tunix/ checkout would shadow the installed package.
cd tests/end_to_end/tpu/perf_parity
```

The engine-side Zero-1 measured in §2 is **PR #5099**, not #5060. On a #5060-only checkout
every engine arm prints `zero1: UNSUPPORTED` or `zero1: DECLINED (...)` and the Zero-1 rows
become a second copy of the baseline — the runner greps for that line so it cannot pass
unnoticed. The other five arms of §2 and all of §1 reproduce on #5060 alone.

## A.1 One command per study

```bash
./run_qwen3_0p6b_zero1.sh   [outdir]   # §2, six arms x GA{1,8}, ~20 min (~10 min --no-trace)
./run_qwen3_5_35b_a3b.sh    [outdir]   # §1, ~90 min
./run_qwen3_0p6b.sh         [outdir]   # the sgd/tp qwen3-0.6b shape, for reference
```

Each prints its step times, peak HBM and trace paths at the end, and writes one log per arm
into the output directory. They run serially — every arm wants the whole host, and two
concurrent runs will fight over the 8 devices.

## A.2 Individual arms — §1, `qwen3.5-35b-a3b`

```bash
SHAPE="--model qwen3.5-35b-a3b --scan --seq 1024 --ga 1"

# 1.1 headline, both trainers at both meshes
python engine_profile.py       $SHAPE --ep 8 --ring-of-experts --ragged-sort --no-trace
python peft_trainer_profile.py $SHAPE --ep 8 --ring-of-experts --ragged-sort --no-trace
python engine_profile.py       $SHAPE --tp 2 --no-trace
python peft_trainer_profile.py $SHAPE --tp 2 --no-trace

# 1.2 mesh ablation
python engine_profile.py $SHAPE --tp 2 --ep 4 --ring-of-experts --ragged-sort --no-trace
python engine_profile.py $SHAPE --ep 8 --no-trace
python engine_profile.py $SHAPE --ep 8 --ragged-sort --no-trace
python engine_profile.py $SHAPE --ep 8 --ring-of-experts --no-trace

# 1.3 gradient accumulation
for GA in 2 4 8; do
  python engine_profile.py       ${SHAPE/--ga 1/--ga $GA} --ep 8 --ring-of-experts --ragged-sort --no-trace
  python peft_trainer_profile.py ${SHAPE/--ga 1/--ga $GA} --ep 8 --ring-of-experts --ragged-sort --no-trace
done

# Device time.
PERF_PARITY_PROFILE_ROOT=/tmp/traces \
  python engine_profile.py $SHAPE --ep 8 --ring-of-experts --ragged-sort --steps 6
```

`--ring-of-experts` and `--ragged-sort` are rejected at `--ep 1`, mirroring MaxText, which
infers the EP rank from `logical_axis_rules` rather than from `ici_expert_parallelism` and
raises *"When EP rank is 1, use_ring_of_experts must be False"*.

The GA table's `PeftTrainer` head column needs tunix past `44a35eeaf`. Installing it without
touching the repo's pin:

```bash
pip install --no-deps "google-tunix @ https://github.com/google/tunix/archive/07dbe293.zip"
```

## A.3 Individual arms — §2, `qwen3-0.6b`

```bash
SHAPE="--model qwen3-0.6b --seq 1024 --opt adamw"

for GA in 1 8; do
  python engine_profile.py       $SHAPE --dp 8            --no-trace --ga $GA  # auto
  python engine_profile.py       $SHAPE --dp 8 --explicit --no-trace --ga $GA  # explicit
  python engine_profile.py       $SHAPE --dp 8 --zero1    --no-trace --ga $GA  # + Zero-1
  python peft_trainer_profile.py $SHAPE --dp 8            --no-trace --ga $GA
  python peft_trainer_profile.py $SHAPE --dp 8 --explicit --no-trace --ga $GA
  python peft_trainer_profile.py $SHAPE --fsdp 8          --no-trace --ga $GA
done

# Device time. Trace the explicit control too -- shard_mode is not only a layout choice
# here, so the control compiles different kernels from the baseline.
PERF_PARITY_PROFILE_ROOT=/tmp/traces \
  python engine_profile.py $SHAPE --dp 8 --zero1 --steps 6 --ga 8
```

`--zero1` implies `--explicit`. `RunSpec` rejects the three combinations the engine would
reject anyway — Zero-1 with FSDP, with `data` of 1, and with `sgd` — rather than letting them
run and quietly measure the baseline.

## A.4 Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model` | `qwen3-0.6b` | any MaxText `model_name`; only `qwen3_0p6b_tunix_profile.py` is model-locked |
| `--dp / --fsdp / --tp / --ep` | 1 / fills devices / 1 / 1 | mesh axes; the product must equal the device count |
| `--ga` | 1 | micro-batches per optimizer step |
| `--seq`, `--steps` | 1024, 23 | tokens per example; optimizer steps including warmup |
| `--scan` | off | MaxText's scanned decoder (its production default; tunix uses a Python loop) |
| `--opt` | `sgd` | `sgd` or `adamw`, matched across arms |
| `--explicit` | off | `shard_mode=explicit` — Explicit rather than Auto mesh axes |
| `--zero1` | off | `shard_optimizer_over_data`; implies `--explicit`; engine arm only |
| `--ring-of-experts`, `--ragged-sort` | off | MoE kernels; require `--ep > 1` |
| `--no-trace` | traces on | skip xprof — **required for any wall-clock number** |
| `--devices N` | all | use only the first N local devices |
| `PERF_PARITY_PROFILE_ROOT` | `./profiles` | where traces are written |

---

# Appendix B: Measurement notes

These are the traps that produce plausible wrong numbers rather than errors. Each was hit.

**Wall clock must come from `--no-trace` runs.** The profiler charges per dispatch, and the two
trainers dispatch very differently — at qwen3-0.6b/GA=8 the traces record **147.3 module
launches per step on the engine against 18.0 on `PeftTrainer`**. Both run nine substantial
kernels; the engine's other ~138 are small eager dispatches, each a separate host round trip.
Traced GA=8 wall clock runs 345–586 ms against 247–372 untraced, and inflates the arms
unevenly — a traced A/B **reverses the GA=8 ranking outright**. On the 35b model the same
overhead is at most 0.08%, because a 2.3 s step with ~14 dispatches gives it nothing to bite on.

**Per-kernel device times are comparable within a GA setting, not across one.** The same HLO
program hash measures 1.8–4.6 ms slower at GA=1 than at GA=8 in all three engine arms. Every
subtraction in §2.2 is taken within one GA column.

**Tracing lowers the engine's peak HBM by ~2.4 G, and only the engine's.** 12.73 G untraced
against 10.36 G traced at GA=8; both `PeftTrainer` arms are identical either way. Step count is
not the variable — re-running untraced at `--steps 6` reproduces the 23-step figures exactly.
The likely cause is dispatch depth (the engine runs further ahead of the device, and the
profiler's synchronization drains the queue), consistent with the engine-only incidence and the
~2.4 G size, but not verified directly. Memory figures come from `--no-trace` runs.

**Read TPU-busy off the `XLA Modules` line, not `XLA Ops`.** On a scanned MoE the Ops line sums
to roughly twice both the module time and the wall step, because ops inside the scan are
emitted underneath a fusion that already covers them.

**Take the maximum per-execution duration, never a mean or a total-over-steps.** Ragged sort
runs on SparseCore and floods the trace buffer (~1.3 M events against ~18 k at tp=2), dropping
most of the `XLA Modules` line; a clipped event then averages in as a fast step. `maybe_trace`
now passes base.yml's `enable_tpu_profiling_options` to `jax.profiler.trace`, which caps the
capture and keeps every execution; `--no-tpu-profiling-options` restores the raw behaviour.

**A bare `jax.sharding.Mesh(create_device_mesh(...), axes)` silently disables Zero-1.**
`_zero1_active` requires *all* `mesh.axis_types` to be `Explicit`, and that constructor leaves
them `Auto`. The arms build their mesh with `maxtext_utils.get_mesh_from_config(...)`, and
every engine arm prints a `zero1: ACTIVE | DECLINED (reason) | UNSUPPORTED` line — a declined
run is the baseline wearing the feature's name, which is invisible in a step time.

**The engine's own `fwd_bwd / update` split does not decompose the step.** `fwd_bwd` dispatches
asynchronously and the blocking wait lands inside `update()`, so at GA=1 the whole step appears
on the update side. Use the device figures.

**`enable_checkpointing=False` does not stop the engine writing a final checkpoint.** `close()`
ends in `save_checkpoint(..., force=True)` and the manager arms itself on `checkpoint_dir` being
non-empty. On the 35b model in f32 that is a ~140 G write per run, which filled this host's disk
and killed the first sweep. `engine_profile.py` now disarms the manager when the config says
checkpointing is off; the write lands after the timed loop either way.
