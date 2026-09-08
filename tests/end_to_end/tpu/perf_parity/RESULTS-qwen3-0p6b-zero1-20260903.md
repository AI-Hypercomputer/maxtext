# qwen3-0.6b under Zero-1 and explicit sharding: `MaxTextTrainingEngine` vs tunix `PeftTrainer v2`

The trainer-vs-trainer comparison of `RESULTS-qwen35-35b-ep-20260902.md`, re-run on
qwen3-0.6b with the engine's optimizer-state sharding (`shard_optimizer_over_data`, Zero-1)
as the thing being varied. `PeftTrainer v2` has no Zero-1, so its arms shard the parameters
instead — plain data parallelism and FSDP — which is the "or fsdp/tp for the tunix side"
half of the comparison. Both trainers drive the same MaxText model.

Six arms, each at GA=1 and GA=8. The arms differ in one thing each, so the cost of the mesh
mode and the cost of the feature come apart:

| Arm | Mesh | `shard_mode` | Zero-1 |
| --- | --- | --- | --- |
| engine, baseline | dp=8 | auto | — |
| engine, explicit control | dp=8 | explicit | — |
| engine, Zero-1 | dp=8 | explicit | on |
| `PeftTrainer v2`, DP | dp=8 | auto | n/a |
| `PeftTrainer v2`, DP explicit | dp=8 | explicit | n/a |
| `PeftTrainer v2`, FSDP | fsdp=8 | auto | n/a |

The `PeftTrainer` explicit arm is what keeps the headline honest: `--explicit` is what lets
the engine defer its all-reduce, and without running the same flag on the other trainer there
is no way to tell an engine capability from a property of Explicit axes.

Run 2026-09-03 at `18f4a2332` — PR #5060's scripts merged with PR #5099's engine
(`engine-ga-unreduced`). **The engine-side Zero-1 support is #5099, not #5060.** This
document and the runner it describes are the #5060 half; on a build without #5099 every
engine arm prints `zero1: UNSUPPORTED` or `DECLINED` and the Zero-1 row silently becomes a
second copy of the baseline. The runner greps for that line for exactly this reason.

## Summary

1. **The headline result is not Zero-1, it is the deferred all-reduce that Zero-1 requires.**
   Switching `shard_mode` from `auto` to `explicit` — changing nothing else — takes GA=8 from
   **371.7 ms to 247.4 ms, a 1.50x speedup**. Under Explicit axes the cotangents come out
   `unreduced`, so the data-parallel all-reduce moves out of every micro-batch and into
   `update()`: one all-reduce per optimizer step instead of eight.
2. **The deferral is an engine capability, not a property of Explicit axes.** The same flag on
   `PeftTrainer` makes it **1.31x slower** at GA=8 (356.8 → 465.7 ms) and 14% slower at GA=1 —
   opposite sign, same mesh, same optimizer. Its per-micro-batch step goes *up*, 42.73 →
   56.04 ms, while its update is unchanged: nothing moves out of the micro-batch, and the
   collectives that GSPMD had been free to rearrange under `auto` now get placed literally.
   Same-mesh, same-mode, GA=8: **247.4 ms against 465.7 ms, 1.88x** — the largest
   trainer-attributable gap in this study.
3. **Zero-1 itself is cheap and buys real memory.** On top of the explicit control it costs
   **3.2 ms (1.3%)** at GA=8 and **4.0 ms (6.4%)** at GA=1, and saves **4.04 G of 12.74 G
   (32%)** at GA=8, **2.20 G of 9.96 G (22%)** at GA=1.
4. **All of Zero-1's cost is in `update()`, provably.** The `jit_first_kernel` and
   `jit_accum_kernel` modules carry *identical HLO program hashes* in the explicit and Zero-1
   traces (`1445966014043162594`, `614701509375139023`). Only `jit__update_kernel` differs,
   and it differs by +2.90 ms at GA=8 and +2.85 ms at GA=1.
5. **At depth the engine beats `PeftTrainer` on the same mesh; at GA=1 it does not.** GA=8:
   250.6 ms engine Zero-1 against 356.8 ms PeftTrainer DP (**1.42x**) and 337.4 ms PeftTrainer
   FSDP (**1.35x**). GA=1: 66.1 ms against 56.0 and 46.4 ms — PeftTrainer leads, because at
   GA=1 there is nothing to defer and the engine still pays its second dispatch and ~19 ms of
   host-side nnx graph work.
6. **FSDP is the better memory lever at this model size, and it is not close.** 2.04 G against
   Zero-1's 8.70 G at GA=8. Zero-1 shards the two Adam moments; FSDP shards the parameters,
   the gradients *and* the moments. At 0.6 B the replicated parameters and gradient
   accumulator are the bulk of what is left, and Zero-1 does not touch them. The engine
   forbids combining the two (`configs/types.py` raises on Zero-1 + FSDP), so this is a choice,
   not a stack.

The practical reading: on this engine `--zero1` implies `--explicit`, so turning Zero-1 on at
GA=8 is a 1.48x win overall — but 1.50x of that is the mesh mode and Zero-1 is a 1.3% tax
paid for a third of the memory back.

## Environment

| | |
| --- | --- |
| Host | 8 x TPU7x (v7-8 Ironwood, 4 chips / 8 JAX devices), single process |
| JAX | 0.11.1 |
| Revision | `18f4a2332` (PR #5060 scripts + PR #5099 engine) |
| Model | `qwen3-0.6b` — 28 layers, emb 1024, 16 query / 8 KV heads, head_dim 128, mlp 3072, vocab 151936, tied embeddings; ~0.6 B params ≈ 2.4 G in f32 |
| Mesh | dp=8 (fsdp=tp=ep=1), unscanned; the FSDP arm is fsdp=8 |
| Shape | micro-batch 8 x seq 1024, f32 compute and weights, `remat_policy=none` |
| Optimizer | `adamw`, b1=0.9 b2=0.95 eps=1e-8 wd=0.1, lr 1e-5 constant, no clipping — matched term for term across both trainers |
| Wall clock | 23 steps, median of the last 19, `--no-trace` |
| Device time | `XLA Modules` line, per-execution, from a separate `--steps 6` traced run |
| Peak HBM | `peak_bytes_in_use` off the TPU allocator, max over the 8 devices, from the `--no-trace` runs |

`adamw` rather than the `sgd` the other runners use: Zero-1 shards *parameter-shaped optimizer
state*, and plain SGD has none, so under `sgd` the feature is vacuous. `--dp 8` rather than
`--fsdp 8` for the same reason — Zero-1 shards over the `data` axis and is mutually exclusive
with FSDP.

Zero-1 is gated on five conditions at once (`maxtext_engine._zero1_active`): the flag on,
`shard_mode=explicit`, a non-None mesh, `mesh.shape['data'] > 1`, and **every**
`mesh.axis_types` being `Explicit`. That last one is why the arms build their mesh with
`maxtext_utils.get_mesh_from_config(config, devices=...)`: a bare
`jax.sharding.Mesh(create_device_mesh(...), axes)` leaves every axis `Auto` and silently
disables the feature. The Zero-1 arms' logs confirm the mesh:

```
mesh: Mesh('diloco': 1, 'data': 8, ..., axis_types=(Explicit, Explicit, ... , Explicit))
zero1: ACTIVE
```

## Reproducing

```bash
cd tests/end_to_end/tpu/perf_parity
./run_qwen3_0p6b_zero1.sh /tmp/z1bench       # all five arms, both GA depths, wall clock + traces
```

or by hand:

```bash
SHAPE="--model qwen3-0.6b --seq 1024 --opt adamw"

# Wall clock. --no-trace, because tracing charges per dispatch.
for GA in 1 8; do
  python engine_profile.py       $SHAPE --dp 8            --no-trace --ga $GA   # baseline
  python engine_profile.py       $SHAPE --dp 8 --explicit --no-trace --ga $GA   # control
  python engine_profile.py       $SHAPE --dp 8 --zero1    --no-trace --ga $GA   # feature
  python peft_trainer_profile.py $SHAPE --dp 8            --no-trace --ga $GA
  python peft_trainer_profile.py $SHAPE --dp 8 --explicit --no-trace --ga $GA   # the control
  python peft_trainer_profile.py $SHAPE --fsdp 8          --no-trace --ga $GA
done

# Device time.
PERF_PARITY_PROFILE_ROOT=/tmp/traces python engine_profile.py $SHAPE --dp 8 --zero1 --steps 6 --ga 8
python xplane_device_summary.py --steps 3 /tmp/traces/<arm>/plugins/profile/<ts>/<host>.xplane.pb
```

Both explicit controls are traced. `shard_mode` is not only a layout choice on this engine —
Explicit axes are what let the gradients come out `unreduced` — so the engine's control and
baseline compile to visibly different kernels, and without it there is no way to say how much
of the Zero-1 arm's time is Zero-1. The `PeftTrainer` explicit arm is the other half of that
control: it is what separates an engine capability from a property of the mesh mode.

## 1. Headline: wall clock and peak HBM, five arms

Median ms/step, `--no-trace`, n=19. Peak HBM per device, of 101.72 G available.

| Arm | GA=1 ms | GA=1 HBM | GA=8 ms | GA=8 HBM |
| --- | --- | --- | --- | --- |
| engine, dp=8 auto (baseline) | 61.0 | 9.95 G | 371.7 | 12.73 G |
| engine, dp=8 explicit | 62.1 | 9.96 G | **247.4** | 12.74 G |
| engine, dp=8 explicit + **Zero-1** | 66.1 | **7.76 G** | 250.6 | **8.70 G** |
| `PeftTrainer v2`, dp=8 | 56.0 | 10.00 G | 356.8 | 12.35 G |
| `PeftTrainer v2`, dp=8 explicit | 64.0 | 10.47 G | 465.7 | 12.39 G |
| `PeftTrainer v2`, fsdp=8 | **46.4** | **1.74 G** | 337.4 | **2.04 G** |

Read down the three engine rows: the mesh mode is worth 1.50x at GA=8 and costs 1.8% at GA=1;
Zero-1 costs 1.3% at GA=8 and 6.4% at GA=1 and returns 32% / 22% of the memory.

Read across the GA=8 column: the engine's explicit path is the fastest arm measured, and
Zero-1 gives up 3.2 ms of that lead to undercut every other arm's memory except FSDP's.

**Hold `shard_mode` equal and the ranking inverts**, which is the one comparison in this table
that isolates the trainer:

| dp=8, GA=8 | Engine | `PeftTrainer v2` | Winner |
| --- | --- | --- | --- |
| `shard_mode=auto` | 371.7 ms | 356.8 ms | Peft, 1.04x |
| `shard_mode=explicit` | **247.4 ms** | 465.7 ms | **engine, 1.88x** |

## 2. Where the time goes

Per-execution device cost off the `--steps 6` traces, core 0, with each module's HLO program
hash. A GA=8 step is one `first_kernel`, seven `accum_kernel`, one `_update_kernel`.

**GA=1**

| Arm | `jit_first_kernel` | `jit__update_kernel` | Total |
| --- | --- | --- | --- |
| engine, auto | 43.25 `(11445164…)` | 7.72 `(14013518…)` | **50.97** |
| engine, explicit | 23.21 `(1445966…)` | 32.39 `(10326704…)` | **55.60** |
| engine, explicit + Zero-1 | 23.15 `(1445966…)` | 35.24 `(8707565…)` | **58.39** |
| `PeftTrainer`, dp=8 | `jit__train_step` 50.83 | — | **50.83** |
| `PeftTrainer`, fsdp=8 | `jit__train_step` 41.07 | — | **41.07** |

**GA=8**

| Arm | `jit_first_kernel` | `jit_accum_kernel` x7 | `jit__update_kernel` | Total |
| --- | --- | --- | --- | --- |
| engine, auto | 43.25 `(11445164…)` | 43.57 `(15271212…)` | 5.95 `(14013518…)` | **354.19** |
| engine, explicit | 23.17 `(1445966…)` | 25.03 `(614701…)` | 27.81 `(10326704…)` | **226.19** |
| engine, explicit + Zero-1 | 23.31 `(1445966…)` | 25.07 `(614701…)` | 30.71 `(8707565…)` | **229.51** |
| `PeftTrainer`, dp=8 | `jit__fwd_bwd_step` 42.73 x8 | | `jit__update_step` 6.82 | **348.66** |
| `PeftTrainer`, dp=8 explicit | `jit__fwd_bwd_step` 56.04 x8 | | `jit__update_step` 6.89 | **455.21** |
| `PeftTrainer`, fsdp=8 | `jit__fwd_bwd_step` 40.79 x8 | | `jit__update_step` 1.69 | **328.01** |

### The deferral moves one all-reduce, and the same ~21 ms shows up on both sides of the move

Within the GA=8 trace, going from `auto` to `explicit`:

* the first kernel drops 43.25 → 23.17, **−20.08 ms**
* each accumulate kernel drops 43.57 → 25.03, **−18.54 ms**, seven times
* the update kernel rises 5.95 → 27.81, **+21.86 ms**, once

Those three numbers are the same quantity seen three times: one all-reduce of the ~2.4 G f32
gradient tree over 8 devices, which this host does in about 20 ms. Under `auto` every
micro-batch pays it; under `explicit` the step pays it once. Net at GA=8, −128.00 ms, a 1.566x
device speedup that matches the 1.502x wall-clock speedup.

Using only GA=8's own kernels, the deferral is behind by 1.78 ms at GA=1 (one first kernel,
no accumulates, against the heavier update) and ahead by 16.76 ms at GA=2. **It pays from
GA=2 on.** The GA=1 trace measures the loss at 4.63 ms rather than 1.78 — see the drift note
in §5.

### Zero-1 costs +2.9 ms, entirely in `update()`

The Zero-1 arm's `jit_first_kernel` and `jit_accum_kernel` are the *same programs* as the
explicit control's — hashes `1445966014043162594` and `614701509375139023` match exactly, at
both GA depths. Nothing about forward or backward changes. `jit__update_kernel` gets its own
hash (`8707565451094719151`) and costs **+2.90 ms** at GA=8 and **+2.85 ms** at GA=1 over the
control's. That is the price of replacing the gradient all-reduce with a reduce-scatter plus
an all-gather of the updated parameters.

The consistency of that figure across two very different step shapes is the strongest
evidence in this document that the feature does what it says: it touches the optimizer step
and only the optimizer step.

### `PeftTrainer` has no deferral — Explicit axes cost it 31%

The control that matters: `peft_trainer_profile.py` honours `--explicit` too, so the flag can
be held equal across trainers. Doing so does not give `PeftTrainer` the deferral. It makes it
slower.

| dp=8, GA=8 | per micro-batch | update | step total | wall |
| --- | --- | --- | --- | --- |
| `PeftTrainer`, auto | 42.73 ms | 6.82 | 348.66 | 356.8 |
| `PeftTrainer`, explicit | **56.04 ms** | 6.89 | 455.21 | 465.7 |
| engine, explicit | **25.03 ms** | 27.81 | 226.19 | 247.4 |

Its per-micro-batch cost goes *up* by 13.31 ms and its update is unchanged to within 0.07 ms.
Nothing moved. Under `auto`, GSPMD is free to place and rearrange the data-parallel
collectives; under `explicit` they are placed as written, and tunix's train step has no path
that emits them anywhere but inside the micro-batch. The engine's step is written to exploit
Explicit axes — its `fwd_bwd` returns `unreduced` cotangents for `update()` to reduce — and
that is a property of the trainer, not of the mesh mode.

This is what rules out the alternative reading of §1: that the 1.50x is something any trainer
would get from `shard_mode=explicit`. The same flag is worth 1.50x to one trainer and −1.31x
to the other.

**FSDP does not supply a deferral either.** It cuts the per-micro-batch cost only from 42.73 to
40.79 ms. Sharding the parameters replaces the gradient all-reduce with a parameter all-gather
plus a gradient reduce-scatter — about the same traffic, still once per micro-batch. The
engine's explicit path runs the same micro-batch in 25.03 ms, **1.63x faster**, because it does
not run the collective there at all.

## 3. Where the memory goes

At ~0.6 B parameters, f32: parameters 2.4 G, Adam `mu` + `nu` 4.8 G, gradients 2.4 G.

| | GA=1 | GA=8 | What is sharded |
| --- | --- | --- | --- |
| engine, dp=8 auto | 9.95 G | 12.73 G | nothing — params, moments and grads all replicated |
| engine, + Zero-1 | 7.76 G | 8.70 G | the two moments, over `data` (4.8 G → 0.6 G) |
| saving | 2.20 G (22%) | **4.04 G (32%)** | |
| `PeftTrainer`, fsdp=8 | 1.74 G | 2.04 G | params, grads and moments all over `fsdp` |

Zero-1's predicted saving is 4.8 − 0.6 = **4.2 G**. At GA=8 it delivers 4.04 G of that. At
GA=1 it delivers only 2.20 G, because with no accumulator live the high-water mark is set
partly by the forward/backward activation peak rather than by the optimizer step — shrinking
the moments below that other peak stops paying. **Zero-1's saving is capped by whatever else
is at the peak**, which is why it looks better the deeper the accumulation goes.

FSDP's 2.04 G is not a better-tuned version of the same idea; it shards a strictly larger set
of tensors. Zero-1's advantage is that it adds no collectives to the forward and backward
pass, which is what §2 measures — not that it saves more bytes. It does not.

## 4. Traces

All eleven `.xplane.pb` files, from the `--steps 6` traced runs:

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

Reading the names: `-maxtext` is the `PeftTrainer v2` arm (the MaxText *model* under tunix's
trainer), `-engine` is `MaxTextTrainingEngine`, `-ga8` is GA=8 and its absence GA=1.

**The last two paths are the fsdp=8 arm, despite not saying so.** `RunSpec.tag()` only
annotates non-default mesh fills, and `--fsdp 8` is the default, so the FSDP arm gets no mesh
suffix while the `--dp 8` arms get `-dp8`. `qwen3-0.6b-maxtext-adamw` is fsdp=8;
`qwen3-0.6b-maxtext-dp8-adamw` is dp=8. Cross-check against §2 if in doubt: the fsdp arm's
GA=1 `jit__train_step` is 41.07 ms, the dp arm's is 50.83 ms.

## 5. Measurement notes

**Per-kernel times are comparable within a GA setting, not across one.** The same HLO program
measures systematically slower at GA=1 than at GA=8 in all three engine arms:
`jit__update_kernel` hash `14013518…` is 7.72 ms at GA=1 and 5.95 at GA=8; `10326704…` is
32.39 against 27.81; `8707565…` is 35.24 against 30.71. Identical programs, a consistent
1.8–4.6 ms spread in one direction. Every subtraction in §2 is therefore taken within one GA
column, and the two estimates of the deferral's GA=1 cost (1.78 ms extrapolated from GA=8's
kernels, 4.63 ms measured in the GA=1 trace) differ by exactly this effect.

**Wall clock comes from `--no-trace` runs, never from traced ones.** The profiler charges per
dispatch, and the two trainers dispatch very differently: at GA=8 the traces record **147.3
module launches per step on the engine against 18.0 on `PeftTrainer`**. Both run nine
substantial kernels; the engine's other ~138 are small eager dispatches (`jit_atleast_1d`,
`jit_append`, `jit_equal` and friends, 0.04–0.2 ms each on device but each one a separate
host round trip). Traced GA=8 wall
clock runs 345–586 ms against 247–371 ms untraced, and it inflates the arms unevenly
(`PeftTrainer` DP goes 356.8 → 585.8, engine explicit 247.4 → 351.9). A traced A/B would have
reversed the GA=8 ranking outright.

**Tracing also lowers the engine's peak HBM, so the memory table uses the untraced runs.**
The engine's GA=8 baseline reports 12.73 G untraced and 10.36 G traced; the Zero-1 arm 8.70 G
against 7.76 G. Both PeftTrainer arms report identical figures either way (12.35 / 2.04 G).
Step count is not the variable — re-running the untraced arms at `--steps 6` reproduces 12.73
and 8.70 exactly, matching the 23-step runs. The likely cause is dispatch depth: the engine
runs further ahead of the device than PeftTrainer does, so more steps' buffers are live at
once, and the profiler's synchronization drains that queue. That is consistent with the
engine-only incidence and with the ~2.4 G size of the gap (one extra in-flight gradient tree),
but it was not verified directly.

**`peak_bytes_in_use` is a process-lifetime high-water mark**, so it covers compilation as
well as steady state, and it is read after the loop. It is reported for all 8 devices rather
than device 0 — Zero-1 shards the moments over `data`, and a lopsided layout across replicas
would show up here and nowhere else. It did not: the spread is ≤0.03 G on every arm.

**The engine pays ~19 ms/step of host-side nnx graph work** that no device trace shows —
`update(state)` 16.4 ms and `update(model)` 2.7–3.2 ms, near-identical across all six engine
runs. At GA=8 that is 5% of the step and invisible; at GA=1 it is a third of it, and it is a
large part of why `PeftTrainer` leads at GA=1 despite a longer device time in the DP arm
(50.83 ms against the engine baseline's 50.97 — a dead heat on device, 56.0 against 61.0 on
the clock).

**Compilation is not counted anywhere above** but is worth knowing: the FSDP arm takes 71.3 s
of `train() total` at GA=1 against the DP arm's 8.8 s, almost all of it XLA partitioning the
sharded model.

## 6. Flags deliberately left alone

`scan_layers=False` and `remat_policy=none` on both arms, matching
`RESULTS-qwen35-35b-20260902.md` and the rest of this rig: tunix builds its decoder layers as
a Python loop, so MaxText's default `nn.scan` would compare compilation strategies rather than
implementations, and tunix's `RematConfig.NONE` does not rematerialize. `dtype=float32` for
the same reason — tunix's `ModelConfig` defaults to f32 while base.yml runs bf16 compute over
f32 weights, and left alone MaxText would win on numerics instead of on implementation.

`attention: autoselected` is *not* equalized. MaxText picks its own TPU kernel; that is part
of what is being compared, and the choice is logged.

## 7. When to turn Zero-1 on

From these five arms, on 8 devices:

* **Not at GA=1.** It costs 6.4% and the memory it frees is capped by the activation peak.
* **At GA≥2, if you cannot use FSDP.** 1.3% for a third of the HBM is a good trade, and
  `--explicit` brings the deferred all-reduce along with it whether or not you want Zero-1.
* **Prefer FSDP if the model is small enough for its collectives to be cheap.** At 0.6 B it
  wins on memory by 4.3x and on GA=1 speed by 1.4x. The engine forbids Zero-1 + FSDP together.
* **The gap to watch is the one Zero-1 does not close**: at GA=8 the engine's explicit path is
  1.36x faster than PeftTrainer FSDP but uses 4.3x the memory. Neither trainer currently
  offers both.

For the sgd/tp shape the other runners use, see `run_qwen3_0p6b.sh` and
`RESULTS-qwen35-35b-20260902.md`.
