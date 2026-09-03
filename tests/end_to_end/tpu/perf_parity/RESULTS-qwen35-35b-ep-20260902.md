# qwen3.5-35b-a3b under expert parallelism: `MaxTextTrainingEngine` vs tunix `PeftTrainer v2`

The same trainer-vs-trainer comparison as `RESULTS-qwen35-35b-20260902.md`, re-run with the
8 devices spent on **expert parallelism** instead of tensor parallelism, and with
`use_ring_of_experts` and `use_ragged_sort` on. Both trainers drive the same MaxText model,
so what is varied is the trainer; the mesh and the two MoE kernel flags are varied
alongside, one at a time, in §3.

Run 2026-09-02 on branch `push-5060-scripts` (`1b75c2479`, PR #5060 plus the scripts commit).
The `--tp 2` rows here were re-measured at that revision rather than quoted from the earlier
document, so every number in §1 and §3 comes from one code state.

## Summary

1. **EP=8 is legal on this model and it is the right mesh.** 256 experts divide 8 ways
   exactly, and the 2-KV-head limit that rules out `--tp 8` does not apply to the expert
   axis. `--ep 8 --ring-of-experts --ragged-sort` runs at **670.3 ms/step** (engine) against
   **2313.8 ms** at the published fsdp=4 x tp=2 — a **3.45x** speedup, all of it device-side.
2. **The two trainers stay at parity, and stay device-bound.** 670.3 ms engine against
   666.6 ms `PeftTrainer v2`, both above 99% device utilization. Cutting the device step 3.5x
   did not expose any host-side difference between them.
3. **The mesh and the kernels contribute separately.** EP=8 with no MoE flags is worth 2.17x
   on its own; ring-of-experts adds 1.48x on top of that; ragged sort adds a further 1.07x
   *only when ring-of-experts is on* — by itself it is worth 1.5%.
4. **The tp=2 x ep=4 fallback is not worth taking.** 1726.9 ms against 670.3 ms at pure
   ep=8. Every device given to the tensor axis is a device taken off the expert axis, and on
   this model the expert axis pays far better.
5. **Gradient accumulation works on both trainers once tunix is current.** At the revision
   MaxText pins (`c4ec573`) `PeftTrainer v2` OOMs at every `--ga > 1` on either mesh, needing
   161.41 G against 94.74 G. Tunix `44a35eeaf` fixes it with one line — `reset()` zeroing the
   accumulator with `v[...] * 0` instead of `jnp.zeros_like`, which loses sharding inside the
   traced update and materialises the full 129 G tree. At tunix head the two trainers are at
   parity at every depth, with `PeftTrainer` 1.2% ahead by GA=8. §6.

## Environment

| | |
| --- | --- |
| Host | 8 x TPU7x (v7-8 Ironwood, 4 chips / 8 JAX devices), single process |
| JAX | 0.11.1 |
| Model | `qwen3.5-35b-a3b` — 40 layers, 256 routed + 1 shared expert, emb 2048, 16 query / 2 KV heads, head_dim 256, vocab 248320 |
| Mesh | fsdp=1 x tp=1 x **ep=8**, scanned (`--ep 8 --scan`); baseline fsdp=4 x tp=2 |
| Shape | micro-batch 8 x seq 1024, f32 compute and weights, `remat_policy=none`, GA=1 |
| Optimizer | `optax.sgd(1e-5)`, constant schedule, no clipping |
| Wall clock | 23 steps, median of the last 19, `--no-trace` |
| Device time | `XLA Modules` line, **max** per-execution, from a separate `--steps 6` traced run — see the first measurement note in §5 |

`fsdp=1` is not a loss of sharding. `base.yml`'s `embed` rule carries `expert`, so the dense
and attention weights still shard 8 ways over the expert axis; `['exp', 'expert']` shards the
routed experts. Nothing is replicated that fsdp=8 would have split.

## Reproducing

```bash
cd tests/end_to_end/tpu/perf_parity
SHAPE="--model qwen3.5-35b-a3b --scan --seq 1024 --ga 1"

# Wall clock. --no-trace, because tracing charges per dispatch.
python engine_profile.py       $SHAPE --ep 8 --ring-of-experts --ragged-sort --no-trace
python peft_trainer_profile.py $SHAPE --ep 8 --ring-of-experts --ragged-sort --no-trace

# Baseline, same revision.
python engine_profile.py       $SHAPE --tp 2 --no-trace
python peft_trainer_profile.py $SHAPE --tp 2 --no-trace

# Device time.
PERF_PARITY_PROFILE_ROOT=/tmp/traces \
  python engine_profile.py $SHAPE --ep 8 --ring-of-experts --ragged-sort --steps 6
python xplane_device_summary.py --steps 3 /tmp/traces/<arm>/plugins/profile/<ts>/<host>.xplane.pb

# §6, gradient accumulation. The engine takes any depth; PeftTrainer OOMs above 1.
for GA in 2 4 8; do
  python engine_profile.py       ${SHAPE/--ga 1/--ga $GA} --ep 8 --ring-of-experts --ragged-sort --no-trace
  python peft_trainer_profile.py ${SHAPE/--ga 1/--ga $GA} --ep 8 --ring-of-experts --ragged-sort --no-trace
done
```

`--ring-of-experts` and `--ragged-sort` are rejected at `--ep 1`. That mirrors MaxText, which
infers the EP rank from `logical_axis_rules` (the `exp` rule maps to the `expert` physical
axis) rather than from `ici_expert_parallelism` directly, and raises *"When EP rank is 1,
use_ring_of_experts must be False"* — `configs/types.py`.

## 1. Headline: trainer vs trainer at EP=8, GA=1

GA=1 because that is the only depth both arms run at the tunix revision MaxText pins. It is not
a limit of the mesh — §6 shows one line of tunix `44a35eeaf` lifts it, and re-measures both
arms up to GA=8.

| Arm (MaxText qwen3.5-35b-a3b on both sides) | Mesh | Median ms/step | Mean | Max | TPU-busy/step | Util | Loop |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MaxText model + **engine** | ep=8, ring+ragged | 670.3 | 670.4 | 671.0 | 664.9 ms | **99.2%** | 19.3 s |
| MaxText model + **`PeftTrainer v2`** | ep=8, ring+ragged | **666.6** | 666.6 | 668.5 | 663.0 ms | **99.5%** | 20.7 s |
| MaxText model + engine | fsdp=4 x tp=2 | 2313.8 | 2313.9 | 2314.7 | 2307.8 ms | 99.7% | 56.3 s |
| MaxText model + `PeftTrainer v2` | fsdp=4 x tp=2 | 2303.1 | 2303.3 | 2305.9 | 2299.5 ms | 99.8% | 55.8 s |

The engine trails `PeftTrainer` by 3.7 ms (0.55%) at ep=8 and by 10.7 ms (0.46%) at tp=2 —
the same gap in both places, and in both places it is the engine's second dispatch. The
engine splits fwd/bwd from the optimizer update into two separately jitted programs, so it
pays `jit__update_kernel` (16.0 ms at ep=8, 16.4 ms at tp=2) as a distinct launch where
`PeftTrainer` folds the update into one `jit__train_step`. That constant, not anything about
the mesh, is the whole difference.

Both arms are still device-bound after the 3.5x cut. There was room for the mesh change to
turn this into a host-bound loop and it did not.

## 2. Speedup over the published tp=2 baseline

| | Engine | `PeftTrainer v2` |
| --- | --- | --- |
| fsdp=4 x tp=2 (baseline) | 2313.8 ms | 2303.1 ms |
| ep=8 + ring of experts + ragged sort | **670.3 ms** | **666.6 ms** |
| Speedup | **3.45x** | **3.46x** |
| Device time, tp=2 -> ep=8 | 2307.8 -> 664.9 ms (3.47x) | 2299.5 -> 663.0 ms (3.47x) |

Wall and device speedups agree to two decimal places on both arms, which is what says the
gain is real work removed rather than overhead rearranged.

## 3. What each piece is worth

Every row is `--model qwen3.5-35b-a3b --scan --seq 1024 --ga 1 --no-trace`, median of 19
steps. "vs previous" compares each row to the one above it on the engine arm.

| Mesh and flags | Engine | `PeftTrainer v2` | vs tp=2 | vs previous |
| --- | --- | --- | --- | --- |
| `--tp 2` (fsdp=4 x tp=2) | 2313.8 | 2303.1 | 1.00x | — |
| `--tp 2 --ep 4 --ring-of-experts --ragged-sort` | 1726.9 | 1704.3 | 1.34x | — |
| `--ep 8` (no MoE flags) | 1063.8 | 1059.7 | **2.17x** | 2.17x over tp=2 |
| `--ep 8 --ragged-sort` | 1047.6 | 1043.4 | 2.21x | 1.02x |
| `--ep 8 --ring-of-experts` | 720.1 | 717.4 | 3.21x | 1.48x over plain ep=8 |
| `--ep 8 --ring-of-experts --ragged-sort` | **670.3** | **666.6** | **3.45x** | 1.07x over ring alone |

Three things fall out of this table.

**The mesh is the larger half of the win.** Moving from tp=2 to a bare ep=8 — no kernel
flags at all — is worth 2.17x. Tensor parallelism on this model splits a 2048-wide embedding
and 2 KV heads across the axis and pays all-reduces for it; expert parallelism splits 256
experts, which is what the model actually has a lot of.

**Ragged sort is not independently useful here; it is a multiplier on ring-of-experts.**
Alone it is worth 1.02x (1063.8 -> 1047.6). On top of ring-of-experts it is worth 1.07x
(720.1 -> 670.3). The reason is in `layers/moe.py`: the ring path selects the
`ring_ragged_sort` / `ring_ragged_unsort` kernels only under
`config.use_ragged_sort and config.use_ring_of_experts`, so without the ring the flag reaches
a much smaller part of the permute path.

**Splitting the mesh between the two axes is the worst of both.** `--tp 2 --ep 4` with both
kernel flags on lands at 1726.9 ms — slower than a bare `--ep 8` with no flags at all
(1063.8 ms). Halving the expert axis costs more than the flags recover.

## 4. Traces

All four `.xplane.pb` files, from the `--steps 6` traced runs:

```
gs://chengnuojin-xprof/ep-parity-qwen35-35b-20260902/qwen3.5-35b-a3b-engine-scan-fsdp1ep8-roe-rsort/plugins/profile/2026_09_02_16_34_54/t1v-n-c9d27794-w-0.xplane.pb
gs://chengnuojin-xprof/ep-parity-qwen35-35b-20260902/qwen3.5-35b-a3b-maxtext-scan-fsdp1ep8-roe-rsort/plugins/profile/2026_09_02_16_37_15/t1v-n-c9d27794-w-0.xplane.pb
gs://chengnuojin-xprof/ep-parity-qwen35-35b-20260902/qwen3.5-35b-a3b-engine-scan-fsdp4tp2/plugins/profile/2026_09_02_16_39_03/t1v-n-c9d27794-w-0.xplane.pb
gs://chengnuojin-xprof/ep-parity-qwen35-35b-20260902/qwen3.5-35b-a3b-maxtext-scan-fsdp4tp2/plugins/profile/2026_09_02_16_40_50/t1v-n-c9d27794-w-0.xplane.pb
```

The `-maxtext` tag is the `PeftTrainer v2` arm — the MaxText *model* under tunix's trainer.
The earlier tp=2 traces at PR head `0d9db2747` are under
`gs://chengnuojin-xprof/pr5060-qwen35-35b-20260902/`.

Per-execution device costs off those traces, core 0:

| Arm | Module | ms/exec |
| --- | --- | --- |
| engine, ep=8 ring+ragged | `jit_first_kernel` + `jit__update_kernel` | 648.9 + 16.0 = **664.9** |
| `PeftTrainer`, ep=8 ring+ragged | `jit__train_step` | **663.0** |
| engine, fsdp=4 x tp=2 | `jit_first_kernel` + `jit__update_kernel` | 2291.4 + 16.4 = **2307.8** |
| `PeftTrainer`, fsdp=4 x tp=2 | `jit__train_step` | **2299.5** |

## 5. Measurement notes

**The EP traces in §4 are truncated, and badly enough that the mean per-execution figure is
wrong.** Ragged sort runs on SparseCore, and the SparseCore planes emit roughly 1.3 M trace
events over a 6-step run (16 `TEC n` lines at ~83 k events each) against ~18 k for the same
run at tp=2. That overflows the device buffer and drops most of the `XLA Modules` line: the
ep=8 `PeftTrainer` trace keeps 2 of 6 executions, at 663.02 ms and 222.30 ms, and the second
is a clipped event, not a fast step. Averaging the two gives 442.7 ms and an apparent 66%
utilization, which is an artifact. **Take the maximum per-execution duration, not the mean,
and never totals divided by steps.** Cross-check against xprof's own `Steps` line, which on
the same trace holds exactly one intact event of 663.0 ms.

**`maybe_trace` now caps the capture, which fixes it.** The rig builds base.yml's
`enable_tpu_profiling_options` advanced configuration -- one chip per task, one SparseCore
tile, two SparseCore cores -- and passes it to `jax.profiler.trace`. A ragged-sort GA=2 run
taken that way keeps **every** execution (5 of 5 `jit_first_kernel`, 5 of 5
`jit_accum_kernel`, 5 of 5 `jit__update_kernel`, all within 1.7 ms of each other), so the
§8 tables are read straight off the production config rather than off a ring-of-experts run
with ragged sort switched off. It also all but removes the tracing overhead: 1338.9 ms
traced against 1338.8 ms untraced. `--no-tpu-profiling-options` restores the raw capture,
which is what produced the §4 traces. The cap is a *coverage* restriction -- the trace holds
one chip's device plane, which is the one these arms read anyway.

`xplane_device_summary.py` reports the mean and divides by `--steps`, so on an EP trace read
its `costliest modules` block and the `ms/exec` column rather than the `per step` line, or
use the per-execution dump the table in §4 came from.

**The engine's `fwd_bwd / update` split does not decompose the step at GA=1.** It reads
7.0 ms per micro / 663.3 ms per step at ep=8. `fwd_bwd` dispatches asynchronously and the
blocking `wait_for_next` lands inside `update()`, so the whole step appears on the update
side. The device figures in §4 are the ones that split it.

**Tracing overhead is negligible on this model at either mesh**, as at tp=2 before: 672.5 ms
traced against 670.3 ms untraced on the engine arm (0.33%), 667.6 against 666.6 on
`PeftTrainer` (0.15%).

**`enable_checkpointing=False` does not stop the engine writing a final checkpoint.**
`close()` ends in `save_checkpoint(..., force=True)`, and the engine's `CheckpointManager`
arms itself on `checkpoint_dir` being non-empty rather than on `enable_checkpointing` —
and `checkpoint_dir` is recomputed by pyconfig from `base_output_directory` + `run_name`, so
it cannot be cleared from the config either. On this model in f32 that is a ~140 G write per
run, which filled this host's disk and killed the first attempt at the sweep.
`engine_profile.py` now disarms the manager when the config says checkpointing is off. This
is not a change in what is measured — the write lands after the timed loop — but it is why
the arm can complete at all here. It arrived with `b21ecb72c`, after the tp=2 numbers in
`RESULTS-qwen35-35b-20260902.md` were taken, which is the other reason those rows were
re-measured rather than quoted.

## 6. Gradient accumulation

Same shape as §1, `--ga N` on both arms, median of 19 steps. The `PeftTrainer` column is split
by tunix revision, because the answer changes between them.

| GA | Engine, ep=8 ring+ragged | per micro | `PeftTrainer`, tunix `c4ec573` | `PeftTrainer`, tunix `07dbe293` | per micro |
| --- | --- | --- | --- | --- | --- |
| 1 | 670.3 ms | 670.3 | 666.6 ms | 667.3 ms | 667.3 |
| 2 | 1338.8 | 669.4 | **OOM** | 1339.6 | 669.8 |
| 4 | 2672.3 | 668.1 | **OOM** | 2651.8 | 663.0 |
| 8 | 5343.0 | 667.9 | **OOM** | 5278.0 | **659.8** |

And at the tp=2 baseline mesh, for the same comparison:

| GA | Engine, fsdp=4 x tp=2 | per micro | `PeftTrainer`, `c4ec573` | `PeftTrainer`, `07dbe293` |
| --- | --- | --- | --- | --- |
| 1 | 2313.8 ms | 2313.8 | 2303.1 ms | 2303.4 ms |
| 2 | 4614.6 | 2307.3 | **OOM** | 4627.8 |
| 8 | 18405.3 | 2300.7 | **OOM** | not run |

`c4ec573` is the revision MaxText pins in `src/dependencies/extra_deps/post_train_github_deps.txt`;
`07dbe293` is tunix `main` as of 2026-09-02, 209 commits later.

**The OOM is fixed at tunix head, and the fix is one line.** Commit **`44a35eeaf`**
(2026-08-19) changes `GradientAccumulator.reset()` from `v.set_value(jnp.zeros_like(v[...]))`
to `v.set_value(v[...] * 0)`, with the comment *"to preserve the buffer's sharding"*. Nothing
else about the accumulator, the update step or the mesh changed.

**That line is the whole cause, tested directly.** Reverting only those three lines on
`07dbe293` — leaving the other 208 commits in place — brings the failure back **byte for
byte**: 161.41 G against 94.74 G, same module, same mesh, same shape. Nothing else in the
209-commit range is load-bearing here.

**Why one `zeros_like` costs 129 G.** The accumulator is correctly sharded when it is *built*:
measured at ep=8/GA=2, `grad_accumulator.grads` is 70 leaves, 129.12 G global, **16.14 G per
device**, with a `NamedSharding` identical leaf-for-leaf to the parameters. Construction was
never the problem — the failure is at compile time for `_update_step`. But `reset()` runs
*inside* that traced function, and there `jnp.zeros_like` does not carry the operand's
sharding through; XLA materialises the full unsharded tree. The arithmetic closes exactly:

```
129.12 G  full parameter tree, unsharded, from the traced zeros_like
+ 16.14 G  the sharded accumulator itself
+ 16.14 G  one more sharded parameter-tree copy
= 161.40 G   measured: 161.41 G
```

`v[...] * 0` is an elementwise op on an already-sharded operand, so the sharding propagates and
the same buffer costs 16.14 G. This is the second time the same trap has been fixed in this
class: `408ca1d95` (2026-08-04) replaced a plain `jnp.zeros` with `jnp.zeros_like` in the
*constructor* "otherwise we see grad_accumulator is put to TPU 0 by default". Eager
`zeros_like` propagates sharding; traced `zeros_like` does not.

It also explains the two facts that made the old figure look inexplicable — the requirement
being identical to the byte across GA=2/4/8 (the buffer is parameter-shaped, not depth-shaped)
and moving only 0.33% between fsdp=4 x tp=2 and ep=8 (129.12 of the 161.41 was never sharded
by any mesh in the first place).

**With it fixed, the two trainers are at parity at every depth, and `PeftTrainer` pulls
slightly ahead as GA grows.** 5278.0 ms against the engine's 5343.0 at GA=8 — 1.2%, or 659.8
against 667.9 per micro-batch. Both amortize their single update over more micro-steps, but
`PeftTrainer` gains more from it: 667.3 -> 659.8 per micro from GA=1 to GA=8 (1.1%) against
670.3 -> 667.9 (0.4%). The engine's fixed `jit__update_kernel` dispatch, worth 16.0 ms at
GA=1, is unchanged in absolute terms and simply divides by more micro-steps on both sides.

**The EP win is independent of accumulation depth.** On the engine, the 3.45x ep=8 buys at
GA=1 is 3.44x at GA=8 (18405.3 -> 5343.0). Per micro-batch the engine is flat to 0.4% across
the whole GA range at ep=8.

**Recommendation:** bump `src/dependencies/extra_deps/post_train_github_deps.txt` past
`44a35eeaf`. That is a dependency change rather than a test-script change, so it is not part
of this branch; the numbers above are from installing `07dbe293` into the venv with
`pip install --no-deps`, which leaves the repo's pin untouched.

## 7. Flags deliberately left alone

| Flag | Why not |
| --- | --- |
| `check_vma` | `base.yml` calls it "recommended for improved performance", but `types.py` makes it mutually exclusive with **both** `use_ragged_sort` and `use_ring_of_experts`. It hard-fails this config. |
| `num_moe_token_chunks` | Legal here (`>1` requires ring-of-experts; 1024 % 4 == 0). Measured on the `PeftTrainer` arm at ep=8 + ring + ragged: **1088.9 ms at 4 chunks against 666.7 ms at 1**, a 1.63x regression at this shape. It also turns `lb_loss` into a per-chunk average, so it is not numerically free either. |
| `moe_chunk_barrier` | A no-op without chunking, and it exists to *defeat* XLA interleaving. Slower by construction. |
| `num_moe_emb_chunks` | Requires `use_gmm_v2=True`, which requires `use_tokamax_gmm=True`. A three-flag change to the MoE backend would stop this being a mesh-only A/B. |
| `ragged_buffer_factor` | Left at `-1.0`, which keeps the buffer dropless. Raising it above 0 drops tokens, so the EP arms would stop being numerically comparable to the tp=2 baseline. |
| `sparse_matmul`, `megablox` | Left at their `base.yml` defaults (both true). All ring-of-experts and ragged-sort code lives under `RoutedMoE.sparse_matmul`; `dense_matmul` contains no reference to either flag, so turning `sparse_matmul` off would silently drop both with no error. |

## 8. Where the engine's own cost sits

The trainer comparison above says the two are level; this section says *why*, and what is
actually worth changing on the engine side. Device figures are per-execution durations from
traces taken in the production config -- ep=8, ring-of-experts, ragged sort -- with the
advanced TPU profiling options of §5 on, so nothing is dropped and no maximum-versus-mean
correction is needed. Memory figures are `Compiled.memory_analysis()` on the engine's own
compiled kernels, per device, against the 94.74 G this chip has.

### Traces

```
gs://chengnuojin-xprof/engine-ga-anatomy-20260902/ga2-roe-rsort-engine/plugins/profile/2026_09_02_22_59_47/t1v-n-c9d27794-w-0.xplane.pb
gs://chengnuojin-xprof/engine-ga-anatomy-20260902/ga2-roe-rsort-peft/plugins/profile/2026_09_02_23_05_04/t1v-n-c9d27794-w-0.xplane.pb
gs://chengnuojin-xprof/engine-ga-anatomy-20260902/ga1-roe-engine/plugins/profile/2026_09_02_22_19_55/t1v-n-c9d27794-w-0.xplane.pb
gs://chengnuojin-xprof/engine-ga-anatomy-20260902/ga1-roe-peft/plugins/profile/2026_09_02_22_18_23/t1v-n-c9d27794-w-0.xplane.pb
```

The first pair is `--ga 2 --ep 8 --ring-of-experts --ragged-sort --steps 5`, taken with the
advanced TPU profiling options on and intact throughout; they are the GA=2 table below. The
second pair is the GA=1 cross-check, ring-of-experts with ragged sort off, from before the
profiler cap existed. `-peft` is the `PeftTrainer v2` arm.

### The kernels

| Kernel | temp | arg | out | alias |
| --- | --- | --- | --- | --- |
| `jit_first_kernel` (fwd/bwd, no accumulator) | 27.01 G | 16.14 G | 16.14 G | 0.00 G |
| `jit_accum_kernel` (fwd/bwd + accumulate) | **49.41 G** | 32.29 G | 16.14 G | 16.14 G |
| `jit__update_kernel` | 0.01 G | 32.29 G | 16.14 G | 16.14 G |

The per-device parameter tree is 16.14 G, so `arg` and `out` are exactly the trees you would
expect and `alias` says donation works: the accumulator and the train state are both written
back in place. The update kernel is free -- 0.01 G of temporaries.

### Device time, GA=2, one step

| | engine | `PeftTrainer` |
| --- | --- | --- |
| fwd/bwd without accumulate | `jit_first_kernel` 649.0 ms | — |
| fwd/bwd with accumulate | `jit_accum_kernel` 667.2 ms | `jit__fwd_bwd_step` 656.6 ms |
| update | `jit__update_kernel` 16.0 ms | `jit__update_step` 21.6 ms |
| **total** | **1332.2 ms** | **1334.8 ms** |
| wall clock for the same step | 1338.9 ms | 1342.1 ms |

Five executions of each, spread under 1.7 ms. The forward/backward is at parity -- take
`jit_first_kernel` as the accumulator-free baseline and `PeftTrainer`'s accumulating step is
649.0 + 7.6 ms against the engine's 649.0 + 18.2. **The whole of the engine's deficit is
that 10.6 ms.** Its update is 5.6 ms cheaper in exchange (`PeftTrainer`'s also zeroes the
accumulator), which is one accumulate's worth, so the two cross over at GA=2: the engine is
2.6 ms ahead there and, projecting 649.0 + 7 x 667.2 + 16.0 = 5335.4 against 8 x 656.6 +
21.6 = 5274.4, 61 ms behind at GA=8. §6 measured 5343.0 against 5278.0.

At GA=1, on a ring-of-experts trace without ragged sort, the engine's two executables come
to 698.4 + 16.0 = 714.4 ms against `PeftTrainer`'s single fused `jit__train_step` at
713.5 ms -- 0.1%, which is what says the split into two dispatches is not itself a cost.

### The accumulate costs 22.40 G, and it is the fusion that costs it

`accum_kernel` needs 22.40 G more temp than `first_kernel` -- 1.39x the parameter tree --
for what is arithmetically one elementwise add into a donated buffer. Four things it is
**not**:

| Ruled out | Evidence |
| --- | --- |
| Donation failing | `alias = 16.14 G`. Removing donation *lowers* temp to 45.80 G but drops the aliasing, so the real peak gets worse. |
| The sharding annotations | Dropping `out_shardings`, dropping `in_shardings` too, and donating only the gradient buffer all give **49.41 G**, identical to the byte. |
| The MoE flags | Plain `--ep 8` with neither ring-of-experts nor ragged sort: 26.50 G -> 48.90 G, the same 22.40 G. |
| `scan_layers` | On dense qwen3-0.6b the delta is 0.001 G scanned and -0.150 G unscanned. XLA fuses the add away completely there. |

What it *is*: the add is being fused into the backward pass and the fusion is what allocates.
Compiled as its own executable the same add needs **0.00 G** of temp and still aliases its
donated accumulator. So splitting the accumulate out of `accum_kernel` would move the step's
peak from 32.29 + 49.41 = 81.70 G to 32.29 + 27.39 (fwd/bwd) + 16.14 (micro-gradients out)
= 75.82 G, about 6% of the chip's HBM, for the same HBM traffic and one extra dispatch. That
is the one change here with a measured payoff.

### A fused GA=1 step would be a pessimization

The obvious next idea -- copy `PeftTrainer`'s `_train_step` and run fwd/bwd + update as one
executable at GA=1 -- was built and measured: **52.97 G of temp against `first_kernel`'s
27.01 G**, because the update's live ranges overlap the backward pass's. Device time is
already a dead heat (714.4 against 713.5 ms). The engine's two-executable split is the
cheaper design at this size; leave it alone.

Likewise, donating the gradient tree into `_update_kernel` -- which the code declines to do,
on the grounds that JAX would only warn -- is worth exactly nothing: compiled with
`donate_argnums=(0, 1)` the kernel reports the same 0.01 G temp, 32.29 G arg, 16.14 G alias.

### Host time

Hidden here, but not zero. The engine's own counters charge 3.1 ms per step to NNX graph
work that survives the pure-state cache (`nnx.update(model)` 1.0 ms, `nnx.update(state)`
2.1 ms; the two `split` calls, 9.5 ms, are cached away). Against a 714 ms step that is 0.4%.
On a small model it would not be.
