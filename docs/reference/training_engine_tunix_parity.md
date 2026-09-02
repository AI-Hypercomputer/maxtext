# MaxText training engine vs. Tunix `PeftTrainer` v2

A head-to-head measurement of `maxtext.training_engine.MaxTextTrainingEngine` against
`tunix.experimental.train.peft_trainer_v2.PeftTrainer`, the trainer introduced in
[google/tunix#1934](https://github.com/google/tunix/pull/1934).

Both trainers are driven over the *same* model instance type, the same loss function, the
same micro-batches and the same `optax` transformation, so every number below is a property
of the trainer, not of the workload.

## Summary

|                                                    | Result                                                                         |
| -------------------------------------------------- | ------------------------------------------------------------------------------ |
| Numerical parity, `gradient_accumulation_steps=1`  | **Equivalent.** Identical loss; gradients agree to `rel_l2` 3.0e-4             |
| Numerical parity, GA > 1 with ragged micro-batches | **Diverge.** `rel_l2` 0.68. MaxText matches the exact gradient; Tunix does not |
| Step time, GA=1, 16 tokens                         | MaxText **34.7 ms** vs. Tunix 64.0 ms (**1.85x**)                              |
| Step time, GA=1, 8192 tokens                       | MaxText 479.6 ms vs. Tunix 493.2 ms (**1.03x — parity**)                       |
| TPU-busy time/step, GA=1, 1024 tokens              | MaxText **44.3 ms** vs. Tunix 63.7 ms (**1.44x**)                              |
| `update()` alone                                   | Tunix **8.2 ms** vs. MaxText 14.6 ms — the one phase Tunix wins                |
| Peak HBM/device, GA=1                              | MaxText **1.78 GiB** vs. Tunix 8.65 GiB (**4.9x**)                             |
| Metrics recorded per step                          | MaxText **23** vs. Tunix 2; `gradient_norm` now matches Tunix bit for bit      |
| Cost of that gradient norm                         | ~0.6 ms/update — 0.7% of a GA=1 step, unmeasurable on device at GA=8           |
| Engine host step path, after the §9 fix            | qwen3-0.6b **1.80x** faster; qwen3.5-35b-a3b 1.004x — a fixed ~70 ms saving    |
| 23-step loop wall clock, after the §9 fix          | qwen3-0.6b 8.9 → **6.3 s**; qwen3.5-35b-a3b 56.3 → **55.7 s**                  |
| Engine host step path at GA=8, after the §9 fix    | qwen3-0.6b **2.19x**/step at identical TPU-busy; 1.45x on loop wall clock      |
| Engine at GA=8, whole PR vs. `59f49ac90^`          | qwen3-0.6b **5.87x** — §9's host fix plus the fused accumulation kernel        |
| Engine vs. `PeftTrainer` at GA=8, identical model  | **2.08x**/step — 589.6 vs. 1225.9 ms, at 96% vs. 50% device utilization        |
| Behaviour change to know about                     | The micro-batch must divide `data x fsdp`, or gradients go NaN — see below     |

**On step time alone the two trainers are equivalent at production sequence lengths.**
MaxText's lead is a short-sequence effect that amortizes away completely: 1.85x at 16
tokens, 1.21x at 2048, **1.03x at 8192** (§4). Do not quote the small-shape numbers as a
general result. The HBM difference, by contrast, is flat in sequence length and does not
amortize.

**The one behaviour change**, as opposed to a timing change, is that
`micro_batch_size_to_train_on` must now be a multiple of `data x fsdp`. Making MaxText's
logical constraints real is what buys the device-side speedup, and it is also what turns an
unshardable batch into NaN gradients under a finite loss:
[the batch has to be shardable](#the-batch-has-to-be-shardable-now).

## Environment

|                           |                                                                              |
| ------------------------- | ---------------------------------------------------------------------------- |
| Host                      | TPU v7x (Ironwood), 4 chips / 8 JAX devices, ~94.7 GiB HBM/device            |
| Mesh                      | `fsdp=8`, all other axes 1                                                   |
| Model                     | Qwen3-0.6B, `scan_layers=True`, f32 weights/activations/grads                |
| Checkpoint                | `gs://maxtext-model-checkpoints/qwen3-0.6b/2025-10-27/scanned/0/items`       |
| Loss                      | `tunix.rl.algo_core.grpo_loss_fn`, `loss_agg_mode="token-mean"`, `beta=0.04` |
| Batch                     | 8 sequences; `--seq` tokens split evenly between prompt and completion       |
| `jax` / `jaxlib` / `flax` | 0.11.1 / 0.11.1 / 0.12.9                                                     |
| MaxText commit            | `671cec44d`                                                                  |
| Tunix                     | source checkout containing `tunix/experimental/train/peft_trainer_v2.py`     |

`gradient_clipping_threshold=0.0` throughout: MaxText clips inside its update kernel and
Tunix never clips, so leaving MaxText's clipping on would mask the very normalization
difference being measured.

## Reproducing

The harness is [`tests/end_to_end/tpu/compare_tunix_trainer.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/tests/end_to_end/tpu/compare_tunix_trainer.py).
Its setup is lifted from `tests/post_training/integration/maxtext_engine_grpo_loss_test.py`;
the ragged-micro-batch method is the one `tests/end_to_end/tpu/compare_training_engine.py`
uses for the engine-vs-`train_step` parity check.

```bash
# Numerics: both trainers in one process, the only way to diff their gradients directly.
python tests/end_to_end/tpu/compare_tunix_trainer.py --trainer=both --ga=1 --batch=8 --iters=8
python tests/end_to_end/tpu/compare_tunix_trainer.py --trainer=both --ga=4 --batch=8 --iters=8 --ragged

# Step time and peak HBM: one trainer per process, so the reading belongs to that trainer
# alone, and no --xprof, so tracing overhead does not land in the wall clock.
python tests/end_to_end/tpu/compare_tunix_trainer.py --trainer=maxtext --ga=1 --batch=8 --seq=1024 --skip-ref
python tests/end_to_end/tpu/compare_tunix_trainer.py --trainer=tunix   --ga=1 --batch=8 --seq=1024 --skip-ref

# Compiled-kernel FLOPs / bytes-accessed, and xplane traces.
python tests/end_to_end/tpu/compare_tunix_trainer.py --trainer=both --ga=1 --skip-ref --kernel-bench
python tests/end_to_end/tpu/compare_tunix_trainer.py --trainer=maxtext --ga=1 --seq=1024 --skip-ref --xprof=/tmp/xprof
```

`--seq` sets the tokens per example, split evenly between prompt and completion. It
defaults to 16 to keep the numerics runs cheap; use a realistic length for any step-time
claim, because the trainers converge as the shape grows (§4).

§9's host-path A/B uses a second harness, `tests/end_to_end/tpu/perf_parity/`, which drives
one arm per process over a synthetic dataset and takes the mesh and model as flags:

```bash
cd tests/end_to_end/tpu/perf_parity   # the arms import perf_parity_common as a sibling

# The two §9 shapes. --no-trace for wall clock; drop it to write an xplane trace.
python engine_profile.py --model qwen3-0.6b       --tp 8          --seq 1024 --no-trace
python engine_profile.py --model qwen3.5-35b-a3b  --tp 2 --scan   --seq 1024 --no-trace

# The same shape under Tunix's trainer, for the trainer-vs-trainer comparison. The tunix
# arm refuses --model/--scan: it implements one architecture and has no scanned variant.
python peft_trainer_profile.py     --tp 8 --seq 1024 --no-trace  # MaxText model, Tunix trainer
python qwen3_0p6b_tunix_profile.py --tp 8 --seq 1024 --no-trace  # Tunix model, Tunix trainer

# The GA=8 rows in §9, on a 4-device host. --ga is the only change; fsdp fills the devices.
# `time` around each of these is where §9's GA=8 process wall clock comes from; the loop
# figure next to it is the arm's own `train() total`.
python engine_profile.py           --ga 8 --no-trace
python peft_trainer_profile.py     --ga 8 --no-trace
python qwen3_0p6b_tunix_profile.py --ga 8 --no-trace

# The matching traces. --steps 6 keeps the xplane under ~1 GiB at this shape.
PERF_PARITY_PROFILE_ROOT=gs://your-bucket/run python engine_profile.py --ga 8 --steps 6
```

Only `qwen3_0p6b_tunix_profile.py` is tied to a model; the other three take `--model`. Two
wrappers run a whole model's sweep end to end and print the steady-state lines and trace
paths at the finish:

```bash
./run_qwen3_0p6b.sh          # all three arms, tp=8 unscanned, GA=1 and GA=8
./run_qwen3_5_35b_a3b.sh     # the two MaxText-side arms, fsdp=4 x tp=2 scanned
```

qwen3.5-35b-a3b is not a drop-in `--model` swap: 2 KV heads make `--tp 8` illegal, the
unscanned decoder OOMs, and `PeftTrainer` cannot accumulate at all on it. Those constraints
and the resulting numbers are in
[`tests/end_to_end/tpu/perf_parity/RESULTS-qwen35-35b-20260902.md`](https://github.com/AI-Hypercomputer/maxtext/blob/main/tests/end_to_end/tpu/perf_parity/RESULTS-qwen35-35b-20260902.md).

For the device side of any of these, `xplane_device_summary.py` reads TPU-busy and module
launches straight off the `.xplane.pb` wire format (no xprof import, which does not resolve
in this venv), and `xplane_host_summary.py` does the same for host events:

```bash
python xplane_device_summary.py --steps 3 <run>/plugins/profile/<ts>/<host>.xplane.pb
```

The "before" rows come from the same commands with only
`src/maxtext/training_engine/{maxtext_engine,inflight_throttler}.py` reverted to the fix's
parent commit, so nothing but the engine differs. §7's cost-of-the-gradient-norm A/B is the
same recipe against the norm commit's parent, at `--ga 1` because that is where one update
per step makes the norm most visible:

```bash
git checkout <norm-commit>^ -- src/maxtext/training_engine/maxtext_engine.py
python engine_profile.py --ga 1 --no-trace   # repeat; the delta is ~0.6 ms/step
git checkout HEAD -- src/maxtext/training_engine/maxtext_engine.py
```

The arms report the steady-state window and nothing else, so §9's wall-clock table was timed
from outside — `time python engine_profile.py …` for the end-to-end row, and the loop
and phase splits from temporary `time.perf_counter()` marks around `engine.compile()` and the
step loop. Reproducing those three rows means re-adding the marks; the last row needs only
`time`.

## 1. End-to-end GRPO test

`tests/post_training/integration/maxtext_engine_grpo_loss_test.py` — **1 passed in 51.67 s**
against the real GCS checkpoint.

Note that this test never calls `engine.compile()`, so it exercises the engine's *eager*
path. That path is roughly 7x slower per update (237 ms vs. 34 ms); see
[Calling `compile()` is load-bearing](#calling-compile-is-load-bearing).

Two things about the test are load-bearing and were not before this PR:

- **The batch is derived from the mesh**, `mesh.shape["data"] * mesh.shape["fsdp"]`, not
  hard-coded. See
  [the batch has to be shardable](#the-batch-has-to-be-shardable-now) — a hard-coded 2 on
  this 4-device host produced NaN gradients under a finite loss.
- **`matmul_precision=highest`.** The KL assertion compares log-probs from two different code
  paths — Tunix's `compute_per_token_logps` for the reference against the engine's sharded
  forward for the policy. At the default bf16 those disagree by ~2e-2 on values near -12.6,
  and `low_var_kl` squares the difference into a KL of ~1.2e-4: noise, but an order of
  magnitude above the 5e-5 the assertion is trying to police. fp32 matmuls put it back under
  the tolerance, and at this model size cost nothing measurable.

## 2. Numerical parity at GA=1

Reference gradient is computed independently and eagerly as the exact
`Σ_i ∇(unreduced_sum_i) / Σ_i denominator_i`.

|                                        | MaxText              | Tunix v2             |
| -------------------------------------- | -------------------- | -------------------- |
| Reported loss                          | -0.26746895909309387 | -0.26746895909309387 |
| Accumulated denominator                | 64.0                 | 1.0                  |
| Effective gradient L2                  | 59.7770076           | 59.7771320           |
| Gradient max abs                       | 4.047993183135986    | 4.047993183135986    |
| Weight-delta `rel_l2` after one update | 0.0016571780955      | 0.0016571779743      |

**MaxText vs. Tunix gradient: `rel_l2` 2.979e-4, max abs diff 0.00339** on gradients whose
max element is 4.05.

Both sit ~5.6e-3 from the eager reference (0.005617 and 0.005610). That residual is
jit + SPMD f32 reduction ordering, not a trainer difference: running MaxText eagerly
reproduces the eager reference bit-exactly.

The two trainers reach the same denominator by different routes — MaxText accumulates the
real token count (64) and divides at the end, Tunix pre-scales each micro-batch and
accumulates a count of micro-steps (1). With one micro-batch these coincide.

## 3. Numerical divergence at GA > 1 with ragged micro-batches

Four micro-batches with valid completion lengths 8 / 2 / 5 / 1, i.e. denominators
64 / 16 / 40 / 8, total 128.

The two candidate normalizations are far apart on this data — computed independently,
mean-of-means vs. sum/sum is **`rel_l2` 0.8876**, max abs diff 2.990.

|                              | Exact reference (sum/sum) | MaxText             | Tunix v2            |
| ---------------------------- | ------------------------- | ------------------- | ------------------- |
| Accumulated denominator      | 128.0                     | **128.0**           | **4.0**             |
| Gradient L2                  | 40.1691955                | 40.1839846          | 52.7629719          |
| `rel_l2` vs. exact reference | —                         | **0.006206**        | **0.887693**        |
| Reported loss                | -0.03790009766817093      | 0.08643750101327896 | 0.08643750101327896 |

**MaxText vs. Tunix gradient: `rel_l2` 0.6760, max abs diff 2.994.**

MaxText reproduces the exact large-batch gradient to the same 6e-3 noise floor it hits at
GA=1. Tunix lands on mean-of-means: its 0.887693 against the exact reference matches the
independently computed mean-of-means distance of 0.887560.

The cause is marked in Tunix's own source. `peft_trainer_v2._fwd_bwd_step` does:

```python
scale = aux.primary_loss.compute_scale()
grads = jax.tree.map(lambda g: g * scale, grads)
# TODO(b/491970038): update denom for sequence packing.
grad_accumulator.add(grads, denom=jnp.asarray(1.0))
```

Each micro-batch is scaled by its *own* `1/denominator` and then accumulated with a
denominator of 1.0, so short micro-batches get the same weight as long ones. MaxText
threads the real per-micro-batch denominator through to the final division.

One caveat on how this shows up in practice: the resulting **weight deltas still agree to
four digits** (`rel_l2` 0.0016488130 vs. 0.0016488636), because Adam normalizes per-element
magnitude. The error is in the gradient *direction*, and the optimizer hides the magnitude
part of it — so this will not show up as a step-size anomaly, only as a slow convergence
difference.

## 4. Step time

Medians over 8 iterations, after 2 untimed warmup updates, one trainer per process and no
profiler attached. `fwd_bwd` is per micro-step, `total` is a whole update.

|             | MaxText `fwd_bwd` | MaxText `update` | MaxText total | Tunix `fwd_bwd` | Tunix `update` | Tunix total | Speedup   |
| ----------- | ----------------- | ---------------- | ------------- | --------------- | -------------- | ----------- | --------- |
| 16 tokens   | 20.4 ms           | 14.3 ms          | **34.7 ms**   | 56.0 ms         | 8.0 ms         | 64.0 ms     | **1.85x** |
| 1024 tokens | 45.6 ms           | 14.0 ms          | **59.7 ms**   | 66.5 ms         | 8.2 ms         | 74.7 ms     | **1.25x** |
| 2048 tokens | 70.3 ms           | 14.0 ms          | **84.5 ms**   | 93.8 ms         | 8.2 ms         | 102.1 ms    | **1.21x** |
| 4096 tokens | 175.1 ms          | 13.9 ms          | **189.1 ms**  | 195.1 ms        | 8.7 ms         | 203.9 ms    | **1.08x** |
| 8192 tokens | 465.0 ms          | 14.4 ms          | **479.6 ms**  | 484.8 ms        | 8.3 ms         | 493.2 ms    | **1.03x** |

These are wall clock with no profiler attached, on purpose (§4b). The 16- and 1024-token
rows were also captured under xprof, so the device-time side of them can be checked:
`gs://chengnuojin-xprof/maxtext-vs-tunix-2026-08-31/{maxtext,tunix}_ga1_bs8_seq{16,1024}/`.
Do not read wall clock off those — attaching the profiler inflates MaxText's more than
Tunix's.

**The step-time advantage amortizes away.** `fwd_bwd` goes 2.75x → 1.46x → 1.34x → 1.11x →
1.04x across the sweep. At 8192 tokens the two trainers are within run-to-run noise of each
other, so **step time is not a reason to prefer either.** The peak-HBM difference (§5) is
flat in sequence length and does *not* amortize; that is the durable one.

The small-shape gap is Tunix's replicated gradient tree — roughly constant extra memory
traffic against a workload that grows. It does not survive: by 4096 tokens the two kernels
access the same bytes and by 8192 MaxText accesses *more* (§6), yet the times have already
converged, because both are compute-bound by then.

`update` is the one phase Tunix wins, by ~6 ms, and it is flat in sequence length — so it
matters at 16 tokens and is a rounding error at 8192. Breaking MaxText's 14.0 ms down:

| MaxText `update()` at 1024 tokens |         |
| --------------------------------- | ------- |
| On device (`jit__update_kernel`)  | 0.71 ms |
| `MetricsRecorder._record_metric`  | ~2.8 ms |
| `nnx.split(self._state)`          | ~1.7 ms |
| Other host-side work              | ~8.8 ms |

Measured by no-op'ing `_record_metric` (13.78 → 10.95 ms) and by timing `nnx.split` on its
own. Tunix's `update` is 8.2 ms wall clock against 0.86 ms on device, so both trainers are
host-bound here; MaxText is just ~6 ms more so. Neither the metrics recorder nor the
`nnx.split` `TODO` in `maxtext_engine.py` accounts for the whole difference on its own.

**§9 closes most of this row.** The `nnx.split` line is now served from a cached pure state
and the metric fetch has been moved off the blocking path; what is left is the two
`nnx.update` publish calls, which are kept on purpose.

This table was measured before §7's always-on gradient norm, so its MaxText column is one
`optax.global_norm`-equivalent reduction short of what the engine does today, while Tunix's
8.2 ms always included one. Re-measured as a two-run A/B at GA=1 on the §9 host — the shape
where the norm is most exposed, since the update kernel runs once per step rather than once
per eight micro-batches — the norm costs **~0.6 ms per update**: 81.8 / 82.1 ms per step with
it against 81.3 / 81.5 without, `update` 65.9 / 66.2 against 65.6 / 65.9 ms. Carrying that
delta across hosts puts this table's `update` at ~14.6 ms against Tunix's 8.2 ms, with the
row now comparing equal work. Nothing else in this section moves: the norm lives entirely
inside `_update_kernel`, so `fwd_bwd` and the totals' sequence-length trend are unchanged.

### Where the `fwd_bwd` gap is *not*

Stripping the Python wrapper off and calling the compiled executable directly:

|          | Full `fwd_bwd` | Bare executable | `nnx.split` cost |
| -------- | -------------- | --------------- | ---------------- |
| MaxText  | 20.5 ms        | 19.1 ms         | 1.49 ms          |
| Tunix v2 | 56.1 ms        | 56.8 ms         | 1.39 ms          |

The wrapper costs ~1.4 ms on both sides. The gap is entirely inside the compiled program.

## 4b. Reading the traces: why xprof flatters Tunix

The xplane traces below are the authority for device time, but the **trace viewer's step
statistics are not usable for MaxText**, and anyone comparing the two side by side will
reach the wrong conclusion unless they know why.

Counting `ph=="X"` events on the `/device:TPU:0` `XLA Modules` line, 8 iterations, GA=1:

| 1024 tokens           | MaxText                            | Tunix                               |
| --------------------- | ---------------------------------- | ----------------------------------- |
| fwd/bwd module        | `jit_first_kernel` 8x **41.74 ms** | `jit__fwd_bwd_step` 8x **62.81 ms** |
| update module         | `jit__update_kernel` 8x 0.71 ms    | `jit__update_step` 8x 0.86 ms       |
| Total TPU-busy        | **354.5 ms**                       | 509.4 ms                            |
| **TPU-busy per step** | **44.3 ms**                        | **63.7 ms** (1.44x)                 |
| XLA module executions | **424**                            | **16**                              |
| `Steps` line entries  | 424, median **0.012 ms**           | 16, median 36.7 ms                  |

Both columns are read straight out of these two traces, which are the same shape captured
back to back:

```text
gs://chengnuojin-xprof/maxtext-vs-tunix-2026-08-31/maxtext_ga1_bs8_seq1024/
gs://chengnuojin-xprof/maxtext-vs-tunix-2026-08-31/tunix_ga1_bs8_seq1024/
```

MaxText runs 424 XLA modules where Tunix runs 16. The extra 408 (~51 per step) are tiny
eager dispatches from `MetricsRecorder._record_metric`, which calls `jnp.atleast_1d` /
`jnp.append` once per metric per micro-step: `jit_atleast_1d` x29, `jit_equal` x4,
`jit__where` x4, `jit_multiply` x5, `jit_maximum` x2, `jit_true_divide` x2, `jit_add` x2.

They cost only ~2.8 ms of wall clock, but they have two effects that matter for profiling:

1. **xprof's step detection splits on module boundaries**, so MaxText's reported step time
   is computed over 424 fake steps with a 0.012 ms median. That statistic is meaningless.
   Read total TPU-busy time instead.
2. In the trace viewer MaxText's device row is a picket fence of sub-0.1 ms slivers next to
   Tunix's two clean bars, which reads as "MaxText is busier and therefore slower". It is
   not: MaxText's row contains 44.3 ms of work per step against Tunix's 63.7 ms.

Attaching the profiler also penalizes MaxText disproportionately, because per-dispatch trace
overhead is multiplied by 51: `update` measures 14.0 ms untraced and 27.0 ms traced, while
Tunix's goes 8.2 → 18.5 ms. **All wall-clock figures in §4 are from untraced runs** for
that reason.

Two further cautions when comparing against traces from elsewhere:

- Trace **after warmup**. Tracing a cold loop drops a ~570 ms
  `PjRtCApiClient::CompileAndLoad(jit__update_kernel)` into the middle of the window, since
  MaxText's donated update kernel re-lowers against the post-donation parameter layout.
  `_time_loop` runs `_WARMUP = 2` untimed updates outside the trace context.
- A MaxText trace and a Tunix trace captured by different people on different models and
  meshes cannot be compared for trainer overhead at all. Only a same-model, same-mesh,
  same-batch, back-to-back run isolates the trainer.

## 5. Peak HBM, and the root cause

Measured one trainer per process.

|                                    | MaxText       | Tunix v2  |
| ---------------------------------- | ------------- | --------- |
| Peak HBM/device, GA=1, 16 tokens   | **1.766 GiB** | 8.636 GiB |
| Peak HBM/device, GA=1, 1024 tokens | **1.782 GiB** | 8.648 GiB |
| Peak HBM/device, GA=1, 2048 tokens | **1.787 GiB** | 8.653 GiB |
| Peak HBM/device, GA=1, 4096 tokens | **1.787 GiB** | 8.652 GiB |
| Peak HBM/device, GA=1, 8192 tokens | **1.791 GiB** | 8.642 GiB |
| Peak HBM/device, GA=2, 1024 tokens | **1.807 GiB** | 4.790 GiB |

Both footprints are flat in sequence length — a 512x increase moves neither, because both
are dominated by parameters and optimizer state, and Tunix's additionally by the replicated
gradient tree. **Unlike the step-time gap (§4), this one does not amortize.**

Gradient-tree placement explains it:

|                                 | MaxText                 | Tunix GA=1   | Tunix GA=2/4      |
| ------------------------------- | ----------------------- | ------------ | ----------------- |
| Largest leaf `[1024, 28, 3072]` | `P('fsdp', None, None)` | **`P()`**    | `P('fsdp',)`      |
| Embedding `[151936, 1024]`      | `P(None, 'fsdp')`       | **`P()`**    | `P(None, 'fsdp')` |
| Gradient tree per device        | 0.278 GiB               | **2.22 GiB** | 0.278 GiB         |

At GA=1 Tunix takes a non-persistent-accumulator fast path: `GradientAccumulator` is built
with `allocate_grads=False`, `self.grads = nnx.data({})`, and `_shard_optimizer`
deliberately skips it, on the reasoning that "the gradients ... are jit outputs whose
sharding XLA derives from the parameters". That does not hold here — `nnx.jit` carries no
`out_shardings`, so the whole gradient tree comes back **fully replicated**, 8x its sharded
size. MaxText pins both `in_shardings` and `out_shardings` on its `_fwd_bwd_kernel`.

**The persistent accumulator that PR #1934 adds fixes the memory, not the compute.** At
GA=2 the accumulator *is* sharded and peak HBM drops 8.65 → 4.79 GiB, but the kernel is
untouched: `fwd_bwd` stays at 71.4 ms and its bytes-accessed at 22.7 GiB, byte-identical to
GA=1.

## 6. Compiled-kernel analysis

From `Compiled.cost_analysis()` / `memory_analysis()` on the `fwd_bwd` executable:

|                      | 16 tok: **MaxText** | 16 tok: Tunix | 1024 tok: **MaxText** | 1024 tok: Tunix |
| -------------------- | ------------------- | ------------- | --------------------- | --------------- |
| GFLOPs               | 17.0                | 18.8          | 1080.7                | 1082.4          |
| Bytes accessed       | **2.24 GiB**        | **16.97 GiB** | **9.58 GiB**          | **22.72 GiB**   |
| Temp size            | 0.257 GiB           | 3.565 GiB     | 1.565 GiB             | 3.928 GiB       |
| Bare-executable time | 19.4 ms             | 56.6 ms       | 43.9 ms               | 67.5 ms         |

Both trainers issue the same arithmetic — at 1024 tokens 1080.7 vs. 1082.4 GFLOP, a 0.2%
difference. What differs at small shapes is memory traffic, and Tunix's excess is roughly
constant, matching the replicated gradient tree it writes and reads regardless of sequence
length:

| Tunix bytes accessed − MaxText | 16 tok        | 1024 tok  | 2048 tok  | 4096 tok | 8192 tok      |
| ------------------------------ | ------------- | --------- | --------- | -------- | ------------- |
|                                | **+14.7 GiB** | +13.1 GiB | +13.0 GiB | −0.7 GiB | **−12.4 GiB** |

The constant explains the small-shape gap and its collapse. It does not explain the large
shapes: by 4096 tokens the two kernels access the same bytes and by 8192 **MaxText accesses
more** (72.3 vs. 59.9 GiB), yet MaxText is still marginally ahead on wall clock. Both are
compute-bound there, and the arithmetic is the same, so the times converge.

Neither kernel is at roofline at the small shapes — 1080 GFLOP in 43.9 ms is ~25 TFLOP/s
across 8 devices — so those numbers characterize kernel *structure* on a 0.6B f32 model
rather than achievable throughput. The 8192-token point is the one to generalize from.

The GA=1 and GA=2 columns are byte-identical (22.72 vs. 22.67 GiB), confirming the
accumulator's sharding never reaches the kernel signature.

The actionable conclusion is that there is nothing to port here: MaxText's explicit
`in_shardings`/`out_shardings` on the fwd/bwd kernel are precisely the thing v2 lacks.

## 7. Metric coverage

MaxText records **23** metrics per step: the full GRPO aux dictionary
(`kl`, `kl_loss`, `entropy`, `ppo_kl`, `pg_clipfrac`, `pg_clipfrac_lower`,
`pg_loss/clipped_mean`, `pg_loss/unclipped_mean`, `reduced_pg_loss`, `unreduced_pg_loss`,
`is_ratio/{min,mean,max}`, `log_ratio/abs_mean`, `sampler_is/{weight_mean,weight_min}`,
`advantage/{min,max,abs_mean,nonzero_frac}`, `loss`) plus `learning_rate` and
`gradient_norm`.

Tunix v2 records **2**: `loss` and `grad_norm`. `PeftTrainer._post_process_train_step` is
`pass` in the base class, so `grpo_loss_fn`'s entire aux dictionary is discarded.

Two further v2 behaviours worth knowing when reading its metrics:

- `_write_train_metrics` deliberately skips the first step so metric I/O overlaps the next
  one. After a single `update()`, `get_metrics()` still returns the empty
  `MetricsBuffer(id=-1)` and every field reads NaN; step 0 is in
  `_prev_buffered_train_metrics`.
- `_log_metrics` calls `np.exp(jax.device_get(loss))`, a host sync on every update.

In the other direction, MaxText's `gradient_norm` used to be emitted only from the
spike-skipping path, so it read NaN whenever `skip_step_on_spikes` was off — which is
base.yml's default. **Fixed**, here and independently in `202a89ab8`, which landed on main
while this was in review: `_update_kernel` now computes it on every update, in float32,
exactly where Tunix's `_update_step` computes `optax.global_norm`. The two agree bit for bit
on the same batch, at GA=1 and under accumulation:

| `--trainer=both`, qwen3-0.6b | MaxText `gradient_norm` | Tunix `grad_norm` |
| ---------------------------- | ----------------------- | ----------------- |
| GA=1, batch 8                | 59.77279281616211       | 59.77279281616211 |
| GA=8, batch 8                | 18.25168800354004       | 18.25168800354004 |

Taken *before* clipping, following `202a89ab8`: that is where Tunix's `optax.global_norm`
sits, since its clipping (when a caller configures any) is a link in the optax chain that
runs afterwards. In `train.py`'s vocabulary this is `learning/raw_grad_norm` rather than
`learning/grad_norm`, and the distinction is invisible in every measurement here —
`gradient_clipping_threshold=0.0` throughout this document, and Tunix never clips at all, so
the two are the same tensor. MaxText therefore records **23** metrics per step, and the
`learning/grad_norm` ↔ `gradient_norm` row of `compare_training_engine.py`'s aux map, which
previously found nothing on the engine side to compare, now runs.

**It costs ~0.6 ms per update, and nothing at all on device.** Measured two ways on
qwen3-0.6b at batch 8 x seq 1024. At GA=1, where one update lands on every step and the norm
is therefore at its most exposed, two runs per arm give 81.8 / 82.1 ms per step with the norm
against 81.3 / 81.5 without — 0.7% of a step. At GA=8 that same fixed cost is spread over
eight micro-batches and disappears: TPU-busy per optimizer step reads 568.0 ms with the norm
against 570.7 without, i.e. below the trace's own noise, and `jit__update_kernel` goes 2.692
→ 2.641 ms. XLA fuses the reduction into an update kernel that already streams every gradient
leaf, so there is no extra pass over the gradients to pay for. **The engine's lead in §9 was
therefore never a matter of skipping work Tunix was doing.**

One consequence worth stating: the always-on norm removed the last reason for the engine's
`_update_marker` helper, a one-element array the update kernel returned purely to give
`InflightThrottler` something cheap to block on. The norm is an output of the same executable,
so it serves that role directly — which is exactly what Tunix v2 does with
`_last_update_grad_norm`.

## 8. Known issue: MaxText logs a different loss than it optimizes

At GA=4 on the ragged batch above:

```text
per-micro losses     [-0.267469, 0.068663, 0.212867, 0.331689]
logged (np.mean)      0.08643750101327896   <- what MaxText reports
correct Σsum/Σdenom  -0.03790009766817093   <- what MaxText's gradient uses (denominator 128)
```

`MetricsRecorder._record_metric` appends one entry per micro-step, so
`WeightedMetric.compute()` returns a vector of per-micro losses rather than a step scalar.
No `aggregation_fn` is registered for `"loss"`, so `MetricsLogger._process_metrics`
(`src/maxtext/training_engine/metrics.py`) falls through to `np.mean` — mean-of-means, the
same normalization the gradient path deliberately avoids. On ragged batches this flips the
sign of the reported loss.

Gradients are unaffected; this is a reporting bug only. The fix is to reduce the weighted
loss metric as `Σ unreduced_sum / Σ denominator` instead of letting it hit the `np.mean`
fallback.

## 9. Host step path: the fix, measured on two models

The `update()` breakdown in §4 and the dispatch count in §7 both point at host work rather
than at kernels. Three changes address it, none of them touching a compiled program: a pure
`nnx.State` mirror carried across steps in place of a per-step `nnx.split` of the module
graph, dropping a `mean_loss` the update kernel discards, and deferring the metrics write
until after the next dispatch so its device-to-host read overlaps live work.

Measured as an A/B on `src/maxtext/training_engine/{maxtext_engine,inflight_throttler}.py`
alone, everything else held fixed. `--no-trace`, 23 steps, last 19 after warmup, batch 8 x
seq 1024, GA=1, f32, `optax.sgd(1e-5)`, no clipping, on the same 8-device v7x host.

| ms/step, untraced                             | median     | mean   | max    | `fwd_bwd` / `update` | trace                                                                              |
| --------------------------------------------- | ---------- | ------ | ------ | -------------------- | ---------------------------------------------------------------------------------- |
| **qwen3-0.6b**, `tp=8`, unscanned — before    | 161.6      | 216.0  | 464.1  | 38.7 / 122.0 ms      | `gs://chengnuojin-xprof/engine-hostpath-20260901/base/qwen3-0.6b-engine-fsdp1tp8/` |
| **qwen3-0.6b**, `tp=8`, unscanned — after     | **89.8**   | 89.8   | 90.2   | 15.9 / 74.0 ms       | `gs://chengnuojin-xprof/engine-hostpath-20260901/head/qwen3-0.6b-engine-fsdp1tp8/` |
| speedup                                       | **1.80x**  | 2.41x  |        |                      |                                                                                    |
| **qwen3.5-35b-a3b**, `tp=2`, scanned — before | 2322.5     | 2339.6 | 2641.8 | 13.6 / 2309.3 ms     | `.../engine-hostpath-20260901/base/qwen3.5-35b-a3b-engine-scan-fsdp4tp2/`          |
| **qwen3.5-35b-a3b**, `tp=2`, scanned — after  | **2314.0** | 2314.0 | 2314.6 | 7.9 / 2306.1 ms      | `.../engine-hostpath-20260901/head/qwen3.5-35b-a3b-engine-scan-fsdp4tp2/`          |
| speedup                                       | 1.004x     | 1.011x |        |                      |                                                                                    |

The `trace` column is the xplane directory each row's shape was separately profiled in;
full paths and the caveat about compilation landing inside those windows are in
[Profiles](#profiles). The timings themselves are from the untraced runs.

Both arms here predate §7's always-on gradient norm, which leaves the A/B itself untouched —
the fix is host-side and the norm is device-side — and puts today's `after` row ~0.6 ms
higher in absolute terms. The GA=8 table below was re-run on top of the norm, so its rows are
current.

**The saving is a fixed quantity of host time, not a percentage**, so the two rows are the
same fix seen from opposite ends. What it is worth on a given workload is set by two things:

- **How big the NNX module graph is.** The cost removed is two graph traversals per step,
  which scale with node count. Scanning collapses a decoder stack into one stacked node
  set, so the *larger* model here has the *smaller* graph — 70 parameter leaves scanned
  against qwen3-0.6b's 310 unscanned — and its `nnx.split` costs 5.0/4.7 ms against
  21.3/22.3 ms. Model size is the wrong intuition for this; `scan_layers` is the right one.
- **How long the device is busy.** Host work that fits inside the step's device time is
  free. qwen3-0.6b at 1024 tokens is ~82 ms of TPU-busy, so ~70 ms of host work was mostly
  exposed; qwen3.5-35b-a3b is 2.3 s of TPU-busy, so nearly all of it hides.

**The tail improves even where the median does not.** qwen3.5-35b-a3b's worst step drops
2641.8 → 2314.6 ms and its mean converges onto its median, a 327 ms jitter reduction on a
model whose median moved 8.5 ms. Each `nnx.split` allocates a large short-lived object
graph twice per step, and the GC pauses that follow land on whichever step is unlucky —
visible as qwen3-0.6b's 464 ms worst step against a 161.6 ms median. Removing the
allocation removes the pauses on both models.

### Gradient accumulation multiplies the saving

The two bullets above set what the fix is worth per *step*. Gradient accumulation changes
the arithmetic, because the removed `nnx.split(model, nnx.Param, ...)` sits in `fwd_bwd`,
which runs once per **micro** step — so the saving scales with GA while the update-side
part does not. GA=1 measures the fix at its weakest.

Re-run at GA=8, everything else held as above and the same A/B on the engine files alone —
`before` is those two files at `98e6886e8^`. A different host from the table above, 4-device
v6e rather than 8-device v7x, so read these rows against each other and not against the GA=1
rows. Both arms compute the per-update gradient norm (§7), so neither is doing less work than
Tunix. Per-micro-batch columns are the same measurement divided by 8: the hook fires once per
micro step and `report(group=8)` sums each run of eight gaps back into an optimizer step.

| qwen3-0.6b, unscanned, batch 8 x seq 1024, GA=8 | ms/step median | ms/step mean | ms/micro median | ms/micro mean | trace                                                                                           |
| ----------------------------------------------- | -------------- | ------------ | --------------- | ------------- | ----------------------------------------------------------------------------------------------- |
| fsdp=4, before                                  | 1289.7         | 1138.2       | 161.2           | 142.3         | `gs://chengnuojin-xprof/engine-gradnorm-ga8-20260901/base-hostpath-only/qwen3-0.6b-engine-ga8/` |
| **fsdp=4, after**                               | **589.6**      | 616.6        | **73.7**        | 77.1          | `gs://chengnuojin-xprof/engine-gradnorm-ga8-20260901/head/qwen3-0.6b-engine-ga8/`               |
| speedup                                         | **2.19x**      | 1.85x        | **2.19x**       | 1.85x         |                                                                                                 |

And the wall clock the same runs took, which is what a caller actually waits for:

| GA=8 wall clock                            | before | after      | speedup   |
| ------------------------------------------ | ------ | ---------- | --------- |
| 23-step loop (`train()` total, compile in) | 39.2 s | **27.1 s** | **1.45x** |
| `time python engine_profile.py --ga 8`     | 64.9 s | **52.6 s** | 1.23x     |

The loop figure tracks the mean, not the median — 23 x (1138.2 - 616.6) = 12.0 s predicted
against 12.1 s observed. The process figure is the same 12.1 s against a fixed ~25 s of
interpreter start, imports, config, tokenizer and engine build that the fix does not touch,
which is the whole reason it reads 1.23x rather than 2.19x.

**2.19x per step at GA=8 against 1.80x at GA=1**, on the same model. The two traces are the
cleanest evidence in this document that the fix is host-only: TPU-busy is **3408.13 vs
3408.12 ms** over the six optimizer steps each trace covers — 568.0 ms/step, agreeing to one
part in 340,000 — and `jit_accum_kernel` carries the *same XLA module hash*
(`13160937502384084184`) on both sides, so the entire 700 ms is host time that stopped being
exposed. Device utilization goes 44% → 96%. What the traces do show changing is the module
*launch* count, 1034 dispatches over 6 steps before against 686 after, which is the deferred
metrics write rather than the graph walk.

The GA scaling is why 2.19x beats 1.80x, but only modestly: the graph walk is per micro step
and so is paid 8 times, while `update`'s share is paid once. Against that, the more
micro-steps there are the more device time there is per optimizer step for host work to hide
behind, which pushes the other way.

**The whole engine change in this PR, not just §9's fix, is worth more.** Taking `before` all
the way back to `59f49ac90^` — both engine commits reverted, which is *not* the A/B this
section is about — gives 3458.9 → 589.6 ms, **5.87x**. That extra factor is not host time:

| qwen3-0.6b GA=8, engine at       | ms/step median | ms/micro mean | TPU-busy/step | launches / 6 steps | fwd/bwd module per micro   |
| -------------------------------- | -------------- | ------------- | ------------- | ------------------ | -------------------------- |
| `59f49ac90^` (before the PR)     | 3458.9         | 415.4         | 2291.0 ms     | 14054              | `jit_kernel` 285.8 ms      |
| `98e6886e8^` (after §9's parent) | 1289.7         | 142.3         | 568.0 ms      | 1034               | `jit_accum_kernel` 71.1 ms |
| PR head                          | **589.6**      | **77.1**      | **568.0 ms**  | 686                | `jit_accum_kernel` 71.9 ms |

Trace: `gs://chengnuojin-xprof/engine-gradnorm-ga8-20260901/base/qwen3-0.6b-engine-ga8/`. On
wall clock that progression is 112.7 → 64.9 → **52.6 s** for the whole process and 87.2 →
39.2 → **27.1 s** for the loop. `59f49ac90` is what moves the device: tracing the kernels
under `nn_partitioning.axis_rules` folds accumulation into the compiled program, so the 14054
launches — 2334 per step, close to 310 parameter leaves x 8 micro-batches of eager `jit_add`
— collapse into one `jit_accum_kernel` per micro step, and the fwd/bwd module itself falls
285.8 → 71.1 ms per micro-batch.

#### The batch has to be shardable now

That speedup has a precondition worth stating separately, because it changes behaviour rather
than just timing. Tracing under `nn_partitioning.axis_rules` is what makes MaxText's logical
constraints on the activations *real*; before, they were inert and the activations simply
stayed unsharded. So a micro-batch that the data x fsdp devices cannot split is no longer
merely slow — XLA pads the batch out, and the padded lanes are all-zero sequences, which mask
themselves out of attention entirely. Their contribution comes back as NaN on the pad token's
embedding row. The loss stays finite; only the gradients are poisoned, which is a bad way to
find out.

The condition is exactly `micro_batch_size_to_train_on % (data x fsdp) == 0`. MaxText's splash
kernel already asserts on it — "Batch dimension should be shardable among the devices in data
and fsdp axis", `src/maxtext/layers/attention_op.py` — but `dot_product` does not, so under
`attention=dot_product` it surfaces as a NaN instead of an error. This is how it first showed
up: §1's test hard-coded a batch of 2 and ran on a 4-device host.

Callers that were relying on an undersized micro-batch working are the group affected. In
practice a batch smaller than the device count was already leaving devices idle, so the fix
is the one they wanted anyway.

It also moves the trainer comparison. §4 found the two trainers equivalent on step time at
production sequence lengths, which held at GA=1; at GA=8 they separate, because
`PeftTrainer` walks the graph per micro step too. All three arms compute a per-update
gradient norm, and all three run one optimizer update per eight micro-batches:

| GA=8, fsdp=4, batch 8 x seq 1024 | ms/step median | ms/step mean | ms/micro mean | loop       | process    | TPU-busy/step | util    |
| -------------------------------- | -------------- | ------------ | ------------- | ---------- | ---------- | ------------- | ------- |
| MaxText model + **engine**       | **589.6**      | 616.6        | **77.1**      | **27.1 s** | **52.6 s** | 568.0 ms      | **96%** |
| MaxText model + `PeftTrainer`    | 1225.9         | 1066.6       | 133.3         | 44.7 s     | 67.0 s     | 617.7 ms      | 50%     |
| tunix model + `PeftTrainer`      | 914.8          | 891.9        | 111.5         | 91.1 s     | 113.5 s    | 728.5 ms      | 80%     |

Trainer traces (the two `PeftTrainer` arms are unchanged by this PR, so these are the
captures from the previous revision):

```text
gs://chengnuojin-xprof/engine-hostpath-ga8-20260901/head/qwen3-0.6b-maxtext-ga8/
gs://chengnuojin-xprof/engine-hostpath-ga8-20260901/head/qwen3-0.6b-tunix-ga8/
```

Against the identical model the engine is **2.08x** on the median, and **1.55x** against the
tunix-model baseline. The loop column separates them much further — 1.65x and 3.36x — but
most of that gap is compilation, not steps: the tunix arm spends 72.6 s of its 91.1 s in its
first two steps. Read the loop column for what a run costs and the median for what a step
costs. The `PeftTrainer` + MaxText-model row is bimodal — it alternates between ~700 ms and
~1230 ms steps, which is why its mean sits 159 ms *below* its median — so treat that row as
a range rather than a point.

The traces make the mechanism explicit, and it is not the one the wall clock suggests. The
**MaxText model is the faster of the two on device** — 617.7 vs 728.5 ms of TPU-busy per step
— and still loses by 311 ms of wall clock under the same trainer, because `PeftTrainer` leaves
it only 50% utilized. Swapping the trainer, not the model, is what recovers that: the same
MaxText model under the engine runs at 96%.

**The gradient norm §7 adds is free here.** It is a reduction over the whole gradient tree on
every update, and it moved TPU-busy from 570.7 ms/step (measured before it existed) to 568.0
— i.e. not at all, within run-to-run noise. XLA fuses it into the update kernel, which
already streams every gradient leaf; `jit__update_kernel` runs 2.64 ms with the norm against
2.69 ms for the pre-fix kernel without it. The engine's lead over `PeftTrainer` therefore
never depended on skipping work Tunix was doing.

### End-to-end wall clock

The per-step figures above are medians; total run time follows the **mean**, so it credits
the fix with the tail as well. Same two shapes, same `--no-trace` runs, timed from outside:

| wall clock                        | qwen3-0.6b `tp=8` | qwen3.5-35b-a3b `tp=2 --scan` |
| --------------------------------- | ----------------- | ----------------------------- |
| 23-step loop — before             | 8.9 s             | 56.3 s                        |
| **23-step loop — after**          | **6.3 s**         | **55.7 s**                    |
| in-process total — before → after | 18.0 → **15.4 s** | 62.6 → **62.0 s**             |
| `python engine_profile.py …`      | 27.7 → **25.3 s** | 72.2 → **71.7 s**             |

The loop times are the means times 23, to within measurement noise: qwen3-0.6b 211.4 → 89.8
ms predicts 2.80 s saved against 2.6 s observed, qwen3.5-35b-a3b 2340.0 → 2314.0 ms predicts
0.60 s against 0.6 s observed. On loop time qwen3-0.6b is **1.41x**, less than its 1.80x
median because the before-arm's mean sits 52 ms above its median — the same GC pauses the
previous paragraph describes, which the median hides and the wall clock does not.

Everything outside the loop is fixed per-run cost that the fix does not touch:

| per-run overhead                 | qwen3-0.6b | qwen3.5-35b-a3b |
| -------------------------------- | ---------- | --------------- |
| interpreter + JAX/MaxText import | ~9.9 s     | ~9.7 s          |
| config + tokenizer + dataset     | ~4.7 s     | ~4.2 s          |
| engine build                     | 3.8 s      | 1.6 s           |
| `engine.compile()`               | 0.6 s      | 0.5 s           |

Two readings there are easy to get wrong. `engine.compile()` looks nearly free because
`jax.jit` lowering is lazy — XLA compilation happens on the first call, inside the loop, and
lands in the warmup steps the steady-state window drops (~4.2 s for qwen3-0.6b, ~2.5 s for
qwen3.5-35b-a3b). And the *larger* model builds in less than half the time of the smaller
one, for the same reason it gains less from the fix: scanned, it compiles and constructs one
decoder layer rather than 28.

**Against `PeftTrainer` driving the identical model**, which is the control that isolates
trainer from model, the engine closes to within a few ms on both:

| ms/step, untraced, median          | qwen3-0.6b `tp=8` | qwen3.5-35b-a3b `tp=2 --scan` |
| ---------------------------------- | ----------------- | ----------------------------- |
| engine, before                     | 161.6             | 2322.5                        |
| **engine, after**                  | **89.8**          | **2314.0**                    |
| same MaxText model + `PeftTrainer` | 83.4              | 2303.1                        |
| engine's remaining deficit         | 1.077x            | 1.005x                        |

**`qwen3.5-35b-a3b` constrains the mesh.** It has 2 KV heads, so `--tp 8` is rejected by the
sharding checks, and unscanned it does not fit in HBM at any batch size or remat policy
tried. `--tp 2 --scan` is the shape that runs on 8 devices. Tunix has no implementation of
this architecture, so its column is absent above: the 35b comparison is engine against
`PeftTrainer` over the same MaxText model, not model against model.

Traced, the same A/B reads 271.4 → 92.8 ms and 2350.0 → 2315.2 ms. Those are the numbers
that match the profiles below, and they overstate the win — see §4b.

## Calling `compile()` is load-bearing

`MaxTextTrainingEngine.compile(dummy_batch)` must be called before the first `fwd_bwd`, or
the engine stays on the eager path and dispatches every optimizer primitive separately:

|                           | `update()` | Total per update |
| ------------------------- | ---------- | ---------------- |
| `engine.compile()` called | 14.2 ms    | 34.6 ms          |
| Eager (no `compile()`)    | 237 ms     | ~258 ms          |

Tunix v2 has no equivalent switch — its `compile()` is `pass` and `fwd_bwd`/`update` always
go through `nnx.jit`. The GRPO integration test does not call `compile()`, so anyone reading
timings off that test is reading the eager path.

## Profiles

xplane traces of the steady-state loop only — 2 untimed warmup updates run before
`start_trace`, so no compilation lands inside the window — with `StepTraceAnnotation` step
boundaries and `fwd_bwd`/`update` region annotations:

```text
gs://chengnuojin-xprof/maxtext-vs-tunix-2026-08-31/maxtext_ga1_bs8_seq16/
gs://chengnuojin-xprof/maxtext-vs-tunix-2026-08-31/tunix_ga1_bs8_seq16/
gs://chengnuojin-xprof/maxtext-vs-tunix-2026-08-31/maxtext_ga1_bs8_seq1024/
gs://chengnuojin-xprof/maxtext-vs-tunix-2026-08-31/tunix_ga1_bs8_seq1024/
gs://chengnuojin-xprof/maxtext-vs-tunix-2026-08-31/maxtext_ga2_bs8_seq1024/
gs://chengnuojin-xprof/maxtext-vs-tunix-2026-08-31/tunix_ga2_bs8_seq1024/
```

Each holds `plugins/profile/<timestamp>/t1v-n-c9d27794-w-0.xplane.pb`. Raw JSON results for
every run, traced and untraced, are under `.../raw_results/`. View with:

```bash
tensorboard --logdir gs://chengnuojin-xprof/maxtext-vs-tunix-2026-08-31/maxtext_ga1_bs8_seq1024
```

The §9 A/B traces, engine-only, `before/` being this fix's parent commit and `after/` the
fix, at the two shapes in that section:

```text
gs://chengnuojin-xprof/engine-hostpath-20260901/base/qwen3-0.6b-engine-fsdp1tp8/
gs://chengnuojin-xprof/engine-hostpath-20260901/head/qwen3-0.6b-engine-fsdp1tp8/
gs://chengnuojin-xprof/engine-hostpath-20260901/base/qwen3.5-35b-a3b-engine-scan-fsdp4tp2/
gs://chengnuojin-xprof/engine-hostpath-20260901/head/qwen3.5-35b-a3b-engine-scan-fsdp4tp2/
```

Compilation *is* inside these windows — the engine arm calls `compile()` under the trace on
purpose, so the first step carries it — so read from the steady-state region, not the whole
trace.

The GA=8 traces backing §9's "Gradient accumulation multiplies the saving", on the 4-device
v6e host. `--steps 6` rather than the default 23, to keep the xplane files under ~1 GiB;
3 warmup steps then 3 measured, so there are 6 optimizer steps and 48 micro-batches in each
window. The three engine arms were re-captured once §7's always-on gradient norm landed, so
that what the traces contain is the code the tables measure:

```text
gs://chengnuojin-xprof/engine-gradnorm-ga8-20260901/base/qwen3-0.6b-engine-ga8/                 # 59f49ac90^, both engine commits reverted
gs://chengnuojin-xprof/engine-gradnorm-ga8-20260901/base-hostpath-only/qwen3-0.6b-engine-ga8/   # 98e6886e8^, §9's own A/B
gs://chengnuojin-xprof/engine-gradnorm-ga8-20260901/head/qwen3-0.6b-engine-ga8/
gs://chengnuojin-xprof/engine-hostpath-ga8-20260901/head/qwen3-0.6b-maxtext-ga8/                # PeftTrainer arms, unchanged by this PR
gs://chengnuojin-xprof/engine-hostpath-ga8-20260901/head/qwen3-0.6b-tunix-ga8/
```

The superseded engine captures, taken before the gradient norm was computed on every update,
are still under `gs://chengnuojin-xprof/engine-hostpath-ga8-20260901/`; they read 570.7 vs.
568.0 ms/step of TPU-busy against the ones above, which is what "the norm is free" means.

Each holds `plugins/profile/<timestamp>/t1v-n-76d392d5-w-0.xplane.pb`. The TPU-busy figures
in §9 are the `XLA Ops` line of one core plane divided by 6; the module-launch counts are the
event count on `XLA Modules`, and the per-module times are that line grouped by name.

These traces show the uneven tracing overhead this section warns about, at GA=8 and from the
side that matters: the engine arm reads 592.2 ms/step traced against 589.6 untraced, while
`PeftTrainer` goes 900.7 → 1080.5 (tunix model) and 1178.0 → 1701.5 (MaxText model). Quote
the `--no-trace` wall clock; take only device time from these.

Read **total TPU-busy time**, not the trace viewer's step statistics — see §4b for why the
latter is not meaningful for MaxText.

Wall clock **from the profiled runs** is inflated by tracing overhead, unevenly between the
two trainers, and is not the performance numbers above. Listed only so the traces can be
matched to numbers:

| Run                       | `fwd_bwd` | `update` | Total    | Peak HBM  | TPU-busy/step |
| ------------------------- | --------- | -------- | -------- | --------- | ------------- |
| `maxtext_ga1_bs8_seq16`   | 23.3 ms   | 25.9 ms  | 49.4 ms  | 1.766 GiB | **19.3 ms**   |
| `tunix_ga1_bs8_seq16`     | 59.0 ms   | 19.0 ms  | 78.2 ms  | 8.636 GiB | 53.0 ms       |
| `maxtext_ga1_bs8_seq1024` | 48.7 ms   | 27.0 ms  | 76.1 ms  | 1.782 GiB | **44.3 ms**   |
| `tunix_ga1_bs8_seq1024`   | 70.7 ms   | 18.5 ms  | 89.2 ms  | 8.648 GiB | 63.7 ms       |
| `maxtext_ga2_bs8_seq1024` | 51.3 ms   | 25.8 ms  | 128.0 ms | 1.807 GiB | —             |
| `tunix_ga2_bs8_seq1024`   | 71.4 ms   | 19.5 ms  | 162.2 ms | 4.790 GiB | —             |
