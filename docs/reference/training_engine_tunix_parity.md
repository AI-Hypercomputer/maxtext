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
| `update()` alone                                   | Tunix **8.2 ms** vs. MaxText 14.0 ms — the one phase Tunix wins                |
| Peak HBM/device, GA=1                              | MaxText **1.78 GiB** vs. Tunix 8.65 GiB (**4.9x**)                             |
| Metrics recorded per step                          | MaxText **22** vs. Tunix 2                                                     |

**On step time alone the two trainers are equivalent at production sequence lengths.**
MaxText's lead is a short-sequence effect that amortizes away completely: 1.85x at 16
tokens, 1.21x at 2048, **1.03x at 8192** (§4). Do not quote the small-shape numbers as a
general result. The HBM difference, by contrast, is flat in sequence length and does not
amortize.

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

## 1. End-to-end GRPO test

`tests/post_training/integration/maxtext_engine_grpo_loss_test.py` — **1 passed in 50.32 s**
against the real GCS checkpoint.

Note that this test never calls `engine.compile()`, so it exercises the engine's *eager*
path. That path is roughly 7x slower per update (237 ms vs. 34 ms); see
[Calling `compile()` is load-bearing](#calling-compile-is-load-bearing).

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

MaxText records **22** metrics per step: the full GRPO aux dictionary
(`kl`, `kl_loss`, `entropy`, `ppo_kl`, `pg_clipfrac`, `pg_clipfrac_lower`,
`pg_loss/clipped_mean`, `pg_loss/unclipped_mean`, `reduced_pg_loss`, `unreduced_pg_loss`,
`is_ratio/{min,mean,max}`, `log_ratio/abs_mean`, `sampler_is/{weight_mean,weight_min}`,
`advantage/{min,max,abs_mean,nonzero_frac}`, `loss`) plus `learning_rate`.

Tunix v2 records **2**: `loss` and `grad_norm`. `PeftTrainer._post_process_train_step` is
`pass` in the base class, so `grpo_loss_fn`'s entire aux dictionary is discarded.

Two further v2 behaviours worth knowing when reading its metrics:

- `_write_train_metrics` deliberately skips the first step so metric I/O overlaps the next
  one. After a single `update()`, `get_metrics()` still returns the empty
  `MetricsBuffer(id=-1)` and every field reads NaN; step 0 is in
  `_prev_buffered_train_metrics`.
- `_log_metrics` calls `np.exp(jax.device_get(loss))`, a host sync on every update.

In the other direction, MaxText's `gradient_norm` is only emitted from the clipping path,
so it reads NaN whenever `gradient_clipping_threshold=0.0`. Tunix reports it
unconditionally (59.777 at GA=1).

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
