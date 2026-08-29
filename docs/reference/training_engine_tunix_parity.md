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
| Step time, GA=1                                    | MaxText **34.9 ms** vs. Tunix 66.8 ms (**1.91x**)                              |
| Step time, GA=4                                    | MaxText **103.6 ms** vs. Tunix 236.6 ms (**2.28x**)                            |
| Peak HBM/device, GA=1                              | MaxText **1.77 GiB** vs. Tunix 7.52 GiB (**4.3x**)                             |
| Metrics recorded per step                          | MaxText **22** vs. Tunix 2                                                     |

## Environment

|                           |                                                                              |
| ------------------------- | ---------------------------------------------------------------------------- |
| Host                      | TPU v7x (Ironwood), 4 chips / 8 JAX devices, ~94.7 GiB HBM/device            |
| Mesh                      | `fsdp=8`, all other axes 1                                                   |
| Model                     | Qwen3-0.6B, `scan_layers=True`, f32 weights/activations/grads                |
| Checkpoint                | `gs://maxtext-model-checkpoints/qwen3-0.6b/2025-10-27/scanned/0/items`       |
| Loss                      | `tunix.rl.algo_core.grpo_loss_fn`, `loss_agg_mode="token-mean"`, `beta=0.04` |
| Batch                     | 8 sequences, `max_target_length=64`, prompt 8 / completion 8 tokens          |
| `jax` / `jaxlib` / `flax` | 0.11.1 / 0.11.1 / 0.12.9                                                     |
| MaxText commit            | `dcafaa2ef`                                                                  |
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

# Peak HBM: one trainer per process, so the reading belongs to that trainer alone.
python tests/end_to_end/tpu/compare_tunix_trainer.py --trainer=maxtext --ga=1 --batch=8 --skip-ref
python tests/end_to_end/tpu/compare_tunix_trainer.py --trainer=tunix   --ga=1 --batch=8 --skip-ref

# Compiled-kernel FLOPs / bytes-accessed, and xplane traces.
python tests/end_to_end/tpu/compare_tunix_trainer.py --trainer=both --ga=1 --skip-ref --kernel-bench
python tests/end_to_end/tpu/compare_tunix_trainer.py --trainer=maxtext --ga=1 --skip-ref --xprof=/tmp/xprof
```

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

Both trainers measured back to back in one process under identical conditions. Times are
medians over 8 iterations; `fwd_bwd` is per micro-step, `total` is a whole update.

|      | MaxText `fwd_bwd` | MaxText `update` | MaxText total | Tunix `fwd_bwd` | Tunix `update` | Tunix total | Speedup   |
| ---- | ----------------- | ---------------- | ------------- | --------------- | -------------- | ----------- | --------- |
| GA=1 | 20.4 ms           | 14.4 ms          | **34.9 ms**   | 57.7 ms         | 8.9 ms         | 66.8 ms     | **1.91x** |
| GA=4 | 20.9 ms           | 18.0 ms          | **103.6 ms**  | 56.9 ms         | 9.3 ms         | 236.6 ms    | **2.28x** |

An independent GA=1 run reproduced 34.6 ms vs. 64.2 ms (1.86x), which is the scale of
run-to-run spread.

Tunix's `update` is genuinely faster (8.9 ms vs. 14.4 ms) — MaxText's `update()` re-runs
`nnx.split(self._state)` on every call, which is a known and already-flagged `TODO` in
`maxtext_engine.py`. It does not come close to paying for the `fwd_bwd` difference.

### Where the `fwd_bwd` gap is *not*

Stripping the Python wrapper off and calling the compiled executable directly:

|          | Full `fwd_bwd` | Bare executable | `nnx.split` cost |
| -------- | -------------- | --------------- | ---------------- |
| MaxText  | 20.5 ms        | 19.1 ms         | 1.49 ms          |
| Tunix v2 | 56.1 ms        | 56.8 ms         | 1.39 ms          |

The wrapper costs ~1.4 ms on both sides. The gap is entirely inside the compiled program.

## 5. Peak HBM, and the root cause

Measured one trainer per process.

|                       | MaxText       | Tunix v2  |
| --------------------- | ------------- | --------- |
| Peak HBM/device, GA=1 | **1.766 GiB** | 7.517 GiB |
| Peak HBM/device, GA=4 | **1.779 GiB** | 3.988 GiB |

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
GA=2 the accumulator *is* sharded and peak HBM drops 7.52 → 3.99 GiB, but the kernel is
untouched and `fwd_bwd` stays at 56.9 ms.

## 6. Compiled-kernel analysis

From `Compiled.cost_analysis()` / `memory_analysis()` on the `fwd_bwd` executable:

|                                                               | MaxText GA=1     | Tunix GA=1       | Tunix GA=2    |
| ------------------------------------------------------------- | ---------------- | ---------------- | ------------- |
| GFLOPs                                                        | 17.0             | 18.8             | 18.8          |
| Bytes accessed                                                | **2.24 GiB**     | **16.97 GiB**    | **16.97 GiB** |
| Argument size                                                 | 0.278 GiB        | 2.498 GiB        | 2.498 GiB     |
| Output size                                                   | 0.278 GiB        | 2.498 GiB        | 2.498 GiB     |
| Temp size                                                     | 0.257 GiB        | 3.565 GiB        | 3.565 GiB     |
| `all-gather` / `all-reduce` / `reduce-scatter` / `all-to-all` | 12 / 6 / 12 / 18 | 10 / 9 / 12 / 17 | —             |
| Bare-executable time                                          | 18.7 ms          | 56.7 ms          | 56.9 ms       |

Same arithmetic (17.0 vs. 18.8 GFLOP), **7.6x the memory traffic** and 13.9x the temp
buffer. The kernel is bandwidth-bound, and 7.6x traffic for 3.0x time is what that looks
like. The GA=1 and GA=2 columns being byte-identical confirms the accumulator's sharding
never reaches the kernel signature.

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

xplane traces of the steady-state loop only (warmup and compilation excluded), with
`StepTraceAnnotation` step boundaries and `fwd_bwd`/`update` region annotations:

```text
gs://chengnuojin-xprof/maxtext-vs-tunix-2026-08-29/maxtext_ga1_bs8/
gs://chengnuojin-xprof/maxtext-vs-tunix-2026-08-29/tunix_ga1_bs8/
gs://chengnuojin-xprof/maxtext-vs-tunix-2026-08-29/maxtext_ga4_bs8/
gs://chengnuojin-xprof/maxtext-vs-tunix-2026-08-29/tunix_ga4_bs8/
```

Each holds `plugins/profile/<timestamp>/t1v-n-c9d27794-w-0.xplane.pb`. Raw JSON results for
the four profiled runs are under `.../raw_results/`. View with:

```bash
tensorboard --logdir gs://chengnuojin-xprof/maxtext-vs-tunix-2026-08-29/maxtext_ga1_bs8
```

Step times **from the profiled runs** are inflated by tracing overhead and are not the
performance numbers above; they are listed here only so the traces can be matched to
numbers:

| Run               | `fwd_bwd` | `update` | Total    | Peak HBM  |
| ----------------- | --------- | -------- | -------- | --------- |
| `maxtext_ga1_bs8` | 23.8 ms   | 25.0 ms  | 48.6 ms  | 1.766 GiB |
| `tunix_ga1_bs8`   | 62.5 ms   | 25.7 ms  | 88.1 ms  | 7.517 GiB |
| `maxtext_ga4_bs8` | 24.1 ms   | 29.5 ms  | 127.8 ms | 1.779 GiB |
| `tunix_ga4_bs8`   | 60.7 ms   | 19.4 ms  | 262.4 ms | 3.988 GiB |
