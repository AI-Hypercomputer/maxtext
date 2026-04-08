# Porting Tunix `linchai_deepscaler` math-verify isolation into MaxText

Reference notes for the change that wraps `math_verify` grading in a
spawn-based multiprocessing pool inside MaxText's RL trainer.

## Problem

MaxText's RL post-training pipeline uses the HuggingFace
[`math_verify`](https://github.com/huggingface/Math-Verify) library to grade
model completions on math datasets such as `nvidia/OpenMathInstruct-2`. Before
this change, [utils_rl.py](../src/maxtext/trainers/post_train/rl/utils_rl.py)
called `math_verify` from a `concurrent.futures.ThreadPoolExecutor` inside the
trainer process:

```python
def math_verify_func(items, timeout=5):
  with concurrent.futures.ThreadPoolExecutor() as executor:
    future_to_index = {executor.submit(verify_math, golds, preds): idx ...}
    for future in concurrent.futures.as_completed(future_to_index):
      try:
        results[index] = future.result(timeout=timeout)
      except (concurrent.futures.TimeoutError, Exception):
        ...
```

Two real problems with this:

1. **Hangs cannot be killed.** `math_verify` calls `sympy.simplify` under the
   hood. On certain pathological inputs (deeply nested radicals/powers, large
   symbolic expressions) sympy can hang inside C-extension code. A
   `ThreadPoolExecutor` `future.result(timeout=...)` only raises `TimeoutError`
   in the *caller* — the worker thread is not interrupted. The grader thread
   keeps running, holds the GIL, and accumulates over the course of training.
   Over a long GRPO run on OpenMathInstruct-2 this manifests as the trainer
   slowly stalling.

2. **Accelerator contention risk.** Anything that runs in the trainer process
   shares JAX/XLA state with the training step. We want grading to be cleanly
   isolated from the trainer's TPU/GPU.

A secondary issue: the `ThreadPoolExecutor` doesn't actually parallelize the
CPU-bound sympy work because of the GIL, so we were paying the latency of
serial grading anyway.

## Why Tunix's `linchai_deepscaler` branch is the right reference

The Tunix branch
[`linchai_deepscaler`](https://github.com/google/tunix/tree/linchai_deepscaler)
hit the same problem during DeepScaler reproduction and added
[`tunix/utils/reward_verify.py`](https://github.com/google/tunix/blob/linchai_deepscaler/tunix/utils/reward_verify.py).
Its design has three pieces worth borrowing:

1. **`silent_worker_init`** — a `Pool` initializer that runs once per spawned
   worker and sets `JAX_PLATFORMS=cpu`, `TPU_VISIBLE_DEVICES=""`,
   `CUDA_VISIBLE_DEVICES=-1`, `XLA_PYTHON_CLIENT_PREALLOCATE=false` *before*
   any JAX-touching import happens in the worker. Without this, every spawned
   worker tries to grab the TPU and crashes the trainer.

2. **`compute_reward(gt, response)`** — constructs a fresh `math_metric(...)`
   *inside* the worker rather than reusing a module-level singleton. Necessary
   because `spawn` does not share Python objects, and because we don't want
   any heavy state pre-imported in the worker.

3. **`math_reward(prompts, completions, answer)`** — the batch entrypoint:
   uses `multiprocessing.get_context("spawn").Pool(initializer=...)`,
   dispatches each item with `apply_async`, enforces per-item `get(timeout=15)`,
   and force-cleans with `pool.close(); pool.terminate(); pool.join()` in a
   `finally`. This is the piece that lets us actually kill a hung sympy call.

### What we did **not** copy from Tunix

After re-reading both codebases, several things in
`reward_verify.py` / `train_deepscaler_new_math_reward.py` are *weaker* than
what MaxText already has on the `rl_dataset` branch and were intentionally
left alone:

| Concern | Tunix branch | MaxText (current) | Decision |
|---|---|---|---|
| Ground-truth normalization | unbox-then-rebox via `math_utils.extract_answer` | full `preprocess_math_string` + `fix_latex_escaping` + `boxed` pipeline | Keep MaxText's. |
| Gold extraction targets | `LatexExtractionConfig` only | `Expr` + `Latex` (more lenient, matches OMI-2's mix of bare numbers and LaTeX) | Keep MaxText's. |
| Multi-answer support | single ground truth | list of acceptable answers per example | Keep MaxText's. |
| Reward shaping | binary 0/1 | tiered: exact / whitespace / verified / numeric ratio / penalty | Keep MaxText's. |
| Batch granularity | one pool per `reward_fn` call | one pool per `check_numbers` call | Equivalent. |

The **only** thing in `reward_verify.py` that MaxText was missing is the
spawn-based process isolation + force-killable timeout. So the port is
narrow: swap the *transport* of `math_verify`, leave everything else alone.

## Changes made

### 1. New module: [`math_verify_pool.py`](../src/maxtext/trainers/post_train/rl/math_verify_pool.py)

A small, self-contained module with **no project-side imports** (no `maxtext`,
no `tunix`, no JAX). This is critical: under `spawn`, every worker re-imports
the parent module of any function it runs. If the worker re-imported
`utils_rl.py`, it would transitively pull in `optax`, `tunix`, and JAX —
defeating the point of CPU-only isolation. By living in its own file with
only stdlib + `math_verify` (lazy-imported inside the worker), the worker's
import surface stays tiny.

Contents:

- **`silent_worker_init()`** — copied verbatim from
  Tunix's `reward_verify.py`. Sets the four env vars that hide accelerators
  and disable XLA preallocation. Wired in as the `Pool(initializer=...)` so it
  runs *once per worker, before any user code* in that worker.

- **`_verify_math_worker(golds, predictions) -> float`** — module-level
  (picklable for `spawn`) worker function. Lazily imports `math_verify` inside
  the function body and runs the same logic as the existing top-level
  `verify_math` in `utils_rl.py` (Expr+Latex parse for both gold and pred,
  `verify(...)` over the cartesian product, returns 1.0 if any match, else
  0.0). Catches all exceptions and returns 0.0 — workers must never raise
  back through the pool.

- **`math_verify_pool(items, timeout=15, num_procs=None, log_fn=None) -> list[float]`**
  — the batch entrypoint. Mirrors `reward_verify.math_reward`'s pool logic
  but parametrized for MaxText's existing `(gen_idx, golds, predictions)`
  tuple shape so callers don't need to change. Key behaviors:
  - `multiprocessing.get_context("spawn")` (not `fork` — `fork` would inherit
    JAX state from the trainer).
  - `num_procs = min(len(items), cpu_count())` by default; capped by an
    optional explicit override.
  - `apply_async` per item, then iterates and calls `job.get(timeout=...)`.
  - On `multiprocessing.TimeoutError` or any other `Exception`: assigns 0.0,
    optionally logs via `log_fn`.
  - `try/finally: pool.close(); pool.terminate(); pool.join()` — `terminate`
    is what actually kills any worker still stuck in sympy. Without it,
    hung workers would leak across batches.

### 2. [`utils_rl.py`](../src/maxtext/trainers/post_train/rl/utils_rl.py): `math_verify_func` rewrite

`math_verify_func` now takes an optional `tmvp_config` parameter and by
default routes through `math_verify_pool`:

```python
def math_verify_func(items, timeout=5, tmvp_config=None):
  if not items:
    return []
  use_pool = True
  num_procs = None
  if tmvp_config is not None:
    use_pool = getattr(tmvp_config, "math_verify_use_pool", True)
    timeout = getattr(tmvp_config, "math_verify_timeout", timeout)
    num_procs = getattr(tmvp_config, "math_verify_num_procs", None)

  if use_pool:
    return math_verify_pool(items, timeout=timeout,
                            num_procs=num_procs, log_fn=max_logging.log)

  # Legacy ThreadPoolExecutor path preserved as a fallback for debugging.
  with concurrent.futures.ThreadPoolExecutor() as executor:
    ...
```

Why keep the legacy thread path: it's a one-line config flip
(`math_verify_use_pool: False`) to bisect any regression caused by the pool
itself versus the surrounding reward shaping. Once the new path is proven in
production, the thread path can be deleted.

### 3. [`utils_rl.py`](../src/maxtext/trainers/post_train/rl/utils_rl.py): `check_numbers` call site

The single existing call site inside `check_numbers` was updated to pass the
config through:

```python
math_verify_results = math_verify_func(math_verify_queue, tmvp_config=tmvp_config)
```

No other call sites of `math_verify_func` exist in the codebase (only a
notebook references it).

### 4. [`configs/types.py`](../src/maxtext/configs/types.py): three new `Reward` fields

```python
math_verify_use_pool: bool = Field(True, ...)         # escape hatch
math_verify_timeout: int = Field(15, ...)             # per-item seconds
math_verify_num_procs: int | None = Field(None, ...)  # None ⇒ cpu_count()
```

The default timeout is bumped from `5` (the old hardcoded thread value) to
`15` because the pool can actually kill hangs now, so a longer timeout
trades zero risk for fewer false-zero gradings on legitimately complex
expressions.

### 5. `__main__` guard

`spawn` on Linux requires the trainer's entry point to be guarded by
`if __name__ == "__main__":`. Verified this is already the case at
[`train_rl.py:779`](../src/maxtext/trainers/post_train/rl/train_rl.py#L779);
no change needed.

## Validation suggestions

- **Hang test:** feed `math_verify_func` an item like
  `(0, ["\\boxed{x}"], ["\\boxed{\\sqrt{x^{x^{x^{x}}}}}"])` and assert it
  returns within `~timeout + 1s` and that `multiprocessing.active_children()`
  is empty after the call.
- **Parity test:** run `check_numbers` over a fixed batch with
  `math_verify_use_pool=True` and `False`; the score lists should be equal.
- **Smoke test:** ~100 GRPO steps on `nvidia/OpenMathInstruct-2`; watch for
  orphan python processes (`ps -ef | grep python`) and step-time regression.
  Pool startup with `spawn` is non-trivial (~hundreds of ms); if this becomes
  the bottleneck, the next optimization is a long-lived pool created once at
  trainer init instead of per `check_numbers` call.

## Files touched

- **New:** `src/maxtext/trainers/post_train/rl/math_verify_pool.py`
- **Modified:** `src/maxtext/trainers/post_train/rl/utils_rl.py` (added
  import; rewrote `math_verify_func`; updated single call site)
- **Modified:** `src/maxtext/configs/types.py` (added three `Reward` fields)

## Rollback

Set `math_verify_use_pool: False` in
`src/maxtext/configs/post_train/rl.yml` (or any RL config). No code change
required.
