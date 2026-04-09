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

### 6. [`utils_rl.py`](../src/maxtext/trainers/post_train/rl/utils_rl.py): preprocessing additions

Tunix's `_strip_string` and `_normalize` ([math_utils.py](https://github.com/google/tunix/blob/linchai_deepscaler/tunix/utils/math_utils.py))
do several normalizations that MaxText's `normalize_final_answer` was
missing. Most matter because they unblock `math_verify`'s LaTeX parser when
it would otherwise fail and force the run into the numeric-ratio fallback.

Added to `SUBSTITUTIONS` (order matters — collapse must come first):

| Rule | Why |
|---|---|
| `\\` → `\` | Collapse double-escaped LaTeX so subsequent rules see canonical form. |
| `\tfrac` / `\dfrac` → `\frac` | Common in MATH/OMI-2; some parsers don't accept the variants. |
| ` or ` / ` and ` → `,` | Set-style answers like "1 or 2" become tuples. |
| `million` → `*10^6`, `billion` → `*10^9`, `trillion` → `*10^12` | Word-form scale factors. |

Added to `REMOVED_EXPRESSIONS`:

| Rule | Why |
|---|---|
| `\left`, `\right` | Pure formatting; major parse blocker for sympy. |
| `\!` (inverse space) | Pure formatting. |
| Singular unit forms: `yard`, `foot`, `mile`, `day`, `week`, `month`, `year`, `hour`, `minute`, `second`, `centimeter`, `meter` (and selected plurals) | MaxText already had plural forms for some units; Tunix removes both. |

New steps inside `normalize_final_answer`:

1. **Mixed-number injection** (`7 3/4` → `7+3/4`) — runs **before** the
   substitutions loop, because that loop strips spaces and would otherwise
   destroy the mixed-number structure.
2. **Leading-zero fix**: `.5` → `0.5` and `{.5` → `{0.5`. Sympy parses
   `0.5` but not bare `.5` in some contexts.
3. **Outer-brace strip**: `{42}` → `42`.
4. **Integer-float collapse**: `2.0` → `2`. Important because Tunix's
   sympy grader requires strict integer match when GT is integer; the same
   discipline helps `math_verify`'s exact-string fast path.

These changes are intentionally conservative — none of them touch the
existing reward shaping or extraction logic, and they only fire on inputs
that previously would have been graded incorrectly or kicked into the
numeric-ratio fallback.

## Files touched

- **New:** `src/maxtext/trainers/post_train/rl/math_verify_pool.py`
- **Modified:** `src/maxtext/trainers/post_train/rl/utils_rl.py` (added
  import; rewrote `math_verify_func`; updated single call site)
- **Modified:** `src/maxtext/configs/types.py` (added three `Reward` fields)

## Rollback

Set `math_verify_use_pool: False` in
`src/maxtext/configs/post_train/rl.yml` (or any RL config). No code change
required.

## Followup: cold-start timeout storm and the persistent-pool fix

### Symptom

After the initial port, the first real run produced log spam like:

```
math_verify_pool timed out for golds: ['\\boxed{\\frac{\\sqrt{41}}{2}}']
                                  preds: ['\\boxed{\\pm\\frac{\\sqrt{41}}{2}}']
math_verify_pool timed out for golds: ['\\boxed{\\frac{\\sqrt{41}}{2}}']
                                  preds: ['\\boxed{-1000000}']
math_verify_pool timed out for golds: ['\\boxed{12\\sqrt{2}}']
                                  preds: ['\\boxed{24}']
...
```

— **every** item timing out at exactly 15 s, even trivially fast inputs like
`\boxed{24}` that sympy grades in microseconds. Alongside that, the trainer
log was repeating

```
tensorflow/core/util/port.cc:153] oneDNN custom operations are on...
```

dozens of times per batch.

### Root cause

Both symptoms had the same cause: **the pool was being created and torn down
on every `check_numbers` call**, and `math_verify`'s import chain
(`math_verify` → `sympy` → `latex2sympy2` → `antlr4` → transitively `tensorflow`)
takes 5–10 seconds per worker process to load on a busy TPU host. With
`spawn`, each worker starts with an empty interpreter and re-imports
everything from scratch.

So on every batch:

1. The pool spawned `cpu_count()` (often 96+) Python interpreters.
2. Each interpreter raced to import the entire math_verify stack at once,
   thrashing memory.
3. None of them finished initializing before the per-item 15 s timeout
   started ticking on `job.get(...)`.
4. Every job timed out → every score was 0.0 → the pool was destroyed →
   the next batch repeated the whole thing.
5. Each fresh worker also re-printed TensorFlow's `oneDNN` init message,
   producing the log spam.

The original 15 s timeout was sized for *steady-state grading* (where
each call should be sub-millisecond), not for absorbing repeated cold
starts of the entire grader stack.

### Fix

Three changes to
[math_verify_pool.py](../src/maxtext/trainers/post_train/rl/math_verify_pool.py),
all in the same file — no caller changes:

1. **Persistent module-level pool.** Added `_POOL` / `_POOL_NUM_PROCS` /
   `_get_pool()` / `_shutdown_pool()`. The pool is created lazily on the
   first `math_verify_pool` call and reused across all subsequent calls.
   Registered `atexit.register(_shutdown_pool)` so it cleans up at process
   exit. The cold-start cost is paid exactly **once per training run**,
   not per batch.

2. **Eager `math_verify` import inside `silent_worker_init`.** The
   initializer now imports `math_verify`, `parse`, `verify`,
   `ExprExtractionConfig`, `LatexExtractionConfig` immediately after
   setting the env vars (which still must come first so JAX/XLA never
   touch the accelerator). This moves the heavy import out of the timed
   `job.get` window — by the time the first real grading task is
   submitted, every worker is already warm. Also added
   `TF_CPP_MIN_LOG_LEVEL=3` and `TF_ENABLE_ONEDNN_OPTS=0` in the
   initializer to silence the `oneDNN` log noise.

3. **Default worker cap of 8** (`_DEFAULT_MAX_PROCS = 8`). TPU hosts
   typically expose 96+ logical CPUs, but spawning that many Python
   interpreters all importing the math stack simultaneously is *slower*
   than a small pool because of memory contention. 8 is a safe default;
   override via `tmvp_config.math_verify_num_procs` if a workload needs
   more.

4. **Rebuild-on-timeout policy.** If any item times out during a batch,
   the pool is torn down in the `finally` block. The next call recreates
   it. This guarantees a hung sympy worker can't poison subsequent
   batches, while keeping the common (no-timeout) case on a warm
   persistent pool. The cold-start cost reappears only after a real
   sympy hang — exactly when we want fresh workers anyway.

### Expected behavior after the fix

| | Before fix | After fix |
|---|---|---|
| First batch latency | ~15 s × len(items) (everything times out) | ~5–10 s once (parallel cold start of 8 workers), then real grading |
| Steady-state batch latency | n/a — every batch was a cold start | sub-second |
| `oneDNN` log lines | ~96 per batch | ≤ 8 total per process |
| Real sympy hangs | masked by cold-start timeouts | caught by the 15 s timeout, score 0.0, pool rebuilt for next batch |

### Tuning knobs (already in `rl.yml` / `types.py`)

- `math_verify_use_pool: True` — keep this on; the legacy
  `ThreadPoolExecutor` path doesn't benefit from the persistent-pool fix.
- `math_verify_timeout: 15` — only matters for (a) the very first batch's
  cold start and (b) real sympy hangs. Bump to 30 or 60 if the first
  batch on a particularly slow host occasionally trips it.
- `math_verify_num_procs: null` — `null` ⇒ the new
  `min(8, len(items), cpu_count())` default. Set explicitly only if
  profiling shows the grader is the bottleneck and there's spare CPU
  headroom.
