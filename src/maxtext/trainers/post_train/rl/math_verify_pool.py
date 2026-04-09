# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Process-isolated, timeout-bounded math_verify grader.

This module is intentionally free of any project-side imports (no JAX, no
maxtext, no tunix). It is re-imported by every spawned worker process, so any
heavy import here would defeat the point of process isolation.

Workers run on CPU only — `silent_worker_init` is set as the Pool initializer
so that JAX/XLA never tries to grab the accelerator from inside a grader
worker, AND so that math_verify (which pulls in sympy, latex2sympy2, antlr,
and transitively tensorflow) is imported exactly once per worker process at
startup rather than on every grading call.

A single module-level Pool is created lazily on the first call and reused
across all subsequent calls. This is critical: spawning a fresh pool per
batch costs ~5–10 s of cold-start latency per worker (the math_verify import
chain is heavy), which would otherwise blow past the per-item timeout on
every batch.

If any item times out — implying a worker is stuck inside sympy and cannot
be reused — the pool is torn down and rebuilt on the next call.
"""

import atexit
import itertools
import multiprocessing
import os

# Module-level persistent pool state.
_POOL = None
_POOL_NUM_PROCS = None
# Sensible default cap. TPU hosts often have 96+ logical CPUs; spawning that
# many Python interpreters all importing math_verify simultaneously thrashes
# memory and is much slower than a small pool.
_DEFAULT_MAX_PROCS = 8


def silent_worker_init():
  """Pool initializer: hide accelerators and pre-import math_verify.

  Runs once per worker immediately after spawn. The env vars must be set
  BEFORE math_verify (and its transitive sympy / tensorflow imports) load,
  so they go first. Then we eagerly import math_verify so the *first* real
  grading call doesn't pay multi-second cold-start latency inside the
  per-item timeout.
  """
  os.environ["JAX_PLATFORMS"] = "cpu"
  os.environ["TPU_VISIBLE_DEVICES"] = ""
  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
  os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
  # Quiet TF init noise in workers.
  os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
  os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
  try:
    # Eagerly import the heavy grader stack so all subsequent
    # `_verify_math_worker` calls in this worker are fast.
    import math_verify  # noqa: F401
    from math_verify import parse, verify  # noqa: F401
    from math_verify.parser import (  # noqa: F401
        ExprExtractionConfig,
        LatexExtractionConfig,
    )
  except Exception:
    # If the import fails, individual jobs will fail too and return 0.0;
    # don't crash the worker at startup.
    pass


def _verify_math_worker(golds, predictions):
  """Worker-side grader. math_verify is already imported by the initializer."""
  try:
    from math_verify import parse, verify
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

    gold_targets = (ExprExtractionConfig(), LatexExtractionConfig())
    pred_targets = (ExprExtractionConfig(), LatexExtractionConfig())

    extracted_predictions = list(
        itertools.chain.from_iterable(
            parse(pred, pred_targets, parsing_timeout=None) for pred in predictions
        )
    )
    extracted_golds = list(
        itertools.chain.from_iterable(
            parse(gold, gold_targets, parsing_timeout=None) for gold in golds
        )
    )
    if not extracted_predictions or not extracted_golds:
      return 0.0

    return max(
        (
            1.0
            if any(verify(gold, pred, timeout_seconds=None) for gold in extracted_golds)
            else 0.0
        )
        for pred in extracted_predictions
    )
  except Exception:
    return 0.0


def _shutdown_pool():
  """Tear down the persistent pool, if any. Safe to call repeatedly."""
  global _POOL, _POOL_NUM_PROCS
  if _POOL is not None:
    try:
      _POOL.terminate()
      _POOL.join()
    except Exception:
      pass
  _POOL = None
  _POOL_NUM_PROCS = None


def _get_pool(num_procs):
  """Return the persistent pool, creating or resizing it as needed."""
  global _POOL, _POOL_NUM_PROCS
  if _POOL is None or _POOL_NUM_PROCS != num_procs:
    _shutdown_pool()
    ctx = multiprocessing.get_context("spawn")
    _POOL = ctx.Pool(processes=num_procs, initializer=silent_worker_init)
    _POOL_NUM_PROCS = num_procs
  return _POOL


atexit.register(_shutdown_pool)


def math_verify_pool(items, timeout=15, num_procs=None, log_fn=None):
  """Grade a batch of (idx, golds, predictions) items in spawned CPU workers.

  Uses a persistent module-level pool. The first call pays the spawn +
  math_verify-import cost (~5–10 s per worker, in parallel); subsequent
  calls reuse warm workers and grade in milliseconds.

  Args:
    items: list of (gen_idx, golds, predictions) tuples. `gen_idx` is unused
      here but preserved for caller-side bookkeeping; results are returned in
      the same order as `items`.
    timeout: per-item wall-clock timeout in seconds. Items that exceed this
      get a 0.0 score; if any item times out the pool is rebuilt on the next
      call to clear potentially-hung workers.
    num_procs: max worker count. None ⇒ min(_DEFAULT_MAX_PROCS, len(items),
      cpu_count()).
    log_fn: optional callable(str) for failure logging.

  Returns:
    list[float] of scores, length == len(items).
  """
  if not items:
    return []

  cpu_count = multiprocessing.cpu_count()
  if num_procs is None:
    num_procs = min(_DEFAULT_MAX_PROCS, len(items), cpu_count)
  else:
    num_procs = max(1, min(num_procs, len(items), cpu_count))

  results = [0.0] * len(items)
  pool = _get_pool(num_procs)
  saw_timeout = False
  try:
    jobs = [
        pool.apply_async(_verify_math_worker, (golds, predictions))
        for (_, golds, predictions) in items
    ]
    for i, job in enumerate(jobs):
      try:
        results[i] = float(job.get(timeout=timeout))
      except multiprocessing.TimeoutError:
        if log_fn is not None:
          log_fn(
              f"math_verify_pool timed out for golds: {items[i][1]} "
              f"and predictions: {items[i][2]}"
          )
        results[i] = 0.0
        saw_timeout = True
      except Exception as e:
        if log_fn is not None:
          log_fn(
              f"math_verify_pool failed ({e}) for golds: {items[i][1]} "
              f"and predictions: {items[i][2]}"
          )
        results[i] = 0.0
  finally:
    # If anything timed out, the worker that owned that task may still be
    # stuck inside sympy. Tear the pool down so the next call gets fresh
    # workers; the cold-start cost is amortized over the next batch.
    if saw_timeout:
      _shutdown_pool()
  return results
