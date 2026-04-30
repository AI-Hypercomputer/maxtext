# Copyright 2023-2026 Google LLC
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

"""Process-isolated, timeout-bounded math_verify grader for RL reward computation.

This module provides a persistent multiprocessing pool that grades model-generated
math answers against ground-truth solutions using the math_verify library. It is
used during GRPO/GSPO training to compute rewards for each example in a batch.

- Workers are spawned as separate CPU-only processes so that JAX/XLA inside
  the trainer never competes with grader workers for accelerator resources.
- math_verify and other heavy dependencies are only imported once per
  worker at startup via `silent_worker_init`, avoiding multi-second cold-start
  latency on every grading call.
- The pool is module-level and persistent across training steps. Workers are
  recycled after `_MAX_TASKS_PER_CHILD` tasks to bound sympy's internal cache
  growth in long-running training jobs.
- Grading is subject to a global wall-clock timeout across all examples in a
  batch (configured via `math_verify_timeout` in rl.yml). Items that do not
  complete within the deadline are dropped; their scores remain at the
  pre-call default. If any items time out, the pool is torn down and recreated
  on the next call to recover from stuck threads.
- Shutdown escalates from SIGTERM to SIGKILL to handle workers blocked inside
  native C extensions that ignore Python signals.
"""

import atexit
import itertools
import multiprocessing
import os
import threading
import time
from typing import Any, Callable, Optional

# Module-level persistent pool state.
_POOL = None
_POOL_NUM_PROCS = None
_DEFAULT_MAX_PROCS = 8
# Recycle a worker after this many tasks. Bounds sympy's internal cache
# growth in long-lived workers without paying the cold-start cost too often.
_MAX_TASKS_PER_CHILD = 100
# SIGTERM grace period before escalating to SIGKILL on stuck workers.
_TERMINATE_GRACE_SECONDS = 2.0


def silent_worker_init() -> None:
  """Pool initializer: hide accelerators and pre-import math_verify.

  Runs once per worker immediately after spawn. The env vars must be set
  before math_verify load. Then we eagerly import math_verify so the first real
  grading call doesn't pay multi-second cold-start latency inside the
  per-item timeout.
  """
  os.environ["JAX_PLATFORMS"] = "cpu"
  os.environ["TPU_VISIBLE_DEVICES"] = ""
  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
  os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
  # Quiet TF / TPU log noise in workers. Override unconditionally — the
  # parent trainer process often sets these to 0 for its own debugging,
  # and `setdefault` would inherit that loud value into every spawned
  # grader worker. We want the workers silent regardless.
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
  os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
  os.environ["TPU_MIN_LOG_LEVEL"] = "3"
  os.environ["TPU_STDERR_LOG_LEVEL"] = "3"
  os.environ["GRPC_VERBOSITY"] = "ERROR"
  try:
    # Eagerly import the heavy grader stack so all subsequent
    # `verify_math_worker` calls in this worker are fast.
    import math_verify  # pylint: disable=import-outside-toplevel,unused-import
    from math_verify import parse, verify  # pylint: disable=import-outside-toplevel,unused-import
    from math_verify.parser import (  # pylint: disable=import-outside-toplevel,unused-import
        ExprExtractionConfig,
        LatexExtractionConfig,
    )
    from sympy.parsing import sympy_parser  # pylint: disable=import-outside-toplevel,unused-import
    from sympy import Basic, MatrixBase  # pylint: disable=import-outside-toplevel,unused-import
  except Exception:  # pylint: disable=broad-exception-caught
    # If the import fails, individual jobs will fail too and return 0.0;
    # don't crash the worker at startup.
    pass


def are_equal_under_sympy(gold: Any, prediction: Any) -> bool:
  """Returns True if gold and prediction are symbolically equal using SymPy.

  Parses both values as symbolic expressions with implicit multiplication
  support (e.g. '2x' == '2*x'). Returns False if parsing fails or the
  expressions differ.
  """
  from sympy.parsing import sympy_parser  # pylint: disable=import-outside-toplevel

  try:
    gold_expr = sympy_parser.parse_expr(str(gold), evaluate=False)
    pred_expr = sympy_parser.parse_expr(str(prediction), evaluate=False)
    if gold_expr == pred_expr:
      return True
  except Exception:  # pylint: disable=broad-exception-caught
    pass
  return False


def verify_math_worker(golds: list[str], predictions: list[str]) -> float:
  """Worker-side math_verify grader."""
  try:  # pylint: disable=too-many-nested-blocks
    from math_verify import parse, verify  # pylint: disable=import-outside-toplevel
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig  # pylint: disable=import-outside-toplevel
    from sympy import Basic, MatrixBase  # pylint: disable=import-outside-toplevel

    gold_targets = (ExprExtractionConfig(), LatexExtractionConfig())
    pred_targets = (ExprExtractionConfig(), LatexExtractionConfig())

    extracted_predictions = list(
        itertools.chain.from_iterable(parse(pred, pred_targets, parsing_timeout=None) for pred in predictions)
    )
    extracted_golds = list(
        itertools.chain.from_iterable(parse(gold, gold_targets, parsing_timeout=None) for gold in golds)
    )
    if not extracted_predictions or not extracted_golds:
      return 0.0

    for gold in extracted_golds:
      for pred in extracted_predictions:
        if isinstance(gold, (Basic, MatrixBase)) and isinstance(pred, (Basic, MatrixBase)):
          if are_equal_under_sympy(gold, pred):
            return 1.0

          if "**" not in str(gold) and "**" not in str(pred):
            if verify(gold, pred, timeout_seconds=None):
              return 1.0
        else:
          if verify(gold, pred, timeout_seconds=None):
            return 1.0
    return 0.0
  except Exception:  # pylint: disable=broad-exception-caught
    return 0.0


def _shutdown_pool() -> None:
  """Tear down the persistent pool, if any. Safe to call repeatedly.

  SIGTERM first with a grace period, then SIGKILL any survivors. SIGKILL is
  handled by the kernel, so workers stuck in sympy/FLINT C extensions (which
  ignore Python signals) are still reaped. Without this, `Pool.terminate()`
  internally calls `p.join()` on each worker and blocks forever on the stuck
  ones.
  """
  global _POOL, _POOL_NUM_PROCS
  pool = _POOL
  _POOL = None
  _POOL_NUM_PROCS = None
  if pool is None:
    return
  try:
    workers = list(getattr(pool, "_pool", []))
    for w in workers:
      if w.is_alive():
        w.terminate()
    deadline = time.monotonic() + _TERMINATE_GRACE_SECONDS
    for w in workers:
      remaining = max(0.0, deadline - time.monotonic())
      w.join(timeout=remaining)
    for w in workers:
      if w.is_alive():
        w.kill()
        w.join(timeout=1.0)
    # Workers SIGKILLed mid-write leave the outqueue lock orphaned, so
    # pool.terminate() / pool.join() block forever on the internal
    # _result_handler / _task_handler threads. Run terminate in a daemon
    # thread with a bounded wait: pool._state flips to TERMINATE so the
    # worker-handler stops spawning replacements, and we return even if the
    # handler threads never unblock. Those threads leak, but they are daemon
    # and cheap; a stuck trainer is not.
    t = threading.Thread(target=pool.terminate, daemon=True)
    t.start()
    t.join(timeout=_TERMINATE_GRACE_SECONDS)
  except Exception:  # pylint: disable=broad-exception-caught
    pass


def _get_pool(num_procs: int) -> multiprocessing.pool.Pool:
  """Return the persistent pool, creating or resizing it as needed."""
  global _POOL, _POOL_NUM_PROCS
  if _POOL is None or _POOL_NUM_PROCS != num_procs:
    _shutdown_pool()
    ctx = multiprocessing.get_context("spawn")
    _POOL = ctx.Pool(
        processes=num_procs,
        initializer=silent_worker_init,
        maxtasksperchild=_MAX_TASKS_PER_CHILD,
    )
    _POOL_NUM_PROCS = num_procs
  return _POOL


# ensures global worker pool is cleanly shut down when program finishes execution
atexit.register(_shutdown_pool)


def math_verify_pool(
    trainer_config: Any,
    items: list[tuple[int, list[str], list[str]]],
    scores: list[float],
    timeout: float = 300,
    num_procs: Optional[int] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> list[float]:
  """Grade a batch of (idx, golds, predictions) items in spawned CPU workers.

  Uses a persistent module-level pool. The first call pays the spawn +
  math_verify-import cost; subsequent calls reuse warm workers and grade.
  """
  if not items:
    return scores

  cpu_count = multiprocessing.cpu_count()
  if num_procs is None:
    num_procs = min(_DEFAULT_MAX_PROCS, len(items), cpu_count)
  else:
    num_procs = max(1, min(num_procs, len(items), cpu_count))

  cnt = 0
  pool = _get_pool(num_procs)
  active_jobs = [(idx, pool.apply_async(verify_math_worker, (golds, predictions))) for (idx, golds, predictions) in items]
  start_time = time.time()
  while active_jobs and (time.time() - start_time < timeout):
    # Iterate backwards to safely remove items from the list without skipping elements
    for i in range(len(active_jobs) - 1, -1, -1):
      idx, job = active_jobs[i]
      if job.ready():
        try:
          # .get(0) returns immediately since ready() was true
          score = job.get(0)
          if score > 0.0:
            scores[idx] = max(scores[idx], trainer_config.reward_exact_answer)
          cnt += 1
        except Exception as e:  # pylint: disable=broad-exception-caught
          if log_fn:
            log_fn(f"math_verify_pool failed ({e}) for idx: {idx}")
        active_jobs.pop(i)

    # Small sleep to prevent high CPU usage during the loop
    time.sleep(0.1)

  if log_fn:
    log_fn(f"math_verify_pool: Processed {cnt}/{len(items)} items ({len(active_jobs)} timed out).")
  if len(active_jobs) > 0:
    _shutdown_pool()
  return scores
