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
worker.
"""

import itertools
import multiprocessing
import os


def silent_worker_init():
  """Pool initializer: hide accelerators from grader workers.

  Runs once per worker immediately after spawn, before math_verify (and its
  transitive sympy / JAX-touching imports) are loaded inside the worker.
  """
  os.environ["JAX_PLATFORMS"] = "cpu"
  os.environ["TPU_VISIBLE_DEVICES"] = ""
  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
  os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def _verify_math_worker(golds, predictions):
  """Worker-side grader. Imports math_verify lazily so the parent need not."""
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


def math_verify_pool(items, timeout=15, num_procs=None, log_fn=None):
  """Grade a batch of (idx, golds, predictions) items in spawned CPU workers.

  Args:
    items: list of (gen_idx, golds, predictions) tuples. `gen_idx` is unused
      here but preserved for caller-side bookkeeping; results are returned in
      the same order as `items`.
    timeout: per-item wall-clock timeout in seconds. Items that exceed this
      get a 0.0 score and the worker is killed via pool.terminate().
    num_procs: max worker count. None ⇒ min(len(items), cpu_count()).
    log_fn: optional callable(str) for failure logging.

  Returns:
    list[float] of scores, length == len(items).
  """
  if not items:
    return []

  ctx = multiprocessing.get_context("spawn")
  if num_procs is None:
    num_procs = min(len(items), ctx.cpu_count())
  else:
    num_procs = max(1, min(num_procs, len(items)))

  results = [0.0] * len(items)
  pool = ctx.Pool(processes=num_procs, initializer=silent_worker_init)
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
      except Exception as e:
        if log_fn is not None:
          log_fn(
              f"math_verify_pool failed ({e}) for golds: {items[i][1]} "
              f"and predictions: {items[i][2]}"
          )
        results[i] = 0.0
  finally:
    pool.close()
    pool.terminate()
    pool.join()
  return results
