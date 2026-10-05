# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Orbax v1 leaf handler that keeps `to_maxtext.LazyTensor` leaves lazy through a save.

This module relies on private Orbax APIs (Orbax v1 does not export its leaf-handler
plumbing publicly). It is tested with orbax-checkpoint 0.12.8 (and works with 0.12.1);
`serialize` below mirrors `NumpyLeafHandler.serialize` in
`orbax/checkpoint/experimental/v1/_src/serialization/numpy_leaf_handler.py`, so diff
against that function on an Orbax bump. Importing this module raises `ImportError`
with a clear message if any private symbol it needs is missing; callers fall back to
the stock (eager) save path.

TODO: replace with a public budgeted/streaming leaf-handler API once Orbax has one.
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from maxtext.checkpoint_conversion.utils.lazy_tensor import LazyTensor
from maxtext.utils import max_logging

TESTED_ORBAX_VERSION = "0.12.8"

try:
  from orbax.checkpoint import v1 as ocp_v1
  from orbax.checkpoint._src.futures import future as ocp_future
  from orbax.checkpoint.experimental.v1._src.serialization import numpy_leaf_handler as ocp_numpy_leaf_handler
  from orbax.checkpoint.experimental.v1._src.serialization import registry as ocp_leaf_registry

  try:  # orbax-checkpoint >= 0.12.8 wraps commit futures in `FuturesAwaitable`.
    from orbax.checkpoint.experimental.v1._src.synchronization import compatibility as ocp_sync_compat
  except ImportError:  # orbax-checkpoint < 0.12.8: `numpy_leaf_handler._async_futures`.
    ocp_sync_compat = None
except ImportError as e:
  raise ImportError(
      f"Lazy checkpoint save needs private Orbax v1 modules that this orbax-checkpoint does not provide ({e}). "
      f"Tested with orbax-checkpoint=={TESTED_ORBAX_VERSION}."
  ) from e

_REQUIRED_SYMBOLS = (
    (ocp_future, "CommitFutureAwaitingContractedSignals"),
    (ocp_numpy_leaf_handler, "NumpyLeafHandler"),
    (ocp_numpy_leaf_handler, "_create_v0_saving_paraminfo"),
    (ocp_numpy_leaf_handler, "_create_v0_savearg"),
    (ocp_leaf_registry, "StandardLeafHandlerRegistry"),
)
_missing = [f"{mod.__name__}.{name}" for mod, name in _REQUIRED_SYMBOLS if not hasattr(mod, name)]
if ocp_sync_compat is not None:
  _missing += [
      f"{ocp_sync_compat.__name__}.{n}" for n in ("FuturesAwaitable", "async_futures") if not hasattr(ocp_sync_compat, n)
  ]
elif not hasattr(ocp_numpy_leaf_handler, "_async_futures"):
  _missing.append(f"{ocp_numpy_leaf_handler.__name__}._async_futures")
if _missing:
  raise ImportError(
      f"Lazy checkpoint save needs private Orbax symbols missing from this orbax-checkpoint: {_missing}. "
      f"Tested with orbax-checkpoint=={TESTED_ORBAX_VERSION}."
  )


def group_indices_by_bytes(nbytes: list[int], budget: int | None) -> list[list[int]]:
  """Splits leaf indices, in order, into groups whose total bytes fit `budget`.

  A leaf larger than the budget gets a group of its own. `budget=None` (or <= 0)
  means unbounded: a single group.
  """
  if not budget or budget <= 0:
    return [list(range(len(nbytes)))] if nbytes else []
  groups, current, current_bytes = [], [], 0
  for i, n in enumerate(nbytes):
    if current and current_bytes + n > budget:
      groups.append(current)
      current, current_bytes = [], 0
    current.append(i)
    current_bytes += n
  if current:
    groups.append(current)
  return groups


class _AbstractLazyArray:
  """Save-only abstract type for lazy leaves; restores go through the np.ndarray handler."""


class BudgetedLazyLeafHandler(ocp_numpy_leaf_handler.NumpyLeafHandler):
  """Orbax v1 leaf handler that writes lazily-loaded leaves within a memory budget.

  Orbax hands a leaf handler all of its leaves in one batch, and the stock
  `NumpyLeafHandler` deep-copies every value up front before writing them all
  concurrently. For `to_maxtext.LazyTensor` proxies that would pull the whole model
  into host RAM (several times over). Instead, the background commit splits the
  leaves, in order, into groups of at most half of `context.memory.
  write_concurrent_bytes`; it loads each group on a thread pool (`LOAD_WORKERS`)
  while the previous group is being written, and drops a group once written.

  Peak memory: at most two groups are live (one being written, the next being
  loaded), plus the loaders' transient copies (up to `LOAD_WORKERS` leaves being
  read/transformed). A leaf larger than half the budget forms its own group, so the
  peak is roughly max(budget, 2 x largest leaf) plus those transients; with
  `scan_layers=True` a single stacked leaf can dominate this.

  Leaves are written as plain NumPy arrays under the "np.ndarray" typestr, so the
  checkpoint is indistinguishable from one saved from materialized arrays.
  """

  LOAD_WORKERS = 8

  async def serialize(self, params, serialization_context):
    """Serializes lazy leaves to disk in budget-sized groups.

    Mirrors `NumpyLeafHandler.serialize` (orbax-checkpoint 0.12.8), but loads and
    writes the leaves group by group inside the background commit.
    """
    # pylint: disable=protected-access
    infos = [ocp_numpy_leaf_handler._create_v0_saving_paraminfo(p, self._context, serialization_context) for p in params]
    save_args = [ocp_numpy_leaf_handler._create_v0_savearg(p, self._context) for p in params]
    lazy_values = [p.value for p in params]
    budget = self._context.memory.write_concurrent_bytes
    group_budget = budget // 2 if budget else None
    nbytes = [int(v.nbytes) for v in lazy_values]
    if group_budget:
      oversized = [lazy_values[i] for i, n in enumerate(nbytes) if n > group_budget]
      if oversized:
        max_logging.log(
            f"WARNING: {len(oversized)} lazy leaves exceed half the write budget ({group_budget / 1e9:.2f} GB), "
            f"largest {max(nbytes) / 1e9:.2f} GB; peak memory may reach ~2x the largest leaf."
        )
    groups = group_indices_by_bytes(nbytes, group_budget)
    v0_handler = self._handler_impl
    # Set when the commit fails, so an in-flight prefetch stops before loading more leaves
    # (a running thread cannot be cancelled; this bounds the wasted work to leaves already started).
    stop = threading.Event()

    def _load_one(i):
      if stop.is_set():
        return None
      return np.asarray(lazy_values[i])

    async def _commit():
      if not groups:
        return
      pool = ThreadPoolExecutor(max_workers=max(1, min(self.LOAD_WORKERS, max(len(g) for g in groups))))

      def _load(group):
        return list(pool.map(_load_one, group))

      pending = asyncio.ensure_future(asyncio.to_thread(_load, groups[0]))
      try:
        for k, group in enumerate(groups):
          values = await pending
          pending = asyncio.ensure_future(asyncio.to_thread(_load, groups[k + 1])) if k + 1 < len(groups) else None
          await v0_handler._background_serialize(values, [infos[i] for i in group], [save_args[i] for i in group])
          del values
      finally:
        stop.set()
        if pending is not None and not pending.done():
          pending.cancel()
        pool.shutdown(wait=False, cancel_futures=True)

    # Same commit plumbing as the stock handler: the future is created here, on the
    # calling thread, so it binds to the current save operation and waits for the
    # item directory to be created before writing.
    commit_future = ocp_future.CommitFutureAwaitingContractedSignals(_commit(), name="lazy_np_type_handler")
    if ocp_sync_compat is None:  # orbax-checkpoint < 0.12.8
      return ocp_numpy_leaf_handler._async_futures([commit_future])
    return ocp_sync_compat.FuturesAwaitable(
        [commit_future],
        lambda: ocp_sync_compat.async_futures([commit_future], operation_name="BudgetedLazyLeafHandler"),
    )


# Orbax v1 has no context/option to inject a leaf-handler registry, and checkpointable
# registries take a handler *class*, so a subclass that passes `leaf_handler_registry`
# is the only way to add a LazyTensor leaf handler. `PyTreeHandler` is marked @final.
class _LazyAwarePyTreeHandler(ocp_v1.handlers.PyTreeHandler):  # pylint: disable=subclassed-final-class
  """`PyTreeHandler` whose leaf registry also accepts `LazyTensor` leaves."""

  def __init__(self, *, context=None, **kwargs):
    leaf_registry = ocp_leaf_registry.StandardLeafHandlerRegistry()
    leaf_registry.add(LazyTensor, _AbstractLazyArray, BudgetedLazyLeafHandler, secondary_typestrs=["np.ndarray"])
    super().__init__(context=context, leaf_handler_registry=leaf_registry, **kwargs)


# Orbax records the saving handler's `module.qualname` in `_CHECKPOINT_METADATA`, and
# loaders resolve the handler from it by name. This subclass only changes how leaves
# are *written*, and the on-disk format is exactly `PyTreeHandler`'s, so record that
# name: checkpoints then load with the stock handler (no conversion code imported,
# no resolution-fallback warning). Side effect: reprs/tracebacks show `PyTreeHandler`.
# Guarded by `test_all_lazy_tree_saves_and_loads_with_stock_handler`.
_LazyAwarePyTreeHandler.__module__ = ocp_v1.handlers.PyTreeHandler.__module__
_LazyAwarePyTreeHandler.__qualname__ = ocp_v1.handlers.PyTreeHandler.__qualname__


def lazy_aware_checkpointables_registry(checkpointable_name: str):
  """Orbax checkpointable registry that saves `checkpointable_name` with `_LazyAwarePyTreeHandler`."""
  registry = ocp_v1.handlers.local_registry(include_global_registry=True)
  registry.add(_LazyAwarePyTreeHandler, checkpointable_name=checkpointable_name)
  return registry
