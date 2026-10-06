# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Checkpointing utilities for MaxText training engine."""

import collections
from collections.abc import Mapping
import dataclasses
import math
import os
import threading
import time
from typing import Any, List
import zlib

from absl import logging
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.configs import pyconfig
from maxtext.training_engine import abstract_engine
import orbax.checkpoint as ocp


@dataclasses.dataclass
class CheckpointState:
  """Container for model, optimizer, accumulated metrics and intra-step states to checkpoint."""

  model: nnx.Module
  optimizer: nnx.optimizer.Optimizer | None = None
  accumulated_metrics: List[abstract_engine.MetricsBuffer] | None = None
  accumulated_grads: Any = None
  # How many micro-batches of `step` are folded into `accumulated_grads`. 0 for a complete
  # step, whose gradients have already been applied and discarded.
  micro_step_count: int = 0
  # Restore target for `accumulated_grads`, as `jax.ShapeDtypeStruct`s with the accumulator's
  # dtype and shardings. Orbax casts each leaf to its target's dtype, so this is needed when the
  # gradients are accumulated in a dtype other than the parameters'. None restores into the
  # parameters' own state.
  accumulated_grads_target: Any = None


_PERSISTENCE = "persistence"
_COLOCATED_PYTHON = "colocated_python"
_PATHWAYS_CHECKPOINTING_IMPLS = (_PERSISTENCE, _COLOCATED_PYTHON)

# Default per-host sidecar SHM dispatch budget (8 GB) when d2h_concurrent_gb is unset.
_COLOCATED_DISPATCH_MAX_BYTES: int = 8 * 10**9

# Which impl this process registered, or None if registration has not succeeded yet. Orbax
# registers type handlers process-globally, so this doubles as the conflict detector.
_REGISTERED_IMPL: str | None = None

# Handlers that actually dispatch writes off the controller under `persistence`. Anything else
# means Orbax fell back to the controller-side handler, which stages every shard through
# proxy-pod host RAM and OOMs at 397B scale.
_PATHWAYS_PERSISTENCE_HANDLERS = ("CloudPathwaysArrayHandler", "PathwaysPersistenceArrayHandler")


class PathwaysCheckpointingUnavailableError(RuntimeError):
  """Pathways checkpointing was explicitly requested but could not be registered."""


class CheckpointRestoreError(RuntimeError):
  """A checkpoint exists at the requested step but could not be restored."""


# Versioned: changing the formula must not turn older fingerprinted checkpoints into mismatches.
_FINGERPRINT_KEY = "fingerprints_v1"


@jax.jit
def _leaf_u32_sum(leaf: jax.Array) -> jax.Array:
  """Wraparound uint32 sum of `leaf`'s raw bits: exact, so identical under any sharding.

  Same-width bitcast first (wider dtypes split into uint32 words), so shapes such as bf16
  `[4096, 15, 1]` never hit a width-changing bitcast. Each element is weighted by an odd
  function of its global index (`iota` is global under SPMD), so elements or shards written
  to the wrong position change the sum, and an all-zero tensor's sum depends on its shape.
  """
  # Offloaded optimizer leaves (optimizer_memory_host_offload) arrive in pinned_host; move them to
  # device inside the jit, one leaf per executable, or XLA:TPU rejects the device output (E1200).
  leaf = jax.device_put(leaf, jax.memory.Space.Device)
  if jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key):
    leaf = jax.random.key_data(leaf)
  if leaf.dtype == jnp.bool_:  # e.g. `is_skipped` in the skip_step_on_spikes optimizer state
    leaf = leaf.astype(jnp.uint8)
  bits = jax.lax.bitcast_convert_type(leaf, {1: jnp.uint8, 2: jnp.uint16}.get(leaf.dtype.itemsize, jnp.uint32))
  bits = bits.astype(jnp.uint32)
  weight = jnp.uint32(0x9E3779B9)
  for dim in range(bits.ndim):
    weight = weight * jnp.uint32(0x01000193) + jax.lax.broadcasted_iota(jnp.uint32, bits.shape, dim)
  return ((bits + jnp.uint32(0x9E3779B9)) * (weight | 1)).sum(dtype=jnp.uint32)


def _tree_fingerprint(state: Any) -> int:
  """Checksum of every leaf's bits, weighted per key path so swapped leaves also change it.

  Per-leaf dispatch (rather than one jit over the tree) tolerates leaves on different device
  sets, and `device_get` of the whole list is a single host round trip.
  """
  flat = jax.tree_util.tree_flatten_with_path(nnx.to_pure_dict(state) if isinstance(state, nnx.State) else state)[0]
  sums = jax.device_get([_leaf_u32_sum(x) for _, x in flat])
  return sum(int(s) * (zlib.crc32(jax.tree_util.keystr(p).encode()) | 1) for (p, _), s in zip(flat, sums)) & 0xFFFFFFFF


def _format_fingerprints(fingerprints: Mapping[str, int]) -> str:
  """`item=0x........` pairs; the save and verify log lines share it so they can be compared verbatim."""
  return " ".join(f"{item}={fp:#010x}" for item, fp in fingerprints.items())


def _orbax_fingerprint_enabled() -> bool:
  """Off by default; True iff ENABLE_ORBAX_FINGERPRINT is "1"/"true". When off, save and restore skip fingerprints."""
  return os.environ.get("ENABLE_ORBAX_FINGERPRINT", "0").strip().lower() in ("1", "true")


def _maybe_register_pathways_persistence(
    impl_name: str = _PERSISTENCE, *, d2h_concurrent_gb: float | None = None
) -> None:
  """Registers the Orbax Pathways array handler for `impl_name`, if applicable.

  When ENABLE_PATHWAYS_PERSISTENCE=1, delegates checkpoint saving directly
  from TPU workers to storage (GCS), bypassing host RAM staging.

  Setting the environment variable is an explicit request, so a failure to install
  the handler raises rather than degrading. The silent fallback is controller-side
  host staging, which exhausts proxy-pod memory at 397B scale, and it surfaces only
  once the first save is attempted -- long after the run looks healthy.

  Args:
    impl_name: Which Pathways implementation to register; one of
      `_PATHWAYS_CHECKPOINTING_IMPLS`. `persistence` is the shipped default;
      `colocated_python` additionally requires a version-matched sidecar container.
    d2h_concurrent_gb: Optional per-host sidecar SHM dispatch budget in GB for
      `colocated_python`.

  Raises:
    ValueError: If `impl_name` is not a known implementation.
    PathwaysCheckpointingUnavailableError: If ENABLE_PATHWAYS_PERSISTENCE=1 but the
      requested handler could not be registered for jax.Array, or if a different
      implementation is already registered in this process.
  """
  global _REGISTERED_IMPL
  if impl_name not in _PATHWAYS_CHECKPOINTING_IMPLS:
    raise ValueError(
        f"unknown pathways_checkpointing_impl {impl_name!r}; expected one of {_PATHWAYS_CHECKPOINTING_IMPLS}."
    )
  if _REGISTERED_IMPL is not None:
    if _REGISTERED_IMPL != impl_name:
      raise PathwaysCheckpointingUnavailableError(
          f"Orbax type handlers are registered process-globally and this process already "
          f"registered {_REGISTERED_IMPL!r}; cannot also serve {impl_name!r}. The impl affects "
          "every save and restore, so mixing them would silently apply one mode to the other's "
          "checkpoints."
      )
    return
  if os.environ.get("ENABLE_PATHWAYS_PERSISTENCE", "") != "1":
    if impl_name == _COLOCATED_PYTHON:
      raise PathwaysCheckpointingUnavailableError(
          "pathways_checkpointing_impl='colocated_python' requires "
          "ENABLE_PATHWAYS_PERSISTENCE=1 in the environment; refusing to fall back "
          "to controller-side host-staged checkpointing."
      )
    return

  try:
    # pylint: disable=g-import-not-at-top,import-outside-toplevel
    import orbax.checkpoint.pathways as ocp_pathways
    from orbax.checkpoint._src.metadata import array_metadata_store as array_metadata_store_lib
    from orbax.checkpoint._src.serialization import type_handler_registry

    # pylint: enable=g-import-not-at-top,import-outside-toplevel

    register_kwargs: dict[str, Any] = {
        "use_single_replica_array_handler": False,
        # Preserve array metadata store required for pytrees with typed PRNG keys.
        "array_metadata_store": array_metadata_store_lib.Store(),
    }
    if impl_name == _COLOCATED_PYTHON:
      impl = ocp_pathways.CheckpointingImpl.COLOCATED_PYTHON
      from orbax.checkpoint._src.serialization import types as serialization_types  # pylint: disable=g-import-not-at-top,import-outside-toplevel

      # Subclass Orbax's no-op default so every status hook the ArrayHandler invokes
      # (on_transfer_start/end, on_write_start/end — the set varies across Orbax versions) is
      # inherited; only the priority is overridden. A hand-rolled class missing `on_write_end`
      # fails every colocated save at commit time (observed on orbax 0.12.6, 397B/32 hosts).
      class _DeprioritizedCallback(serialization_types.DefaultSerializationStatusCallback):
        """Routes all arrays through Orbax's memory-limited (deprioritized) D2H batching."""

        def key_priority(self, _: Any) -> serialization_types.TransferPriority:
          return serialization_types.TransferPriority.ASYNCHRONOUS_DEPRIORITIZED

      register_kwargs["callback"] = _DeprioritizedCallback()
    else:
      impl = ocp_pathways.CheckpointingImpl.PERSISTENCE

    register_kwargs["checkpointing_impl"] = impl
    ocp_pathways.register_type_handlers(**register_kwargs)

    handler = type_handler_registry.get_type_handler(jax.Array)
    handler_name = type(handler).__name__
  except (ImportError, AttributeError, ModuleNotFoundError, NotImplementedError) as e:
    raise PathwaysCheckpointingUnavailableError(
        f"ENABLE_PATHWAYS_PERSISTENCE=1 explicitly requested Pathways checkpointing "
        f"(impl={impl_name!r}), but registering the Orbax Pathways array handler failed "
        f"({type(e).__name__}: {e}). Refusing to fall back to controller-side host-staged "
        "checkpointing, which exhausts proxy-pod memory at 397B scale. Unset "
        "ENABLE_PATHWAYS_PERSISTENCE to run without Pathways checkpointing."
    ) from e

  if impl_name == _COLOCATED_PYTHON:
    # COLOCATED_PYTHON is the only impl in this Orbax build that attaches a dispatcher, so
    # has_dispatcher() distinguishes it from NO_DISPATCHER -- the controller-side fallback --
    # without reaching into Orbax internals. Checking the class name alone would not: the
    # NO_DISPATCHER fallback yields an ArrayHandler too, just without a dispatcher.
    registered_as_requested = handler_name == "ArrayHandler" and handler.has_dispatcher()
    expected = "an ArrayHandler with a dispatcher attached"
  else:
    registered_as_requested = handler_name in _PATHWAYS_PERSISTENCE_HANDLERS
    expected = f"one of {_PATHWAYS_PERSISTENCE_HANDLERS}"

  if not registered_as_requested:
    raise PathwaysCheckpointingUnavailableError(
        f"ENABLE_PATHWAYS_PERSISTENCE=1 explicitly requested Pathways checkpointing "
        f"(impl={impl_name!r}), but after registration jax.Array is handled by {handler_name}, "
        f"not {expected}. This is the controller-side staging path that exhausts proxy-pod "
        "memory at 397B scale; aborting before the first save rather than OOM-ing mid-run."
    )

  if impl_name == _COLOCATED_PYTHON:
    _configure_colocated_python_handler(handler, d2h_concurrent_gb=d2h_concurrent_gb)

  store = getattr(handler, "_array_metadata_store", None)
  if store is None:
    logging.error(
        "Registered %s but array metadata store is None; saving may fail on typed PRNG keys.",
        handler_name,
    )
  else:
    logging.info(
        "Registered Pathways array handler (impl=%s, %s, store=%s); " "TPUs will write directly to storage.",
        impl.name,
        handler_name,
        type(store).__name__,
    )

  # Latch only on success, so a failed attempt re-raises on the next construction
  # instead of being swallowed by the early return above.
  _REGISTERED_IMPL = impl_name


def _normalize_colocated_cpu_shardings() -> None:
  """Strips `pinned_host` memory kind when mapping shardings to sidecar CPU devices."""
  try:
    from orbax.checkpoint._src.multihost import colocated_transport  # pylint: disable=g-import-not-at-top,import-outside-toplevel
  except (ImportError, AttributeError):
    return
  if getattr(colocated_transport, "_maxtext_Normalized", False):
    return

  def _strip_pinned(fn: Any) -> Any:
    return lambda shd: fn(
        shd.with_memory_kind("device")
        if getattr(shd, "memory_kind", None) == "pinned_host"
        else shd
    )

  colocated_transport.colocated_cpu_sharding = _strip_pinned(colocated_transport.colocated_cpu_sharding)
  colocated_transport._normalize_single_device_sharding_to_colocated_cpu = _strip_pinned(
      colocated_transport._normalize_single_device_sharding_to_colocated_cpu
  )
  colocated_transport._maxtext_Normalized = True


def _est_host_bytes(spec: jax.ShapeDtypeStruct) -> int:
  """Estimates per-host bytes for an array spec, handling extended dtypes like key<fry>."""
  itemsize = spec.dtype.itemsize
  if spec.sharding is not None:
    devices = spec.sharding.device_set
    max_devs_per_host = 8 if any("v7" in d.device_kind.lower() for d in devices) else 4
    devs_per_host = min(max_devs_per_host, len(devices))
    return devs_per_host * math.prod(spec.sharding.shard_shape(spec.shape)) * itemsize
  return math.prod(spec.shape) * itemsize


def _configure_colocated_python_handler(handler: Any, *, d2h_concurrent_gb: float | None = None) -> None:
  """Configures colocated Python handler with sharding normalization and serialized batched dispatch."""
  max_bytes = (
      int(d2h_concurrent_gb * 10**9)
      if d2h_concurrent_gb is not None and d2h_concurrent_gb > 0
      else _COLOCATED_DISPATCH_MAX_BYTES
  )
  _normalize_colocated_cpu_shardings()
  dispatcher = getattr(handler, "_dispatcher", None)
  if dispatcher is None or getattr(dispatcher, "_maxtext_wrapped", False):
    return
  lock = threading.Lock()
  orig_dispatch = dispatcher.dispatch

  def _run_dispatch(func: Any, input_arrays: Any, specs: Any, func_args: Any, kw: Any) -> Any:
    res = orig_dispatch(func, input_arrays=input_arrays, result_specs=specs, func_args=func_args, func_kwargs=kw)
    try:
      jax.block_until_ready(res)
    except Exception:  # pylint: disable=broad-except
      pass
    return res

  def locked_dispatch(
      func: Any,
      *,
      input_arrays: Any = None,
      result_specs: Any = None,
      func_args: Any = (),
      func_kwargs: Any = None,
  ) -> Any:
    with lock:
      if (
          getattr(func, "__name__", "") == "_sync_deserialize_arrays"
          and isinstance(result_specs, (list, tuple))
          and len(result_specs) > 1
          and func_kwargs is not None
          and {"infos", "args", "shardings"} <= func_kwargs.keys()
      ):
        batch: list[Any] = []
        b_bytes = 0
        out: list[Any] = []

        def _flush() -> None:
          infos, args, shds, specs = map(list, zip(*batch))
          kw = {**func_kwargs, "infos": infos, "args": args, "shardings": shds}
          out.extend(_run_dispatch(func, input_arrays, specs, func_args, kw))
          batch.clear()

        for item in zip(
            func_kwargs["infos"], func_kwargs["args"], func_kwargs["shardings"], result_specs
        ):
          arr_bytes = _est_host_bytes(item[3])
          if batch and b_bytes + arr_bytes >= max_bytes:
            _flush()
            b_bytes = 0
          batch.append(item)
          b_bytes += arr_bytes
        if batch:
          _flush()
        return out
      return _run_dispatch(func, input_arrays, result_specs, func_args, func_kwargs)

  dispatcher.dispatch = locked_dispatch
  dispatcher._maxtext_wrapped = True


def _assert_uniform_device_set(tree: Any, *, item: str) -> None:
  """Raises PathwaysCheckpointingUnavailableError if leaves in `tree` span differing device sets.

  Orbax's ColocatedPythonDispatcher dispatches by array sharding device_set and asserts a single
  uniform device set per dispatched batch. A 0-D scalar leaf (such as optax `count` or `step`) left
  on a 1-device SingleDeviceSharding causes the dispatcher to fail inside Orbax with a generic
  ValueError naming neither the item nor the offending leaf keypaths.

  Args:
    tree: The PyTree or NNX State to inspect (pure metadata read, zero device transfer).
    item: Name of the checkpoint item being saved (e.g., "model_params" or "optimizer_state").

  Raises:
    PathwaysCheckpointingUnavailableError: If any leaf's sharding.device_set differs from the
      majority device set across `tree`, listing up to 5 offending keypaths.
  """
  flat = jax.tree_util.tree_flatten_with_path(
      nnx.to_pure_dict(tree) if isinstance(tree, nnx.State) else tree
  )[0]
  sharded: list[tuple[Any, frozenset[Any]]] = []
  for path, leaf in flat:
    sharding = getattr(leaf, "sharding", None)
    device_set = getattr(sharding, "device_set", None)
    if device_set is not None:
      sharded.append((path, frozenset(device_set)))
  if len(sharded) <= 1:
    return
  majority_set, _ = collections.Counter(ds for _, ds in sharded).most_common(1)[0]
  offenders = [
      f"{jax.tree_util.keystr(p)} (len(device_set)={len(ds)})"
      for p, ds in sharded
      if ds != majority_set
  ]
  if offenders:
    sample = ", ".join(offenders[:5])
    more = f" (+{len(offenders) - 5} more)" if len(offenders) > 5 else ""
    raise PathwaysCheckpointingUnavailableError(
        f"colocated_python checkpoint save for item={item!r} requires every leaf to share "
        f"the same sharding.device_set (majority len={len(majority_set)}), but {len(offenders)} "
        f"leaf/leaves differ: {sample}{more}."
    )


def _stage_to_pinned_host(tree: Any) -> Any:
  """Stages jax.Array leaves with non-pinned_host sharding to pinned_host."""
  def _stage_leaf(x: Any) -> Any:
    if isinstance(x, jax.Array) and not jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
      sharding = getattr(x, "sharding", None)
      if sharding is not None and getattr(sharding, "memory_kind", None) != "pinned_host":
        try:
          return jax.device_put(x, sharding.with_memory_kind("pinned_host"))
        except (ValueError, RuntimeError):
          pass
    return x

  return jax.tree.map(_stage_leaf, tree)


class CheckpointManager:
  """CheckpointManager wrapper for MaxText training engine."""

  def __init__(
      self,
      checkpoint_dir: str,
      config: pyconfig.HyperParameters,
  ) -> None:
    """Initializes the CheckpointManager.

    Args:
      checkpoint_dir: The root directory for saving checkpoints.
      config: The training configuration.
    """
    self._checkpoint_manager: ocp.CheckpointManager | None = None
    self._async_checkpointing = bool(config.async_checkpointing)
    # Whether a background save that fails is raised from the next checkpoint call (False, the
    # default) or logged and dropped (True). See `_drain_in_flight_save`.
    self._abandon_failed_saves = config.abandon_failed_checkpoint_saves
    # The step of the save most recently handed to Orbax, so a failure that surfaces later can
    # be attributed to it.
    self._in_flight_step: int | None = None
    # The failure dropped by the most recent abandonment, if no later save has replaced it in
    # Orbax; the one failure a wait with no save in flight may drop again (see `_drain_in_flight_save`).
    self._abandoned_failure: BaseException | None = None
    if checkpoint_dir:
      if os.environ.get("ENABLE_PATHWAYS_PERSISTENCE") == "1" and not str(checkpoint_dir).startswith("gs://"):
        raise ValueError(
            "ENABLE_PATHWAYS_PERSISTENCE=1 dispatches persistence writes to every "
            f"pathways-worker; checkpoint_dir must be a gs:// URI, got {checkpoint_dir!r}."
        )
      impl = getattr(config, "pathways_checkpointing_impl", _PERSISTENCE)
      if impl == _COLOCATED_PYTHON:
        _maybe_register_pathways_persistence(
            impl, d2h_concurrent_gb=getattr(config, "checkpoint_storage_device_host_concurrent_gb", None)
        )
      else:
        _maybe_register_pathways_persistence(impl)

      # Use configured array format (e.g. use_ocdbt=False for Pathways).
      # Build a fresh handler per item as Orbax handlers carry per-item state.
      def _pytree_handler() -> ocp.PyTreeCheckpointHandler:
        return ocp.PyTreeCheckpointHandler(
            use_ocdbt=config.checkpoint_storage_use_ocdbt,
            use_zarr3=config.checkpoint_storage_use_zarr3,
            save_device_host_concurrent_gb=config.checkpoint_storage_device_host_concurrent_gb,
        )

      self._checkpoint_manager = ocp.CheckpointManager(
          directory=checkpoint_dir,
          options=ocp.CheckpointManagerOptions(
              save_interval_steps=config.checkpoint_period,
              max_to_keep=config.max_num_checkpoints_to_keep,
              enable_async_checkpointing=config.async_checkpointing,
              # Deadline for the background half of a save (storage writes and finalization).
              async_options=ocp.AsyncOptions(timeout_secs=config.async_checkpointing_timeout_secs),
          ),
          item_handlers={
              "model_params": _pytree_handler(),
              "optimizer_state": _pytree_handler(),
              "accumulated_metrics": _pytree_handler(),
              "accumulated_grads": _pytree_handler(),
          },
      )

  def get_latest_step(self) -> int | None:
    """Returns the latest checkpoint step."""
    if self._checkpoint_manager:
      return self._checkpoint_manager.latest_step()
    return None

  def wait_until_finished(self) -> None:
    """Waits for any ongoing async checkpoint save; a failed one is raised or abandoned (see `_drain_in_flight_save`)."""
    self._drain_in_flight_save("explicit wait")

  def _drain_in_flight_save(self, reason: str) -> bool:
    """Blocks until the save most recently handed to Orbax has finished, attributing its failure to its step.

    Orbax finishes an async save on a background thread with a deadline (`async_checkpointing_timeout_secs`,
    1200 s by default). That thread's failure is stored and raised from whichever later call waits on
    it -- the next `save`, `delete`, `wait_until_finished` or `close` -- once per thread that waits
    (`CheckpointManager._FinalizeThread.join` keeps a per-thread flag and never clears the stored
    exception; only a later save replaces the thread). A timed-out save therefore surfaces from the
    *next* checkpoint call, while the training state in memory is intact. So every wait in this
    wrapper goes through here. By default the failure is logged against the step it belongs to and
    re-raised, which takes that next call -- and, through the trainer worker, the run -- down with it.
    With `abandon_failed_checkpoint_saves=true` it is logged and the save dropped instead: the next
    save proceeds normally. Only a failure attributable to a save this wrapper handed over is ever
    dropped: the one in flight, or the already abandoned one surfacing again on another thread (it is
    the same exception object). Anything else is raised regardless of the flag.

    Dropping a save stops nothing that is still running: the deadline only ends Orbax's wait, and the
    storage writes it was waiting for (with Pathways persistence, uploads already issued to the
    workers) finish on their own, leaving the step directory uncommitted. Nothing awaits them, so a
    late failure cannot reach this thread.

    Args:
      reason: Why the wait is happening, for the log line.

    Returns:
      True if nothing was in flight (or checkpointing is disabled) or it finished; False if it failed
      and has been abandoned.

    Raises:
      The save's failure, unless `abandon_failed_checkpoint_saves` is True; any failure that cannot be
      attributed to a save this wrapper handed over.
    """
    if self._checkpoint_manager is None:
      return True
    step = self._in_flight_step
    try:
      self._checkpoint_manager.wait_until_finished()
    except Exception as e:  # pylint: disable=broad-except
      self._in_flight_step = None
      if step is None:
        if e is self._abandoned_failure:
          logging.warning(
              "The already abandoned checkpoint save surfaced again on this thread (%s): %s: %s",
              reason,
              type(e).__name__,
              e,
          )
          return False
        logging.error(
            "Waiting on the checkpoint manager failed with no save in flight (%s): %s: %s. This is not a background save "
            "failure this wrapper can attribute to a step, so it is raised regardless of abandon_failed_checkpoint_saves.",
            reason,
            type(e).__name__,
            e,
        )
        raise
      if not self._abandon_failed_saves:
        logging.error(
            "The background half of the checkpoint save at step %s failed with %s: %s; raising it from here (%s). "
            "The training state is intact; set abandon_failed_checkpoint_saves=true to drop such a save and continue.",
            step,
            type(e).__name__,
            e,
            reason,
        )
        raise
      self._abandoned_failure = e
      logging.error(
          "Abandoning the checkpoint save at step %s (%s): its background half failed with %s: %s. "
          "The training state is intact and the next save will proceed; step %s is not restorable.",
          step,
          reason,
          type(e).__name__,
          e,
          step,
      )
      return False
    self._in_flight_step = None
    return True

  def get_saved_micro_step_count(self, step: int) -> int:
    """Returns how far into `step` the checkpoint already on disk got.

    Args:
      step: The step whose saved checkpoint should be inspected.

    Returns:
      0 if that checkpoint covers a complete step, otherwise the number of micro-batches
      accumulated into it.
    """
    if self._checkpoint_manager is None:
      return 0
    try:
      metadata = self._checkpoint_manager.metadata(step)
    except Exception as e:  # pylint: disable=broad-except
      logging.warning("Could not read metadata for step %d, treating it as complete: %s", step, e)
      return 0
    custom_metadata = getattr(metadata, "custom_metadata", None)
    if not isinstance(custom_metadata, Mapping):
      return 0
    saved = custom_metadata.get("micro_step_count", 0)
    return saved if isinstance(saved, int) else 0

  def _supersedes_saved_checkpoint(self, step: int, micro_step_count: int) -> bool:
    """Returns whether a new checkpoint is more complete than the one saved at `step`.

    A complete step is never superseded: once the optimizer update for `step` has been
    checkpointed there is nothing more to record for it. A partial one is superseded by a
    complete step, and by a partial one that got further through the same step.

    Args:
      step: The step both checkpoints belong to.
      micro_step_count: The new checkpoint's progress through `step`.

    Returns:
      Whether the new checkpoint should replace the saved one.
    """
    saved_micro_step_count = self.get_saved_micro_step_count(step)
    if saved_micro_step_count == 0:
      return False
    return micro_step_count == 0 or micro_step_count > saved_micro_step_count

  def _latest_step_with_optimizer_state(self, latest_step: int) -> int:
    """Returns the newest step that resumes training with its optimizer state.

    Saves made with `save_optimizer_state=False` hold only the model params, and resuming from
    one would restart the optimizer from scratch. So fall back to the newest checkpoint that has
    the optimizer state, and delete the params-only steps after it: the resumed run redoes them,
    and Orbax refuses to write a step that already exists. If no checkpoint holds the optimizer
    state, `latest_step` is returned unchanged and the restore raises.

    Args:
      latest_step: The latest step on disk.

    Returns:
      The step to restore from.
    """
    steps = sorted(self._checkpoint_manager.all_steps(), reverse=True)
    for step in steps:
      if "optimizer_state" in self._checkpoint_manager.metadata(step).item_metadata:
        break
    else:
      return latest_step
    newer = [s for s in steps if s > step]
    if newer:
      logging.warning(
          "Latest checkpoint step %d has no optimizer state; resuming from step %d and deleting the"
          " params-only steps %s after it.",
          latest_step,
          step,
          sorted(newer),
      )
      self._drain_in_flight_save("before deleting params-only steps")
      for s in newer:
        self._checkpoint_manager.delete(s)
    return step

  def _delete_saved_step(self, step: int) -> None:
    """Deletes the checkpoint at `step` to make room for a more complete one.

    Orbax refuses to write a step that already exists, so superseding one means removing it
    first. An async save for this step may still be in flight, so drain before deleting.

    Args:
      step: The step to delete.
    """
    logging.info("Deleting intra-step checkpoint at step %d so a more complete one can replace it.", step)
    in_flight = self._in_flight_step
    if not self._drain_in_flight_save(f"before deleting step {step}") and in_flight == step:
      # The save being superseded was the one just abandoned: Orbax has already dropped it from
      # its bookkeeping (`delete` would raise FileNotFoundError), and the forced save that follows
      # removes whatever it left in the step directory.
      logging.info("The abandoned save at step %d is no longer registered; nothing to delete.", step)
      return
    self._checkpoint_manager.delete(step)

  def save_checkpoint(
      self,
      step: int,
      checkpoint_state: CheckpointState,
      custom_metadata: Any = None,
      **kwargs,
  ) -> bool:
    """Saves the params for the given step along with optional intra-step state.

    Args:
      step: The step to save the params for.
      checkpoint_state: CheckpointState object containing model, optimizer, and
        optional intra_step_state.
      custom_metadata: Custom metadata to save with the checkpoint.

    Returns:
      Whether the checkpoint was saved.
    """
    if self._checkpoint_manager is None:
      logging.info("Checkpointing is disabled, skipping save.")
      return False

    # Record micro_step_count on every checkpoint, complete or not, so that a later save at the same step
    # can tell whether it supersedes what is already on disk.
    custom_metadata = dict(custom_metadata) if custom_metadata else {}
    custom_metadata["micro_step_count"] = checkpoint_state.micro_step_count

    # Orbax waits for the previous save only for a step it is going to save ("must happen after
    # `should_save` to avoid blocking callers"); mirror that, so a step the interval policy declines
    # neither blocks on the in-flight save nor takes its failure. The wait is also where the previous
    # save's failure surfaces -- absorbed or raised per `abandon_failed_checkpoint_saves` -- rather
    # than from inside Orbax's `save`, which would take this save down with it; and it drops a
    # failed step from `get_latest_step()` before the check below.
    if kwargs.get("force") or self._checkpoint_manager.should_save(step):
      self._drain_in_flight_save(f"before saving step {step}")

    # A checkpoint already exists at this step. Skip, unless this one is more complete --
    # the case that matters is a step resumed from an intra-step checkpoint and then run to
    # completion, whose finished state would otherwise never reach disk.
    if self.get_latest_step() == step:
      if not self._supersedes_saved_checkpoint(step, checkpoint_state.micro_step_count):
        logging.info(
            "Checkpoint already saved at step %d, skipping save.",
            step,
        )
        return False
      self._delete_saved_step(step)
      # Orbax's save-interval policy declines a step it has already saved, so the
      # replacement has to be forced through.
      kwargs["force"] = True

    params = nnx.state(checkpoint_state.model)
    if _REGISTERED_IMPL == _COLOCATED_PYTHON:
      _assert_uniform_device_set(params, item="model_params")

    # Taken from the values handed to Orbax, before the async upload overlaps the next step, so
    # restore can prove the persisted bytes are exactly these.
    fingerprints_enabled = _orbax_fingerprint_enabled()
    fingerprint_start = time.perf_counter()
    fingerprints = {"model_params": _tree_fingerprint(params)} if fingerprints_enabled else {}
    fingerprint_seconds = time.perf_counter() - fingerprint_start

    if _REGISTERED_IMPL == _COLOCATED_PYTHON and self._async_checkpointing:
      params = _stage_to_pinned_host(params)
    jax.block_until_ready(params)
    model_cp_args = ocp.args.PyTreeSave(
        item=params,
        save_args=jax.tree.map(lambda _: ocp.SaveArgs(), params),
    )
    save_args = {"model_params": model_cp_args}

    if checkpoint_state.optimizer:
      optimizer_state = nnx.state(checkpoint_state.optimizer, nnx.optimizer.OptState)
      if _REGISTERED_IMPL == _COLOCATED_PYTHON:
        _assert_uniform_device_set(optimizer_state, item="optimizer_state")
      jax.block_until_ready(optimizer_state)
      optimizer_cp_args = ocp.args.PyTreeSave(
          item=optimizer_state,
          save_args=jax.tree.map(lambda _: ocp.SaveArgs(), optimizer_state),
      )
      save_args["optimizer_state"] = optimizer_cp_args
      if fingerprints_enabled:
        fingerprint_start = time.perf_counter()
        fingerprints["optimizer_state"] = _tree_fingerprint(optimizer_state)
        fingerprint_seconds += time.perf_counter() - fingerprint_start
    if fingerprints_enabled:
      custom_metadata[_FINGERPRINT_KEY] = fingerprints

    if checkpoint_state.accumulated_metrics:
      jax.block_until_ready(checkpoint_state.accumulated_metrics)
      metrics_cp_args = ocp.args.PyTreeSave(
          item=checkpoint_state.accumulated_metrics,
          save_args=jax.tree.map(
              lambda _: ocp.SaveArgs(),
              checkpoint_state.accumulated_metrics,
          ),
      )
      save_args["accumulated_metrics"] = metrics_cp_args

    if checkpoint_state.accumulated_grads:
      jax.block_until_ready(checkpoint_state.accumulated_grads)
      grads_cp_args = ocp.args.PyTreeSave(
          item=checkpoint_state.accumulated_grads,
          save_args=jax.tree.map(
              lambda _: ocp.SaveArgs(),
              checkpoint_state.accumulated_grads,
          ),
      )
      save_args["accumulated_grads"] = grads_cp_args

    saved = self._checkpoint_manager.save(
        step=step,
        args=ocp.args.Composite(**save_args),
        custom_metadata=custom_metadata,
        **kwargs,
    )
    if saved:
      self._in_flight_step = step
      self._abandoned_failure = None  # Orbax's finalize thread is replaced; the old failure cannot surface again.
    if saved and not fingerprints_enabled:
      logging.info("Checkpoint step=%d saved without fingerprints (ENABLE_ORBAX_FINGERPRINT=0).", step)
    elif saved:  # Orbax's interval policy may decline; only an accepted save carries these values.
      logging.info(
          "Checkpoint %s step=%d %s (%.2fs)",
          _FINGERPRINT_KEY,
          step,
          _format_fingerprints(fingerprints),
          fingerprint_seconds,
      )
    return saved

  def restore_checkpoint(
      self,
      checkpoint_state: CheckpointState,
      step: int | None = None,
  ) -> tuple[int | None, CheckpointState, Any]:
    """Restores items from the checkpoint at the given step.

    Args:
      checkpoint_state: CheckpointState object containing model and optimizer.
      step: Optional step index to restore from.

    Returns:
      A tuple of (step, checkpoint_state, custom metadata).

    Raises:
      CheckpointRestoreError: If a checkpoint exists at `step` but cannot be restored.
    """
    if self._checkpoint_manager is None:
      logging.info("Checkpointing is disabled, skipping restore.")
      return None, checkpoint_state, None

    if step is None:
      step = self.get_latest_step()
      if step is None:
        logging.info("No checkpoint found, skipping restore.")
        return None, checkpoint_state, None
      if checkpoint_state.optimizer is not None:
        step = self._latest_step_with_optimizer_state(step)

    metadata = self._checkpoint_manager.metadata(step)
    restore_args: dict[str, Any] = {}

    abstract_params = nnx.state(checkpoint_state.model)
    restore_args["model_params"] = ocp.args.PyTreeRestore(
        item=abstract_params,
        restore_args=ocp.checkpoint_utils.construct_restore_args(target=abstract_params),
    )

    if checkpoint_state.optimizer is not None and "optimizer_state" not in metadata.item_metadata:
      # Resuming training with a freshly initialized optimizer would silently change the run.
      raise CheckpointRestoreError(
          f"Checkpoint at step {step} has no optimizer state (saved with save_optimizer_state=False);"
          " restore it with optimizer=None, or resume from a step that has the optimizer state."
      )
    if checkpoint_state.optimizer is not None:
      optimizer_state = nnx.state(checkpoint_state.optimizer, nnx.optimizer.OptState)

      # `CloudPathwaysArrayHandler.deserialize` ignores `memory_kind="pinned_host"` for physical
      # placement (allocating restored buffers in device HBM) while keeping `pinned_host` on the
      # returned array's sharding metadata. Request `device` placement explicitly so the restored
      # sharding matches physical placement; `MaxTextTrainingEngine.restore_checkpoint` then
      # offloads the restored optimizer state to `pinned_host`.
      def _device_restore_target(leaf: Any) -> Any:
        sharding = getattr(leaf, "sharding", None)
        if getattr(sharding, "memory_kind", None) == "pinned_host":
          return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=sharding.with_memory_kind("device"))
        return leaf

      optimizer_target = jax.tree.map(_device_restore_target, optimizer_state)
      restore_args["optimizer_state"] = ocp.args.PyTreeRestore(
          item=optimizer_target,
          restore_args=ocp.checkpoint_utils.construct_restore_args(target=optimizer_target),
      )

    if "accumulated_metrics" in metadata.item_metadata:
      restore_args["accumulated_metrics"] = ocp.args.PyTreeRestore()

    if "accumulated_grads" in metadata.item_metadata:
      accumulated_grads_target = checkpoint_state.accumulated_grads_target
      if accumulated_grads_target is None:
        accumulated_grads_target = nnx.state(checkpoint_state.model, nnx.Param)
      restore_args["accumulated_grads"] = ocp.args.PyTreeRestore(
          item=accumulated_grads_target,
          restore_args=ocp.checkpoint_utils.construct_restore_args(target=accumulated_grads_target),
      )

    custom_metadata = None
    if metadata and hasattr(metadata, "custom_metadata"):
      custom_metadata = metadata.custom_metadata

    restore_start = time.perf_counter()
    try:
      restored_items = self._checkpoint_manager.restore(
          step=step,
          args=ocp.args.Composite(**restore_args),
      )
    except Exception as e:  # pylint: disable=broad-except
      # Returning "no checkpoint" here would make the orchestrator start a fresh run from the
      # base weights while the checkpoint it failed to read is still on disk.
      raise CheckpointRestoreError(
          f"Checkpoint at step {step} exists but could not be restored; refusing to silently "
          f"start a fresh run over it. {type(e).__name__}: {e}"
      ) from e
    restore_seconds = time.perf_counter() - restore_start

    saved_fingerprints = custom_metadata.get(_FINGERPRINT_KEY) if isinstance(custom_metadata, Mapping) else None
    if not _orbax_fingerprint_enabled():
      logging.info(
          "Checkpoint at step %d: fingerprint verification disabled (ENABLE_ORBAX_FINGERPRINT=0); skipping verification"
          " (restore %.1fs).",
          step,
          restore_seconds,
      )
    elif saved_fingerprints is None:
      logging.info(
          "Checkpoint at step %d predates save-time fingerprints; skipping verification (restore %.1fs).",
          step,
          restore_seconds,
      )
    else:
      verify_start = time.perf_counter()
      verified = {}  # Only items actually restored and recomputed, so the log never claims more.
      # Checked before `nnx.update`, so a mismatch leaves the live state untouched.
      for item, want in saved_fingerprints.items():
        if item not in restored_items:
          continue
        if (got := _tree_fingerprint(restored_items[item])) != want:
          raise CheckpointRestoreError(
              f"Checkpoint at step {step}: restored {item!r} fingerprint {got:#010x} != {want:#010x} "
              "recorded at save time; the persisted bytes differ from what the trainer held (or the "
              "restore target's dtypes differ from the saved ones)."
          )
        verified[item] = got
      logging.info(
          "Verified checkpoint %s step=%d %s (restore %.1fs, verify %.2fs)",
          _FINGERPRINT_KEY,
          step,
          _format_fingerprints(verified),
          restore_seconds,
          time.perf_counter() - verify_start,
      )

    if "model_params" in restored_items:
      nnx.update(checkpoint_state.model, restored_items["model_params"])
    if checkpoint_state.optimizer is not None and "optimizer_state" in restored_items:
      nnx.update(checkpoint_state.optimizer, restored_items["optimizer_state"])
    if "accumulated_metrics" in restored_items:
      checkpoint_state.accumulated_metrics = restored_items["accumulated_metrics"]
    if "accumulated_grads" in restored_items:
      checkpoint_state.accumulated_grads = restored_items["accumulated_grads"]

    return step, checkpoint_state, custom_metadata

  def close(self) -> None:
    """Closes the checkpoint manager once the in-flight save, if any, has finished (a failed one is raised or abandoned)."""
    if self._checkpoint_manager:
      self._drain_in_flight_save("before close")
      self._checkpoint_manager.close()
