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

"""Per-shape AOT executable cache for jitted inference entry points.

A JAX persistent-compilation-cache hit still pays trace + lowering +
cache-key hashing on every process start (~seconds per big executable).
This module serializes the compiled executable itself
(``jax.experimental.serialize_executable``) after the first warmup, so
subsequent processes deserialize on a background thread and call it
directly: no trace, no lowering, no cache lookup.

Design (ported from DiffusionServing runners/torchax_aot.py, PR #38/#39,
minus the torch interop):
  * ``cached_jit`` replaces ``jax.jit`` at the definition site. Until
    ``install()`` is called it delegates to plain ``jax.jit`` with zero
    behavioral difference, so tests and trainers are unaffected.
  * One executable is kept PER dynamic input signature (shapes/dtypes of
    array leaves + treedef + non-array leaves). Different resolutions or
    frame counts never collide on disk.
  * Unknown signature -> silent jit fallback; the first call's args are
    recorded so ``save_pending()`` (call it after warmup, synchronously)
    can lower + serialize that shape without touching other shapes.
  * ``deserialize_and_load`` must receive ``execution_devices`` in the
    mesh's topology-pinned order; the default ``jax.devices()`` order can
    bind logical slots to the wrong physical chips and abort in C++.
  * A deserialized ``Compiled`` does not auto-reshard inputs like jit
    does; inputs are aligned to ``compiled.input_shardings`` in Python
    before the call.

Usage::

    @partial(aot_cache.cached_jit, static_argnames=("guidance_scale",))
    def transformer_forward_pass(...):
        ...

    # at pipeline construction (e.g. generate_wan.run):
    aot_cache.install(cache_dir, meta={...config fingerprint...}, mesh=mesh)
    # ... run warmup ...
    aot_cache.save_pending()
"""

from __future__ import annotations

import contextlib
import glob
import hashlib
import inspect
import os
import json
import pickle
import re
import threading
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax.experimental import serialize_executable

from maxtext.utils import max_logging

_FORMAT_VERSION = 1


def _metadata_fingerprint(meta: dict[str, Any]) -> str:
  """Returns the stable filename fingerprint for install-time metadata."""
  serialized = json.dumps(meta, sort_keys=True, default=str)
  return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def _dynamic_signature(args: tuple, kwargs: dict) -> str:
  """Deterministic digest of everything that selects an executable.

  Structure is captured by each leaf's KEY PATH (names, order, count) --
  NOT by ``repr(treedef)``: an nnx GraphDef's repr embeds object
  addresses and hash-order-dependent content that differ per process and
  made signatures never match across restarts (measured: every array
  part stable, only the treedef part unstable). Static graph metadata
  not visible in key paths (attention kernel, dtypes, model path) is
  covered by the install-time config fingerprint in the filename.
  Array leaves contribute shape/dtype; non-array leaves (python scalars,
  None flags) contribute an address-stripped repr.
  """
  leaves_with_paths = jax.tree_util.tree_flatten_with_path((args, kwargs))[0]
  parts = []
  for path, leaf in leaves_with_paths:
    if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
      desc = f"{tuple(leaf.shape)}:{leaf.dtype}"
    else:
      desc = re.sub(r"0x[0-9a-fA-F]+", "@", repr(leaf))
    parts.append(f"{jax.tree_util.keystr(path)}={desc}")
  return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


class _AotEntry:
  """Executables for one wrapped fn, keyed by dynamic input signature."""

  def __init__(self, name: str, fn: Callable, static_argnames: tuple):
    self.name = name
    self.fn = fn
    self.static_argnames = tuple(static_argnames)
    self.py_signature = inspect.signature(fn)
    self.jitted = jax.jit(fn, static_argnames=static_argnames or None)
    self._compiled: dict[str, Any] = {}
    self._out_specs: dict[str, Any] = {}
    self._pending: dict[str, tuple] = {}
    self._adapters: dict[str, Any] = {}
    self._on_disk: set[str] = set()
    self._lock = threading.Lock()

  def _zeros_output(self, signature: str):
    """Builds all-zero outputs matching a compiled signature's out specs.

    Compilation only needs avals -- executing a 14B transformer to warm the
    pipeline wastes the full step compute. In warmup mode the caller gets
    zeros with the executable's exact shapes/dtypes/shardings, so every
    DOWNSTREAM executable (scheduler step, VAE) still compiles against
    faithful inputs. Returns None if out specs are unknown for this
    signature (caller must execute normally).
    """
    spec = self._out_specs.get(signature)
    if spec is None:
      return None
    out_treedef, shapes_dtypes, out_shardings = spec
    zeros = [
        jax.device_put(jnp.zeros(shape, dtype), sharding) for (shape, dtype), sharding in zip(shapes_dtypes, out_shardings)
    ]
    return jax.tree_util.tree_unflatten(out_treedef, zeros)

  def _adapter_for(self, signature: str, treedef: Any, static: dict):
    """Returns the per-signature flat-leaf-list jit of fn.

    The input treedef (which may embed unpicklable statics, e.g. an nnx
    GraphDef holding initializer closures) stays inside this process's
    closure and is never serialized -- the adapter's own in/out trees are
    plain lists/tuples of arrays. Warmup compiles this adapter, so
    ``save_pending``'s lower().compile() hits the in-memory pjit cache
    instead of recompiling.
    """
    adapter_jit = self._adapters.get(signature)
    if adapter_jit is None:

      def adapter(flat, _treedef=treedef, _static=static):
        return self.fn(**jax.tree_util.tree_unflatten(_treedef, flat), **_static)

      adapter_jit = jax.jit(adapter)
      with self._lock:
        self._adapters.setdefault(signature, adapter_jit)
        adapter_jit = self._adapters[signature]
    return adapter_jit

  def _canonicalize(self, args: tuple, kwargs: dict) -> tuple[dict, dict]:
    """Splits a call into (dynamic kwargs, static kwargs) by param name.

    A deserialized ``Compiled`` must be called with the static args
    STRIPPED (they are baked into the graph and absent from its input
    pytree), and positional/keyword form must match how it was lowered.
    Canonicalizing every call to keyword form on both the lower and call
    paths makes the pytrees agree by construction.
    """
    bound = self.py_signature.bind(*args, **kwargs)
    dynamic, static = {}, {}
    for name, val in bound.arguments.items():
      (static if name in self.static_argnames else dynamic)[name] = val
    return dynamic, static

  def _compile_and_record(self, signature: str, leaves: list, treedef: Any, static: dict):
    """Lower+compile one signature (no execution) and capture out specs."""
    lowered = self._adapter_for(signature, treedef, static).lower(leaves)
    compiled = lowered.compile()
    info_leaves, out_treedef = jax.tree_util.tree_flatten(lowered.out_info)
    shapes_dtypes = [(tuple(x.shape), jnp.dtype(x.dtype)) for x in info_leaves]
    out_shardings = jax.tree_util.tree_leaves(compiled.output_shardings)
    with self._lock:
      self._compiled[signature] = compiled
      self._out_specs[signature] = (out_treedef, shapes_dtypes, out_shardings)
    return compiled

  # ---------------------------------------------------------------- call
  def __call__(self, *args, **kwargs):
    if not _STATE.enabled:
      return self.jitted(*args, **kwargs)
    dynamic, static = self._canonicalize(args, kwargs)
    leaves, treedef = jax.tree_util.tree_flatten(dynamic)
    if any(isinstance(leaf, jax.core.Tracer) for leaf in leaves):
      # Under an outer trace a deserialized executable cannot be applied
      # and tracers must not be recorded -- inline like a nested jit.
      return self.jitted(**dynamic, **static)
    signature = _dynamic_signature((), {**dynamic, **static})
    if _STATE.warmup_only:
      # Compilation only needs avals; skip the (possibly seconds-long)
      # real execution and hand back correctly-shaped/sharded zeros so
      # downstream executables still warm against faithful inputs.
      if signature not in self._compiled:
        self._compile_and_record(signature, leaves, treedef, static)
        with self._lock:
          if signature not in self._on_disk:
            self._pending[signature] = (leaves, treedef, static)
      zeros = self._zeros_output(signature)
      if zeros is not None:
        return zeros
    compiled = self._compiled.get(signature)
    if compiled is not None:
      flat = self._align_inputs(compiled, leaves)
      if flat is not None:
        return compiled(flat)
      # Fewer expected shardings than leaves: XLA pruned unused inputs
      # (e.g. encoder params in a decode-only executable). Compiled keeps
      # the full in_tree and prunes internally, so hand it the raw leaves;
      # sharding/structure problems surface as catchable Python errors.
      try:
        return compiled(leaves)
      except Exception as e:  # noqa: BLE001 - any failure means "use jit"
        max_logging.log(f"[aot] {self.name}: compiled call failed ({e}); using jit")
    with self._lock:
      if signature not in self._pending and signature not in self._compiled:
        self._pending[signature] = (leaves, treedef, static)
    return self._adapter_for(signature, treedef, static)(leaves)

  def _align_inputs(self, compiled: Any, leaves: list):
    """Reshards the flat input leaves onto the executable's shardings.

    jit auto-commits mismatched inputs; a deserialized Compiled does not --
    a placement mismatch aborts inside PjRt (uncatchable C++). Weights
    already carry final shardings; in practice this only moves small
    fresh-off-host activations. Returns the aligned leaf list, or None on
    structural mismatch (caller falls back to jit).
    """
    try:
      flat_expected = jax.tree_util.tree_leaves(compiled.input_shardings)
      if len(flat_expected) != len(leaves):
        # Fewer expected shardings than leaves = XLA pruned unused inputs;
        # the caller retries via Compiled's own pruning path. Not an error.
        return None
      aligned = []
      for leaf, expected in zip(leaves, flat_expected):
        if not hasattr(leaf, "shape"):  # python scalar traced as weak array
          leaf = jnp.asarray(leaf)
        sharding = getattr(leaf, "sharding", None)
        if sharding is not None and sharding.is_equivalent_to(expected, leaf.ndim):
          aligned.append(leaf)
        else:
          aligned.append(jax.device_put(leaf, expected))
      return aligned
    except Exception as e:  # noqa: BLE001 - any failure means "use jit"
      max_logging.log(f"[aot] {self.name}: cannot align inputs ({e}); using jit")
      return None

  # ---------------------------------------------------------------- disk
  def _path_for(self, signature: str) -> str:
    return os.path.join(_STATE.cache_dir, f"{self.name}-{_STATE.fingerprint}-{signature}.aotx")

  def load_from_disk(self) -> None:
    """Deserializes every on-disk executable for this fn. Never raises."""
    pattern = os.path.join(_STATE.cache_dir, f"{self.name}-{_STATE.fingerprint}-*.aotx")
    for path in glob.glob(pattern):
      try:
        with open(path, "rb") as f:
          blob = pickle.load(f)
        if blob["format_version"] != _FORMAT_VERSION:
          continue
        # Topology-pinned device order; the default reconstruction binds
        # logical slots to the wrong physical chips and aborts in C++.
        execution_devices = list(_STATE.mesh.devices.flatten()) if _STATE.mesh is not None else None
        compiled = serialize_executable.deserialize_and_load(
            blob["payload"],
            blob["in_tree"],
            blob["out_tree"],
            execution_devices=execution_devices,
        )
        signature = blob["dynamic_signature"]
        with self._lock:
          self._compiled[signature] = compiled
          self._on_disk.add(signature)
          if "out_shapes_dtypes" in blob:
            # Out specs let warmup mode return zeros instead of executing.
            self._out_specs[signature] = (
                blob["out_tree"],
                [(tuple(shape), jnp.dtype(dtype)) for shape, dtype in blob["out_shapes_dtypes"]],
                jax.tree_util.tree_leaves(compiled.output_shardings),
            )
        max_logging.log(f"[aot] {self.name}: loaded {os.path.basename(path)} ({len(blob['payload']) / 1e6:.1f}MB)")
      except Exception as e:  # noqa: BLE001 - fall back to jit for this shape
        max_logging.log(f"[aot] {self.name}: load failed for {os.path.basename(path)} ({e}); will re-jit")

  def save_pending(self) -> int:
    """Lowers + serializes every recorded signature. Returns count saved."""
    saved = 0
    with self._lock:
      pending, self._pending = self._pending, {}
    for signature, (leaves, treedef, static) in pending.items():
      if signature in self._on_disk:
        # Background deserialization landed after this shape was recorded.
        continue
      try:
        compiled = self._compiled.get(signature)
        if compiled is None:
          # Re-lowering retraces on this thread (the compile itself hits
          # the in-memory cache from warmup); sharding constraints inside
          # the model need the mesh context that warmup provided.
          mesh_ctx = _STATE.mesh if _STATE.mesh is not None else contextlib.nullcontext()
          with mesh_ctx:
            compiled = self._compile_and_record(signature, leaves, treedef, static)
        payload, in_tree, out_tree = serialize_executable.serialize(compiled)
        blob = {
            "format_version": _FORMAT_VERSION,
            "payload": payload,
            "in_tree": in_tree,
            "out_tree": out_tree,
            "dynamic_signature": signature,
        }
        out_spec = self._out_specs.get(signature)
        if out_spec is not None:
          blob["out_shapes_dtypes"] = [(list(shape), str(dtype)) for shape, dtype in out_spec[1]]
        path = self._path_for(signature)
        tmp_path = f"{path}.tmp.{os.getpid()}"
        with open(tmp_path, "wb") as f:
          pickle.dump(blob, f)
        os.replace(tmp_path, path)
        with self._lock:
          self._compiled[signature] = compiled
          self._on_disk.add(signature)
        saved += 1
        max_logging.log(f"[aot] {self.name}: serialized {os.path.basename(path)} ({len(payload) / 1e6:.1f}MB)")
      except Exception as e:  # noqa: BLE001 - saving is best-effort
        max_logging.log(f"[aot] {self.name}: serialize failed ({e}); shape stays on jit")
    return saved


class _State:
  """Process-global install state (null until install() is called)."""

  def __init__(self):
    self.enabled = False
    self.cache_dir = ""
    self.fingerprint = ""
    self.mesh = None
    self.warmup_only = False


_STATE = _State()
_REGISTRY: list[_AotEntry] = []
_LOAD_THREADS: list[threading.Thread] = []


def cached_jit(fn: Callable, static_argnames: tuple = ()) -> Callable:
  """Drop-in replacement for ``jax.jit`` with an optional AOT layer.

  Behaves exactly like ``jax.jit(fn, static_argnames=...)`` until
  ``install()`` enables the executable cache.
  """
  # Qualify by module: same-named fns (e.g. the VACE and base
  # transformer_forward_pass) must not glob each other's files.
  name = f"{fn.__module__.rsplit('.', 1)[-1]}.{fn.__name__}"
  entry = _AotEntry(name, fn, static_argnames)
  _REGISTRY.append(entry)
  return entry


def install(cache_dir: str, meta: dict[str, Any], mesh: Any) -> None:
  """Enables the AOT cache and starts background deserialization.

  Args:
    cache_dir: Directory for .aotx files (created if missing).
    meta: Everything the executables depend on beyond input shapes:
      model path, mesh shape, sharding/attention config, jax version.
      Hashed into the filename so incompatible executables never load.
    mesh: The pipeline mesh; pins device order for deserialization and
      provides the context for re-lowering at save time.
  """
  # A process may construct multiple pipelines with different cache settings.
  # Finish any previous loads, then reset all install-scoped state even when
  # the new cache directory is empty.
  wait_for_loads()
  _STATE.cache_dir = ""
  _STATE.fingerprint = ""
  _STATE.mesh = None
  _STATE.enabled = False
  for entry in _REGISTRY:
    with entry._lock:
      entry._compiled.clear()
      entry._out_specs.clear()
      entry._pending.clear()
      entry._adapters.clear()
      entry._on_disk.clear()

  if not cache_dir:
    return
  os.makedirs(cache_dir, exist_ok=True)
  _STATE.cache_dir = cache_dir
  _STATE.fingerprint = _metadata_fingerprint(meta)
  _STATE.mesh = mesh
  _STATE.enabled = True
  for entry in _REGISTRY:
    thread = threading.Thread(target=entry.load_from_disk, name=f"aot-load-{entry.name}", daemon=True)
    thread.start()
    _LOAD_THREADS.append(thread)


def wait_for_loads() -> None:
  """Blocks until background deserialization finishes (call before timing)."""
  for thread in _LOAD_THREADS:
    thread.join()
  _LOAD_THREADS.clear()


def save_pending() -> int:
  """Serializes all recorded shapes across wrapped fns. Call after warmup,
  synchronously -- a background save competes with the first real request
  (DiffusionServing PR#39 first-generation-stall lesson)."""
  if not _STATE.enabled:
    return 0
  return sum(entry.save_pending() for entry in _REGISTRY)


@contextlib.contextmanager
def warmup_mode():
  """Zero-execution warmup: wrapped fns lower+compile but never execute.

  Compilation only needs avals, not values -- a 14B transformer step costs
  seconds to run and nothing to skip. Inside this context a wrapped call
  compiles its signature (or reuses the deserialized executable) and
  returns all-zero outputs with the executable's exact shapes, dtypes and
  shardings, so downstream executables (scheduler step, VAE decode) still
  compile against faithful inputs. Outputs of a warmup pass are garbage by
  design; callers must discard them. No-op when the cache is disabled.
  """
  _STATE.warmup_only = _STATE.enabled
  try:
    yield
  finally:
    _STATE.warmup_only = False


def in_warmup() -> bool:
  """True inside `warmup_mode()`, so callers can add warmup-only work."""
  return _STATE.warmup_only


@contextlib.contextmanager
def real_execution():
  """Temporarily suspends zero-execution warmup so a call really runs.

  Warmup skips execution, leaving first-execution costs to the first real
  request. Wrap a targeted priming call in this to pay them during warmup.
  """
  previous = _STATE.warmup_only
  _STATE.warmup_only = False
  try:
    yield
  finally:
    _STATE.warmup_only = previous
