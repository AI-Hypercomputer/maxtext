# Copyright 2026 Google LLC
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

"""Path-transparent pinned-host offload and runtime TPU HBM telemetry for m3.

`HostOffloadedOptimizer` parks `opt_state` (such as FP32 AdamW first and second
moments) in `"pinned_host"` memory between steps and transfers them to `"device"`
inside `update`, preserving identical `nnx.state(model, nnx.Param)` paths and
step-for-step numerical trajectories when compiled with buffer donation
(`@nnx.jit(donate_argnums=(0, 1))`). `maybe_offload_parameters` optionally moves
`nnx.Param` state to `"pinned_host"` — either all parameters
(`parameter_memory_host_offload=True`) or only parameters whose `"/"`-joined
paths match `host_offload_param_patterns` (`selective_host_offload=True`),
leaving all other parameters and optimizer state on device.

For selectively offloaded `nnx.Embed` tables, `HostOffloadedEmbed` replaces the
module path-transparently with a static `nnx.Embed` subclass that gathers only
the needed rows per step from `pinned_host` to device via
`jax.experimental.compute_on("device_host")` (shard-locally inside
`jax.shard_map` + ICI `psum` when row-sharded). Under `value_and_grad` /
`pure_value_and_grad`, backward produces a row-sparse `SparseHostGrad` without
materializing a dense full-table gradient in HBM. Exact dense AdamW updates
(`host_compute` or `streamed_chunks`) remain the default; `lazy_sparse` is an
opt-in approximation that updates only rows touched in the current step and
skips weight/moment decay on untouched rows.

Inside traced regions, `to_memory` lowers transfers via
`jax.device_put(x, jax.memory.Space.*)` so XLA schedules host-device DMA copies
within the compiled program; outside tracing, concrete sharded arrays use
`x.sharding.with_memory_kind(kind)`.

Cluster trade-off: optimizer host offload is a memory-capacity mechanism rather
than a throughput optimization. On 64-chip TPU7x MoE runs, offloading FP32 AdamW
moments saves ~23.6-26.8 GiB/device of HBM (enabling fit within the 94.7 GiB HBM
ceiling), but incurs ~1.0-1.83 s/step (~21.8% step time) of PCIe transfer cost
when device HBM already accommodates resident moments.

`get_device_hbm_stats` queries live `device.memory_stats()` and reports both
`peak_bytes_in_use` and `allocated_bytes = bytes_in_use + bytes_reserved`,
capturing XLA HLO temporary workspace held in `bytes_reserved`.
"""

from collections.abc import Callable, Sequence
import contextlib
import functools
import inspect
import re
import threading
from typing import Any

from flax import nnx
import jax
from jax.experimental import compute_on
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np
import optax

from maxtext.m3.infra import named_scopes

_GIB = float(2**30)
_HOST_KINDS = frozenset({"pinned_host", "unpinned_host"})
_SPACES = {
    "pinned_host": jax.memory.Space.Host,
    "device": jax.memory.Space.Device,
}
_OPTIMIZER_MODES = frozenset({"host_compute", "streamed_chunks", "lazy_sparse"})
_SELECTIVE_HOST_PATH_KEY = "_m3_selective_host_path"
_SELECTIVE_HOST_STAGE_KEY = "_m3_selective_host_stage"
_OFFLOADED_MODEL_CLASSES: dict[type[nnx.Module], type[nnx.Module]] = {}
_HOST_VARIABLE_CLASSES: dict[type[nnx.Variable], type[nnx.Variable]] = {}
_PREGATHERED_STATE = threading.local()
_CAPTURE_EMBED_IDS = threading.local()
_HOST_SQNORM_CAPTURE = threading.local()
_ACTIVE_PARAM_SHARDINGS = threading.local()


class _RecaptureEmbedIds(Exception):
  """Internal signal requesting per-call token-ID capture when pre-gather assumptions do not hold."""


@contextlib.contextmanager
def active_param_shardings(param_shardings: Any):
  """Provides physical parameter shardings during `Auto`-mode `jit` traces (e.g. `MaxTextTrainingEngine`)."""
  prev = getattr(_ACTIVE_PARAM_SHARDINGS, "map", None)
  if param_shardings is None:
    yield
    return
  spec_map: dict[tuple[Any, ...], Any] = {}
  for path, leaf in jax.tree_util.tree_leaves_with_path(
      param_shardings, is_leaf=lambda x: isinstance(x, (nnx.Variable, jax.sharding.Sharding, P))
  ):
    if isinstance(leaf, nnx.Variable):
      leaf = leaf.get_raw_value()
    spec = getattr(leaf, "spec", leaf if isinstance(leaf, P) else None)
    if spec is not None:
      spec_map[tuple(_key_value(k) for k in path)] = spec
  _ACTIVE_PARAM_SHARDINGS.map = spec_map
  try:
    yield
  finally:
    _ACTIVE_PARAM_SHARDINGS.map = prev


def _resolve_compute_on_decorator(mod: Any = compute_on) -> Any | None:
  """Returns the `compute_on` decorator accepting `out_memory_spaces` (`0.11.0` `compute_on2` or `0.11.1+`)."""
  for attr in ("compute_on", "compute_on2"):
    fn = getattr(mod, attr, None)
    if callable(fn):
      try:
        if "out_memory_spaces" in inspect.signature(fn).parameters:
          return fn
      except (TypeError, ValueError):
        continue
  return None


_COMPUTE_ON_DECORATOR = _resolve_compute_on_decorator()


def _device_host_compute(*, out_memory_spaces: Any, compute_on_mod: Any = compute_on):
  """Decorates a `device_host` host-compute function with explicit output memory spaces across JAX versions."""
  decorator_fn = _COMPUTE_ON_DECORATOR if compute_on_mod is compute_on else _resolve_compute_on_decorator(compute_on_mod)
  if decorator_fn is not None:
    return decorator_fn(compute_type="device_host", out_memory_spaces=out_memory_spaces)

  def _decorator(fn: Any) -> Any:
    @functools.wraps(fn)
    def _wrapped(*args: Any, **kwargs: Any) -> Any:
      with compute_on_mod.compute_on("device_host"):
        out = fn(*args, **kwargs)
      if isinstance(out_memory_spaces, tuple) and isinstance(out, tuple):
        return tuple(jax.device_put(v, sp) for v, sp in zip(out, out_memory_spaces))
      return jax.tree.map(lambda v: jax.device_put(v, out_memory_spaces), out)

    return _wrapped

  return _decorator


def _call_shard_map(
    fn: Any,
    *,
    in_specs: Any,
    out_specs: Any,
    args: tuple[Any, ...],
    mesh: Any | None = None,
) -> Any:
  """Runs `jax.shard_map` under `jax.jit` so eager and `jax.checkpoint` callers can use `Space.Host` transfers."""
  amb = jax.sharding.get_abstract_mesh()
  use_explicit_mesh = (amb is None or amb.empty) and mesh is not None
  smap = (
      jax.shard_map(fn, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False)
      if use_explicit_mesh
      else jax.shard_map(fn, in_specs=in_specs, out_specs=out_specs, check_vma=False)
  )
  return jax.jit(smap)(*args)


def _coerce_sds_memory_kind(value: Any, kind: str | None) -> Any:
  """Returns `value` with `sharding.memory_kind == kind` when `value` is a `ShapeDtypeStruct`."""
  if kind and isinstance(value, jax.ShapeDtypeStruct):
    sharding = getattr(value, "sharding", None)
    if sharding is not None and getattr(sharding, "memory_kind", None) != kind:
      return jax.ShapeDtypeStruct(
          value.shape,
          value.dtype,
          sharding=sharding.with_memory_kind(kind),
      )
  return value


def _coerce_value_for_host_kind(
    raw_val: Any,
    mem_kind: str | None,
    metadata: dict[str, Any],
    *,
    eager_sharding: bool,
) -> Any:
  """Coerces `ShapeDtypeStruct`, `np.ndarray`, or misplaced `jax.Array` to `mem_kind`."""
  if isinstance(raw_val, jax.ShapeDtypeStruct):
    return _coerce_sds_memory_kind(raw_val, mem_kind)
  if mem_kind not in _HOST_KINDS:
    return raw_val
  target_space = _validate_kind(mem_kind)
  if isinstance(raw_val, jax.core.Tracer):  # jmp-waiver: TRACER_GUARD
    return raw_val
  if isinstance(raw_val, np.ndarray):
    mesh = metadata.get("mesh")
    if not isinstance(mesh, jax.sharding.Mesh):
      try:
        mesh = jax.sharding.get_mesh()
      except ValueError:
        mesh = None
    out_sh = metadata.get("out_sharding")
    if eager_sharding and isinstance(mesh, jax.sharding.Mesh) and not mesh.empty and isinstance(out_sh, P):
      return jax.device_put(
          raw_val,
          jax.sharding.NamedSharding(mesh, out_sh, memory_kind=mem_kind),
      )
    return _move_raw_value(raw_val, mem_kind, target_space)
  if isinstance(raw_val, jax.Array):
    sh = getattr(raw_val, "sharding", None)
    if getattr(sh, "memory_kind", None) != mem_kind:
      return _move_raw_value(raw_val, mem_kind, target_space)
  return raw_val


class _HostMemoryKindVariableMixin:
  """Preserves `memory_kind` across `nnx.eval_shape` (`_new`/`set_value`) and `replace_by_pure_dict`."""

  __slots__ = ()

  def __init__(
      self,
      value: Any,
      *,
      hijax: bool | None = None,
      ref: bool | None = None,
      eager_sharding: bool | None = None,
      **metadata: Any,
  ):
    if isinstance(value, nnx.VariableMetadata):
      raw_val = value.raw_value
      merged_meta = {**dict(value.metadata), **metadata}
    else:
      raw_val = value
      merged_meta = dict(metadata)
    merged_meta.pop("on_get_value", None)
    merged_meta.pop("on_set_value", None)
    if "hijax" in merged_meta:
      hijax = merged_meta.pop("hijax")
    if "ref" in merged_meta:
      ref = merged_meta.pop("ref")
    orig_eager = merged_meta.pop("eager_sharding", eager_sharding)
    if orig_eager is None:
      orig_eager = nnx.using_eager_sharding()
    mem_kind = merged_meta.get("memory_kind")
    raw_val = _coerce_value_for_host_kind(raw_val, mem_kind, merged_meta, eager_sharding=bool(orig_eager))
    if mem_kind in _HOST_KINDS:
      # Bypass `core_spmd.shard_value` during `Variable.__init__` so explicit/auto meshes never
      # reshard `pinned_host` values back onto `Space.Device` during `replace_by_pure_dict`.
      super().__init__(  # type: ignore[misc]
          nnx.VariableMetadata(raw_val, metadata=merged_meta),
          hijax=hijax,
          ref=ref,
          eager_sharding=False,
      )
      self.set_metadata("eager_sharding", bool(orig_eager))
      return
    super().__init__(  # type: ignore[misc]
        nnx.VariableMetadata(raw_val, metadata=merged_meta),
        hijax=hijax,
        ref=ref,
        eager_sharding=orig_eager,
    )

  @classmethod
  def _new(cls, value: Any, metadata: dict[str, Any]) -> Any:
    value = _coerce_sds_memory_kind(value, metadata.get("memory_kind"))
    return super()._new(value, metadata)  # type: ignore[misc]

  @classmethod
  def from_metadata(cls, value: Any, attributes: dict[str, Any]) -> Any:
    """Rebuilds a host-resident variable while preserving its `memory_kind` placement."""
    meta = dict(attributes)
    meta.pop("on_get_value", None)
    meta.pop("on_set_value", None)
    meta.setdefault("hijax", False)
    mem_kind = meta.get("memory_kind")
    eager = bool(meta.get("eager_sharding", nnx.using_eager_sharding()))
    value = _coerce_value_for_host_kind(value, mem_kind, meta, eager_sharding=eager)
    return super().from_metadata(value, meta)  # type: ignore[misc]

  def get_value(self, *, index: Any = nnx.variablelib.MISSING) -> Any:
    """Returns `value`, preserving `memory_kind` on `ShapeDtypeStruct` shardings."""
    raw_value = super().get_value()  # type: ignore[misc]
    if isinstance(raw_value, jax.ShapeDtypeStruct):
      coerced = _coerce_sds_memory_kind(raw_value, self.get_metadata("memory_kind", default=None))
      return coerced if isinstance(index, nnx.variablelib.Missing) or index is ... else coerced[index]
    return super().get_value(index=index)  # type: ignore[misc]

  def set_value(self, value: Any, *, index: Any = nnx.variablelib.MISSING) -> None:
    """Assigns `value` while keeping host-kind leaves off device HBM."""
    meta = self.get_metadata()
    mem_kind = meta.get("memory_kind")
    eager = bool(meta.get("eager_sharding", nnx.using_eager_sharding()))
    if isinstance(index, nnx.variablelib.Missing) or index is ...:
      value = _coerce_value_for_host_kind(value, mem_kind, meta, eager_sharding=eager)
    else:
      value = _coerce_sds_memory_kind(value, mem_kind)
    super().set_value(value, index=index)  # type: ignore[misc]

  def set_raw_value(self, value: Any, *, _unsafe_bypass_check: bool = False) -> None:
    """Assigns raw `value`, preserving `memory_kind` on `ShapeDtypeStruct` shardings."""
    value = _coerce_sds_memory_kind(value, self.get_metadata("memory_kind", default=None))
    super().set_raw_value(value, _unsafe_bypass_check=_unsafe_bypass_check)  # type: ignore[misc]


class HostOffloadedOptVariable(_HostMemoryKindVariableMixin, nnx.OptVariable):
  """`nnx.OptVariable` subclass keeping offloaded optimizer moments in `pinned_host`."""

  __slots__ = ()


class HostOffloadedOptArray(_HostMemoryKindVariableMixin, nnx.training.optimizer.OptArray):
  """`OptArray` subclass preserving `memory_kind` across `eval_shape`."""

  __slots__ = ()


class HostOffloadedOptState(_HostMemoryKindVariableMixin, nnx.OptState):
  """`nnx.OptState` subclass preserving `memory_kind` across `eval_shape`."""

  __slots__ = ()


class HostOffloadedVariable(_HostMemoryKindVariableMixin, nnx.Variable):
  """`nnx.Variable` subclass preserving `memory_kind` across `eval_shape`."""

  __slots__ = ()


def _host_variable_class(cls: type[nnx.Variable]) -> type[nnx.Variable]:
  """Returns a cached `_HostMemoryKindVariableMixin` subclass of `cls`."""
  if issubclass(cls, _HostMemoryKindVariableMixin):
    return cls
  known: dict[type[nnx.Variable], type[nnx.Variable]] = {
      nnx.Param: HostOffloadedParam,
      nnx.OptVariable: HostOffloadedOptVariable,
      nnx.training.optimizer.OptArray: HostOffloadedOptArray,
      nnx.OptState: HostOffloadedOptState,
      nnx.Variable: HostOffloadedVariable,
  }
  if cls in known:
    return known[cls]
  subcls = _HOST_VARIABLE_CLASSES.get(cls)
  if subcls is None:
    subcls = type(cls)(
        f"HostOffloaded{cls.__name__}",
        (_HostMemoryKindVariableMixin, cls),
        {"__slots__": (), "__module__": __name__},
    )
    _HOST_VARIABLE_CLASSES[cls] = subcls
  return subcls


def _to_memory_kind_variable(
    var: nnx.Variable,
    kind: str,
    *,
    raw_value: Any = nnx.variablelib.MISSING,
    extra_metadata: dict[str, Any] | None = None,
) -> nnx.Variable:
  """Constructs a `_HostMemoryKindVariableMixin` variable in `kind` without mutating `type(var)`."""
  meta = dict(var.get_metadata())
  meta.pop("on_get_value", None)
  meta.pop("on_set_value", None)
  meta.pop("_offload_saved_eager", None)
  if extra_metadata:
    meta.update(extra_metadata)
  meta["memory_kind"] = kind
  val = var.get_raw_value() if isinstance(raw_value, nnx.variablelib.Missing) else raw_value
  target_cls = _host_variable_class(type(var))
  return target_cls.from_metadata(val, meta)  # type: ignore[attr-defined]


def _is_host_leaf(x: Any) -> bool:
  """Returns True when `x` (or its wrapped value) resides in host memory."""
  val = x.get_raw_value() if isinstance(x, nnx.Variable) else x
  if isinstance(val, jax.core.Tracer):  # jmp-waiver: TRACER_GUARD
    aval = getattr(val, "aval", None)
    if getattr(aval, "memory_space", None) == jax.memory.Space.Host:
      return True
    return isinstance(x, nnx.Variable) and x.get_metadata("memory_kind", default=None) in _HOST_KINDS
  sharding = getattr(val, "sharding", None)
  if getattr(sharding, "memory_kind", None) in _HOST_KINDS:
    return True
  return isinstance(x, nnx.Variable) and x.get_metadata("memory_kind", default=None) in _HOST_KINDS


def _has_host_leaves(tree: Any) -> bool:
  """Returns True if any array or variable leaf in `tree` is in host memory."""
  leaves = jax.tree.leaves(tree, is_leaf=lambda x: isinstance(x, (nnx.Variable, SparseHostGrad)))
  return any(isinstance(leaf, SparseHostGrad) or _is_host_leaf(leaf) for leaf in leaves)


def _validate_kind(kind: str) -> Any:
  if kind not in _SPACES:
    raise ValueError(f"Unsupported memory kind {kind!r}; expected one of {sorted(_SPACES)}.")
  return _SPACES[kind]


def _move_raw_value(x: Any, kind: str, target_space: Any) -> Any:
  """Places a single raw leaf value in `kind` while leaving scalars/`None` unchanged."""
  if x is None or isinstance(x, (bool, int, float, complex, str, bytes)):
    return x
  if isinstance(x, jax.core.Tracer):  # jmp-waiver: TRACER_GUARD
    return jax.device_put(x, target_space)
  if isinstance(x, jax.ShapeDtypeStruct):
    sharding = getattr(x, "sharding", None)
    if sharding is not None:
      return jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sharding.with_memory_kind(kind))
    return x
  if isinstance(x, jax.Array):
    sharding = getattr(x, "sharding", None)
    target = sharding.with_memory_kind(kind) if sharding is not None else target_space
    if getattr(x, "ndim", -1) == 0:
      try:
        return jax.device_put(x, target)
      except jax.errors.JaxRuntimeError:
        if kind == "device":
          return jax.device_put(np.asarray(x), target)
        return x
    return jax.device_put(x, target)
  if isinstance(x, np.ndarray):
    return jax.device_put(x, target_space)
  return x


def to_memory(tree: Any, kind: str) -> Any:
  """Returns `tree` with array and `nnx.Variable` leaves placed in `kind`.

  Args:
    tree: Arbitrary PyTree containing `nnx.Variable`, traced/concrete JAX
      arrays, `jax.ShapeDtypeStruct`, NumPy arrays, or Python scalars/`None`.
    kind: Target memory kind (`"pinned_host"` or `"device"`).
  """
  target_space = _validate_kind(kind)

  def move(x: Any) -> Any:
    if isinstance(x, nnx.Variable):
      return _to_memory_kind_variable(x, kind, raw_value=_move_raw_value(x.get_raw_value(), kind, target_space))
    return _move_raw_value(x, kind, target_space)

  return jax.tree.map(move, tree, is_leaf=lambda x: isinstance(x, nnx.Variable))


def _move_variables_in_place(tree: Any, kind: str) -> None:
  """Moves `nnx.Variable` values in `tree` to `kind` without replacing Variable instances."""
  target_space = _validate_kind(kind)
  for leaf in jax.tree.leaves(tree, is_leaf=lambda x: isinstance(x, nnx.Variable)):
    if isinstance(leaf, nnx.Variable):
      leaf.set_metadata("memory_kind", kind)
      leaf.set_raw_value(_move_raw_value(leaf.get_raw_value(), kind, target_space))


def _key_value(key: Any) -> Any:
  """Extracts the raw Python key from a JAX PyTree key entry."""
  if hasattr(key, "key"):
    return key.key
  if hasattr(key, "name"):
    return key.name
  if hasattr(key, "idx"):
    return key.idx
  return key


def _format_param_path(path: Sequence[Any]) -> str:
  return "/".join(str(k) for k in path)


def is_selective_host_offload_enabled(config: Any) -> bool:
  """Returns True when selective path-pattern host offload is enabled in `config`."""
  return bool(getattr(config, "selective_host_offload", False))  # jmp-waiver: SILENT_CONFIG_FALLBACK


def get_host_offload_param_patterns(config: Any) -> tuple[str, ...]:
  """Returns normalized regex patterns selecting parameters for host offload."""
  raw = getattr(config, "host_offload_param_patterns", None) or ()  # jmp-waiver: SILENT_CONFIG_FALLBACK
  if isinstance(raw, str):
    return (raw,) if raw else ()
  return tuple(str(p) for p in raw if str(p))


def _get_optimizer_mode(config: Any) -> str:
  """Returns validated selective host-offload optimizer mode from `config`."""
  raw_mode = getattr(config, "selective_host_offload_optimizer_mode", None)  # jmp-waiver: SILENT_CONFIG_FALLBACK
  mode = "host_compute" if raw_mode is None else str(raw_mode)
  if mode not in _OPTIMIZER_MODES:
    raise ValueError(
        f"Unsupported selective_host_offload_optimizer_mode {mode!r}; expected one of {sorted(_OPTIMIZER_MODES)}."
    )
  return mode


def _get_num_chunks(config: Any) -> int:
  raw_chunks = getattr(config, "selective_host_offload_num_chunks", None)  # jmp-waiver: SILENT_CONFIG_FALLBACK
  chunks = 16 if raw_chunks is None else int(raw_chunks)
  if chunks < 1:
    raise ValueError(f"selective_host_offload_num_chunks must be >= 1, got {chunks}.")
  return chunks


def _match_param_paths(params_state: Any, patterns: Sequence[str]) -> frozenset[tuple[Any, ...]]:
  compiled = [re.compile(p) for p in patterns]
  matched: list[tuple[Any, ...]] = []
  for path, _ in nnx.to_flat_state(params_state):
    path_str = _format_param_path(path)
    if any(rx.search(path_str) for rx in compiled):
      matched.append(tuple(path))
  return frozenset(matched)


def selected_host_param_paths(model_or_state: Any) -> frozenset[tuple[Any, ...]]:
  """Returns canonical paths of `nnx.Param` variables marked for selective host offload."""
  state = nnx.state(model_or_state, nnx.Param) if isinstance(model_or_state, nnx.Module) else model_or_state
  paths: list[tuple[Any, ...]] = []
  for leaf in jax.tree.leaves(state, is_leaf=lambda x: isinstance(x, nnx.Variable)):
    if isinstance(leaf, nnx.Variable):
      marked = leaf.get_metadata(_SELECTIVE_HOST_PATH_KEY, default=None)
      if marked is not None:
        paths.append(tuple(marked))
  return frozenset(paths)


def has_selective_host_offload(model_or_state: Any) -> bool:
  """Returns True when `model_or_state` has selectively host-offloaded parameters."""
  return bool(selected_host_param_paths(model_or_state))


def _extract_mesh_from_obj(obj: Any) -> Any | None:
  """Extracts a multi-device `Mesh` from a Variable, Array, Tracer, SparseHostGrad, or Sharding."""
  if obj is None:
    return None
  if isinstance(obj, nnx.Variable):
    out_sh = obj.get_metadata("out_sharding", default=None)
    m = getattr(out_sh, "mesh", None)
    if m is not None and not getattr(m, "empty", False) and getattr(m, "size", 1) > 1:
      return m
    obj = obj.get_raw_value()
  if isinstance(obj, SparseHostGrad):
    return _extract_mesh_from_obj(obj.ids) or _extract_mesh_from_obj(obj.d_gathered)
  sh = getattr(obj, "sharding", None)
  if sh is None and isinstance(obj, jax.core.Tracer):  # jmp-waiver: TRACER_GUARD
    sh = getattr(jax.typeof(obj), "sharding", None)
  if sh is None and isinstance(obj, jax.sharding.Sharding):
    sh = obj
  m = getattr(sh, "mesh", None)
  if m is not None and not getattr(m, "empty", False) and getattr(m, "size", 1) > 1:
    return m
  return None


def _active_mesh(*candidates: Any) -> Any | None:
  """Returns the active multi-device mesh from `get_abstract_mesh()` or any candidate sharding."""
  mesh = jax.sharding.get_abstract_mesh()
  if mesh is not None and not mesh.empty and mesh.size > 1:
    return mesh
  for cand in candidates:
    cand_mesh = _extract_mesh_from_obj(cand)
    if cand_mesh is not None:
      return cand_mesh
  return None


def _is_explicit_mesh(mesh: Any | None) -> bool:
  return mesh is not None and all(t == jax.sharding.AxisType.Explicit for t in mesh.axis_types)


def _normalize_axis_entry(entry: Any, mesh: Any) -> tuple[str, ...]:
  if entry is None:
    return ()
  names = (entry,) if isinstance(entry, str) else tuple(entry)
  return tuple(ax for ax in names if ax in mesh.shape and mesh.shape[ax] > 1)


def _pspec_dim(axes: tuple[str, ...]) -> Any:
  if not axes:
    return None
  return axes[0] if len(axes) == 1 else axes


def _spec_dim_for_mesh(entry: Any, mesh: Any) -> Any:
  """Returns a `shard_map` `PartitionSpec` entry, preserving size-1 axes on `Explicit` meshes."""
  if entry is None:
    return None
  names = (entry,) if isinstance(entry, str) else tuple(entry)
  if _is_explicit_mesh(mesh):
    kept = tuple(ax for ax in names if ax in mesh.shape)
  else:
    kept = tuple(ax for ax in names if ax in mesh.shape and mesh.shape[ax] > 1)
  return _pspec_dim(kept)


def _axes_in_spec_dim(entry: Any) -> tuple[str, ...]:
  if entry is None:
    return ()
  return (entry,) if isinstance(entry, str) else tuple(entry)


def _resolve_2d_specs_and_axes(
    var_or_arr: Any,
    mesh: Any | None,
    sharding_hint: Any = None,
) -> tuple[Any, Any, tuple[str, ...], tuple[str, ...]]:
  """Returns `(r_spec, c_spec, row_axes, col_axes)` for a 2D table variable or array."""
  if mesh is None:
    return None, None, (), ()
  raw = var_or_arr.get_raw_value() if isinstance(var_or_arr, nnx.Variable) else var_or_arr
  spec = None
  if _is_explicit_mesh(mesh) and hasattr(raw, "ndim"):
    if isinstance(raw, jax.ShapeDtypeStruct):
      sh = getattr(raw, "sharding", None)
    else:
      sh = getattr(jax.typeof(raw), "sharding", None)
    spec = getattr(sh, "spec", None)
  if spec is None and sharding_hint is not None:
    hint = sharding_hint.get_raw_value() if isinstance(sharding_hint, nnx.Variable) else sharding_hint
    spec = getattr(hint, "spec", hint if isinstance(hint, P) else None)
  if spec is None and isinstance(var_or_arr, nnx.Variable):
    active_map = getattr(_ACTIVE_PARAM_SHARDINGS, "map", None)
    sel_path = var_or_arr.get_metadata(_SELECTIVE_HOST_PATH_KEY, default=None)
    if active_map and sel_path is not None and sel_path in active_map:
      spec = active_map[sel_path]
    if spec is None:
      spec = var_or_arr.get_metadata("out_sharding", default=None)
  if spec is None:
    sh = getattr(raw, "sharding", None)
    if sh is None and isinstance(raw, jax.core.Tracer):  # jmp-waiver: TRACER_GUARD
      sh = getattr(jax.typeof(raw), "sharding", None)
    spec = getattr(sh, "spec", None)
  if spec is None:
    return None, None, (), ()
  d0 = spec[0] if len(spec) >= 1 else None
  d1 = spec[1] if len(spec) >= 2 else None
  r_spec = _spec_dim_for_mesh(d0, mesh)
  c_spec = _spec_dim_for_mesh(d1, mesh)
  row_axes = _normalize_axis_entry(d0, mesh)
  col_axes = _normalize_axis_entry(d1, mesh)
  return r_spec, c_spec, row_axes, col_axes


def _resolve_2d_axes(
    var_or_arr: Any,
    mesh: Any | None,
    sharding_hint: Any = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
  """Returns active mesh `(row_axes, col_axes)` for a 2D table variable or array."""
  _, _, row_axes, col_axes = _resolve_2d_specs_and_axes(var_or_arr, mesh, sharding_hint=sharding_hint)
  return row_axes, col_axes


def _leaf_pspec(x: Any, mesh: Any | None, fallback_2d_spec: P | None = None) -> P:
  """Returns the `shard_map` `PartitionSpec` for an array leaf on `mesh`."""
  if mesh is None or not hasattr(x, "ndim"):
    return P()
  if _is_explicit_mesh(mesh):
    sh = getattr(x, "sharding", None) if isinstance(x, jax.ShapeDtypeStruct) else getattr(jax.typeof(x), "sharding", None)
    spec = getattr(sh, "spec", None)
    if spec is not None:
      return spec
  if fallback_2d_spec is not None and x.ndim == 2 and _is_host_leaf(x):
    return fallback_2d_spec
  return P(*([None] * x.ndim)) if x.ndim > 0 else P()


def _embed_ids_and_out_specs(ids: jax.Array, r_spec: Any, c_spec: Any, mesh: Any) -> tuple[P, P]:
  """Computes `(ids_in_spec, out_spec)` preserving non-conflicting batch/seq sharding on Explicit meshes."""
  if _is_explicit_mesh(mesh):
    ids_spec = _leaf_pspec(ids, mesh)
    tbl_axes = set(_axes_in_spec_dim(r_spec)) | set(_axes_in_spec_dim(c_spec))
    clean_dims = []
    for k in range(ids.ndim):
      entry = ids_spec[k] if k < len(ids_spec) else None
      kept = tuple(ax for ax in _axes_in_spec_dim(entry) if ax in mesh.shape and ax not in tbl_axes)
      clean_dims.append(_pspec_dim(kept))
    return P(*clean_dims), P(*clean_dims, c_spec)
  rep_dims = [None] * ids.ndim
  return P(*rep_dims), P(*rep_dims, c_spec)


def _maybe_reshard(x: jax.Array, spec: P, mesh: Any | None) -> jax.Array:
  if _is_explicit_mesh(mesh):
    amb = jax.sharding.get_abstract_mesh()
    target = spec if (amb is not None and not amb.empty) else jax.sharding.NamedSharding(mesh, spec)
    return jax.sharding.reshard(x, target)
  return x


def _can_move_vars(vars_list: Sequence[nnx.Variable]) -> bool:
  raw_leaves = [v.get_raw_value() for v in vars_list]
  return bool(raw_leaves) and not any(
      isinstance(v, jax.core.Tracer)  # jmp-waiver: TRACER_GUARD
      or (isinstance(v, jax.ShapeDtypeStruct) and getattr(v, "sharding", None) is None)
      for v in raw_leaves
  )


# ---------------------------------------------------------------------------
# Row-sparse host gather, host scatter-add, and SparseHostGrad container
# ---------------------------------------------------------------------------


@_device_host_compute(out_memory_spaces=jax.memory.Space.Device)
def _replicated_host_gather(table_h: jax.Array, ids_h: jax.Array) -> jax.Array:
  return table_h[ids_h]


@_device_host_compute(out_memory_spaces=jax.memory.Space.Host)
def _replicated_host_scatter_add(table_like_h: jax.Array, ids_h: jax.Array, d_out_h: jax.Array) -> jax.Array:
  zeros_h = jax.device_put(jnp.zeros(table_like_h.shape, dtype=table_like_h.dtype), jax.memory.Space.Host)
  return zeros_h.at[ids_h].add(d_out_h.astype(table_like_h.dtype))


@_device_host_compute(out_memory_spaces=jax.memory.Space.Device)
def _local_host_gather(local_table_h: jax.Array, ids_h: jax.Array, start_h: jax.Array) -> jax.Array:
  n_local = local_table_h.shape[0]
  rel = ids_h - start_h
  in_range = (rel >= 0) & (rel < n_local)
  zero_i = jax.device_put(jnp.int32(0), jax.memory.Space.Host)
  safe = jnp.where(in_range, rel, zero_i)
  zero_v = jax.device_put(jnp.zeros((), dtype=local_table_h.dtype), jax.memory.Space.Host)
  return jnp.where(in_range[..., None], local_table_h[safe], zero_v)


@_device_host_compute(out_memory_spaces=jax.memory.Space.Host)
def _local_host_scatter_add(
    local_table_like_h: jax.Array, ids_h: jax.Array, d_out_h: jax.Array, start_h: jax.Array
) -> jax.Array:
  """Scatters `d_out_h` into a zero-initialized local shard on host."""
  n_local = local_table_like_h.shape[0]
  rel = ids_h - start_h
  in_range = (rel >= 0) & (rel < n_local)
  zero_i = jax.device_put(jnp.int32(0), jax.memory.Space.Host)
  safe = jnp.where(in_range, rel, zero_i)
  zero_v = jax.device_put(jnp.zeros((), dtype=local_table_like_h.dtype), jax.memory.Space.Host)
  vals = jnp.where(in_range[..., None], d_out_h.astype(local_table_like_h.dtype), zero_v)
  zeros_h = jax.device_put(jnp.zeros(local_table_like_h.shape, dtype=local_table_like_h.dtype), jax.memory.Space.Host)
  return zeros_h.at[safe].add(vals)


def _host_embed_gather(
    table: jax.Array,
    ids: jax.Array,
    *,
    dtype: Any,
    row_axes: tuple[str, ...],
    col_axes: tuple[str, ...],
) -> jax.Array:
  """Gathers only rows indexed by `ids` from host-resident `table` onto device."""
  mesh = _active_mesh(table, ids)
  if mesh is not None:
    # Always isolate host compute in shard_map so collectives run on device ICI, not MegaScale host.
    if _is_explicit_mesh(mesh):
      r_spec, c_spec, row_axes, _ = _resolve_2d_specs_and_axes(table, mesh)
    else:
      r_spec, c_spec = _pspec_dim(row_axes), _pspec_dim(col_axes)
    ids_in_spec, out_spec = _embed_ids_and_out_specs(ids, r_spec, c_spec, mesh)
    ids_in = _maybe_reshard(ids, ids_in_spec, mesh)

    def _shard_gather(local_table: jax.Array, local_ids: jax.Array) -> jax.Array:
      ids_h = jax.device_put(local_ids, jax.memory.Space.Host)
      if row_axes:
        start = jax.lax.axis_index(row_axes).astype(jnp.int32) * jnp.int32(local_table.shape[0])
        part = _local_host_gather(local_table, ids_h, jax.device_put(start, jax.memory.Space.Host))
        return jax.lax.psum(part, row_axes).astype(dtype)
      return _replicated_host_gather(local_table, ids_h).astype(dtype)

    return _call_shard_map(
        _shard_gather,
        in_specs=(P(r_spec, c_spec), ids_in_spec),
        out_specs=out_spec,
        args=(table, ids_in),
        mesh=mesh,
    )

  gathered = _replicated_host_gather(table, jax.device_put(ids, jax.memory.Space.Host))
  return gathered.astype(dtype)


def _host_embed_scatter_add(
    table_like: jax.Array,
    ids: jax.Array,
    d_out: jax.Array,
    *,
    row_axes: tuple[str, ...],
    col_axes: tuple[str, ...],
) -> jax.Array:
  """Scatters `d_out` into a host-resident gradient matching `table_like`."""
  mesh = _active_mesh(table_like, ids, d_out)
  if mesh is not None:
    if _is_explicit_mesh(mesh):
      r_spec, c_spec, row_axes, _ = _resolve_2d_specs_and_axes(table_like, mesh)
    else:
      r_spec, c_spec = _pspec_dim(row_axes), _pspec_dim(col_axes)
    rep_ids_spec = P(*([None] * ids.ndim))
    out_spec = P(*([None] * ids.ndim), c_spec)
    ids_in = _maybe_reshard(ids, rep_ids_spec, mesh)
    d_out_in = _maybe_reshard(d_out, out_spec, mesh)

    def _shard_scatter(local_tbl: jax.Array, local_ids: jax.Array, local_g: jax.Array) -> jax.Array:
      ids_h = jax.device_put(local_ids, jax.memory.Space.Host)
      g_h = jax.device_put(local_g, jax.memory.Space.Host)
      if row_axes:
        start = jax.lax.axis_index(row_axes).astype(jnp.int32) * jnp.int32(local_tbl.shape[0])
        return _local_host_scatter_add(local_tbl, ids_h, g_h, jax.device_put(start, jax.memory.Space.Host))
      return _replicated_host_scatter_add(local_tbl, ids_h, g_h)

    return _call_shard_map(
        _shard_scatter,
        in_specs=(P(r_spec, c_spec), rep_ids_spec, out_spec),
        out_specs=P(r_spec, c_spec),
        args=(table_like, ids_in, d_out_in),
        mesh=mesh,
    )

  return _replicated_host_scatter_add(
      table_like,
      jax.device_put(ids, jax.memory.Space.Host),
      jax.device_put(d_out, jax.memory.Space.Host),
  )


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4))
def _host_embed_lookup_vjp(
    table: jax.Array,
    ids: jax.Array,
    dtype: Any,
    row_axes: tuple[str, ...],
    col_axes: tuple[str, ...],
) -> jax.Array:
  return _host_embed_gather(table, ids, dtype=dtype, row_axes=row_axes, col_axes=col_axes)


def _host_embed_lookup_fwd(
    table: jax.Array,
    ids: jax.Array,
    dtype: Any,
    row_axes: tuple[str, ...],
    col_axes: tuple[str, ...],
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
  out = _host_embed_gather(table, ids, dtype=dtype, row_axes=row_axes, col_axes=col_axes)
  return out, (table, ids)


def _host_embed_lookup_bwd(
    dtype: Any,
    row_axes: tuple[str, ...],
    col_axes: tuple[str, ...],
    res: tuple[jax.Array, jax.Array],
    g: jax.Array,
) -> tuple[jax.Array, None]:
  """VJP backward scattering device cotangent directly into a host gradient table."""
  del dtype
  table, ids = res
  d_table = _host_embed_scatter_add(table, ids, g, row_axes=row_axes, col_axes=col_axes)
  return (d_table, None)


_host_embed_lookup_vjp.defvjp(_host_embed_lookup_fwd, _host_embed_lookup_bwd)


@jax.tree_util.register_pytree_node_class
class SparseHostGrad:
  """Row-sparse gradient for a host-offloaded embedding table.

  Stores only the accessed row indices `ids` and device cotangent `d_gathered`
  without materializing a dense `(num_embeddings, features)` gradient in HBM.
  """

  _UNSET = object()

  def __init__(
      self,
      ids: jax.Array,
      d_gathered: jax.Array,
      *,
      table_shape: tuple[int, int],
      table_dtype: Any,
      row_axes: tuple[str, ...] = (),
      col_axes: tuple[str, ...] = (),
      grad_dtype: Any = None,
      col_spec: Any = _UNSET,
  ):
    self.ids = ids
    self.d_gathered = d_gathered
    self.table_shape = tuple(int(d) for d in table_shape)
    self.table_dtype = jnp.dtype(table_dtype)
    self.row_axes = tuple(row_axes)
    self.col_axes = tuple(col_axes)
    self.grad_dtype = jnp.dtype(grad_dtype) if grad_dtype is not None else self.table_dtype
    self.col_spec = _pspec_dim(self.col_axes) if col_spec is self._UNSET else col_spec

  def resolved_col_spec(self, mesh: Any | None) -> Any:
    if _is_explicit_mesh(mesh):
      dg_spec = _leaf_pspec(self.d_gathered, mesh)
      if len(dg_spec) == self.d_gathered.ndim and self.d_gathered.ndim >= 1:
        return dg_spec[-1]
      return self.col_spec
    return _pspec_dim(self.col_axes)

  def with_d_gathered(self, new_d_gathered: jax.Array, *, grad_dtype: Any = None) -> "SparseHostGrad":
    return SparseHostGrad(
        self.ids,
        new_d_gathered,
        table_shape=self.table_shape,
        table_dtype=self.table_dtype,
        row_axes=self.row_axes,
        col_axes=self.col_axes,
        grad_dtype=grad_dtype if grad_dtype is not None else self.grad_dtype,
        col_spec=self.col_spec,
    )

  def tree_flatten(self):
    children = (self.ids, self.d_gathered)
    aux = (self.table_shape, self.table_dtype, self.row_axes, self.col_axes, self.grad_dtype, self.col_spec)
    return children, aux

  @classmethod
  def tree_unflatten(cls, aux, children):
    ids, d_gathered = children
    table_shape, table_dtype, row_axes, col_axes, grad_dtype, col_spec = aux
    return cls(
        ids,
        d_gathered,
        table_shape=table_shape,
        table_dtype=table_dtype,
        row_axes=row_axes,
        col_axes=col_axes,
        grad_dtype=grad_dtype,
        col_spec=col_spec,
    )


def materialize_sparse_grad(grad: SparseHostGrad) -> jax.Array:  # public-api
  """Scatters a `SparseHostGrad` into a dense `(num_embeddings, features)` array."""
  flat_ids = grad.ids.reshape(-1)
  flat_dg = grad.d_gathered.reshape(-1, grad.table_shape[1]).astype(grad.grad_dtype)
  return jnp.zeros(grad.table_shape, dtype=grad.grad_dtype).at[flat_ids].add(flat_dg)


@contextlib.contextmanager
def _pregathered_context(mapping: dict[tuple[Any, ...], dict[str, Any]]):
  """Installs thread-local pre-gathered embedding row slots for one forward/backward pass."""
  prev = getattr(_PREGATHERED_STATE, "mapping", None)
  # Copy per-path mutable cursor state so nested/repeated traces start at call index 0.
  state = {}
  for p, entry in mapping.items():
    if isinstance(entry, tuple):
      state[p] = {
          "idx": 0,
          "slots": [entry],
          "check_identity": False,
          "has_custom_ids_fn": False,
          "dense_proxy": None,
      }
    else:
      state[p] = {
          "idx": 0,
          "slots": list(entry["slots"]),
          "check_identity": bool(entry.get("check_identity", False)),
          "has_custom_ids_fn": bool(entry.get("has_custom_ids_fn", False)),
          "dense_proxy": entry.get("dense_proxy"),
      }
  _PREGATHERED_STATE.mapping = state
  try:
    yield
  finally:
    _PREGATHERED_STATE.mapping = prev


@contextlib.contextmanager
def _capture_embed_ids_context(skip_paths: frozenset[tuple[Any, ...]] = frozenset()):
  """Records per-call `HostOffloadedEmbed` input IDs for outer-scope recapture."""
  prev = getattr(_CAPTURE_EMBED_IDS, "recording", None)
  prev_skip = getattr(_CAPTURE_EMBED_IDS, "skip_paths", frozenset())
  recorded: dict[Any, Any] = {}
  _CAPTURE_EMBED_IDS.recording = recorded
  _CAPTURE_EMBED_IDS.skip_paths = skip_paths
  try:
    yield recorded
  finally:
    _CAPTURE_EMBED_IDS.recording = prev
    _CAPTURE_EMBED_IDS.skip_paths = prev_skip


@jax.custom_vjp
def _host_to_device_weight(w_h: jax.Array) -> jax.Array:
  return jax.device_put(w_h, jax.memory.Space.Device)


def _host_to_device_weight_fwd(w_h: jax.Array) -> tuple[jax.Array, None]:
  return jax.device_put(w_h, jax.memory.Space.Device), None


def _host_to_device_weight_bwd(_: None, g_dev: jax.Array) -> tuple[jax.Array]:
  return (jax.device_put(g_dev, jax.memory.Space.Host),)


_host_to_device_weight.defvjp(_host_to_device_weight_fwd, _host_to_device_weight_bwd)


class HostOffloadedParam(_HostMemoryKindVariableMixin, nnx.Param):
  """Path-transparent `nnx.Param` subclass keeping host tables off HBM outside `[...]` reads."""

  __slots__ = ()

  def get_value(self, *, index: Any = nnx.variablelib.MISSING) -> Any:
    """Returns host weight directly on `.get_value()`, staging to HBM only on `[...]` reads."""
    param_path = self.get_metadata(_SELECTIVE_HOST_PATH_KEY, default=None)
    if param_path is None and not bool(self.get_metadata(_SELECTIVE_HOST_STAGE_KEY, default=False)):
      return super().get_value(index=index)
    raw_value = super().get_value()
    if isinstance(raw_value, jax.ShapeDtypeStruct):
      coerced = _coerce_sds_memory_kind(raw_value, self.get_metadata("memory_kind", default=None))
      return coerced if isinstance(index, nnx.variablelib.Missing) or index is ... else coerced[index]
    # Unindexed `.get_value()` calls (`to_pure_dict`, `to_checkpoint_dict`, `tx.init`, `eval_shape`)
    # must never stage `pinned_host` weights onto device HBM.
    if isinstance(index, nnx.variablelib.Missing):
      return raw_value
    capture_rec = getattr(_CAPTURE_EMBED_IDS, "recording", None)
    if capture_rec is not None and param_path is not None:
      capture_rec.setdefault("__dense_paths__", set()).add(param_path)
      staged = jax.device_put(raw_value, jax.memory.Space.Device) if _is_host_leaf(raw_value) else raw_value
      return staged if index is ... else staged[index]
    active_map = getattr(_PREGATHERED_STATE, "mapping", None)
    if param_path is not None and active_map is not None and param_path in active_map:
      entry = active_map[param_path]
      dense_proxy = entry.get("dense_proxy")
      if dense_proxy is not None and getattr(raw_value, "shape", None) == getattr(dense_proxy, "shape", None):
        return dense_proxy if index is ... else dense_proxy[index]
      if dense_proxy is None:
        if entry["check_identity"]:
          raise _RecaptureEmbedIds()
        raise ValueError(f"HostOffloadedEmbed at {param_path} accessed densely via `[...]` without a dense device proxy.")
    staged = _host_to_device_weight(raw_value) if _is_host_leaf(raw_value) else raw_value
    return staged if index is ... else staged[index]


def _to_host_offloaded_param(var: nnx.Param) -> HostOffloadedParam:
  """Constructs a selective `HostOffloadedParam` from `var` without in-place type mutation."""
  meta = dict(var.get_metadata())
  meta.pop("on_get_value", None)
  meta.pop("on_set_value", None)
  meta[_SELECTIVE_HOST_STAGE_KEY] = True
  if isinstance(var, HostOffloadedParam):
    object.__setattr__(var, "_var_metadata", meta)
    return var
  return HostOffloadedParam.from_metadata(var.get_raw_value(), meta)


class HostOffloadedEmbed(nnx.Embed):
  """Path-transparent `nnx.Embed` subclass performing row-sparse host gathers."""

  def __init__(self, embed: nnx.Embed, *, scope: str | None = None):  # pylint: disable=super-init-not-called
    self.num_embeddings = embed.num_embeddings
    self.features = embed.features
    self.dtype = embed.dtype
    self.param_dtype = embed.param_dtype
    self.promote_dtype = embed.promote_dtype
    emb_var = embed.embedding
    if isinstance(emb_var, nnx.Param) and (
        emb_var.get_metadata(_SELECTIVE_HOST_PATH_KEY, default=None) is not None or _is_host_leaf(emb_var)
    ):
      emb_var = _to_host_offloaded_param(emb_var)
      embed.embedding = emb_var
    self.embedding = emb_var
    self._m3_embed_scope = scope

  @classmethod
  def from_embed(cls, embed: nnx.Embed, *, scope: str | None = None) -> "HostOffloadedEmbed":
    return cls(embed, scope=scope)

  def __call__(self, inputs: jax.Array, out_sharding: Any = None) -> jax.Array:
    if not jnp.issubdtype(inputs.dtype, jnp.integer):
      raise ValueError("Input type must be an integer or unsigned integer.")
    scope = getattr(self, "_m3_embed_scope", None)
    scope_ctx = jax.named_scope(scope) if scope else contextlib.nullcontext()
    with scope_ctx:
      param_path = self.embedding.get_metadata(_SELECTIVE_HOST_PATH_KEY, default=None)
      capture_rec = getattr(_CAPTURE_EMBED_IDS, "recording", None)
      if capture_rec is not None and param_path is not None:
        skip_paths = getattr(_CAPTURE_EMBED_IDS, "skip_paths", frozenset())
        if param_path not in skip_paths:
          capture_rec.setdefault(param_path, []).append(inputs)
        table = self.embedding.get_raw_value()
        target_dtype = jnp.dtype(self.dtype) if self.dtype is not None else table.dtype
        mesh = _active_mesh(self.embedding, table, inputs)
        out_shape = (*inputs.shape, self.features)
        if _is_explicit_mesh(mesh):
          r_spec, c_spec, _, _ = _resolve_2d_specs_and_axes(self.embedding, mesh)
          _, embed_out_spec = _embed_ids_and_out_specs(inputs, r_spec, c_spec, mesh)
          amb = jax.sharding.get_abstract_mesh()
          sh_arg = (
              embed_out_spec if (amb is not None and not amb.empty) else jax.sharding.NamedSharding(mesh, embed_out_spec)
          )
          out = jnp.zeros(out_shape, dtype=target_dtype, out_sharding=sh_arg)
        else:
          out = jnp.zeros(out_shape, dtype=target_dtype)
        if out_sharding is not None:
          out = jax.lax.with_sharding_constraint(out, out_sharding)
        return out
      active_map = getattr(_PREGATHERED_STATE, "mapping", None)
      if param_path is not None and active_map is not None and param_path in active_map:
        entry = active_map[param_path]
        idx = entry["idx"]
        slots = entry["slots"]
        if idx >= len(slots):
          if entry["check_identity"]:
            raise _RecaptureEmbedIds()
          raise ValueError(
              f"HostOffloadedEmbed at {param_path} received more calls ({idx + 1}) than pre-gathered slots ({len(slots)})."
          )
        pre_ids, pre_gathered = slots[idx]
        if entry["check_identity"] and not entry["has_custom_ids_fn"] and inputs is not pre_ids:
          raise _RecaptureEmbedIds()
        if inputs.shape != pre_ids.shape:
          if entry["check_identity"]:
            raise _RecaptureEmbedIds()
          raise ValueError(
              f"HostOffloadedEmbed pre-gather shape mismatch for {param_path}: "
              f"received inputs.shape={inputs.shape}, expected {pre_ids.shape}."
          )
        entry["idx"] = idx + 1
        out = pre_gathered
        if out_sharding is not None:
          out = jax.lax.with_sharding_constraint(out, out_sharding)
        return out
      if _is_host_leaf(self.embedding):
        table = self.embedding.get_raw_value()
        target_dtype = jnp.dtype(self.dtype) if self.dtype is not None else table.dtype
        mesh = _active_mesh(self.embedding, table, inputs)
        row_axes, col_axes = _resolve_2d_axes(self.embedding, mesh)
        out = _host_embed_lookup_vjp(table, inputs, target_dtype, row_axes, col_axes)
        if out_sharding is not None:
          out = jax.lax.with_sharding_constraint(out, out_sharding)
        return out
      return super().__call__(inputs, out_sharding=out_sharding)


class _HostOffloadedParametersMeta(type):
  """Metaclass recognizing single-inheritance parameter-offloaded model subclasses."""

  def __instancecheck__(cls, instance: Any) -> bool:
    return bool(getattr(instance, "_is_host_offloaded_params", False))

  def __subclasscheck__(cls, subclass: type) -> bool:
    return bool(getattr(subclass, "_is_host_offloaded_params", False))


class HostOffloadedParameters(metaclass=_HostOffloadedParametersMeta):
  """Marker type for models whose `nnx.Param`s are parked in `pinned_host`."""


def _offloaded_model_class(cls: type[nnx.Module]) -> type[nnx.Module]:
  """Returns a cached path-transparent single-inheritance subclass of `cls`."""
  if getattr(cls, "_is_host_offloaded_params", False):
    return cls
  subcls = _OFFLOADED_MODEL_CLASSES.get(cls)
  if subcls is None:
    parent_call = cls.__call__
    parent_vocab_tiling = getattr(cls, "logits_from_hidden_states_for_vocab_tiling", None)

    def _with_device_params(self: Any, fn: Any, *args: Any, **kwargs: Any) -> Any:
      # Never pull selectively offloaded `HostOffloadedEmbed` tables into HBM here.
      embed_var_ids = {
          id(node.embedding)
          for _, node in nnx.iter_graph(self)
          if isinstance(node, HostOffloadedEmbed) and isinstance(node.embedding, nnx.Variable)
      }
      param_vars = [
          v
          for v in jax.tree.leaves(nnx.state(self, nnx.Param), is_leaf=lambda x: isinstance(x, nnx.Variable))
          if isinstance(v, nnx.Variable) and id(v) not in embed_var_ids and _is_host_leaf(v)
      ]
      if not param_vars:
        return fn(self, *args, **kwargs)
      saved_raw = [v.get_raw_value() for v in param_vars]
      saved_kinds = [v.get_metadata("memory_kind", default=None) for v in param_vars]
      for v in param_vars:
        v.set_metadata("memory_kind", "device")
        v.set_raw_value(_move_raw_value(v.get_raw_value(), "device", jax.memory.Space.Device))
      try:
        return fn(self, *args, **kwargs)
      finally:
        for v, raw, kind in zip(param_vars, saved_raw, saved_kinds):
          v.set_metadata("memory_kind", kind)
          v.set_raw_value(raw)

    def _call(self: Any, *args: Any, **kwargs: Any) -> Any:
      return _with_device_params(self, parent_call, *args, **kwargs)

    def _vocab_tiling(self: Any, *args: Any, **kwargs: Any) -> Any:
      return _with_device_params(self, parent_vocab_tiling, *args, **kwargs)

    namespace: dict[str, Any] = {
        "__slots__": (),
        "__module__": cls.__module__,
        "__qualname__": cls.__qualname__,
        "_is_host_offloaded_params": True,
        "__call__": _call,
    }
    if parent_vocab_tiling is not None:
      namespace["logits_from_hidden_states_for_vocab_tiling"] = _vocab_tiling

    subcls = type(cls)(
        cls.__name__,
        (cls,),
        namespace,
    )
    _OFFLOADED_MODEL_CLASSES[cls] = subcls
  return subcls


def _replace_object_references(root: nnx.Module, replacements: dict[int, Any]) -> None:
  """Replaces references matching `id(obj) in replacements` across `root`'s NNX graph."""
  if not replacements:
    return
  for _, node in list(nnx.iter_graph(root)):
    if isinstance(node, (list, nnx.List)):
      for idx, item in enumerate(node):
        if id(item) in replacements:
          node[idx] = replacements[id(item)]
    elif isinstance(node, (dict, nnx.Dict)):
      for key, item in list(node.items()):
        if id(item) in replacements:
          node[key] = replacements[id(item)]
    elif hasattr(node, "__dict__"):
      for key, item in list(vars(node).items()):
        if key.startswith("_pytree") or key.startswith("_object__"):
          continue
        if id(item) in replacements:
          setattr(node, key, replacements[id(item)])
        elif isinstance(item, tuple) and any(id(elem) in replacements for elem in item):
          setattr(node, key, tuple(replacements.get(id(elem), elem) for elem in item))


def _apply_selective_parameter_offload(model: nnx.Module, config: Any) -> tuple[nnx.Module, bool]:
  """Offloads pattern-matched parameters to `pinned_host` and wraps matched `nnx.Embed` modules."""
  patterns = get_host_offload_param_patterns(config)
  if not patterns:
    raise ValueError("selective_host_offload=True requires non-empty host_offload_param_patterns.")
  params = nnx.state(model, nnx.Param)
  selected_paths = _match_param_paths(params, patterns)
  if not selected_paths:
    raise ValueError(
        f"selective_host_offload=True matched 0 parameters for host_offload_param_patterns={list(patterns)!r}."
    )

  selected_pairs: list[tuple[tuple[Any, ...], nnx.Variable]] = []
  for path, var in nnx.to_flat_state(params):
    path_tuple = tuple(path)
    if path_tuple in selected_paths and isinstance(var, nnx.Variable):
      raw = var.get_raw_value()
      if getattr(raw, "ndim", None) != 2:
        joined = "/".join(str(k) for k in path_tuple)
        raise ValueError(
            f"selective_host_offload only supports 2D weight/embedding matrices, but matched parameter "
            f"{joined!r} has shape {getattr(raw, 'shape', None)} (ndim={getattr(raw, 'ndim', None)})."
        )
      selected_pairs.append((path_tuple, var))
  param_replacements: dict[int, nnx.Variable] = {}
  selected_vars: list[nnx.Variable] = []
  seen_selected_ids: set[int] = set()
  for path_tuple, var in selected_pairs:
    if id(var) not in param_replacements:
      if isinstance(var, nnx.Param):
        new_var: nnx.Variable = _to_host_offloaded_param(var)
        new_var.set_metadata(_SELECTIVE_HOST_PATH_KEY, path_tuple)
      else:
        new_var = _to_memory_kind_variable(var, "pinned_host", extra_metadata={_SELECTIVE_HOST_PATH_KEY: path_tuple})
      param_replacements[id(var)] = new_var
    target_var = param_replacements[id(var)]
    if id(target_var) not in seen_selected_ids:
      seen_selected_ids.add(id(target_var))
      selected_vars.append(target_var)

  if _can_move_vars(selected_vars):
    _move_variables_in_place(selected_vars, "pinned_host")
  else:
    for v in selected_vars:
      v.set_metadata("memory_kind", "pinned_host")
      raw = v.get_raw_value()
      if isinstance(raw, jax.core.Tracer) and not _is_host_leaf(raw):  # jmp-waiver: TRACER_GUARD
        v.set_raw_value(jax.device_put(raw, jax.memory.Space.Host))
      else:
        v.set_raw_value(_coerce_sds_memory_kind(raw, "pinned_host"))

  _replace_object_references(model, param_replacements)

  # Replace matching `nnx.Embed` modules path-transparently with `HostOffloadedEmbed` across all
  # parent references (preserving shared tied-embedding module identity).
  embed_replacements: dict[int, HostOffloadedEmbed] = {}
  covered_param_ids: set[int] = set()
  embed_paths_covered: set[tuple[Any, ...]] = set()
  for mod_path, node in list(nnx.iter_graph(model)):
    if not mod_path or not isinstance(node, nnx.Embed):
      continue
    emb_param_path = (*mod_path, "embedding")
    emb_obj = getattr(node, "embedding", None)
    if emb_param_path not in selected_paths and id(emb_obj) not in seen_selected_ids:
      continue
    embed_paths_covered.add(emb_param_path)
    if emb_obj is not None:
      covered_param_ids.add(id(emb_obj))
    if not isinstance(node, HostOffloadedEmbed) and id(node) not in embed_replacements:
      scope = named_scopes.scope_of(node)
      wrapped_embed = HostOffloadedEmbed.from_embed(node, scope=scope)
      covered_param_ids.add(id(wrapped_embed.embedding))
      embed_replacements[id(node)] = wrapped_embed

  _replace_object_references(model, embed_replacements)

  for path_tuple, var in selected_pairs:
    mapped = param_replacements.get(id(var), var)
    if id(mapped) in covered_param_ids:
      embed_paths_covered.add(path_tuple)

  non_embed_selected = selected_paths - embed_paths_covered
  if non_embed_selected:
    bad_joined = ", ".join("/".join(str(k) for k in p) for p in sorted(non_embed_selected))
    raise ValueError(
        "selective_host_offload only supports 2D nnx.Embed embedding tables, but host_offload_param_patterns "
        f"matched non-embedding parameter(s): {bad_joined}."
    )
  return model, False


def maybe_offload_parameters(model: nnx.Module, config: Any) -> nnx.Module:
  """Moves all or pattern-selected `nnx.Param` state to `pinned_host` per `config`."""
  full_offload = bool(getattr(config, "parameter_memory_host_offload", False))  # jmp-waiver: SILENT_CONFIG_FALLBACK
  selective_offload = is_selective_host_offload_enabled(config)
  if full_offload and selective_offload:
    raise ValueError("selective_host_offload=True cannot be combined with parameter_memory_host_offload=True.")
  if selective_offload:
    model, needs_full_wrap = _apply_selective_parameter_offload(model, config)
    if not needs_full_wrap:
      return model
  elif not full_offload:
    return model
  else:
    params = nnx.state(model, nnx.Param)
    param_leaves = [
        v for v in jax.tree.leaves(params, is_leaf=lambda x: isinstance(x, nnx.Variable)) if isinstance(v, nnx.Variable)
    ]
    should_move = _can_move_vars(param_leaves)
    param_replacements: dict[int, nnx.Variable] = {}
    for v in param_leaves:
      if id(v) in param_replacements:
        continue
      raw = v.get_raw_value()
      if should_move:
        new_raw = _move_raw_value(raw, "pinned_host", jax.memory.Space.Host)
      elif isinstance(raw, jax.core.Tracer) and not _is_host_leaf(v):  # jmp-waiver: TRACER_GUARD
        new_raw = jax.device_put(raw, jax.memory.Space.Host)
      else:
        new_raw = _coerce_sds_memory_kind(raw, "pinned_host")
      param_replacements[id(v)] = _to_memory_kind_variable(v, "pinned_host", raw_value=new_raw)
    _replace_object_references(model, param_replacements)
  if not isinstance(model, HostOffloadedParameters):
    subcls = _offloaded_model_class(type(model))
    clone = object.__new__(subcls)
    for key, val in vars(model).items():
      object.__setattr__(clone, key, val)
    return clone
  return model


def _build_opt_state_out_shardings(params_state: Any, tx: optax.GradientTransformation, kind: str) -> Any:
  """Derives `out_shardings` in `kind` matching each `tx.init` leaf's source parameter sharding."""
  sharding_table: list[tuple[Any, int]] = []
  first_sharding = None

  def _tag_param(x: Any) -> Any:
    nonlocal first_sharding
    if not isinstance(x, nnx.Variable):
      return x
    raw = x.get_raw_value()
    sharding = getattr(raw, "sharding", None)
    if first_sharding is None and sharding is not None:
      first_sharding = sharding
    idx = len(sharding_table)
    sharding_table.append((sharding, getattr(raw, "ndim", 0)))
    sds_sharding = (
        sharding.with_memory_kind("device")
        if sharding is not None and hasattr(sharding, "with_memory_kind")
        else sharding
    )
    tagged = x.replace(jax.ShapeDtypeStruct(raw.shape, raw.dtype, sharding=sds_sharding))
    tagged.set_metadata("_offload_sharding_idx", idx)
    return tagged

  tagged_params = jax.tree.map(_tag_param, params_state, is_leaf=lambda x: isinstance(x, nnx.Variable))
  if isinstance(first_sharding, jax.sharding.NamedSharding):
    replicated: Any = jax.sharding.NamedSharding(first_sharding.mesh, jax.sharding.PartitionSpec(), memory_kind=kind)
  elif first_sharding is not None and hasattr(first_sharding, "with_memory_kind"):
    replicated = first_sharding.with_memory_kind(kind)
  else:
    replicated = _SPACES[kind]

  abs_opt = jax.eval_shape(tx.init, tagged_params)

  def _out_sharding(leaf: Any) -> Any:
    if isinstance(leaf, nnx.Variable):
      sds = leaf.get_raw_value()
      if not hasattr(sds, "shape"):
        return None
      idx = leaf.get_metadata("_offload_sharding_idx", default=None)
      if idx is not None:
        src_sharding, src_ndim = sharding_table[idx]
        if src_sharding is not None and hasattr(src_sharding, "with_memory_kind") and src_ndim == sds.ndim:
          return src_sharding.with_memory_kind(kind)
      return replicated
    if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
      return replicated
    return None

  return jax.tree.map(_out_sharding, abs_opt, is_leaf=lambda x: isinstance(x, nnx.Variable))


def _init_opt_state_for_kind(params_state: Any, tx: optax.GradientTransformation, kind: str) -> Any:
  """Initializes `tx.init(params_state)` directly into `kind` without staging `params + mu + nu` in HBM."""
  out_shardings = _build_opt_state_out_shardings(params_state, tx, kind)
  return jax.jit(tx.init, out_shardings=out_shardings)(params_state)


def has_host_leaves(tree: Any) -> bool:
  """Returns True if any array or variable leaf in `tree` resides in host memory."""
  return _has_host_leaves(tree)


def apply_selective_memory_kind_to_shardings(abs_var_state: Any, sharding_state: Any) -> Any:
  """Propagates `pinned_host` metadata from traced abstract vars onto `NamedSharding` trees."""

  def _is_leaf(x: Any) -> bool:
    return isinstance(x, (nnx.Variable, jax.sharding.Sharding, jax.ShapeDtypeStruct))

  def _apply(a_leaf: Any, s_leaf: Any) -> Any:
    if s_leaf is None:
      return s_leaf
    kind = a_leaf.get_metadata("memory_kind", default=None) if isinstance(a_leaf, nnx.Variable) else None
    if kind not in _HOST_KINDS:
      raw = a_leaf.get_raw_value() if isinstance(a_leaf, nnx.Variable) else a_leaf
      sh = getattr(raw, "sharding", None)
      if getattr(sh, "memory_kind", None) in _HOST_KINDS:
        kind = sh.memory_kind
    if kind not in _HOST_KINDS:
      return s_leaf

    def _to_host_sharding(sh: Any) -> Any:
      if isinstance(sh, jax.sharding.NamedSharding) and sh.memory_kind != kind:
        return sh.with_memory_kind(kind)
      if isinstance(sh, jax.ShapeDtypeStruct):
        inner = getattr(sh, "sharding", None)
        if isinstance(inner, jax.sharding.NamedSharding) and inner.memory_kind != kind:
          return jax.ShapeDtypeStruct(sh.shape, sh.dtype, sharding=inner.with_memory_kind(kind))
      return sh

    def _is_sh_leaf(x: Any) -> bool:
      return isinstance(x, (jax.sharding.Sharding, jax.ShapeDtypeStruct))

    if isinstance(s_leaf, nnx.Variable):
      updated = jax.tree.map(_to_host_sharding, s_leaf.get_raw_value(), is_leaf=_is_sh_leaf)
      return s_leaf.replace(updated)
    return jax.tree.map(_to_host_sharding, s_leaf, is_leaf=_is_sh_leaf)

  return jax.tree.map(_apply, abs_var_state, sharding_state, is_leaf=_is_leaf)


# ---------------------------------------------------------------------------
# Sparse value_and_grad (row-sparse backward with zero dense HBM gradient)
# ---------------------------------------------------------------------------


def _collect_host_embeds(model: nnx.Module) -> list[tuple[tuple[Any, ...], HostOffloadedEmbed, Any]]:
  """Returns `(param_path, embed_module, parent_module)` for each `HostOffloadedEmbed`."""
  nodes = dict(nnx.iter_graph(model))
  specs: list[tuple[tuple[Any, ...], HostOffloadedEmbed, Any]] = []
  for mod_path, node in nodes.items():
    if isinstance(node, HostOffloadedEmbed) and isinstance(node.embedding, nnx.Variable):
      param_path = node.embedding.get_metadata(_SELECTIVE_HOST_PATH_KEY, default=None)
      if param_path is None:
        param_path = (*mod_path, "embedding")
      parent = nodes.get(mod_path[:-1]) if mod_path else None
      specs.append((tuple(param_path), node, parent))
  return specs


def _extract_tokens_and_segments(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    input_tokens: jax.Array | None,
    segment_ids: jax.Array | None,
) -> tuple[jax.Array | None, jax.Array | None]:
  """Extracts token and optional segment ID arrays from training-step batch arguments."""
  if input_tokens is not None:
    return input_tokens, segment_ids
  for candidate in (*args, *kwargs.values()):
    if isinstance(candidate, dict):
      toks = candidate.get("inputs")
      if toks is None:
        toks = candidate.get("input_tokens")
      if toks is None:
        toks = candidate.get("decoder_input_tokens")
      if toks is not None:
        segs = segment_ids
        if segs is None:
          segs = candidate.get("inputs_segmentation")
        if segs is None:
          segs = candidate.get("decoder_segment_ids")
        if segs is None:
          segs = candidate.get("segment_ids")
        return toks, segs
  return None, segment_ids


def _prepare_host_embed_gathers(
    model: nnx.Module,
    input_tokens: jax.Array,
    segment_ids: jax.Array | None,
    *,
    recorded_ids_map: dict[tuple[Any, ...], list[jax.Array]] | None = None,
) -> tuple[dict[tuple[Any, ...], tuple[jax.Array, ...]], dict[tuple[Any, ...], dict[str, Any]]]:
  """Pre-gathers rows for every `HostOffloadedEmbed` call in `model`."""
  gathered_map: dict[tuple[Any, ...], tuple[jax.Array, ...]] = {}
  meta_map: dict[tuple[Any, ...], dict[str, Any]] = {}
  for param_path, embed_mod, parent_mod in _collect_host_embeds(model):
    ids_fn = getattr(parent_mod, "embed_ids_from_tokens", None) if parent_mod is not None else None
    has_custom_ids_fn = callable(ids_fn)
    if has_custom_ids_fn:
      ids_list = [ids_fn(input_tokens, segment_ids=segment_ids)]  # pylint: disable=not-callable
    elif recorded_ids_map is not None:
      ids_list = list(recorded_ids_map.get(param_path, ()))
    else:
      ids_list = [input_tokens]
    table = embed_mod.embedding.get_raw_value()
    target_dtype = jnp.dtype(embed_mod.dtype) if embed_mod.dtype is not None else table.dtype
    mesh = _active_mesh(embed_mod.embedding, table, input_tokens)
    _, c_spec, row_axes, col_axes = _resolve_2d_specs_and_axes(embed_mod.embedding, mesh)
    gathered_tuple = tuple(
        _host_embed_gather(table, call_ids, dtype=target_dtype, row_axes=row_axes, col_axes=col_axes)
        for call_ids in ids_list
    )
    gathered_map[param_path] = gathered_tuple
    meta_map[param_path] = {
        "ids_list": ids_list,
        "has_custom_ids_fn": has_custom_ids_fn,
        "table_shape": table.shape,
        "table_dtype": table.dtype,
        "target_dtype": target_dtype,
        "row_axes": row_axes,
        "col_axes": col_axes,
        "col_spec": c_spec,
        "mesh": mesh,
    }
  return gathered_map, meta_map


def _combine_call_ids_and_cotangents(
    ids_list: list[jax.Array],
    dg_tuple: tuple[jax.Array, ...],
    *,
    features: int,
    target_dtype: Any,
    col_spec: Any,
    mesh: Any | None,
) -> tuple[jax.Array, jax.Array]:
  """Combines `(ids, d_gathered)` across 0, 1, or multiple calls to the same `HostOffloadedEmbed`."""
  if not ids_list:
    ids_zero = jnp.zeros((1,), dtype=jnp.int32)
    if _is_explicit_mesh(mesh):
      amb = jax.sharding.get_abstract_mesh()
      dg_spec = P(None, col_spec)
      sh_arg = dg_spec if (amb is not None and not amb.empty) else jax.sharding.NamedSharding(mesh, dg_spec)
      dg_zero = jnp.zeros((1, features), dtype=target_dtype, out_sharding=sh_arg)
    else:
      dg_zero = jnp.zeros((1, features), dtype=target_dtype)
    return ids_zero, dg_zero
  if len(ids_list) == 1:
    return ids_list[0], dg_tuple[0]
  b0 = ids_list[0].shape[0] if ids_list[0].ndim >= 1 else None
  if b0 is not None and all(x.ndim >= 1 and x.shape[0] == b0 for x in ids_list):
    ids_cat = jnp.concatenate([x.reshape(b0, -1) for x in ids_list], axis=1)
    dg_cat = jnp.concatenate([g.reshape(b0, -1, features) for g in dg_tuple], axis=1)
    return ids_cat, dg_cat
  flat_ids = [_maybe_reshard(x, P(*([None] * x.ndim)), mesh).reshape(-1) for x in ids_list]
  flat_dgs = [_maybe_reshard(g, P(*([None] * (g.ndim - 1)), col_spec), mesh).reshape(-1, features) for g in dg_tuple]
  return jnp.concatenate(flat_ids, axis=0), jnp.concatenate(flat_dgs, axis=0)


def _build_pregather_ctx_map(
    g_map: dict[tuple[Any, ...], tuple[jax.Array, ...]],
    meta_map: dict[tuple[Any, ...], dict[str, Any]],
    *,
    check_identity: bool,
    dense_proxies: dict[tuple[Any, ...], jax.Array] | None = None,
) -> dict[tuple[Any, ...], dict[str, Any]]:
  return {
      p: {
          "slots": list(zip(meta_map[p]["ids_list"], g_map[p])),
          "check_identity": check_identity,
          "has_custom_ids_fn": meta_map[p]["has_custom_ids_fn"],
          "dense_proxy": dense_proxies.get(p) if dense_proxies else None,
      }
      for p in g_map
  }


def _make_dense_proxies(
    host_param_map: dict[tuple[Any, ...], Any], dense_paths: frozenset[tuple[Any, ...]]
) -> dict[tuple[Any, ...], jax.Array]:
  proxies: dict[tuple[Any, ...], jax.Array] = {}
  for p in sorted(dense_paths):
    leaf = host_param_map[p]
    raw = leaf.get_raw_value() if isinstance(leaf, nnx.Variable) else leaf
    proxies[p] = jax.device_put(raw, jax.memory.Space.Device) if _is_host_leaf(raw) else raw
  return proxies


def _partition_state_by_paths(
    state: nnx.State, excluded_paths: frozenset[tuple[Any, ...]]
) -> tuple[nnx.State, nnx.State]:
  """Splits `state` into `(kept_state, excluded_state)` by canonical leaf path."""
  kept_items = []
  excl_items = []
  for path, val in nnx.to_flat_state(state):
    if tuple(path) in excluded_paths:
      excl_items.append((path, val))
    else:
      kept_items.append((path, val))
  return nnx.from_flat_state(kept_items), nnx.from_flat_state(excl_items)


def _assemble_grads_with_sparse(
    dev_grads: nnx.State,
    d_gathered_map: dict[tuple[Any, ...], tuple[jax.Array, ...]],
    meta_map: dict[tuple[Any, ...], dict[str, Any]],
    *,
    d_dense_map: dict[tuple[Any, ...], jax.Array] | None = None,
    host_param_map: dict[tuple[Any, ...], Any] | None = None,
    grad_dtype: Any = None,
) -> nnx.State:
  """Merges device gradients with `SparseHostGrad` (or dense host tied-embed) leaves."""
  items = list(nnx.to_flat_state(dev_grads))
  for param_path, dg_tuple in d_gathered_map.items():
    meta = meta_map[param_path]
    t_dtype = meta["table_dtype"]
    eff_grad_dtype = (
        jnp.dtype(grad_dtype) if grad_dtype is not None and jnp.dtype(t_dtype) == jnp.float32 else jnp.dtype(t_dtype)
    )
    if d_dense_map and param_path in d_dense_map:
      p_leaf = host_param_map[param_path] if host_param_map is not None else None
      d_dense_h = jax.device_put(d_dense_map[param_path].astype(eff_grad_dtype), jax.memory.Space.Host)
      if meta["ids_list"]:
        ids_comb, dg_comb = _combine_call_ids_and_cotangents(
            meta["ids_list"],
            dg_tuple,
            features=meta["table_shape"][1],
            target_dtype=meta["target_dtype"],
            col_spec=meta["col_spec"],
            mesh=meta["mesh"],
        )
        d_embed_h = _host_embed_scatter_add(
            d_dense_h, ids_comb, dg_comb, row_axes=meta["row_axes"], col_axes=meta["col_axes"]
        )
        d_total_h = _apply_host_elementwise(
            _host_add_arrays, p_leaf if p_leaf is not None else d_dense_h, (d_embed_h, d_dense_h)
        )
      else:
        d_total_h = d_dense_h
      grad_leaf = p_leaf.replace(d_total_h) if isinstance(p_leaf, nnx.Variable) else d_total_h
      items.append((param_path, grad_leaf))
      continue

    ids_comb, dg_comb = _combine_call_ids_and_cotangents(
        meta["ids_list"],
        dg_tuple,
        features=meta["table_shape"][1],
        target_dtype=meta["target_dtype"],
        col_spec=meta["col_spec"],
        mesh=meta["mesh"],
    )
    items.append(
        (
            param_path,
            SparseHostGrad(
                ids_comb,
                dg_comb,
                table_shape=meta["table_shape"],
                table_dtype=t_dtype,
                row_axes=meta["row_axes"],
                col_axes=meta["col_axes"],
                grad_dtype=eff_grad_dtype,
                col_spec=meta["col_spec"],
            ),
        )
    )
  return nnx.from_flat_state(items)


def value_and_grad(
    loss_fn: Callable[..., Any],
    model: nnx.Module,
    *args: Any,
    has_aux: bool = False,
    input_tokens: jax.Array | None = None,
    segment_ids: jax.Array | None = None,
    **kwargs: Any,
) -> tuple[Any, Any]:
  """Computes `(value, grads)` with row-sparse gradients for `HostOffloadedEmbed` tables."""
  if not has_selective_host_offload(model) or not _collect_host_embeds(model):
    return nnx.value_and_grad(loss_fn, has_aux=has_aux)(model, *args, **kwargs)
  tokens, segs = _extract_tokens_and_segments(args, kwargs, input_tokens, segment_ids)
  if tokens is None:
    raise ValueError(
        "m3_offload.value_and_grad requires token IDs (via `input_tokens` or a batch dict with "
        "'inputs'/'input_tokens') when HostOffloadedEmbed modules are present; refusing dense HBM fallback."
    )

  gathered_map, meta_map = _prepare_host_embed_gathers(model, tokens, segs)
  if not gathered_map:
    raise ValueError(
        "m3_offload.value_and_grad found HostOffloadedEmbed modules but pre-gather produced no active gathers."
    )

  host_embed_paths = frozenset(gathered_map.keys())
  graphdef, params, rest = nnx.split(model, nnx.Param, ...)
  dev_params, host_params = _partition_state_by_paths(params, host_embed_paths)
  host_param_map = {tuple(p): v for p, v in nnx.to_flat_state(host_params)}

  def _run_grad(
      cur_gathered: dict[tuple[Any, ...], tuple[jax.Array, ...]],
      cur_meta: dict[tuple[Any, ...], dict[str, Any]],
      *,
      check_identity: bool,
      dense_paths: frozenset[tuple[Any, ...]] = frozenset(),
  ):
    dense_proxies = _make_dense_proxies(host_param_map, dense_paths) if dense_paths else {}

    def _diff_fn(
        dev_p: nnx.State,
        g_map: dict[tuple[Any, ...], tuple[jax.Array, ...]],
        d_proxies: dict[tuple[Any, ...], jax.Array],
    ) -> Any:
      full_p = nnx.merge_state(dev_p, host_params) if jax.tree.leaves(host_params) else dev_p
      local_model = nnx.merge(graphdef, full_p, rest, copy=True)
      ctx_map = _build_pregather_ctx_map(g_map, cur_meta, check_identity=check_identity, dense_proxies=d_proxies)
      with _pregathered_context(ctx_map):
        return loss_fn(local_model, *args, **kwargs)

    return jax.value_and_grad(_diff_fn, argnums=(0, 1, 2), has_aux=has_aux)(dev_params, cur_gathered, dense_proxies)

  try:
    val_out, (dev_grads, d_gathered_map, d_dense_map) = _run_grad(gathered_map, meta_map, check_identity=True)
  except _RecaptureEmbedIds:
    custom_skip = frozenset(p for p, m in meta_map.items() if m["has_custom_ids_fn"])
    with _capture_embed_ids_context(skip_paths=custom_skip) as recorded_ids:
      capture_model = nnx.merge(graphdef, params, rest, copy=True)
      loss_fn(capture_model, *args, **kwargs)
    dense_paths = frozenset(recorded_ids.pop("__dense_paths__", set()))
    gathered_map, meta_map = _prepare_host_embed_gathers(model, tokens, segs, recorded_ids_map=recorded_ids)
    val_out, (dev_grads, d_gathered_map, d_dense_map) = _run_grad(
        gathered_map, meta_map, check_identity=False, dense_paths=dense_paths
    )
  grads = _assemble_grads_with_sparse(
      dev_grads, d_gathered_map, meta_map, d_dense_map=d_dense_map, host_param_map=host_param_map
  )
  return val_out, grads


def pure_value_and_grad(
    diff_wrapper: Callable[..., Any],
    model_graphdef: Any,
    curr_params: nnx.State,
    *,
    custom_params: nnx.State,
    rest: nnx.State,
    config: Any,
    data: dict[str, Any],
    has_aux: bool = True,
) -> tuple[Any, tuple[nnx.State, nnx.State]]:
  """Pure-state `value_and_grad` for `train.py` with row-sparse host-embed gradients."""
  probe_model = nnx.merge(model_graphdef, curr_params, custom_params, rest, copy=True)
  if not _collect_host_embeds(probe_model):
    return jax.value_and_grad(diff_wrapper, argnums=(0, 1), has_aux=has_aux)(
        curr_params, custom_params, rest, config, data
    )
  tokens, segs = _extract_tokens_and_segments((data,), {}, None, None)
  if tokens is None:
    raise ValueError(
        "m3_offload.pure_value_and_grad requires token IDs in `data` ('inputs'/'input_tokens') "
        "when HostOffloadedEmbed modules are present; refusing dense HBM fallback."
    )
  gathered_map, meta_map = _prepare_host_embed_gathers(probe_model, tokens, segs)
  if not gathered_map:
    raise ValueError(
        "m3_offload.pure_value_and_grad found HostOffloadedEmbed modules but pre-gather produced no active gathers."
    )

  host_embed_paths = frozenset(gathered_map.keys())
  dev_params, host_params = _partition_state_by_paths(curr_params, host_embed_paths)
  host_param_map = {tuple(p): v for p, v in nnx.to_flat_state(host_params)}

  def _run_grad(
      cur_gathered: dict[tuple[Any, ...], tuple[jax.Array, ...]],
      cur_meta: dict[tuple[Any, ...], dict[str, Any]],
      *,
      check_identity: bool,
      dense_paths: frozenset[tuple[Any, ...]] = frozenset(),
  ):
    dense_proxies = _make_dense_proxies(host_param_map, dense_paths) if dense_paths else {}

    def _wrapped(
        dev_p: nnx.State,
        cust_p: nnx.State,
        g_map: dict[tuple[Any, ...], tuple[jax.Array, ...]],
        d_proxies: dict[tuple[Any, ...], jax.Array],
    ) -> Any:
      full_p = nnx.merge_state(dev_p, host_params) if jax.tree.leaves(host_params) else dev_p
      ctx_map = _build_pregather_ctx_map(g_map, cur_meta, check_identity=check_identity, dense_proxies=d_proxies)
      with _pregathered_context(ctx_map):
        return diff_wrapper(full_p, cust_p, rest, config, data)

    return jax.value_and_grad(_wrapped, argnums=(0, 1, 2, 3), has_aux=has_aux)(
        dev_params, custom_params, cur_gathered, dense_proxies
    )

  try:
    val_out, (dev_grads, custom_grads, d_gathered_map, d_dense_map) = _run_grad(
        gathered_map, meta_map, check_identity=True
    )
  except _RecaptureEmbedIds:
    custom_skip = frozenset(p for p, m in meta_map.items() if m["has_custom_ids_fn"])
    with _capture_embed_ids_context(skip_paths=custom_skip) as recorded_ids:
      diff_wrapper(curr_params, custom_params, nnx.clone(rest), config, data)
    dense_paths = frozenset(recorded_ids.pop("__dense_paths__", set()))
    gathered_map, meta_map = _prepare_host_embed_gathers(probe_model, tokens, segs, recorded_ids_map=recorded_ids)
    val_out, (dev_grads, custom_grads, d_gathered_map, d_dense_map) = _run_grad(
        gathered_map, meta_map, check_identity=False, dense_paths=dense_paths
    )
  raw_grads = _assemble_grads_with_sparse(
      dev_grads,
      d_gathered_map,
      meta_map,
      d_dense_map=d_dense_map,
      host_param_map=host_param_map,
      grad_dtype=getattr(config, "grad_dtype", None),  # jmp-waiver: SILENT_CONFIG_FALLBACK
  )
  return val_out, (raw_grads, custom_grads)


def engine_value_and_grad(
    diff_wrapper: Callable[..., Any],
    model_graphdef: Any,
    params: nnx.State,
    rest: nnx.State,
    batch: Any,
    *,
    param_shardings: Any = None,
    freeze_mask: Any = None,
) -> tuple[Any, nnx.State]:
  """Engine `fwd_bwd` differentiation avoiding `<host>` cotangents across scan/cond boundaries."""
  with active_param_shardings(param_shardings):
    full_model = nnx.merge(model_graphdef, params, rest, copy=True)
    if not _collect_host_embeds(full_model):
      return jax.value_and_grad(diff_wrapper, argnums=0, has_aux=True)(params, rest, batch)
    tokens, segs = _extract_tokens_and_segments((batch,), {}, None, None)
    if tokens is None:
      raise ValueError(
          "m3_offload.engine_value_and_grad requires token IDs in `batch` ('inputs'/'input_tokens') "
          "when HostOffloadedEmbed modules are present; refusing dense HBM fallback."
      )
    gathered_map, meta_map = _prepare_host_embed_gathers(full_model, tokens, segs)
    if not gathered_map:
      raise ValueError(
          "m3_offload.engine_value_and_grad found HostOffloadedEmbed modules but pre-gather produced no active gathers."
      )

    frozen_paths: frozenset[tuple[Any, ...]] = frozenset()
    if freeze_mask is not None:

      def _flag(v: Any) -> bool:
        return bool(v.get_raw_value() if isinstance(v, nnx.Variable) else v)

      if isinstance(freeze_mask, nnx.State):
        frozen_paths = frozenset(tuple(p) for p, is_frz in nnx.to_flat_state(freeze_mask) if _flag(is_frz))
      else:
        p_paths = [tuple(p) for p, _ in nnx.to_flat_state(params)]
        m_leaves = jax.tree.leaves(freeze_mask)
        if len(p_paths) == len(m_leaves):
          frozen_paths = frozenset(p for p, is_frz in zip(p_paths, m_leaves) if _flag(is_frz))

    host_embed_paths = frozenset(gathered_map.keys())
    dev_params, host_params = _partition_state_by_paths(params, host_embed_paths)
    host_param_map = {tuple(p): v for p, v in nnx.to_flat_state(host_params)}

    def _run_grad(
        cur_gathered: dict[tuple[Any, ...], tuple[jax.Array, ...]],
        cur_meta: dict[tuple[Any, ...], dict[str, Any]],
        *,
        check_identity: bool,
        dense_paths: frozenset[tuple[Any, ...]] = frozenset(),
    ):
      dense_proxies = _make_dense_proxies(host_param_map, dense_paths) if dense_paths else {}

      def _wrapped(
          dev_p: nnx.State,
          g_map: dict[tuple[Any, ...], tuple[jax.Array, ...]],
          d_proxies: dict[tuple[Any, ...], jax.Array],
      ) -> Any:
        full_p = nnx.merge_state(dev_p, host_params) if jax.tree.leaves(host_params) else dev_p
        eff_g_map = {
            p: tuple(jax.lax.stop_gradient(x) for x in xs) if p in frozen_paths else xs for p, xs in g_map.items()
        }
        eff_d_proxies = {p: jax.lax.stop_gradient(x) if p in frozen_paths else x for p, x in d_proxies.items()}
        ctx_map = _build_pregather_ctx_map(
            eff_g_map, cur_meta, check_identity=check_identity, dense_proxies=eff_d_proxies
        )
        with _pregathered_context(ctx_map):
          return diff_wrapper(full_p, rest, batch)

      return jax.value_and_grad(_wrapped, argnums=(0, 1, 2), has_aux=True)(dev_params, cur_gathered, dense_proxies)

    try:
      val_out, (dev_grads, d_gathered_map, d_dense_map) = _run_grad(gathered_map, meta_map, check_identity=True)
    except _RecaptureEmbedIds:
      custom_skip = frozenset(p for p, m in meta_map.items() if m["has_custom_ids_fn"])
      with _capture_embed_ids_context(skip_paths=custom_skip) as recorded_ids:
        diff_wrapper(params, nnx.clone(rest), batch)
      dense_paths = frozenset(recorded_ids.pop("__dense_paths__", set()))
      gathered_map, meta_map = _prepare_host_embed_gathers(full_model, tokens, segs, recorded_ids_map=recorded_ids)
      val_out, (dev_grads, d_gathered_map, d_dense_map) = _run_grad(
          gathered_map, meta_map, check_identity=False, dense_paths=dense_paths
      )
    grad_items = list(nnx.to_flat_state(dev_grads))
    for param_path, dg_tuple in d_gathered_map.items():
      meta = meta_map[param_path]
      p_leaf = host_param_map[param_path]
      table = p_leaf.get_raw_value() if isinstance(p_leaf, nnx.Variable) else p_leaf
      if param_path in frozen_paths:
        d_table_h = _apply_host_elementwise(_host_zeros_like_array, p_leaf, (table,), (jnp.zeros((), dtype=table.dtype),))
      elif param_path in d_dense_map:
        d_dense_h = jax.device_put(d_dense_map[param_path].astype(table.dtype), jax.memory.Space.Host)
        if meta["ids_list"]:
          ids_comb, dg_comb = _combine_call_ids_and_cotangents(
              meta["ids_list"],
              dg_tuple,
              features=meta["table_shape"][1],
              target_dtype=meta["target_dtype"],
              col_spec=meta["col_spec"],
              mesh=meta["mesh"],
          )
          d_embed_h = _host_embed_scatter_add(
              d_dense_h, ids_comb, dg_comb, row_axes=meta["row_axes"], col_axes=meta["col_axes"]
          )
          d_table_h = _apply_host_elementwise(_host_add_arrays, p_leaf, (d_embed_h, d_dense_h))
        else:
          d_table_h = d_dense_h
      else:
        ids_comb, dg_comb = _combine_call_ids_and_cotangents(
            meta["ids_list"],
            dg_tuple,
            features=meta["table_shape"][1],
            target_dtype=meta["target_dtype"],
            col_spec=meta["col_spec"],
            mesh=meta["mesh"],
        )
        d_table_h = _host_embed_scatter_add(
            table, ids_comb, dg_comb, row_axes=meta["row_axes"], col_axes=meta["col_axes"]
        )
      grad_leaf = p_leaf.replace(d_table_h) if isinstance(p_leaf, nnx.Variable) else d_table_h
      grad_items.append((param_path, grad_leaf))
    return val_out, nnx.from_flat_state(grad_items)


# ---------------------------------------------------------------------------
# Exact global L2 norm, clipping, and host-safe gradient ops
# ---------------------------------------------------------------------------


def _sparse_grad_sqnorm(grad: SparseHostGrad) -> jax.Array:
  """Exact squared L2 norm `||scatter_add(0, ids, d_gathered)||_2^2` in O(M*D) memory."""
  mesh = _active_mesh(grad)

  def _local_sqnorm(ids_arr: jax.Array, dg_arr: jax.Array, start: jax.Array, n_local: int) -> jax.Array:
    flat_ids = ids_arr.reshape(-1)
    flat_dg = dg_arr.reshape(-1, dg_arr.shape[-1]).astype(jnp.float32)
    m = flat_ids.shape[0]
    if m == 0 or flat_dg.shape[-1] == 0:
      return jnp.float32(0.0)
    rel = flat_ids - start
    in_range = (rel >= 0) & (rel < n_local)
    # Send out-of-shard entries to a unique sentinel so they stay masked out cleanly.
    sentinel = jnp.int32(n_local)
    key = jnp.where(in_range, rel, sentinel)
    vals = jnp.where(in_range[:, None], flat_dg, jnp.zeros((), dtype=jnp.float32))
    order = jnp.argsort(key)
    sorted_key = key[order]
    sorted_vals = vals[order]
    is_new = jnp.concatenate([jnp.array([True]), sorted_key[1:] != sorted_key[:-1]])
    seg_ids = jnp.cumsum(is_new.astype(jnp.int32)) - 1
    seg_sums = jax.ops.segment_sum(sorted_vals, seg_ids, num_segments=m)
    # Match exact grad_dtype rounding before squaring.
    seg_sums = seg_sums.astype(grad.grad_dtype).astype(jnp.float32)
    return jnp.sum(seg_sums * seg_sums)

  if mesh is not None:
    c_spec = grad.resolved_col_spec(mesh)
    rep_ids_spec = P(*([None] * grad.ids.ndim))
    dg_spec = P(*([None] * grad.ids.ndim), c_spec)
    ids_in = _maybe_reshard(grad.ids, rep_ids_spec, mesh)
    dg_in = _maybe_reshard(grad.d_gathered, dg_spec, mesh)
    n_shards = 1
    for ax in grad.row_axes:
      n_shards *= mesh.shape[ax]
    n_local = grad.table_shape[0] // n_shards
    sum_axes = (*grad.row_axes, *grad.col_axes)

    def _body(local_ids: jax.Array, local_dg: jax.Array) -> jax.Array:
      start = jax.lax.axis_index(grad.row_axes).astype(jnp.int32) * jnp.int32(n_local) if grad.row_axes else jnp.int32(0)
      local_sq = _local_sqnorm(local_ids, local_dg, start, n_local)
      return jax.lax.psum(local_sq, sum_axes) if sum_axes else local_sq

    return _call_shard_map(_body, in_specs=(rep_ids_spec, dg_spec), out_specs=P(), args=(ids_in, dg_in), mesh=mesh)

  return _local_sqnorm(grad.ids, grad.d_gathered, jnp.int32(0), grad.table_shape[0])


def _sparse_grad_max_abs(grad: SparseHostGrad) -> jax.Array:
  """Exact `max |scatter_add(0, ids, d_gathered)|` over accumulated rows in O(M*D) memory."""
  mesh = _active_mesh(grad)

  def _local_max_abs(ids_arr: jax.Array, dg_arr: jax.Array, start: jax.Array, n_local: int) -> jax.Array:
    flat_ids = ids_arr.reshape(-1)
    flat_dg = dg_arr.reshape(-1, dg_arr.shape[-1]).astype(jnp.float32)
    m = flat_ids.shape[0]
    if m == 0 or flat_dg.shape[-1] == 0:
      return jnp.float32(0.0)
    rel = flat_ids - start
    in_range = (rel >= 0) & (rel < n_local)
    sentinel = jnp.int32(n_local)
    key = jnp.where(in_range, rel, sentinel)
    vals = jnp.where(in_range[:, None], flat_dg, jnp.zeros((), dtype=jnp.float32))
    order = jnp.argsort(key)
    sorted_key = key[order]
    sorted_vals = vals[order]
    is_new = jnp.concatenate([jnp.array([True]), sorted_key[1:] != sorted_key[:-1]])
    seg_ids = jnp.cumsum(is_new.astype(jnp.int32)) - 1
    seg_sums = jax.ops.segment_sum(sorted_vals, seg_ids, num_segments=m)
    seg_sums = seg_sums.astype(grad.grad_dtype).astype(jnp.float32)
    return jnp.max(jnp.abs(seg_sums))

  if mesh is not None:
    c_spec = grad.resolved_col_spec(mesh)
    rep_ids_spec = P(*([None] * grad.ids.ndim))
    dg_spec = P(*([None] * grad.ids.ndim), c_spec)
    ids_in = _maybe_reshard(grad.ids, rep_ids_spec, mesh)
    dg_in = _maybe_reshard(grad.d_gathered, dg_spec, mesh)
    n_shards = 1
    for ax in grad.row_axes:
      n_shards *= mesh.shape[ax]
    n_local = grad.table_shape[0] // n_shards
    red_axes = (*grad.row_axes, *grad.col_axes)

    def _body(local_ids: jax.Array, local_dg: jax.Array) -> jax.Array:
      start = jax.lax.axis_index(grad.row_axes).astype(jnp.int32) * jnp.int32(n_local) if grad.row_axes else jnp.int32(0)
      local_m = _local_max_abs(local_ids, local_dg, start, n_local)
      return jax.lax.pmax(local_m, red_axes) if red_axes else local_m

    return _call_shard_map(_body, in_specs=(rep_ids_spec, dg_spec), out_specs=P(), args=(ids_in, dg_in), mesh=mesh)

  return _local_max_abs(grad.ids, grad.d_gathered, jnp.int32(0), grad.table_shape[0])


@_device_host_compute(out_memory_spaces=jax.memory.Space.Device)
def _host_array_sqnorm_raw(x_h: jax.Array) -> jax.Array:
  x32 = x_h.astype(jnp.float32)
  return jnp.sum(x32 * x32)


@_device_host_compute(out_memory_spaces=jax.memory.Space.Device)
def _host_array_max_abs_raw(x_h: jax.Array) -> jax.Array:
  if x_h.size == 0:
    return jnp.float32(0.0)
  return jnp.max(jnp.abs(x_h)).astype(jnp.float32)


def _host_array_sqnorm(x: jax.Array, var_or_arr: Any = None, *, sharding_hint: Any = None) -> jax.Array:
  """Computes squared L2 norm of a host-resident array on host without moving it to HBM."""
  mesh = _active_mesh(var_or_arr, x, sharding_hint)
  r_spec, c_spec, row_axes, col_axes = _resolve_2d_specs_and_axes(
      var_or_arr if var_or_arr is not None else x, mesh, sharding_hint=sharding_hint
  )
  if mesh is not None and x.ndim == 2:
    sum_axes = (*row_axes, *col_axes)

    def _body(local_x: jax.Array) -> jax.Array:
      local_sq = _host_array_sqnorm_raw(local_x)
      return jax.lax.psum(local_sq, sum_axes) if sum_axes else local_sq

    return _call_shard_map(_body, in_specs=(P(r_spec, c_spec),), out_specs=P(), args=(x,), mesh=mesh)
  return _host_array_sqnorm_raw(x)


def _host_array_max_abs(x: jax.Array, var_or_arr: Any = None, *, sharding_hint: Any = None) -> jax.Array:
  """Computes max absolute value of a host-resident array on host without moving it to HBM."""
  mesh = _active_mesh(var_or_arr, x, sharding_hint)
  r_spec, c_spec, row_axes, col_axes = _resolve_2d_specs_and_axes(
      var_or_arr if var_or_arr is not None else x, mesh, sharding_hint=sharding_hint
  )
  if mesh is not None and x.ndim == 2:
    red_axes = (*row_axes, *col_axes)

    def _body(local_x: jax.Array) -> jax.Array:
      local_m = _host_array_max_abs_raw(local_x)
      return jax.lax.pmax(local_m, red_axes) if red_axes else local_m

    return _call_shard_map(_body, in_specs=(P(r_spec, c_spec),), out_specs=P(), args=(x,), mesh=mesh)
  return _host_array_max_abs_raw(x)


def _apply_host_elementwise(
    host_fn: Any,
    ref_leaf: Any,
    host_arrays: tuple[jax.Array, ...],
    device_scalars: tuple[jax.Array, ...] = (),
    *,
    sharding_hint: Any = None,
) -> jax.Array:
  """Isolates an elementwise host-compute op inside `shard_map` when a mesh is active."""
  mesh = _active_mesh(ref_leaf, *host_arrays, sharding_hint)
  arr0 = host_arrays[0]
  if mesh is not None and arr0.ndim == 2:
    r_spec, c_spec, _, _ = _resolve_2d_specs_and_axes(ref_leaf, mesh, sharding_hint=sharding_hint)
    tbl_spec = P(r_spec, c_spec)
    arr_specs = tuple(_leaf_pspec(a, mesh, tbl_spec) for a in host_arrays)
    sc_specs = tuple(_leaf_pspec(s, mesh) for s in device_scalars)
    in_specs = arr_specs + sc_specs
    n_arr = len(host_arrays)

    def _body(*local_args: jax.Array) -> jax.Array:
      loc_arrs = local_args[:n_arr]
      loc_scalars = tuple(jax.device_put(s, jax.memory.Space.Host) for s in local_args[n_arr:])
      return host_fn(*loc_arrs, *loc_scalars)

    return _call_shard_map(
        _body, in_specs=in_specs, out_specs=arr_specs[0], args=(*host_arrays, *device_scalars), mesh=mesh
    )
  scalars_h = tuple(jax.device_put(s, jax.memory.Space.Host) for s in device_scalars)
  return host_fn(*host_arrays, *scalars_h)


@_device_host_compute(out_memory_spaces=jax.memory.Space.Host)
def _host_cast_array(x_h: jax.Array, dtype_holder_h: jax.Array) -> jax.Array:
  return x_h.astype(dtype_holder_h.dtype)


@_device_host_compute(out_memory_spaces=jax.memory.Space.Host)
def _host_zeros_like_array(x_h: jax.Array, dtype_holder_h: jax.Array) -> jax.Array:
  return jnp.zeros(x_h.shape, dtype=dtype_holder_h.dtype)


def _alloc_host_zeros_for_var(param_var: nnx.Variable, target_dtype: Any, *, out_pspec: Any = None) -> jax.Array:
  """Allocates zero moments directly in `pinned_host` via `compute_on('device_host')` without HBM or TPU `lax.full`."""
  raw = param_var.get_raw_value()
  dtype_holder = jnp.zeros((), dtype=target_dtype)
  mesh = _active_mesh(param_var, raw)
  if mesh is not None and getattr(raw, "ndim", 0) == 2:
    zeros_h = _apply_host_elementwise(_host_zeros_like_array, param_var, (raw,), (dtype_holder,))
    if isinstance(mesh, jax.sharding.Mesh) and isinstance(out_pspec, P):
      cur_spec = getattr(getattr(zeros_h, "sharding", None), "spec", None)
      if cur_spec != out_pspec:
        zeros_h = jax.device_put(
            zeros_h,
            jax.sharding.NamedSharding(mesh, out_pspec, memory_kind="pinned_host"),
        )
    return zeros_h
  sh = getattr(raw, "sharding", None)
  out_sh = sh.with_memory_kind("pinned_host") if sh is not None else jax.memory.Space.Host
  return jax.jit(_host_zeros_like_array, out_shardings=out_sh)(raw, jax.device_put(dtype_holder, jax.memory.Space.Host))


@_device_host_compute(out_memory_spaces=jax.memory.Space.Host)
def _host_add_arrays(a_h: jax.Array, b_h: jax.Array) -> jax.Array:
  return a_h + b_h.astype(a_h.dtype)


@_device_host_compute(out_memory_spaces=jax.memory.Space.Host)
def _host_scale_array(x_h: jax.Array, scale_h: jax.Array) -> jax.Array:
  return x_h * scale_h.astype(x_h.dtype)


@_device_host_compute(out_memory_spaces=jax.memory.Space.Host)
def _host_clip_array(x_h: jax.Array, trigger_h: jax.Array, inv_norm_h: jax.Array) -> jax.Array:
  scaled = x_h * inv_norm_h.astype(x_h.dtype)
  return jnp.where(trigger_h, x_h, scaled)


@_device_host_compute(out_memory_spaces=jax.memory.Space.Host)
def _host_where_arrays(true_h: jax.Array, false_h: jax.Array, cond_h: jax.Array) -> jax.Array:
  return jnp.where(cond_h, true_h, false_h)


@_device_host_compute(out_memory_spaces=jax.memory.Space.Host)
def _host_zero_if(arr_h: jax.Array, cond_h: jax.Array) -> jax.Array:
  return jnp.where(cond_h, jax.device_put(jnp.zeros((), dtype=arr_h.dtype), jax.memory.Space.Host), arr_h)


def _is_tree_leaf(x: Any) -> bool:
  return isinstance(x, (nnx.Variable, SparseHostGrad, jax.sharding.Sharding, P))


def _is_tree_leaf_or_none(x: Any) -> bool:
  return x is None or _is_tree_leaf(x)


def select_tree_if(cond: jax.Array, true_tree: Any, false_tree: Any, *, shardings: Any = None) -> Any:
  """Elementwise `jnp.where(cond, true_tree, false_tree)` keeping host leaves on host."""

  def _sel(t_leaf: Any, f_leaf: Any, sh_leaf: Any = None) -> Any:
    if t_leaf is None or f_leaf is None:
      return f_leaf
    if isinstance(f_leaf, SparseHostGrad):
      t_dg = t_leaf.d_gathered if isinstance(t_leaf, SparseHostGrad) else jnp.zeros_like(f_leaf.d_gathered)
      return f_leaf.with_d_gathered(jnp.where(cond, t_dg, f_leaf.d_gathered))
    t_val = t_leaf.get_raw_value() if isinstance(t_leaf, nnx.Variable) else t_leaf
    f_val = f_leaf.get_raw_value() if isinstance(f_leaf, nnx.Variable) else f_leaf
    if not hasattr(f_val, "shape") or not hasattr(f_val, "dtype"):
      return f_leaf
    if _is_host_leaf(t_leaf) or _is_host_leaf(f_leaf):
      t_h = t_val if _is_host_leaf(t_leaf) else jax.device_put(t_val, jax.memory.Space.Host)
      f_h = f_val if _is_host_leaf(f_leaf) else jax.device_put(f_val, jax.memory.Space.Host)
      out = _apply_host_elementwise(_host_where_arrays, f_leaf, (t_h, f_h), (cond,), sharding_hint=sh_leaf)
      return f_leaf.replace(out) if isinstance(f_leaf, nnx.Variable) else out
    out = jnp.where(cond, t_val, f_val)
    return f_leaf.replace(out) if isinstance(f_leaf, nnx.Variable) else out

  if shardings is not None:
    return jax.tree.map(_sel, true_tree, false_tree, shardings, is_leaf=_is_tree_leaf_or_none)
  return jax.tree.map(_sel, true_tree, false_tree, is_leaf=_is_tree_leaf_or_none)


def zero_tree_if(cond: jax.Array, tree: Any, *, shardings: Any = None) -> Any:
  """Zeros leaves when `cond` is True without copying host leaves onto HBM."""

  def _zero(leaf: Any, sh_leaf: Any = None) -> Any:
    if leaf is None:
      return None
    if isinstance(leaf, SparseHostGrad):
      return leaf.with_d_gathered(jnp.where(cond, jnp.zeros_like(leaf.d_gathered), leaf.d_gathered))
    val = leaf.get_raw_value() if isinstance(leaf, nnx.Variable) else leaf
    if not hasattr(val, "shape") or not hasattr(val, "dtype"):
      return leaf
    if _is_host_leaf(leaf):
      out = _apply_host_elementwise(_host_zero_if, leaf, (val,), (cond,), sharding_hint=sh_leaf)
      return leaf.replace(out) if isinstance(leaf, nnx.Variable) else out
    out = jnp.where(cond, jnp.zeros_like(val), val)
    return leaf.replace(out) if isinstance(leaf, nnx.Variable) else out

  if shardings is not None:
    return jax.tree.map(_zero, tree, shardings, is_leaf=_is_tree_leaf_or_none)
  return jax.tree.map(_zero, tree, is_leaf=_is_tree_leaf_or_none)


def cast_grad_leaf(leaf: Any, grad_dtype: Any, *, sharding: Any = None) -> Any:
  """Casts a gradient leaf (`device`, `pinned_host`, or `SparseHostGrad`) without moving host arrays to HBM."""
  target = jnp.dtype(grad_dtype)
  if isinstance(leaf, SparseHostGrad):
    if leaf.table_dtype == jnp.float32 and leaf.grad_dtype != target:
      return leaf.with_d_gathered(leaf.d_gathered, grad_dtype=target)
    return leaf
  val = leaf.get_raw_value() if isinstance(leaf, nnx.Variable) else leaf
  if not hasattr(val, "dtype") or val.dtype == target:
    return leaf
  if _is_host_leaf(leaf):
    casted = _apply_host_elementwise(
        _host_cast_array, leaf, (val,), (jnp.zeros((), dtype=target),), sharding_hint=sharding
    )
    return leaf.replace(casted) if isinstance(leaf, nnx.Variable) else casted
  casted = val.astype(target)
  return leaf.replace(casted) if isinstance(leaf, nnx.Variable) else casted


def cast_grad_tree(grads: Any, grad_dtype: Any) -> Any:
  """Casts float32 gradient leaves to `grad_dtype`, keeping host leaves on host."""
  target = jnp.dtype(grad_dtype)

  def _cast(leaf: Any) -> Any:
    if isinstance(leaf, SparseHostGrad):
      return cast_grad_leaf(leaf, target)
    val = leaf.get_raw_value() if isinstance(leaf, nnx.Variable) else leaf
    if hasattr(val, "dtype") and val.dtype == jnp.float32 and target != jnp.float32:
      return cast_grad_leaf(leaf, target)
    return leaf

  return jax.tree.map(_cast, grads, is_leaf=lambda x: isinstance(x, (nnx.Variable, SparseHostGrad)))


def accumulate_grad_leaves(acc: Any, g: Any, sharding: Any = None) -> Any:
  """Adds gradient leaf `g` into `acc` on host when both reside in host memory."""
  acc_val = acc.get_raw_value() if isinstance(acc, nnx.Variable) else acc
  g_val = g.get_raw_value() if isinstance(g, nnx.Variable) else g
  if _is_host_leaf(acc) or _is_host_leaf(g):
    acc_h = jax.device_put(acc_val, jax.memory.Space.Host) if not _is_host_leaf(acc) else acc_val
    g_h = jax.device_put(g_val, jax.memory.Space.Host) if not _is_host_leaf(g) else g_val
    out = _apply_host_elementwise(_host_add_arrays, acc, (acc_h, g_h), sharding_hint=sharding)
    return acc.replace(out) if isinstance(acc, nnx.Variable) else out
  out = acc_val + g_val.astype(acc_val.dtype)
  return acc.replace(out) if isinstance(acc, nnx.Variable) else out


def scale_grad_leaf(leaf: Any, factor: jax.Array, *, sharding: Any = None) -> Any:
  """Multiplies a gradient leaf by scalar `factor`, preserving host placement."""
  if isinstance(leaf, SparseHostGrad):
    return leaf.with_d_gathered(leaf.d_gathered * factor.astype(leaf.d_gathered.dtype))
  val = leaf.get_raw_value() if isinstance(leaf, nnx.Variable) else leaf
  if _is_host_leaf(leaf):
    out = _apply_host_elementwise(_host_scale_array, leaf, (val,), (factor,), sharding_hint=sharding)
    return leaf.replace(out) if isinstance(leaf, nnx.Variable) else out
  out = val * factor.astype(val.dtype)
  return leaf.replace(out) if isinstance(leaf, nnx.Variable) else out


def l2norm_pytree(tree: Any, *, shardings: Any = None) -> jax.Array:
  """Exact global L2 norm across device arrays, `SparseHostGrad`, and `pinned_host` leaves."""
  sq_terms: list[jax.Array] = []

  def _accum_sq(leaf: Any, sh_leaf: Any = None) -> None:
    if leaf is None:
      return
    if isinstance(leaf, SparseHostGrad):
      sq_terms.append(_sparse_grad_sqnorm(leaf))
      return
    val = leaf.get_raw_value() if isinstance(leaf, nnx.Variable) else leaf
    if not hasattr(val, "dtype") or not hasattr(val, "shape"):
      return
    if _is_host_leaf(leaf):
      sq_terms.append(_host_array_sqnorm(val, leaf, sharding_hint=sh_leaf))
    else:
      v32 = val.astype(jnp.float32)
      sq_terms.append(jnp.sum(v32 * v32))

  if shardings is not None:
    jax.tree.map(_accum_sq, tree, shardings, is_leaf=_is_tree_leaf_or_none)
  else:
    jax.tree.map(_accum_sq, tree, is_leaf=_is_tree_leaf_or_none)
  if not sq_terms:
    return jnp.array(0.0, dtype=jnp.float32)
  return jnp.sqrt(functools.reduce(jnp.add, sq_terms))


def has_host_or_sparse_grad_leaves(tree: Any) -> bool:
  """Returns True if `tree` contains any `SparseHostGrad` or `pinned_host` leaves."""
  for leaf in jax.tree.leaves(tree, is_leaf=_is_tree_leaf):
    if isinstance(leaf, SparseHostGrad) or _is_host_leaf(leaf):
      return True
  return False


def max_abs_grad_pytree(tree: Any, *, shardings: Any = None) -> jax.Array:
  """Exact global max |gradient| across device arrays, `SparseHostGrad`, and `pinned_host` leaves."""
  max_terms: list[jax.Array] = []

  def _accum_max(leaf: Any, sh_leaf: Any = None) -> None:
    if leaf is None:
      return
    if isinstance(leaf, SparseHostGrad):
      max_terms.append(_sparse_grad_max_abs(leaf))
      return
    val = leaf.get_raw_value() if isinstance(leaf, nnx.Variable) else leaf
    if not hasattr(val, "dtype") or not jnp.issubdtype(val.dtype, jnp.inexact):
      return
    if _is_host_leaf(leaf):
      max_terms.append(_host_array_max_abs(val, leaf, sharding_hint=sh_leaf))
    elif val.size:
      max_terms.append(jnp.max(jnp.abs(val)).astype(jnp.float32))

  if shardings is not None:
    jax.tree.map(_accum_max, tree, shardings, is_leaf=_is_tree_leaf_or_none)
  else:
    jax.tree.map(_accum_max, tree, is_leaf=_is_tree_leaf_or_none)
  if not max_terms:
    return jnp.float32(0.0)
  return jnp.max(jnp.stack(max_terms))


def clip_grads_and_norms(grads: Any, max_norm: float, *, shardings: Any = None) -> tuple[Any, jax.Array, jax.Array]:
  """Clips a mixed gradient tree in one pass, returning `(clipped, raw_norm, clipped_norm)`."""
  g_norm = l2norm_pytree(grads, shardings=shardings)
  if max_norm <= 0:
    return grads, g_norm, g_norm
  max_norm_arr = jnp.asarray(max_norm, dtype=jnp.float32)
  trigger = jnp.squeeze(g_norm < max_norm_arr)
  safe_norm = jnp.where(trigger, jnp.ones_like(g_norm), g_norm)
  scale_f32 = max_norm_arr / safe_norm

  def _clip(leaf: Any, sh_leaf: Any = None) -> Any:
    if leaf is None:
      return None
    if isinstance(leaf, SparseHostGrad):
      # Clip in `grad_dtype` (e.g. fp32 when `weight_dtype=float32, dtype=bfloat16`) to avoid bf16 pre-rounding.
      dg = leaf.d_gathered.astype(leaf.grad_dtype)
      clipped_dg = jax.lax.select(trigger, dg, (dg / safe_norm.astype(dg.dtype)) * max_norm_arr.astype(dg.dtype))
      return leaf.with_d_gathered(clipped_dg)
    val = leaf.get_raw_value() if isinstance(leaf, nnx.Variable) else leaf
    if not hasattr(val, "dtype") or not hasattr(val, "shape"):
      return leaf
    if _is_host_leaf(leaf):
      clipped = _apply_host_elementwise(_host_clip_array, leaf, (val,), (trigger, scale_f32), sharding_hint=sh_leaf)
      return leaf.replace(clipped) if isinstance(leaf, nnx.Variable) else clipped
    clipped = jax.lax.select(trigger, val, (val / safe_norm.astype(val.dtype)) * max_norm_arr.astype(val.dtype))
    return leaf.replace(clipped) if isinstance(leaf, nnx.Variable) else clipped

  if shardings is not None:
    clipped = jax.tree.map(_clip, grads, shardings, is_leaf=_is_tree_leaf_or_none)
  else:
    clipped = jax.tree.map(_clip, grads, is_leaf=_is_tree_leaf_or_none)
  clipped_norm = jnp.where(trigger, g_norm, jnp.where(jnp.isfinite(g_norm), max_norm_arr, g_norm))
  return clipped, g_norm, clipped_norm


def clip_grads_by_global_norm(grads: Any, max_norm: float, *, shardings: Any = None) -> Any:
  """Clips a mixed device/`SparseHostGrad`/`pinned_host` gradient tree by global L2 norm."""
  clipped, _, _ = clip_grads_and_norms(grads, max_norm, shardings=shardings)
  return clipped


@contextlib.contextmanager
def capture_host_param_sqnorms():
  """Captures post-update squared L2 norms of host-offloaded params inside the optimizer host pass."""
  prev = getattr(_HOST_SQNORM_CAPTURE, "sqnorms", None)
  captured: dict[tuple[Any, ...], jax.Array] = {}
  _HOST_SQNORM_CAPTURE.sqnorms = captured
  try:
    yield captured
  finally:
    _HOST_SQNORM_CAPTURE.sqnorms = prev


def l2norm_pytree_with_overrides(
    tree: Any,
    sqnorm_overrides: dict[tuple[Any, ...], jax.Array],
    *,
    shardings: Any = None,
) -> jax.Array:
  """Exact global L2 norm substituting pre-computed squared norms for captured host parameters."""
  if not sqnorm_overrides:
    return l2norm_pytree(tree, shardings=shardings)
  sharding_map: dict[tuple[Any, ...], Any] = {}
  if shardings is not None:
    for sh_path, sh_leaf in jax.tree_util.tree_leaves_with_path(
        shardings, is_leaf=lambda x: isinstance(x, (nnx.Variable, jax.sharding.Sharding, P))
    ):
      sharding_map[tuple(_key_value(k) for k in sh_path)] = sh_leaf
  sq_terms: list[jax.Array] = []
  for path, leaf in nnx.to_flat_state(tree):
    path_tuple = tuple(path)
    if path_tuple in sqnorm_overrides:
      sq_terms.append(sqnorm_overrides[path_tuple])
      continue
    if leaf is None:
      continue
    if isinstance(leaf, SparseHostGrad):
      sq_terms.append(_sparse_grad_sqnorm(leaf))
      continue
    val = leaf.get_raw_value() if isinstance(leaf, nnx.Variable) else leaf
    if not hasattr(val, "dtype") or not hasattr(val, "shape"):
      continue
    if _is_host_leaf(leaf):
      sq_terms.append(_host_array_sqnorm(val, leaf, sharding_hint=sharding_map.get(path_tuple)))
    else:
      v32 = val.astype(jnp.float32)
      sq_terms.append(jnp.sum(v32 * v32))
  if not sq_terms:
    return jnp.array(0.0, dtype=jnp.float32)
  return jnp.sqrt(functools.reduce(jnp.add, sq_terms))


# ---------------------------------------------------------------------------
# Selective optimizer state slicing & host-compute / streamed-chunks / lazy-sparse updates
# ---------------------------------------------------------------------------


def _nest_path(path: tuple[Any, ...], value: Any) -> dict[Any, Any]:
  node: Any = value
  for key in reversed(path):
    node = {key: node}
  return node


def _unnest_path(path: tuple[Any, ...], tree: Any) -> Any:
  node = tree
  for key in path:
    node = node[key]
  return node


def _filter_mapping_by_paths(
    mapping: dict[Any, Any],
    keep_fn: Callable[[tuple[Any, ...]], bool],
    prefix: tuple[Any, ...] = (),
) -> dict[Any, Any]:
  """Filters a nested parameter/state dict keeping only leaves where `keep_fn(path)` is True."""
  out: dict[Any, Any] = {}
  for k, v in mapping.items():
    sub_path = (*prefix, k)
    if isinstance(v, (dict, nnx.State)):
      sub_dict = dict(v) if isinstance(v, nnx.State) else v
      filtered = _filter_mapping_by_paths(sub_dict, keep_fn, sub_path)
      if filtered:
        out[k] = filtered
    elif keep_fn(sub_path):
      out[k] = v
  return out


def _flat_dict_paths(mapping: dict[Any, Any], prefix=()) -> set[tuple[Any, ...]]:
  paths: set[tuple[Any, ...]] = set()
  for k, v in mapping.items():
    sub_path = (*prefix, k)
    if isinstance(v, (dict, nnx.State)):
      sub_dict = dict(v) if isinstance(v, nnx.State) else v
      paths |= _flat_dict_paths(sub_dict, sub_path)
    else:
      paths.add(sub_path)
  return paths


def _slice_opt_state(
    opt_state: Any,
    all_param_paths: frozenset[tuple[Any, ...]],
    keep_fn: Callable[[tuple[Any, ...]], bool],
) -> Any:
  """Filters parameter-shaped subtrees inside `opt_state` while keeping scalars intact."""
  if isinstance(opt_state, (dict, nnx.State)):
    as_dict = dict(opt_state) if isinstance(opt_state, nnx.State) else opt_state
    leaf_paths = _flat_dict_paths(as_dict)
    if leaf_paths and leaf_paths.issubset(all_param_paths):
      return _filter_mapping_by_paths(as_dict, keep_fn)
    return {k: _slice_opt_state(v, all_param_paths, keep_fn) for k, v in as_dict.items()}
  if isinstance(opt_state, tuple) and hasattr(opt_state, "_fields"):
    return type(opt_state)(
        *[_slice_opt_state(getattr(opt_state, f), all_param_paths, keep_fn) for f in opt_state._fields]
    )
  if isinstance(opt_state, tuple):
    return tuple(_slice_opt_state(v, all_param_paths, keep_fn) for v in opt_state)
  if isinstance(opt_state, list):
    return [_slice_opt_state(v, all_param_paths, keep_fn) for v in opt_state]
  return opt_state


def _lookup_path_in_tree(tree: Any, key_tuple: tuple[Any, ...]) -> Any:
  """Traverses a nested opt_state PyTree along `key_tuple`."""
  node = tree
  for k in key_tuple:
    if isinstance(node, (dict, nnx.State)):
      node = node[k]
    elif isinstance(node, (tuple, list)) and isinstance(k, int):
      node = node[k]
    elif hasattr(node, str(k)):
      node = getattr(node, str(k))
    else:
      node = node[k]
  return node


_OPTAX_PARAM_ROOT_KEYS: frozenset[str] = frozenset(
    {
        "mu",
        "nu",
        "trace",
        "ema",
        "sum_of_squares",
        "v_row",
        "v_col",
        "v",
        "m",
        "grad_ema",
        "nu_ema",
        "accumulator",
        "params",
    }
)
_OPTAX_WRAPPER_KEYS: frozenset[str] = frozenset(
    {
        "inner_state",
        "inner_opt_state",
        "inner_states",
    }
)


def _is_optax_prefix(prefix: tuple[Any, ...]) -> bool:
  """Returns True when `prefix` is a valid Optax state path preceding a parameter root."""
  if not prefix:
    return True
  last = prefix[-1]
  if not isinstance(last, int) and last not in _OPTAX_PARAM_ROOT_KEYS:
    return False
  idx = 0
  limit = len(prefix) - 1
  while idx < limit:
    elem = prefix[idx]
    if elem == "inner_states" and idx + 1 < limit:
      idx += 2
      continue
    if isinstance(elem, int) or elem in _OPTAX_WRAPPER_KEYS or elem in _OPTAX_PARAM_ROOT_KEYS:
      idx += 1
      continue
    return False
  return True


def _is_matching_opt_leaf(
    key_tuple: tuple[Any, ...],
    selected_paths: frozenset[tuple[Any, ...]],
    all_param_paths: frozenset[tuple[Any, ...]] | None = None,
) -> tuple[Any, ...] | None:
  """Returns the selected parameter path for `key_tuple` anchored at the Optax/NNX boundary."""
  if all_param_paths is not None:
    for start in range(len(key_tuple)):
      cand = key_tuple[start:]
      if cand in all_param_paths and _is_optax_prefix(key_tuple[:start]):
        return cand if cand in selected_paths else None
    for start in range(len(key_tuple)):
      cand = key_tuple[start:]
      if cand in all_param_paths:
        return cand if cand in selected_paths else None
    return None

  best: tuple[Any, ...] | None = None
  for p_path in selected_paths:
    n = len(p_path)
    if len(key_tuple) >= n and key_tuple[-n:] == p_path:
      prefix = key_tuple[:-n]
      if _is_optax_prefix(prefix) and (best is None or len(p_path) > len(best)):
        best = p_path
  return best


def _run_tx_on_single_leaf(
    tx: optax.GradientTransformation,
    p_path: tuple[Any, ...],
    grad_arr: jax.Array,
    opt_state_sub: Any,
    param_arr: jax.Array,
    **kwargs: Any,
) -> tuple[jax.Array, Any]:
  """Runs `tx.update` + `apply_updates` on a single parameter wrapped at its exact path."""
  g_nested = _nest_path(p_path, grad_arr)
  p_nested = _nest_path(p_path, param_arr)
  updates_nested, new_st = tx.update(g_nested, opt_state_sub, p_nested, **kwargs)
  upd_arr = _unnest_path(p_path, updates_nested)
  if _is_host_leaf(param_arr) and not _is_host_leaf(upd_arr):
    upd_arr = jax.device_put(upd_arr, jax.memory.Space.Host)
  new_p = optax.apply_updates(param_arr, upd_arr)
  return new_p, new_st


def _selective_host_compute_update(
    param: jax.Array,
    grad: Any,
    opt_state_sub: Any,
    *,
    tx: optax.GradientTransformation,
    p_path: tuple[Any, ...],
    row_axes: tuple[str, ...],
    col_axes: tuple[str, ...],
    kwargs_arrays: dict[str, Any],
) -> tuple[jax.Array, Any]:
  """Runs exact dense optimizer update on host CPU with zero HBM full-table traffic."""
  mesh = _active_mesh(param, grad)
  is_sparse = isinstance(grad, SparseHostGrad)

  capture_map = getattr(_HOST_SQNORM_CAPTURE, "sqnorms", None)
  capture_sqnorm = capture_map is not None

  @_device_host_compute(out_memory_spaces=jax.memory.Space.Host)
  def host_sparse_fn(p_loc, st_loc_h, sparse_h, start_h, kw_h):
    ids_h, dg_h = sparse_h
    n_local = p_loc.shape[0]
    g_dtype = grad.grad_dtype
    if row_axes:
      rel = ids_h - start_h
      in_range = (rel >= 0) & (rel < n_local)
      zero_i = jax.device_put(jnp.int32(0), jax.memory.Space.Host)
      safe = jnp.where(in_range, rel, zero_i)
      zero_v = jax.device_put(jnp.zeros((), dtype=g_dtype), jax.memory.Space.Host)
      vals = jnp.where(in_range[..., None], dg_h.astype(g_dtype), zero_v)
      z_buf = jax.device_put(jnp.zeros(p_loc.shape, dtype=g_dtype), jax.memory.Space.Host)
      g_loc = z_buf.at[safe].add(vals)
    else:
      z_buf = jax.device_put(jnp.zeros(p_loc.shape, dtype=g_dtype), jax.memory.Space.Host)
      g_loc = z_buf.at[ids_h].add(dg_h.astype(g_dtype))
    p_new, st_new = _run_tx_on_single_leaf(tx, p_path, g_loc, st_loc_h, p_loc, **kw_h)
    st_new = jax.tree.map(lambda x: jax.device_put(x, jax.memory.Space.Host) if hasattr(x, "ndim") else x, st_new)
    if capture_sqnorm:
      p32 = p_new.astype(jnp.float32)
      return p_new, st_new, jnp.sum(p32 * p32)
    return p_new, st_new

  @_device_host_compute(out_memory_spaces=jax.memory.Space.Host)
  def host_dense_fn(p_loc, st_loc_h, g_loc, kw_h):
    p_new, st_new = _run_tx_on_single_leaf(tx, p_path, g_loc, st_loc_h, p_loc, **kw_h)
    st_new = jax.tree.map(lambda x: jax.device_put(x, jax.memory.Space.Host) if hasattr(x, "ndim") else x, st_new)
    if capture_sqnorm:
      p32 = p_new.astype(jnp.float32)
      return p_new, st_new, jnp.sum(p32 * p32)
    return p_new, st_new

  st_orig_leaves, _ = jax.tree_util.tree_flatten(opt_state_sub)
  st_dev_mask = tuple(hasattr(x, "ndim") and not _is_host_leaf(x) for x in st_orig_leaves)
  kw_orig_leaves, _ = jax.tree_util.tree_flatten(kwargs_arrays)
  kw_dev_mask = tuple(hasattr(x, "ndim") and not _is_host_leaf(x) for x in kw_orig_leaves)
  has_spike_wrapper = (
      "loss" in kwargs_arrays
      and kwargs_arrays["loss"] is not None
      and isinstance(opt_state_sub, dict)
      and "inner_state" in opt_state_sub
      and "is_skipped" in opt_state_sub
  )
  kw_inner_keys = tuple(k for k in kwargs_arrays if k not in ("loss", "grad_norm"))

  def _move_masked(tree: Any, mask: tuple[bool, ...], space: Any) -> Any:
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    moved = [jax.device_put(x, space) if m else x for x, m in zip(leaves, mask)]
    return jax.tree_util.tree_unflatten(treedef, moved)

  def _exec_local(p_loc: jax.Array, st_loc: Any, grad_arg: Any, start_dev: jax.Array, kw_loc: dict[str, Any]):
    if not has_spike_wrapper:
      st_h = _move_masked(st_loc, st_dev_mask, jax.memory.Space.Host)
      kw_h = _move_masked(kw_loc, kw_dev_mask, jax.memory.Space.Host)
      if is_sparse:
        ids_loc, dg_loc = grad_arg
        res = host_sparse_fn(
            p_loc,
            st_h,
            (jax.device_put(ids_loc, jax.memory.Space.Host), jax.device_put(dg_loc, jax.memory.Space.Host)),
            jax.device_put(start_dev, jax.memory.Space.Host),
            kw_h,
        )
      else:
        res = host_dense_fn(p_loc, st_h, grad_arg, kw_h)
      st_out = _move_masked(res[1], st_dev_mask, jax.memory.Space.Device)
      if capture_sqnorm:
        return res[0], st_out, jax.device_put(res[2], jax.memory.Space.Device)
      return res[0], st_out

    # Evaluate rolling spike statistics and skip predicate on Device via a 1x1 dummy tensor,
    # then invoke host AdamW without `loss`/`grad_norm` only when the step is accepted.
    g_probe_dtype = grad.grad_dtype if is_sparse else grad.dtype
    st_dummy = {
        k: (
            jax.tree.map(lambda x: jnp.zeros((1, 1), dtype=x.dtype) if _is_host_leaf(x) else x, v)
            if k == "inner_state"
            else jax.tree.map(lambda x: jax.device_put(x, jax.memory.Space.Device) if _is_host_leaf(x) else x, v)
        )
        for k, v in st_loc.items()
    }
    _, st_probe = _run_tx_on_single_leaf(
        tx,
        p_path,
        jnp.zeros((1, 1), dtype=g_probe_dtype),
        st_dummy,
        jnp.zeros((1, 1), dtype=p_loc.dtype),
        **kw_loc,
    )

    def _skip_branch():
      if capture_sqnorm:
        return p_loc, st_loc["inner_state"], _host_array_sqnorm_raw(p_loc)
      return p_loc, st_loc["inner_state"]

    def _apply_branch():
      kw_inner_h = {
          k: (jax.device_put(kw_loc[k], jax.memory.Space.Host) if hasattr(kw_loc[k], "ndim") else kw_loc[k])
          for k in kw_inner_keys
      }
      st_h = _move_masked(st_loc, st_dev_mask, jax.memory.Space.Host)
      if is_sparse:
        ids_loc, dg_loc = grad_arg
        res = host_sparse_fn(
            p_loc,
            st_h,
            (jax.device_put(ids_loc, jax.memory.Space.Host), jax.device_put(dg_loc, jax.memory.Space.Host)),
            jax.device_put(start_dev, jax.memory.Space.Host),
            kw_inner_h,
        )
      else:
        res = host_dense_fn(p_loc, st_h, grad_arg, kw_inner_h)
      st_full_dev = _move_masked(res[1], st_dev_mask, jax.memory.Space.Device)
      if capture_sqnorm:
        return res[0], st_full_dev["inner_state"], jax.device_put(res[2], jax.memory.Space.Device)
      return res[0], st_full_dev["inner_state"]

    cond_out = jax.lax.cond(st_probe["is_skipped"], _skip_branch, _apply_branch)
    st_final_raw = {**st_probe, "inner_state": cond_out[1]}
    st_final_leaves, st_final_treedef = jax.tree_util.tree_flatten(st_final_raw)
    st_final = jax.tree_util.tree_unflatten(
        st_final_treedef,
        [
            jax.device_put(fl, jax.memory.Space.Host)
            if hasattr(ol, "ndim") and _is_host_leaf(ol) and not _is_host_leaf(fl)
            else fl
            for ol, fl in zip(st_orig_leaves, st_final_leaves)
        ],
    )
    if capture_sqnorm:
      return cond_out[0], st_final, cond_out[2]
    return cond_out[0], st_final

  if mesh is not None:
    if _is_explicit_mesh(mesh):
      r_spec, c_spec, row_axes, col_axes = _resolve_2d_specs_and_axes(param, mesh)
    else:
      r_spec, c_spec = _pspec_dim(row_axes), _pspec_dim(col_axes)
    tbl_spec = P(r_spec, c_spec)
    st_spec = jax.tree.map(lambda x: _leaf_pspec(x, mesh, tbl_spec), opt_state_sub)
    kw_spec = jax.tree.map(lambda x: _leaf_pspec(x, mesh), kwargs_arrays)
    sum_axes = (*row_axes, *col_axes)
    out_specs = (tbl_spec, st_spec, P()) if capture_sqnorm else (tbl_spec, st_spec)
    if is_sparse:
      dg_c_spec = grad.resolved_col_spec(mesh) if _is_explicit_mesh(mesh) else c_spec
      rep_ids_spec = P(*([None] * grad.ids.ndim))
      dg_spec = P(*([None] * grad.ids.ndim), dg_c_spec)
      ids_in = _maybe_reshard(grad.ids, rep_ids_spec, mesh)
      dg_in = _maybe_reshard(grad.d_gathered, dg_spec, mesh)

      def _smap_sparse(p_loc, st_loc, ids_loc, dg_loc, kw_loc):
        start = jax.lax.axis_index(row_axes).astype(jnp.int32) * jnp.int32(p_loc.shape[0]) if row_axes else jnp.int32(0)
        loc_out = _exec_local(p_loc, st_loc, (ids_loc, dg_loc), start, kw_loc)
        if capture_sqnorm:
          sq_tot = jax.lax.psum(loc_out[2], sum_axes) if sum_axes else loc_out[2]
          return loc_out[0], loc_out[1], sq_tot
        return loc_out

      out = _call_shard_map(
          _smap_sparse,
          in_specs=(tbl_spec, st_spec, rep_ids_spec, dg_spec, kw_spec),
          out_specs=out_specs,
          args=(param, opt_state_sub, ids_in, dg_in, kwargs_arrays),
          mesh=mesh,
      )
      if capture_sqnorm:
        capture_map[p_path] = out[2]
        return out[0], out[1]
      return out

    g_spec = _leaf_pspec(grad, mesh, tbl_spec)

    def _smap_dense(p_loc, st_loc, g_loc, kw_loc):
      loc_out = _exec_local(p_loc, st_loc, g_loc, jnp.int32(0), kw_loc)
      if capture_sqnorm:
        sq_tot = jax.lax.psum(loc_out[2], sum_axes) if sum_axes else loc_out[2]
        return loc_out[0], loc_out[1], sq_tot
      return loc_out

    out = _call_shard_map(
        _smap_dense,
        in_specs=(tbl_spec, st_spec, g_spec, kw_spec),
        out_specs=out_specs,
        args=(param, opt_state_sub, grad, kwargs_arrays),
        mesh=mesh,
    )
    if capture_sqnorm:
      capture_map[p_path] = out[2]
      return out[0], out[1]
    return out

  loc_out = _exec_local(
      param,
      opt_state_sub,
      (grad.ids, grad.d_gathered) if is_sparse else grad,
      jnp.int32(0),
      kwargs_arrays,
  )
  if capture_sqnorm:
    capture_map[p_path] = loc_out[2]
    return loc_out[0], loc_out[1]
  return loc_out


def _selective_streamed_chunks_update(
    param: jax.Array,
    grad: Any,
    opt_state_sub: Any,
    *,
    tx: optax.GradientTransformation,
    p_path: tuple[Any, ...],
    num_chunks: int,
    row_axes: tuple[str, ...],
    col_axes: tuple[str, ...],
    kwargs_arrays: dict[str, Any],
) -> tuple[jax.Array, Any]:
  """Streams shard-local row chunks through device inside `jax.shard_map` (zero MegaScale collectives)."""
  mesh = _active_mesh(param, grad)
  if mesh is not None and not row_axes and not col_axes:
    row_axes, col_axes = _resolve_2d_axes(param, mesh)
  is_sparse = isinstance(grad, SparseHostGrad)
  capture_map = getattr(_HOST_SQNORM_CAPTURE, "sqnorms", None)
  capture_sqnorm = capture_map is not None

  def _run_local_stream(p_loc, st_loc, grad_arg, shard_start, kw_loc):
    """Slices only the device-local shard `p_loc` (`(n_local, width)`) on host without cross-host communication."""
    n_local, width = p_loc.shape
    k = min(num_chunks, n_local)
    cs = (n_local + k - 1) // k
    k_eff = (n_local + cs - 1) // cs

    st_leaves, st_treedef = jax.tree_util.tree_flatten(st_loc)
    arr_indices = [
        idx
        for idx, lf in enumerate(st_leaves)
        if _is_host_leaf(lf) and hasattr(lf, "shape") and tuple(lf.shape) == (n_local, width)
    ]
    host_arrs_init = (p_loc, *[st_leaves[idx] for idx in arr_indices])

    @_device_host_compute(out_memory_spaces=(jax.memory.Space.Device,) * len(host_arrs_init))
    def _read_chunk(arrs_h, off_h):
      return tuple(jax.lax.dynamic_slice(a, (off_h, 0), (cs, width)) for a in arrs_h)

    @_device_host_compute(out_memory_spaces=(jax.memory.Space.Host,) * len(host_arrs_init))
    def _write_chunk(arrs_h, chunks_h, off_h):
      return tuple(jax.lax.dynamic_update_slice(a, c, (off_h, 0)) for a, c in zip(arrs_h, chunks_h))

    @_device_host_compute(out_memory_spaces=jax.memory.Space.Device)
    def _read_grad_chunk(g_h, off_h):
      return jax.lax.dynamic_slice(g_h, (off_h, 0), (cs, width))

    def _scan_body(carry, i):
      arrs_h, sq_acc = carry
      raw_off = i * jnp.int32(cs)
      slice_off = jnp.minimum(raw_off, jnp.int32(n_local - cs))
      off_h = jax.device_put(slice_off, jax.memory.Space.Host)
      chunks_dev = _read_chunk(arrs_h, off_h)
      p_c = chunks_dev[0]
      st_arr_c = chunks_dev[1:]
      if is_sparse:
        ids_loc, dg_loc = grad_arg
        g_dtype = grad.grad_dtype
        rel = ids_loc - (shard_start + slice_off)
        in_chunk = (rel >= 0) & (rel < cs)
        safe = jnp.where(in_chunk, rel, jnp.int32(0))
        vals = jnp.where(in_chunk[..., None], dg_loc.astype(g_dtype), jnp.zeros((), dtype=g_dtype))
        g_c = jnp.zeros((cs, width), dtype=g_dtype).at[safe].add(vals)
      else:
        g_c = _read_grad_chunk(grad_arg, off_h)

      cur_leaves = list(st_leaves)
      for pos, idx in enumerate(arr_indices):
        cur_leaves[idx] = st_arr_c[pos]
      st_c = jax.tree_util.tree_unflatten(st_treedef, cur_leaves)
      p_new_c, st_new_c = _run_tx_on_single_leaf(tx, p_path, g_c, st_c, p_c, **kw_loc)
      new_leaves, _ = jax.tree_util.tree_flatten(st_new_c)
      new_arr_c = (p_new_c, *[new_leaves[idx] for idx in arr_indices])

      # Mask overlapping rows on the final chunk so every row updates and contributes to sqnorm once.
      active = ((slice_off + jnp.arange(cs, dtype=jnp.int32)) >= raw_off)[:, None]
      masked_chunks = tuple(jnp.where(active, new_c, old_c) for new_c, old_c in zip(new_arr_c, chunks_dev))
      if capture_sqnorm:
        p_active_f32 = jnp.where(active, p_new_c.astype(jnp.float32), jnp.float32(0.0))
        sq_acc = sq_acc + jnp.sum(p_active_f32 * p_active_f32)
      chunks_h = tuple(jax.device_put(c, jax.memory.Space.Host) for c in masked_chunks)
      arrs_h = _write_chunk(arrs_h, chunks_h, off_h)
      scalar_out = tuple(lf for idx, lf in enumerate(new_leaves) if idx not in arr_indices)
      return (arrs_h, sq_acc), scalar_out

    (final_arrs_h, local_sq), scalar_hist = jax.lax.scan(
        _scan_body, (host_arrs_init, jnp.zeros((), dtype=jnp.float32)), jnp.arange(k_eff, dtype=jnp.int32)
    )
    final_scalars = [s[-1] for s in scalar_hist]
    out_leaves = list(st_leaves)
    arr_set = set(arr_indices)
    sc_iter = iter(final_scalars)
    arr_iter = iter(final_arrs_h[1:])
    for idx, _ in enumerate(out_leaves):
      out_leaves[idx] = next(arr_iter) if idx in arr_set else next(sc_iter)
    st_out = jax.tree_util.tree_unflatten(st_treedef, out_leaves)
    if capture_sqnorm:
      return final_arrs_h[0], st_out, local_sq
    return final_arrs_h[0], st_out

  if mesh is not None:
    if _is_explicit_mesh(mesh):
      r_spec, c_spec, row_axes, col_axes = _resolve_2d_specs_and_axes(param, mesh)
    else:
      r_spec, c_spec = _pspec_dim(row_axes), _pspec_dim(col_axes)
    tbl_spec = P(r_spec, c_spec)
    st_spec = jax.tree.map(lambda x: _leaf_pspec(x, mesh, tbl_spec), opt_state_sub)
    kw_spec = jax.tree.map(lambda x: _leaf_pspec(x, mesh), kwargs_arrays)
    sum_axes = (*row_axes, *col_axes)
    out_specs = (tbl_spec, st_spec, P()) if capture_sqnorm else (tbl_spec, st_spec)
    if is_sparse:
      dg_c_spec = grad.resolved_col_spec(mesh) if _is_explicit_mesh(mesh) else c_spec
      rep_ids_spec = P(*([None] * grad.ids.ndim))
      dg_spec = P(*([None] * grad.ids.ndim), dg_c_spec)
      ids_in = _maybe_reshard(grad.ids, rep_ids_spec, mesh)
      dg_in = _maybe_reshard(grad.d_gathered, dg_spec, mesh)

      def _smap_sparse(p_loc, st_loc, ids_loc, dg_loc, kw_loc):
        start = jax.lax.axis_index(row_axes).astype(jnp.int32) * jnp.int32(p_loc.shape[0]) if row_axes else jnp.int32(0)
        res = _run_local_stream(p_loc, st_loc, (ids_loc, dg_loc), start, kw_loc)
        if capture_sqnorm:
          sq_tot = jax.lax.psum(res[2], sum_axes) if sum_axes else res[2]
          return res[0], res[1], sq_tot
        return res

      out = _call_shard_map(
          _smap_sparse,
          in_specs=(tbl_spec, st_spec, rep_ids_spec, dg_spec, kw_spec),
          out_specs=out_specs,
          args=(param, opt_state_sub, ids_in, dg_in, kwargs_arrays),
          mesh=mesh,
      )
      if capture_sqnorm:
        capture_map[p_path] = out[2]
        return out[0], out[1]
      return out

    g_spec = _leaf_pspec(grad, mesh, tbl_spec)

    def _smap_dense(p_loc, st_loc, g_loc, kw_loc):
      start = jax.lax.axis_index(row_axes).astype(jnp.int32) * jnp.int32(p_loc.shape[0]) if row_axes else jnp.int32(0)
      res = _run_local_stream(p_loc, st_loc, g_loc, start, kw_loc)
      if capture_sqnorm:
        sq_tot = jax.lax.psum(res[2], sum_axes) if sum_axes else res[2]
        return res[0], res[1], sq_tot
      return res

    out = _call_shard_map(
        _smap_dense,
        in_specs=(tbl_spec, st_spec, g_spec, kw_spec),
        out_specs=out_specs,
        args=(param, opt_state_sub, grad, kwargs_arrays),
        mesh=mesh,
    )
    if capture_sqnorm:
      capture_map[p_path] = out[2]
      return out[0], out[1]
    return out

  grad_arg = (grad.ids, grad.d_gathered) if is_sparse else grad
  res = _run_local_stream(param, opt_state_sub, grad_arg, jnp.int32(0), kwargs_arrays)
  if capture_sqnorm:
    capture_map[p_path] = res[2]
    return res[0], res[1]
  return res


def _selective_lazy_sparse_update(
    param: jax.Array,
    grad: SparseHostGrad,
    opt_state_sub: Any,
    *,
    tx: optax.GradientTransformation,
    p_path: tuple[Any, ...],
    row_axes: tuple[str, ...],
    col_axes: tuple[str, ...],
    kwargs_arrays: dict[str, Any],
) -> tuple[jax.Array, Any]:
  """Opt-in sparse-row AdamW approximation updating only touched rows (skips decay on untouched rows)."""
  mesh = _active_mesh(param, grad)
  capture_map = getattr(_HOST_SQNORM_CAPTURE, "sqnorms", None)
  capture_sqnorm = capture_map is not None

  def _run_local_sparse(p_loc, st_loc, sparse_loc, shard_start, kw_loc):
    ids_loc, dg_loc = sparse_loc
    n_local, width = p_loc.shape
    g_dtype = grad.grad_dtype
    flat_ids = ids_loc.reshape(-1)
    flat_dg = dg_loc.reshape(-1, width).astype(g_dtype)
    m = flat_ids.shape[0]
    if m == 0 or width == 0:
      if capture_sqnorm:
        return p_loc, st_loc, _host_array_sqnorm_raw(p_loc)
      return p_loc, st_loc
    rel = flat_ids - shard_start
    in_range = (rel >= 0) & (rel < n_local)
    sentinel = jnp.int32(n_local)
    key = jnp.where(in_range, rel, sentinel)
    vals = jnp.where(in_range[:, None], flat_dg, jnp.zeros((), dtype=g_dtype))
    order = jnp.argsort(key)
    sorted_key = key[order]
    sorted_vals = vals[order]
    is_new = jnp.concatenate([jnp.array([True]), sorted_key[1:] != sorted_key[:-1]])
    seg_ids = jnp.cumsum(is_new.astype(jnp.int32)) - 1
    unique_grads = jax.ops.segment_sum(sorted_vals, seg_ids, num_segments=m)
    unique_ids = jnp.full((m,), sentinel, dtype=jnp.int32).at[seg_ids].min(sorted_key)
    valid_row = unique_ids < sentinel
    safe_ids = jnp.where(valid_row, unique_ids, jnp.int32(0))

    st_leaves, st_treedef = jax.tree_util.tree_flatten(st_loc)
    arr_indices = [
        idx
        for idx, lf in enumerate(st_leaves)
        if _is_host_leaf(lf) and hasattr(lf, "shape") and tuple(lf.shape) == (n_local, width)
    ]
    host_arrs_init = (p_loc, *[st_leaves[idx] for idx in arr_indices])

    @_device_host_compute(out_memory_spaces=(jax.memory.Space.Device,) * len(host_arrs_init))
    def _gather_rows(arrs_h, idx_h):
      return tuple(a[idx_h] for a in arrs_h)

    scatter_out_spaces = (
        (*((jax.memory.Space.Host,) * len(host_arrs_init)), jax.memory.Space.Device)
        if capture_sqnorm
        else (jax.memory.Space.Host,) * len(host_arrs_init)
    )

    @_device_host_compute(out_memory_spaces=scatter_out_spaces)
    def _scatter_rows(arrs_h, rows_h, write_idx_h):
      # Out-of-bounds index `n_local` with `mode="drop"` avoids clobbering row 0.
      updated = tuple(a.at[write_idx_h].set(r, mode="drop") for a, r in zip(arrs_h, rows_h))
      if capture_sqnorm:
        p32 = updated[0].astype(jnp.float32)
        return (*updated, jnp.sum(p32 * p32))
      return updated

    rows_dev = _gather_rows(host_arrs_init, jax.device_put(safe_ids, jax.memory.Space.Host))
    cur_leaves = list(st_leaves)
    for pos, idx in enumerate(arr_indices):
      cur_leaves[idx] = rows_dev[pos + 1]
    st_rows = jax.tree_util.tree_unflatten(st_treedef, cur_leaves)
    p_rows_new, st_rows_new = _run_tx_on_single_leaf(tx, p_path, unique_grads, st_rows, rows_dev[0], **kw_loc)
    new_leaves, _ = jax.tree_util.tree_flatten(st_rows_new)
    new_rows_h = tuple(
        jax.device_put(r, jax.memory.Space.Host) for r in (p_rows_new, *[new_leaves[idx] for idx in arr_indices])
    )
    write_idx = jnp.where(valid_row, unique_ids, sentinel)
    scatter_res = _scatter_rows(host_arrs_init, new_rows_h, jax.device_put(write_idx, jax.memory.Space.Host))
    updated_arrs_h = scatter_res[:-1] if capture_sqnorm else scatter_res

    out_leaves = list(new_leaves)
    for pos, idx in enumerate(arr_indices):
      out_leaves[idx] = updated_arrs_h[pos + 1]
    st_out = jax.tree_util.tree_unflatten(st_treedef, out_leaves)
    if capture_sqnorm:
      return updated_arrs_h[0], st_out, scatter_res[-1]
    return updated_arrs_h[0], st_out

  if mesh is not None:
    if _is_explicit_mesh(mesh):
      r_spec, c_spec, row_axes, col_axes = _resolve_2d_specs_and_axes(param, mesh)
      dg_c_spec = grad.resolved_col_spec(mesh)
    else:
      r_spec, c_spec = _pspec_dim(row_axes), _pspec_dim(col_axes)
      dg_c_spec = c_spec
    tbl_spec = P(r_spec, c_spec)
    st_spec = jax.tree.map(lambda x: _leaf_pspec(x, mesh, tbl_spec), opt_state_sub)
    rep_ids_spec = P(*([None] * grad.ids.ndim))
    dg_spec = P(*([None] * grad.ids.ndim), dg_c_spec)
    kw_spec = jax.tree.map(lambda x: _leaf_pspec(x, mesh), kwargs_arrays)
    sum_axes = (*row_axes, *col_axes)
    out_specs = (tbl_spec, st_spec, P()) if capture_sqnorm else (tbl_spec, st_spec)
    ids_in = _maybe_reshard(grad.ids, rep_ids_spec, mesh)
    dg_in = _maybe_reshard(grad.d_gathered, dg_spec, mesh)

    def _smap(p_loc, st_loc, ids_loc, dg_loc, kw_loc):
      start = jax.lax.axis_index(row_axes).astype(jnp.int32) * jnp.int32(p_loc.shape[0]) if row_axes else jnp.int32(0)
      res = _run_local_sparse(p_loc, st_loc, (ids_loc, dg_loc), start, kw_loc)
      if capture_sqnorm:
        sq_tot = jax.lax.psum(res[2], sum_axes) if sum_axes else res[2]
        return res[0], res[1], sq_tot
      return res

    out = _call_shard_map(
        _smap,
        in_specs=(tbl_spec, st_spec, rep_ids_spec, dg_spec, kw_spec),
        out_specs=out_specs,
        args=(param, opt_state_sub, ids_in, dg_in, kwargs_arrays),
        mesh=mesh,
    )
    if capture_sqnorm:
      capture_map[p_path] = out[2]
      return out[0], out[1]
    return out

  res = _run_local_sparse(param, opt_state_sub, (grad.ids, grad.d_gathered), jnp.int32(0), kwargs_arrays)
  if capture_sqnorm:
    capture_map[p_path] = res[2]
    return res[0], res[1]
  return res


class HostOffloadedOptimizer(nnx.Optimizer):
  """An `nnx.Optimizer` parking all or selected `opt_state`/params in `pinned_host` between steps."""

  def __init__(
      self,
      model: nnx.Module,
      tx: optax.GradientTransformation,
      wrt: Any = nnx.Param,
      *,
      offload_opt_state: bool = True,
      selected_param_paths: frozenset[tuple[Any, ...]] = frozenset(),
      optimizer_mode: str = "host_compute",
      num_chunks: int = 16,
      clip_max_norm: float = 0.0,
  ):
    if optimizer_mode not in _OPTIMIZER_MODES:
      raise ValueError(f"Unsupported optimizer_mode {optimizer_mode!r}; expected one of {sorted(_OPTIMIZER_MODES)}.")
    self._offload_opt_state = offload_opt_state
    self._selected_param_paths = frozenset(tuple(p) for p in selected_param_paths)
    self._optimizer_mode = optimizer_mode
    self._num_chunks = int(num_chunks)
    self._clip_max_norm = float(clip_max_norm)

    if self._selected_param_paths:
      params_for_init = nnx.state(model, wrt)
      all_init_paths = frozenset(tuple(p) for p, _ in nnx.to_flat_state(params_for_init))
      selected_param_map: dict[tuple[Any, ...], nnx.Variable] = {}
      for p, v in nnx.to_flat_state(params_for_init):
        pt = tuple(p)
        if pt in self._selected_param_paths and isinstance(v, nnx.Variable):
          selected_param_map[pt] = v
      selected_param_vars = list(selected_param_map.values())
      concrete_selected = bool(selected_param_vars) and _can_move_vars(selected_param_vars)
      saved_param_state: list[tuple[nnx.Variable, Any, dict[str, Any]]] = []
      if concrete_selected:
        # Substitute tiny (1, 1) unsharded device placeholders during `tx.init` so concrete
        # initialization neither stages full tables/moments onto HBM nor invokes eager
        # `jnp.zeros_like` on `pinned_host` buffers (which fails TPU PjRt `lax.full`).
        for v in selected_param_vars:
          orig_raw = v.get_raw_value()
          orig_meta = dict(v.get_metadata())
          saved_param_state.append((v, orig_raw, orig_meta))
          tmp_meta = {**orig_meta, "eager_sharding": False}
          object.__setattr__(v, "_var_metadata", tmp_meta)
          object.__setattr__(v, "_raw_value", jnp.zeros((1, 1), dtype=orig_raw.dtype))
      try:
        super().__init__(model, tx, wrt=wrt)
      finally:
        for v, orig_raw, orig_meta in saved_param_state:
          object.__setattr__(v, "_var_metadata", orig_meta)
          object.__setattr__(v, "_raw_value", orig_raw)

      def _transform_opt_leaf(path: tuple[Any, ...], leaf: Any) -> Any:
        if not isinstance(leaf, nnx.Variable):
          return leaf
        raw = leaf.get_raw_value()
        if not hasattr(raw, "ndim") or raw.ndim == 0:
          return leaf
        key_tuple = tuple(_key_value(k) for k in path)
        matched_path = _is_matching_opt_leaf(key_tuple, self._selected_param_paths, all_init_paths)
        if matched_path is not None:
          extra_meta: dict[str, Any] = {"memory_kind": "pinned_host"}
          new_raw = raw
          if concrete_selected and matched_path in selected_param_map:
            p_var = selected_param_map[matched_path]
            orig_eager = p_var.get_metadata("eager_sharding", default=nnx.using_eager_sharding())
            extra_meta["eager_sharding"] = bool(orig_eager)
            out_pspec = leaf.get_metadata("out_sharding", default=None)
            new_raw = _alloc_host_zeros_for_var(p_var, raw.dtype, out_pspec=out_pspec)
          elif _can_move_vars([leaf]):
            new_raw = _move_raw_value(raw, "pinned_host", jax.memory.Space.Host)
          elif isinstance(raw, jax.core.Tracer) and not _is_host_leaf(leaf):  # jmp-waiver: TRACER_GUARD
            new_raw = jax.device_put(raw, jax.memory.Space.Host)
          return _to_memory_kind_variable(leaf, "pinned_host", raw_value=new_raw, extra_metadata=extra_meta)
        if self._offload_opt_state and _is_matching_opt_leaf(key_tuple, all_init_paths, all_init_paths) is not None:
          extra_meta = {"memory_kind": "pinned_host"}
          new_raw = raw
          if _can_move_vars([leaf]):
            new_raw = _move_raw_value(raw, "pinned_host", jax.memory.Space.Host)
          elif isinstance(raw, jax.core.Tracer) and not _is_host_leaf(leaf):  # jmp-waiver: TRACER_GUARD
            new_raw = jax.device_put(raw, jax.memory.Space.Host)
          return _to_memory_kind_variable(leaf, "pinned_host", raw_value=new_raw, extra_metadata=extra_meta)
        return _to_memory_kind_variable(leaf, "device")

      self.opt_state = nnx.data(
          jax.tree_util.tree_map_with_path(
              _transform_opt_leaf,
              self.opt_state,
              is_leaf=lambda x: isinstance(x, nnx.Variable),
          )
      )
      return

    target_kind = "pinned_host" if self._offload_opt_state else "device"
    param_vars = [
        v
        for v in jax.tree.leaves(nnx.state(model, wrt), is_leaf=lambda x: isinstance(x, nnx.Variable))
        if isinstance(v, nnx.Variable)
    ]
    has_concrete_arrays = bool(param_vars) and all(
        not isinstance(v.get_raw_value(), jax.core.Tracer)  # jmp-waiver: TRACER_GUARD
        and isinstance(v.get_raw_value(), jax.Array)
        for v in param_vars
    )
    if has_concrete_arrays:

      def _jit_init(ps: Any) -> Any:
        raw_opt = _init_opt_state_for_kind(ps, tx, target_kind)
        for leaf in jax.tree.leaves(raw_opt, is_leaf=lambda x: isinstance(x, nnx.Variable)):
          if isinstance(leaf, nnx.Variable):
            leaf.set_metadata("_offload_saved_eager", bool(leaf.eager_sharding))
            leaf.set_metadata("eager_sharding", False)
        return raw_opt

      super().__init__(model, optax.GradientTransformation(_jit_init, tx.update), wrt=wrt)
      self.tx = tx

      def _wrap_concrete_opt_leaf(leaf: Any) -> Any:
        if not isinstance(leaf, nnx.Variable):
          return leaf
        extra: dict[str, Any] = {}
        if isinstance(leaf, nnx.training.optimizer.OptVariable):
          saved = leaf.get_metadata("_offload_saved_eager", default=True)
          extra["eager_sharding"] = bool(saved)
        return _to_memory_kind_variable(leaf, target_kind, extra_metadata=extra)

      self.opt_state = nnx.data(
          jax.tree.map(_wrap_concrete_opt_leaf, self.opt_state, is_leaf=lambda x: isinstance(x, nnx.Variable))
      )
      return

    restore_host_params = any(_is_host_leaf(v) for v in param_vars)
    saved_raw = [v.get_raw_value() for v in param_vars] if restore_host_params else []
    saved_kinds = [v.get_metadata("memory_kind", default="device") for v in param_vars] if restore_host_params else []
    if restore_host_params:
      for v in param_vars:
        v.set_metadata("memory_kind", "device")
        raw = v.get_raw_value()
        if (
            not isinstance(raw, jax.core.Tracer)  # jmp-waiver: TRACER_GUARD
            or getattr(getattr(raw, "aval", None), "memory_space", None) == jax.memory.Space.Host
        ):
          v.set_raw_value(_move_raw_value(raw, "device", jax.memory.Space.Device))
    try:
      super().__init__(model, tx, wrt=wrt)
    finally:
      if restore_host_params:
        for v, raw, kind in zip(param_vars, saved_raw, saved_kinds):
          v.set_metadata("memory_kind", kind)
          v.set_raw_value(raw)
    target_space = _validate_kind(target_kind)
    self.opt_state = nnx.data(
        jax.tree.map(
            lambda leaf: (
                _to_memory_kind_variable(
                    leaf,
                    target_kind,
                    raw_value=_move_raw_value(leaf.get_raw_value(), target_kind, target_space),
                )
                if isinstance(leaf, nnx.Variable)
                else leaf
            ),
            self.opt_state,
            is_leaf=lambda x: isinstance(x, nnx.Variable),
        )
    )

  def _update_selective(self, model: nnx.Module, grads: Any, **kwargs: Any) -> None:
    """Updates device and selectively host-offloaded parameters in place."""
    params_state = nnx.state(model, self.wrt, graph=self.graph)
    if self._clip_max_norm > 0.0:
      grads = clip_grads_by_global_norm(
          grads, self._clip_max_norm, shardings=None if isinstance(grads, nnx.Module) else params_state
      )
    param_var_map = {tuple(p): v for p, v in nnx.to_flat_state(params_state)}
    all_param_paths = frozenset(param_var_map.keys())
    selected = self._selected_param_paths & all_param_paths

    grads_state = nnx.state(grads, self.wrt, graph=self.graph) if isinstance(grads, nnx.Module) else grads
    grad_val_map = {
        tuple(p): (v.get_raw_value() if isinstance(v, nnx.Variable) else v) for p, v in nnx.to_flat_state(grads_state)
    }
    opt_state_pure = nnx.as_pure(self.opt_state)
    kwargs_arrays = nnx.as_pure(kwargs)

    # 1. Update non-selected parameters on device.
    dev_paths = all_param_paths - selected
    updated_dev_opt_state: Any = None
    if dev_paths:
      dev_params_pure = _filter_mapping_by_paths(nnx.as_pure(params_state), lambda p: p in dev_paths)
      dev_grads_pure = _filter_mapping_by_paths(nnx.as_pure(grads_state), lambda p: p in dev_paths)
      dev_opt_in = _slice_opt_state(opt_state_pure, all_param_paths, lambda p: p in dev_paths)
      dev_opt_in_leaves, _ = jax.tree_util.tree_flatten(dev_opt_in)
      dev_opt_host_mask = tuple(hasattr(x, "ndim") and _is_host_leaf(x) for x in dev_opt_in_leaves)
      if any(dev_opt_host_mask):
        dev_opt_in = to_memory(dev_opt_in, "device")
      dev_updates, updated_dev_opt_state = self.tx.update(dev_grads_pure, dev_opt_in, dev_params_pure, **kwargs_arrays)
      new_dev_params = optax.apply_updates(dev_params_pure, dev_updates)
      for p_path in dev_paths:
        param_var_map[p_path].set_raw_value(_unnest_path(p_path, new_dev_params))
      if any(dev_opt_host_mask):
        upd_leaves, upd_treedef = jax.tree_util.tree_flatten(updated_dev_opt_state)
        updated_dev_opt_state = jax.tree_util.tree_unflatten(
            upd_treedef,
            [jax.device_put(u, jax.memory.Space.Host) if m else u for u, m in zip(upd_leaves, dev_opt_host_mask)],
        )

    # 2. Update each selectively host-offloaded parameter without materializing dense HBM gradients.
    updated_host_opt_states: dict[tuple[Any, ...], Any] = {}
    for p_path in sorted(selected):
      p_var = param_var_map[p_path]
      p_val = p_var.get_raw_value()
      g_val = grad_val_map[p_path]
      st_sub = _slice_opt_state(opt_state_pure, all_param_paths, lambda p, target=p_path: p == target)
      mesh = _active_mesh(p_var, p_val, g_val)
      row_axes, col_axes = _resolve_2d_axes(p_var, mesh)

      if self._optimizer_mode == "streamed_chunks":
        p_new, st_new = _selective_streamed_chunks_update(
            p_val,
            g_val,
            st_sub,
            tx=self.tx,
            p_path=p_path,
            num_chunks=self._num_chunks,
            row_axes=row_axes,
            col_axes=col_axes,
            kwargs_arrays=kwargs_arrays,
        )
      elif self._optimizer_mode == "lazy_sparse":
        if not isinstance(g_val, SparseHostGrad):
          raise ValueError(
              f"selective_host_offload_optimizer_mode='lazy_sparse' requires SparseHostGrad for {p_path}, "
              f"got dense {type(g_val).__name__}. Use 'host_compute' or 'streamed_chunks' with dense host "
              "gradients (such as MaxTextTrainingEngine)."
          )
        p_new, st_new = _selective_lazy_sparse_update(
            p_val,
            g_val,
            st_sub,
            tx=self.tx,
            p_path=p_path,
            row_axes=row_axes,
            col_axes=col_axes,
            kwargs_arrays=kwargs_arrays,
        )
      else:
        p_new, st_new = _selective_host_compute_update(
            p_val,
            g_val,
            st_sub,
            tx=self.tx,
            p_path=p_path,
            row_axes=row_axes,
            col_axes=col_axes,
            kwargs_arrays=kwargs_arrays,
        )
      p_var.set_raw_value(p_new)
      updated_host_opt_states[p_path] = st_new

    # 3. Write updated optimizer state back into existing Variables in place.
    fallback_scalar_state = (
        updated_dev_opt_state if updated_dev_opt_state is not None else next(iter(updated_host_opt_states.values()))
    )
    for path, leaf in jax.tree_util.tree_leaves_with_path(self.opt_state, is_leaf=lambda x: isinstance(x, nnx.Variable)):
      if not isinstance(leaf, nnx.Variable):
        continue
      key_tuple = tuple(_key_value(k) for k in path)
      matched_p = _is_matching_opt_leaf(key_tuple, selected, all_param_paths)
      if matched_p is not None:
        leaf.set_raw_value(_lookup_path_in_tree(updated_host_opt_states[matched_p], key_tuple))
      else:
        leaf.set_raw_value(_lookup_path_in_tree(fallback_scalar_state, key_tuple))
    self.step[...] += 1

  def update(self, model: nnx.Module, grads: Any, **kwargs: Any) -> Any:  # pylint: disable=arguments-differ
    """Performs an optimizer step, handling full or selective pinned-host offload."""
    if self._selected_param_paths:
      return self._update_selective(model, grads, **kwargs)
    if self._offload_opt_state:
      _move_variables_in_place(self.opt_state, "device")
    params = nnx.state(model, self.wrt)
    offload_params = _has_host_leaves(params)
    if offload_params:
      _move_variables_in_place(params, "device")
    if offload_params or _has_host_leaves(grads):
      grads = to_memory(grads, "device")
    super().update(model, grads, **kwargs)
    if offload_params:
      _move_variables_in_place(nnx.state(model, self.wrt), "pinned_host")
    if self._offload_opt_state:
      _move_variables_in_place(self.opt_state, "pinned_host")
    return None


def make_optimizer(
    model: nnx.Module,
    tx: optax.GradientTransformation,
    config: Any,
    wrt: Any = nnx.Param,
    *,
    clip_max_norm: float = 0.0,
    dense_host_grads: bool = False,
) -> nnx.Optimizer:
  """Builds the optimizer, using `HostOffloadedOptimizer` when any host offload mode is active."""
  offload_opt = bool(getattr(config, "optimizer_memory_host_offload", False))  # jmp-waiver: SILENT_CONFIG_FALLBACK
  selective = is_selective_host_offload_enabled(config) or has_selective_host_offload(model)
  if selective:
    if getattr(config, "opt_type", None) == "muon":  # jmp-waiver: SILENT_CONFIG_FALLBACK
      raise ValueError("selective_host_offload=True is not supported with opt_type='muon'.")
    mode = _get_optimizer_mode(config)
    if dense_host_grads and mode == "lazy_sparse":
      raise ValueError(
          "selective_host_offload_optimizer_mode='lazy_sparse' is incompatible with dense pinned_host "
          "gradients (MaxTextTrainingEngine); use 'host_compute' or 'streamed_chunks'."
      )
    selected_paths = selected_host_param_paths(model)
    if not selected_paths:
      patterns = get_host_offload_param_patterns(config)
      selected_paths = _match_param_paths(nnx.state(model, wrt), patterns)
    return HostOffloadedOptimizer(
        model,
        tx,
        wrt=wrt,
        offload_opt_state=offload_opt,
        selected_param_paths=selected_paths,
        optimizer_mode=mode,
        num_chunks=_get_num_chunks(config),
        clip_max_norm=clip_max_norm,
    )
  offload_params = bool(  # jmp-waiver: SILENT_CONFIG_FALLBACK
      getattr(config, "parameter_memory_host_offload", False)  # jmp-waiver: SILENT_CONFIG_FALLBACK
  ) or _has_host_leaves(nnx.state(model, wrt))
  if offload_opt or offload_params:
    return HostOffloadedOptimizer(model, tx, wrt=wrt, offload_opt_state=offload_opt)
  return nnx.Optimizer(model, tx, wrt=wrt)


def get_device_hbm_stats(device: Any = None) -> dict[str, float] | None:
  """Queries live `device.memory_stats()` including reserved XLA HLO workspace.

  Returns `None` on backends (such as CPU or non-addressable Pathways IFRT devices)
  where `memory_stats()` returns `None`, raises `RuntimeError`, or omits
  `bytes_in_use` / `peak_bytes_in_use`.
  """
  if device is None:
    local_devices = jax.local_devices()
    if not local_devices:
      return None
    device = local_devices[0]
  stats_fn = getattr(device, "memory_stats", None)
  if not callable(stats_fn):
    return None
  try:
    stats = stats_fn()  # pylint: disable=not-callable
  except RuntimeError:
    return None
  if not stats or "bytes_in_use" not in stats or "peak_bytes_in_use" not in stats:
    return None
  bytes_in_use = int(stats["bytes_in_use"])
  bytes_reserved = int(stats.get("bytes_reserved", 0))
  allocated_bytes = bytes_in_use + bytes_reserved
  peak_bytes_in_use = int(stats["peak_bytes_in_use"])
  peak_allocated_bytes = max(allocated_bytes, peak_bytes_in_use, int(stats.get("peak_allocated_bytes", 0)))
  return {
      "bytes_in_use": float(bytes_in_use),
      "bytes_reserved": float(bytes_reserved),
      "allocated_bytes": float(allocated_bytes),
      "peak_bytes_in_use": float(peak_bytes_in_use),
      "peak_allocated_bytes": float(peak_allocated_bytes),
      "hbm_allocated_gib": allocated_bytes / _GIB,
      "hbm_peak_in_use_gib": peak_bytes_in_use / _GIB,
      "hbm_peak_allocated_gib": peak_allocated_bytes / _GIB,
  }
