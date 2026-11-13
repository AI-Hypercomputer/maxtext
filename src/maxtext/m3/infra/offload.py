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
`nnx.Param` state to `"pinned_host"` and gathers parameters to `"device"` for
forward and backward execution via a dynamic subclass mixin without adding
wrapper segments to module paths.

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

from typing import Any

from flax import nnx
import jax
import numpy as np
import optax

_GIB = float(2**30)
_HOST_KINDS = frozenset({"pinned_host", "unpinned_host"})
_SPACES = {
    "pinned_host": jax.memory.Space.Host,
    "device": jax.memory.Space.Device,
}
_OFFLOADED_MODEL_CLASSES: dict[type[nnx.Module], type[nnx.Module]] = {}
_HOST_VARIABLE_CLASSES: dict[type[nnx.Variable], type[nnx.Variable]] = {}


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
    if (
        eager_sharding
        and isinstance(mesh, jax.sharding.Mesh)
        and not mesh.empty
        and isinstance(out_sh, jax.sharding.PartitionSpec)
    ):
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
      # reshard `pinned_host` values onto `Space.Device` during `replace_by_pure_dict`.
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
    """Rebuilds a variable while preserving its `memory_kind` placement."""
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


class HostOffloadedParam(_HostMemoryKindVariableMixin, nnx.Param):
  """`nnx.Param` subclass preserving host `memory_kind` across `eval_shape`."""

  __slots__ = ()


class HostOffloadedOptVariable(_HostMemoryKindVariableMixin, nnx.OptVariable):
  """`nnx.OptVariable` subclass preserving `memory_kind` across `eval_shape`."""

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


_KNOWN_HOST_VAR_CLASSES: dict[type[nnx.Variable], type[nnx.Variable]] = {
    nnx.Param: HostOffloadedParam,
    nnx.OptVariable: HostOffloadedOptVariable,
    nnx.training.optimizer.OptArray: HostOffloadedOptArray,
    nnx.OptState: HostOffloadedOptState,
    nnx.Variable: HostOffloadedVariable,
}


def _host_variable_class(cls: type[nnx.Variable]) -> type[nnx.Variable]:
  """Returns a cached `_HostMemoryKindVariableMixin` subclass of `cls`."""
  if issubclass(cls, _HostMemoryKindVariableMixin):
    return cls
  known = _KNOWN_HOST_VAR_CLASSES.get(cls)
  if known is not None:
    return known
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
  """Constructs a `_HostMemoryKindVariableMixin` variable in `kind` without mutating `__class__`."""
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
  leaves = jax.tree.leaves(tree, is_leaf=lambda x: isinstance(x, nnx.Variable))
  return any(_is_host_leaf(leaf) for leaf in leaves)


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
      param_vars = [
          v
          for v in jax.tree.leaves(nnx.state(self, nnx.Param), is_leaf=lambda x: isinstance(x, nnx.Variable))
          if isinstance(v, nnx.Variable)
      ]
      if not any(_is_host_leaf(v) for v in param_vars):
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


def maybe_offload_parameters(model: nnx.Module, config: Any) -> nnx.Module:
  """Moves `nnx.Param` state to `pinned_host` when `config.parameter_memory_host_offload` is set."""
  if not getattr(config, "parameter_memory_host_offload", False):  # jmp-waiver: SILENT_CONFIG_FALLBACK
    return model
  params = nnx.state(model, nnx.Param)
  param_leaves = [
      v for v in jax.tree.leaves(params, is_leaf=lambda x: isinstance(x, nnx.Variable)) if isinstance(v, nnx.Variable)
  ]
  raw_leaves = [v.get_raw_value() for v in param_leaves]
  should_move = bool(raw_leaves) and not any(
      isinstance(v, jax.core.Tracer)  # jmp-waiver: TRACER_GUARD
      or (isinstance(v, jax.ShapeDtypeStruct) and getattr(v, "sharding", None) is None)
      for v in raw_leaves
  )
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


class HostOffloadedOptimizer(nnx.Optimizer):
  """An `nnx.Optimizer` parking `opt_state` (and host-offloaded params) in `pinned_host` between steps."""

  def __init__(
      self,
      model: nnx.Module,
      tx: optax.GradientTransformation,
      wrt: Any = nnx.Param,
      *,
      offload_opt_state: bool = True,
  ):
    self._offload_opt_state = offload_opt_state
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

  def update(self, model: nnx.Module, grads: Any, **kwargs: Any) -> None:  # pylint: disable=arguments-differ
    """Performs an optimizer step, gathering host-offloaded state to device."""
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


def make_optimizer(
    model: nnx.Module,
    tx: optax.GradientTransformation,
    config: Any,
    wrt: Any = nnx.Param,
) -> nnx.Optimizer:
  """Builds the optimizer, using `HostOffloadedOptimizer` when optimizer or parameter offload is active."""
  offload_opt = bool(getattr(config, "optimizer_memory_host_offload", False))  # jmp-waiver: SILENT_CONFIG_FALLBACK
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
