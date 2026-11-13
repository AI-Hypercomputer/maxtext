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


def _apply_memory_kind_hook(var: nnx.Variable, value: Any) -> Any:
  """Preserves `memory_kind` metadata when `nnx.eval_shape` assigns abstract shardings."""
  kind = var.get_metadata("memory_kind", default=None)
  if not kind:
    return value
  if isinstance(value, jax.ShapeDtypeStruct):
    sharding = getattr(value, "sharding", None)
    if sharding is not None and getattr(sharding, "memory_kind", None) != kind:
      return jax.ShapeDtypeStruct(
          value.shape,
          value.dtype,
          sharding=sharding.with_memory_kind(kind),
      )
  return value


def _attach_memory_kind_metadata(var: nnx.Variable, kind: str) -> None:
  """Records `memory_kind` on `var` and registers `_apply_memory_kind_hook` for `nnx.eval_shape`."""
  var.set_metadata("memory_kind", kind)
  hooks = tuple(var.get_metadata("on_set_value", default=()) or ())
  if _apply_memory_kind_hook not in hooks:
    var.set_metadata("on_set_value", (*hooks, _apply_memory_kind_hook))


def _is_host_leaf(x: Any) -> bool:
  """Returns True when `x` (or its wrapped value) resides in host memory."""
  val = x.get_raw_value() if isinstance(x, nnx.Variable) else x
  if isinstance(val, jax.core.Tracer):
    aval = getattr(val, "aval", None)
    return getattr(aval, "memory_space", None) == jax.memory.Space.Host
  sharding = getattr(val, "sharding", None)
  return getattr(sharding, "memory_kind", None) in _HOST_KINDS


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
  if isinstance(x, jax.core.Tracer):
    return jax.device_put(x, target_space)
  if isinstance(x, jax.ShapeDtypeStruct):
    sharding = getattr(x, "sharding", None)
    if sharding is not None:
      return jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sharding.with_memory_kind(kind))
    return x
  if isinstance(x, jax.Array):
    sharding = getattr(x, "sharding", None)
    if sharding is not None:
      return jax.device_put(x, sharding.with_memory_kind(kind))
    return jax.device_put(x, target_space)
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
      moved = x.replace(_move_raw_value(x.get_raw_value(), kind, target_space))
      _attach_memory_kind_metadata(moved, kind)
      return moved
    return _move_raw_value(x, kind, target_space)

  return jax.tree.map(move, tree, is_leaf=lambda x: isinstance(x, nnx.Variable))


def _move_variables_in_place(tree: Any, kind: str) -> None:
  """Moves `nnx.Variable` values in `tree` to `kind` without replacing Variable instances."""
  target_space = _validate_kind(kind)
  for leaf in jax.tree.leaves(tree, is_leaf=lambda x: isinstance(x, nnx.Variable)):
    if isinstance(leaf, nnx.Variable):
      _attach_memory_kind_metadata(leaf, kind)
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
      for v in param_vars:
        v.set_raw_value(_move_raw_value(v.get_raw_value(), "device", jax.memory.Space.Device))
      try:
        return fn(self, *args, **kwargs)
      finally:
        for v, raw in zip(param_vars, saved_raw):
          v.set_raw_value(raw)

    def _call(self: Any, *args: Any, **kwargs: Any) -> Any:
      return _with_device_params(self, parent_call, *args, **kwargs)

    def _vocab_tiling(self: Any, *args: Any, **kwargs: Any) -> Any:
      if parent_vocab_tiling is None:
        raise AttributeError(f"{cls.__name__} has no 'logits_from_hidden_states_for_vocab_tiling' method.")
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


def maybe_offload_parameters(model: nnx.Module, config: Any) -> nnx.Module:
  """Moves `nnx.Param` state to `pinned_host` when `config.parameter_memory_host_offload` is set."""
  if not getattr(config, "parameter_memory_host_offload", False):
    return model
  params = nnx.state(model, nnx.Param)
  param_leaves = jax.tree.leaves(params, is_leaf=lambda x: isinstance(x, nnx.Variable))
  raw_leaves = [v.get_raw_value() if isinstance(v, nnx.Variable) else v for v in param_leaves]
  # Inside `nnx.eval_shape`, attach the dispatch subclass without moving abstract leaves prematurely.
  should_move = bool(raw_leaves) and not any(
      isinstance(v, jax.core.Tracer) or (isinstance(v, jax.ShapeDtypeStruct) and getattr(v, "sharding", None) is None)
      for v in raw_leaves
  )
  if should_move:
    _move_variables_in_place(params, "pinned_host")
  else:
    for v in param_leaves:
      if isinstance(v, nnx.Variable):
        _attach_memory_kind_metadata(v, "pinned_host")
  if not isinstance(model, HostOffloadedParameters):
    object.__setattr__(model, "__class__", _offloaded_model_class(type(model)))
  return model


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
    param_vars = [
        v
        for v in jax.tree.leaves(nnx.state(model, wrt), is_leaf=lambda x: isinstance(x, nnx.Variable))
        if isinstance(v, nnx.Variable)
    ]
    restore_host_params = any(_is_host_leaf(v) for v in param_vars)
    saved_raw = [v.get_raw_value() for v in param_vars] if restore_host_params else []
    if restore_host_params:
      _move_variables_in_place(param_vars, "device")
    try:
      super().__init__(model, tx, wrt=wrt)
    finally:
      if restore_host_params:
        for v, raw in zip(param_vars, saved_raw):
          _attach_memory_kind_metadata(v, "pinned_host")
          v.set_raw_value(raw)
    if self._offload_opt_state:
      _move_variables_in_place(self.opt_state, "pinned_host")

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
  offload_opt = bool(getattr(config, "optimizer_memory_host_offload", False))
  offload_params = bool(getattr(config, "parameter_memory_host_offload", False)) or _has_host_leaves(
      nnx.state(model, wrt)
  )
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
  peak_allocated_bytes = max(allocated_bytes, peak_bytes_in_use)
  return {
      "bytes_in_use": float(bytes_in_use),
      "bytes_reserved": float(bytes_reserved),
      "allocated_bytes": float(allocated_bytes),
      "peak_bytes_in_use": float(peak_bytes_in_use),
      "peak_allocated_bytes": float(peak_allocated_bytes),
      "hbm_allocated_gib": allocated_bytes / _GIB,
      "hbm_peak_in_use_gib": peak_bytes_in_use / _GIB,
  }
