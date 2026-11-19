# Copyright 2023–2026 Google LLC
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

"""The NNX Unified TrainState."""

import math
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding

from maxtext.utils import max_logging


class TrainStateNNX(nnx.Module):
  """A unified container for NNX models and optimizers.

  This replaces Linen's TrainState for checkpointing.

  Linen TrainState pytree:
    {"params": {...}, "opt_state": {}...}
  TrainStateNNX state pytree:
    {"model": {...}, "optimizer": {"opt_state": {...}}}
  """

  def __init__(
      self,
      model: nnx.Module,
      optimizer: nnx.Optimizer | None,
  ):
    self.model = model
    self.optimizer = optimizer

  def apply_gradients(self, grads: Any, **kwargs):
    """Mimics the Linen apply_gradients function.

    Updates the optimizer state, applies updates to parameters, and increments
    the step counter. Only updates `self.model`. Extra kwargs (e.g. loss/grad_norm
    for the skip-step-on-spikes optimizer) are forwarded to the optax update.
    """
    if self.optimizer is None:
      raise RuntimeError(
          "Cannot call apply_gradients on a TrainStateNNX initialized without"
          " an optimizer. This usually happens when the state was created for"
          " inference only."
      )
    self.optimizer.update(self.model, grads, **kwargs)


# Residual16 master-weight storage (`fp32_master_storage="residual16"`).
#
# Working weights in `model` stay in `bfloat16` (2 bytes/param), while `Residual16Optimizer`
# stores signed `int16` low-16-bit residuals in a sibling `self.master_residuals` `OptState`
# attribute (`state["optimizer"]["master_residuals"]`), keeping `self.opt_state` structurally
# identical to `nnx.Optimizer.opt_state`.

_RECONSTRUCT_SAVE_CHUNK_BYTES = 256 * 1024 * 1024


def _typeof_spec(x: Any) -> jax.sharding.PartitionSpec | None:
  """Safely extracts the PartitionSpec from `jax.typeof(x).sharding` when available."""
  try:
    aval = jax.typeof(x)
  except TypeError:
    aval = None
  return getattr(getattr(aval, "sharding", None), "spec", None)


def _key_tuple(path: tuple[Any, ...]) -> tuple[Any, ...]:
  """Normalizes a JAX tree path or Flax flat-state path into raw key primitives."""
  out = []
  for entry in path:
    if hasattr(entry, "key"):
      out.append(entry.key)
    elif hasattr(entry, "idx"):
      out.append(entry.idx)
    elif hasattr(entry, "name"):
      out.append(entry.name)
    else:
      out.append(entry)
  return tuple(out)


def _flatten_leaves_by_key_tuple(tree: Any) -> dict[tuple[Any, ...], Any]:
  """Flattens a nested dict / nnx.State tree into `{key_tuple: leaf}`."""
  if not tree:
    return {}
  return {
      _key_tuple(p): leaf
      for p, leaf in jax.tree_util.tree_leaves_with_path(
          tree,
          is_leaf=lambda x: isinstance(x, (jax.Array, jax.ShapeDtypeStruct, nnx.Variable)),
      )
  }


def _is_cpu_pjrt_layout_error(exc: RuntimeError) -> bool:
  """Returns True when `exc` is CPU PjRt's `{...:S(5)}` pinned_host buffer-copy limitation."""
  msg = str(exc)
  return jax.default_backend() == "cpu" or ":S(5)" in msg or "PjRt CPU" in msg


def _transfer_sharding(arr: jax.Array, dst_sharding: Any) -> jax.Array:
  """Places `arr` on `dst_sharding`, working around CPU PjRt's `{...:S(5)}` pinned_host copy error."""
  try:
    return jax.device_put(arr, dst_sharding)
  except RuntimeError as exc:
    if not _is_cpu_pjrt_layout_error(exc):
      raise
    try:
      return jax.jit(lambda x: x, in_shardings=getattr(arr, "sharding", None), out_shardings=dst_sharding)(arr)
    except RuntimeError as jit_exc:
      if _is_cpu_pjrt_layout_error(jit_exc) and getattr(arr, "is_fully_addressable", False):
        import numpy as np  # pylint: disable=import-outside-toplevel

        return jax.device_put(np.asarray(arr), dst_sharding)
      raise


def _device_put_staged(arr: jax.Array, target_sharding: Any) -> jax.Array:
  """Places `arr` onto `target_sharding`, staging cross-spec reshards through `device` memory."""
  if target_sharding is None or getattr(arr, "sharding", None) == target_sharding:
    return arr
  src_s = getattr(arr, "sharding", None)
  if isinstance(src_s, NamedSharding) and src_s.memory_kind == "pinned_host":
    arr = _transfer_sharding(arr, src_s.with_memory_kind("device"))
    src_s = getattr(arr, "sharding", None)
  if (
      isinstance(target_sharding, NamedSharding)
      and target_sharding.memory_kind == "pinned_host"
      and isinstance(src_s, NamedSharding)
      and src_s.spec != target_sharding.spec
  ):
    arr = _transfer_sharding(arr, target_sharding.with_memory_kind("device"))
  return _transfer_sharding(arr, target_sharding)


def split_f32_to_bf16_residual16(x_f32: jax.Array) -> tuple[jax.Array, jax.Array]:
  """Splits a float32 array into a rounded bfloat16 working weight and signed int16 residual.

  Every normal IEEE-754 float32 (`|x| >= 2^-126`), `+/-0.0`, exact halfway tie (rounded away
  from zero so `-32768` fits in signed int16), `+/-finfo.max`, and `+/-Inf` reconstructs
  bit-for-bit via `reconstruct_f32_from_bf16_residual16`. IEEE-754 float32 subnormals
  (`0 < |x| < 2^-126`, exponent `0`) are flushed to signed `+/-0.0` before splitting to match
  TPU VPU flush-to-zero semantics and avoid producing `bfloat16` subnormals paired with negative
  `int16` residuals. `NaN` bit patterns are canonicalized to quiet `NaN` (`0x7FC00000`) so adding
  `0x8000` never wraps payload bits such as `0xFFFFFFFF` into `+0.0`.
  """
  x_f32 = jnp.asarray(x_f32, dtype=jnp.float32)
  bits = jax.lax.bitcast_convert_type(x_f32, jnp.uint32)
  bits = jnp.where(jnp.isnan(x_f32), jnp.uint32(0x7FC00000), bits)
  bits = jnp.where((bits & jnp.uint32(0x7F800000)) == jnp.uint32(0), bits & jnp.uint32(0x80000000), bits)
  w_u16 = ((bits + jnp.uint32(0x8000)) >> jnp.uint32(16)).astype(jnp.uint16)
  w_bf16 = jax.lax.bitcast_convert_type(w_u16, jnp.bfloat16)
  res_i16 = jax.lax.bitcast_convert_type((bits & jnp.uint32(0xFFFF)).astype(jnp.uint16), jnp.int16)
  return w_bf16, res_i16


def reconstruct_f32_from_bf16_residual16(w_bf16: jax.Array, res_i16: jax.Array) -> jax.Array:
  """Reconstructs the float32 master weight from a `(bfloat16, int16)` residual16 pair.

  Clamps the low-16-bit residual to `0` whenever `w_bf16` is `+/-0.0` (`(w_u32 & 0x7FFF) == 0`)
  so any `(+/-0.0, res_i16)` pair (including externally FTZ'd `bfloat16` subnormals paired with
  negative `res_i16`) reconstructs cleanly to signed `+/-0.0` and never wraps into `NaN`.
  """
  w_bf16 = jnp.asarray(w_bf16, dtype=jnp.bfloat16)
  res_i16 = jnp.asarray(res_i16, dtype=jnp.int16)
  abstract_mesh = jax.sharding.get_abstract_mesh()
  if not abstract_mesh.empty and any(t == jax.sharding.AxisType.Explicit for t in abstract_mesh.axis_types):
    r_spec = _typeof_spec(res_i16)
    if r_spec is not None and _typeof_spec(w_bf16) != r_spec:
      w_bf16 = jax.sharding.reshard(w_bf16, r_spec)
  w_u32 = jax.lax.bitcast_convert_type(w_bf16, jnp.uint16).astype(jnp.uint32)
  r_u32 = jax.lax.bitcast_convert_type(res_i16.astype(jnp.int32), jnp.uint32)
  r_u32 = jnp.where((w_u32 & jnp.uint32(0x7FFF)) == jnp.uint32(0), jnp.uint32(0), r_u32)
  bits = (w_u32 << jnp.uint32(16)) + r_u32
  return jax.lax.bitcast_convert_type(bits, jnp.float32)


@jax.jit
def _reconstruct_f32_jitted(w_bf16: jax.Array, res_i16: jax.Array) -> jax.Array:
  """Fused JIT wrapper around `reconstruct_f32_from_bf16_residual16` for eager checkpoint saves."""
  return reconstruct_f32_from_bf16_residual16(w_bf16, res_i16)


def _reconstruct_leaf_f32_bytes(w_bf16: Any) -> int:
  """Returns per-device byte footprint of the reconstructed float32 tensor for `w_bf16`."""
  shape = getattr(w_bf16, "shape", ())
  sharding = getattr(w_bf16, "sharding", None)
  shard_shape = sharding.shard_shape(shape) if isinstance(sharding, NamedSharding) else shape
  return int(math.prod(int(d) for d in shard_shape)) * 4


def _launch_f32_concrete_for_save(w_bf16: jax.Array, res_i16: jax.Array) -> jax.Array:
  """Launches float32 reconstruction and async transfer to `pinned_host` without blocking."""
  if isinstance(w_bf16, jax.core.Tracer) or isinstance(res_i16, jax.core.Tracer):
    return reconstruct_f32_from_bf16_residual16(w_bf16, res_i16)
  w_sharding = getattr(w_bf16, "sharding", None)
  compute_sharding = None
  if hasattr(w_sharding, "with_memory_kind"):
    try:
      compute_sharding = w_sharding.with_memory_kind("device")
    except (ValueError, RuntimeError):
      compute_sharding = w_sharding if isinstance(w_sharding, NamedSharding) else None
  w_dev = (
      _device_put_staged(w_bf16, compute_sharding)
      if compute_sharding is not None and getattr(w_sharding, "memory_kind", None) == "pinned_host"
      else w_bf16
  )
  r_dev = (
      _device_put_staged(res_i16, compute_sharding)
      if compute_sharding is not None and getattr(res_i16, "sharding", None) != compute_sharding
      else res_i16
  )
  f32_arr = _reconstruct_f32_jitted(w_dev, r_dev)
  del w_dev, r_dev
  if hasattr(w_sharding, "with_memory_kind"):
    try:
      host_sharding = w_sharding.with_memory_kind("pinned_host")
      f32_host = _device_put_staged(f32_arr, host_sharding)
      del f32_arr
      return f32_host
    except (ValueError, RuntimeError):
      pass
  return f32_arr


def _reconstruct_f32_tree_for_save(
    model_params: Any,
    flat_residuals: dict[tuple[Any, ...], Any],
    chunk_bytes: int | None = None,
) -> Any:
  """Reconstructs residual16 `(bf16, int16)` model leaves in byte-budgeted chunks.

  Grouping concrete leaves into chunks of at most `chunk_bytes` of reconstructed float32 per device
  (with any single leaf larger than `chunk_bytes` placed in its own 1-leaf chunk) bounds temporary
  device HBM to `<= max(chunk_bytes, max_single_leaf_f32_bytes)` while cutting host-device
  `jax.block_until_ready` syncs from `O(num_leaves)` to `O(total_f32_bytes / chunk_bytes)`.
  """
  leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(
      model_params,
      is_leaf=lambda x: isinstance(x, (jax.Array, jax.ShapeDtypeStruct)),
  )
  out_leaves: list[Any] = [None] * len(leaves_with_path)
  pending_concrete: list[tuple[int, jax.Array, jax.Array, int]] = []

  for idx, (jax_path, leaf) in enumerate(leaves_with_path):
    key_tuple = _key_tuple(jax_path)
    if key_tuple not in flat_residuals:
      out_leaves[idx] = leaf
    elif isinstance(leaf, jax.ShapeDtypeStruct):
      out_leaves[idx] = jax.ShapeDtypeStruct(leaf.shape, jnp.float32, sharding=getattr(leaf, "sharding", None))
    elif isinstance(leaf, jax.core.Tracer) or isinstance(flat_residuals[key_tuple], jax.core.Tracer):
      out_leaves[idx] = reconstruct_f32_from_bf16_residual16(leaf, flat_residuals[key_tuple])
    else:
      res_leaf = flat_residuals[key_tuple]
      pending_concrete.append((idx, leaf, res_leaf, _reconstruct_leaf_f32_bytes(leaf)))

  if pending_concrete:
    limit_bytes = max(int(_RECONSTRUCT_SAVE_CHUNK_BYTES if chunk_bytes is None else chunk_bytes), 1)
    pos = 0
    n_pending = len(pending_concrete)
    while pos < n_pending:
      chunk_indices: list[int] = []
      chunk_outputs: list[jax.Array] = []
      cur_bytes = 0
      while pos < n_pending:
        idx, w_leaf, r_leaf, leaf_bytes = pending_concrete[pos]
        if chunk_indices and cur_bytes + leaf_bytes > limit_bytes:
          break
        chunk_indices.append(idx)
        chunk_outputs.append(_launch_f32_concrete_for_save(w_leaf, r_leaf))
        cur_bytes += leaf_bytes
        pos += 1
      chunk_outputs = jax.block_until_ready(chunk_outputs)
      for idx, f32_out in zip(chunk_indices, chunk_outputs):
        out_leaves[idx] = f32_out
      del chunk_indices, chunk_outputs

  return jax.tree_util.tree_unflatten(treedef, out_leaves)


def _split_f32_concrete_for_restore(
    raw_val: jax.Array,
    target_w_sharding: Any,
    target_res_sharding: Any,
) -> tuple[jax.Array, jax.Array]:
  """Splits a concrete restored weight into `(w_bf16, res_i16)` placed on target shardings."""
  raw_sharding = getattr(raw_val, "sharding", None)
  raw_dev = (
      _device_put_staged(raw_val, raw_sharding.with_memory_kind("device"))
      if isinstance(raw_sharding, NamedSharding) and raw_sharding.memory_kind == "pinned_host"
      else raw_val
  )
  w_bf16, res_i16 = split_f32_to_bf16_residual16(raw_dev)
  if target_w_sharding is not None:
    w_bf16 = _device_put_staged(w_bf16, target_w_sharding)
  if target_res_sharding is not None:
    res_i16 = _device_put_staged(res_i16, target_res_sharding)
  return w_bf16, res_i16


def should_use_residual16_for_param(path: tuple[Any, ...], var: nnx.Variable, config: Any = None) -> bool:
  """Returns whether `var` should be stored as `bf16` working weight + `int16` residual."""
  del path
  val = var.get_value() if isinstance(var, nnx.Variable) else var
  dtype = getattr(val, "dtype", None)
  if dtype not in (jnp.float32, jnp.bfloat16):
    return False
  if isinstance(var, nnx.Variable) and var.get_metadata().get("keep_fp32_master", False):
    return False
  if config is not None and config.weight_dtype in (jnp.bfloat16, "bfloat16"):
    # Under weight_dtype=bfloat16, leave intentionally float32 parameters in float32.
    return dtype == jnp.bfloat16
  return True


def _record_mu_tree_specs(mu_tree: Any, out: dict[tuple[Any, ...], Any]) -> None:
  """Records non-None PartitionSpecs from a single Optax `mu` subtree into `out`."""
  for p, leaf in jax.tree_util.tree_leaves_with_path(mu_tree):
    if hasattr(leaf, "dtype"):
      spec = _typeof_spec(leaf)
      if spec is not None:
        out.setdefault(_key_tuple(p), spec)


def _collect_mu_specs(opt_state_node: Any, out: dict[tuple[Any, ...], Any]) -> None:
  """Collects per-parameter PartitionSpecs from `mu` trees inside an Optax state."""
  if isinstance(opt_state_node, (dict, nnx.State)):
    if "mu" in opt_state_node and isinstance(opt_state_node["mu"], (dict, nnx.State)):
      _record_mu_tree_specs(opt_state_node["mu"], out)
    for v in opt_state_node.values():
      _collect_mu_specs(v, out)
  elif hasattr(opt_state_node, "_fields"):
    for field_name in opt_state_node._fields:
      val = getattr(opt_state_node, field_name)
      if field_name == "mu" and isinstance(val, (dict, nnx.State)):
        _record_mu_tree_specs(val, out)
      else:
        _collect_mu_specs(val, out)
  elif isinstance(opt_state_node, (tuple, list)):
    for v in opt_state_node:
      _collect_mu_specs(v, out)


class Residual16Optimizer(nnx.Optimizer):
  """NNX Optimizer storing working weights in bfloat16 and low-16-bit residuals in int16.

  Model parameters stay in `bfloat16` (2 bytes/param) for forward/backward passes, while
  `self.master_residuals` stores the signed `int16` low 16 bits (2 bytes/param) as a sibling
  `OptState` subtree alongside standard `self.opt_state`. During `update`, exact `float32`
  master weights are reconstructed on the fly, passed to `self.tx.update` + `apply_updates`,
  and split back into `(bfloat16, int16)` — saving 2 bytes/param vs keeping both `float32`
  master weights (4 bytes) and `bfloat16` working weights.
  """

  def __init__(
      self,
      model: nnx.Module,
      tx: Any,
      config: Any = None,
      *,
      wrt: nnx.filterlib.Filter = nnx.Param,
      graph: bool | None = None,
  ):
    # pylint: disable=super-init-not-called
    self.graph = graph
    self.step = nnx.OptState(jnp.array(0, dtype=jnp.uint32))
    self.tx = tx
    self.wrt = wrt

    params_state = nnx.state(model, wrt, graph=graph)
    split_cache: dict[tuple[Any, ...], tuple[jax.Array, jax.Array, jax.Array]] = {}

    def _prepare_model_var(jax_path, var):
      key_tuple = _key_tuple(jax_path)
      if not should_use_residual16_for_param(key_tuple, var, config):
        return var
      val = var.get_value()
      if val.dtype == jnp.float32:
        w_bf16, res_i16 = split_f32_to_bf16_residual16(val)
        f32_val = val
      else:
        w_bf16 = jnp.asarray(val, dtype=jnp.bfloat16)
        res_i16 = jnp.zeros(val.shape, dtype=jnp.int16)
        f32_val = w_bf16.astype(jnp.float32)
      split_cache[key_tuple] = (w_bf16, res_i16, f32_val)
      return var.replace(value=w_bf16)

    updated_model_params = jax.tree_util.tree_map_with_path(
        _prepare_model_var,
        params_state,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )
    f32_params_for_tx = jax.tree_util.tree_map_with_path(
        lambda jax_path, var: (
            var.replace(value=split_cache[_key_tuple(jax_path)][2]) if _key_tuple(jax_path) in split_cache else var
        ),
        params_state,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )
    eligible_state = params_state.filter(lambda path, var: _key_tuple(path) in split_cache)
    residual_vars = jax.tree_util.tree_map_with_path(
        lambda jax_path, var: var.replace(value=split_cache[_key_tuple(jax_path)][1]),
        eligible_state,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )

    nnx.update(model, updated_model_params)
    to_opt_state = nnx.training.optimizer.to_opt_state
    self.opt_state = nnx.data(to_opt_state(tx.init(f32_params_for_tx)))
    self.master_residuals = nnx.data(to_opt_state(residual_vars) if split_cache else {})

  def update(self, model: nnx.Module, grads: Any, /, **kwargs) -> Any:
    """Reconstructs float32 master weights, runs `tx.update`, and splits back to `(bf16, int16)`."""
    param_arrays = nnx.as_pure(nnx.state(model, self.wrt, graph=self.graph))
    grad_arrays = nnx.as_pure(nnx.state(grads, self.wrt, graph=self.graph))
    opt_state_arrays = nnx.as_pure(self.opt_state)
    residuals_pure = nnx.as_pure(self.master_residuals)
    kwargs_arrays = nnx.as_pure(kwargs)

    flat_residuals = _flatten_leaves_by_key_tuple(residuals_pure)

    abstract_mesh = jax.sharding.get_abstract_mesh()
    has_explicit_axes = not abstract_mesh.empty and any(
        t == jax.sharding.AxisType.Explicit for t in abstract_mesh.axis_types
    )
    opt_specs: dict[tuple[Any, ...], Any] = {}
    if has_explicit_axes:
      for k, res_leaf in flat_residuals.items():
        res_spec = _typeof_spec(res_leaf)
        if res_spec is not None:
          opt_specs[k] = res_spec
      _collect_mu_specs(opt_state_arrays, opt_specs)

    def _reconstruct_param(jax_path, w):
      key_tuple = _key_tuple(jax_path)
      if key_tuple in flat_residuals:
        return reconstruct_f32_from_bf16_residual16(w, flat_residuals[key_tuple])
      target_spec = opt_specs.get(key_tuple) if has_explicit_axes else None
      if target_spec is not None and _typeof_spec(w) != target_spec:
        w = jax.sharding.reshard(w, target_spec)
      return w

    f32_params = jax.tree_util.tree_map_with_path(_reconstruct_param, param_arrays)
    if has_explicit_axes and opt_specs:
      grad_arrays = jax.tree_util.tree_map_with_path(
          lambda p, g: (
              jax.sharding.reshard(g, opt_specs[_key_tuple(p)])
              if opt_specs.get(_key_tuple(p)) is not None and _typeof_spec(g) != opt_specs[_key_tuple(p)]
              else g
          ),
          grad_arrays,
      )

    updates, new_opt_state = self.tx.update(grad_arrays, opt_state_arrays, f32_params, **kwargs_arrays)
    new_f32_params = jax.tree.map(
        lambda p, u: None if p is None else jnp.asarray(p + u, dtype=p.dtype), f32_params, updates
    )

    updated_residuals_by_key: dict[tuple[Any, ...], jax.Array] = {}

    def _split_updated_param(jax_path, p_new):
      key_tuple = _key_tuple(jax_path)
      if key_tuple in flat_residuals:
        w_bf16, res_i16 = split_f32_to_bf16_residual16(p_new)
        updated_residuals_by_key[key_tuple] = res_i16
        return w_bf16
      return p_new

    new_working_params = jax.tree_util.tree_map_with_path(_split_updated_param, new_f32_params)
    nnx.update(model, new_working_params)
    nnx.update(self.opt_state, nnx.state(new_opt_state, graph=self.graph))
    if flat_residuals:
      new_residuals = jax.tree_util.tree_map_with_path(
          lambda jax_path, _: updated_residuals_by_key[_key_tuple(jax_path)],
          residuals_pure,
      )
      nnx.update(self.master_residuals, nnx.state(new_residuals, graph=self.graph))
    self.step[...] += 1
    return updates

  def apply_master_param_transform(self, model: nnx.Module, transform_fn) -> None:
    """Applies `transform_fn(jax_path, param_f32)` on reconstructed master weights and re-splits."""
    param_arrays = nnx.as_pure(nnx.state(model, self.wrt, graph=self.graph))
    residuals_pure = nnx.as_pure(self.master_residuals)
    flat_residuals = _flatten_leaves_by_key_tuple(residuals_pure)

    f32_params = jax.tree_util.tree_map_with_path(
        lambda p, w: (
            reconstruct_f32_from_bf16_residual16(w, flat_residuals[_key_tuple(p)])
            if _key_tuple(p) in flat_residuals
            else w
        ),
        param_arrays,
    )
    transformed_f32 = jax.tree_util.tree_map_with_path(transform_fn, f32_params)

    updated_residuals_by_key: dict[tuple[Any, ...], jax.Array] = {}

    def _resplit(jax_path, p_new):
      key_tuple = _key_tuple(jax_path)
      if key_tuple in flat_residuals:
        w_bf16, res_i16 = split_f32_to_bf16_residual16(p_new)
        updated_residuals_by_key[key_tuple] = res_i16
        return w_bf16
      return p_new

    new_working_params = jax.tree_util.tree_map_with_path(_resplit, transformed_f32)
    nnx.update(model, new_working_params)
    if flat_residuals:
      new_residuals = jax.tree_util.tree_map_with_path(
          lambda p, _: updated_residuals_by_key[_key_tuple(p)],
          residuals_pure,
      )
      nnx.update(self.master_residuals, nnx.state(new_residuals, graph=self.graph))


def create_train_state(
    model: nnx.Module,
    tx: Any,
    config: Any = None,
    *,
    wrt: nnx.filterlib.Filter = nnx.Param,
) -> TrainStateNNX:
  """Creates a `TrainStateNNX` using `Residual16Optimizer` when `fp32_master_storage == 'residual16'`."""
  if config is not None and config.fp32_master_storage == "residual16":
    optimizer: nnx.Optimizer = Residual16Optimizer(model, tx, config=config, wrt=wrt)
  else:
    optimizer = nnx.Optimizer(model, tx, wrt=wrt)
  return TrainStateNNX(model, optimizer)


def reshape_restored_weights_for_residual16(weights: dict[str, Any], linen_state: nnx.State) -> dict[str, Any]:
  """Splits restored float32 checkpoint weights into `(bf16, int16)` when `linen_state` uses residual16."""
  if "optimizer" not in linen_state or "master_residuals" not in linen_state["optimizer"]:
    return weights

  abstract_model_pure = linen_state["model"].to_pure_dict() if "model" in linen_state else {}
  res_state = linen_state["optimizer"]["master_residuals"]
  abstract_res_pure = res_state.to_pure_dict() if hasattr(res_state, "to_pure_dict") else res_state
  flat_abstract_model = _flatten_leaves_by_key_tuple(abstract_model_pure)
  flat_abstract_res = _flatten_leaves_by_key_tuple(abstract_res_pure)
  if not flat_abstract_res:
    return weights

  out = dict(weights)
  split_residuals_by_key: dict[tuple[Any, ...], Any] = {}
  if "model" in out:

    def _split_model_leaf(jax_path, leaf):
      key_tuple = _key_tuple(jax_path)
      if key_tuple in flat_abstract_res and isinstance(leaf, jax.Array):
        target_w_sds = flat_abstract_model.get(key_tuple)
        target_res_sds = flat_abstract_res[key_tuple]
        w_bf16, res_i16 = _split_f32_concrete_for_restore(
            leaf,
            getattr(target_w_sds, "sharding", None),
            getattr(target_res_sds, "sharding", None),
        )
        split_residuals_by_key[key_tuple] = res_i16
        return w_bf16
      return leaf

    out["model"] = jax.tree_util.tree_map_with_path(
        _split_model_leaf,
        out["model"],
        is_leaf=lambda x: isinstance(x, (jax.Array, jax.ShapeDtypeStruct)),
    )

  restored_residuals = jax.tree_util.tree_map_with_path(
      lambda p, sds: split_residuals_by_key.get(_key_tuple(p), sds),
      abstract_res_pure,
      is_leaf=lambda x: isinstance(x, (jax.Array, jax.ShapeDtypeStruct)),
  )
  opt_dict = dict(out.get("optimizer", {}))
  opt_dict["master_residuals"] = restored_residuals
  out["optimizer"] = opt_dict
  return out


def promote_abstract_params_for_residual16(model_params_state: nnx.State, abstract_state: Any) -> nnx.State:
  """Promotes residual16 parameter `ShapeDtypeStruct`s to `float32` for params-only Orbax restore."""
  if not isinstance(abstract_state, (nnx.State, dict)) or "optimizer" not in abstract_state:
    return model_params_state
  opt = abstract_state["optimizer"]
  if not isinstance(opt, (dict, nnx.State)) or "master_residuals" not in opt:
    return model_params_state

  res_state = opt["master_residuals"]
  res_pure = res_state.to_pure_dict() if hasattr(res_state, "to_pure_dict") else res_state
  flat_res = _flatten_leaves_by_key_tuple(res_pure)
  if not flat_res:
    return model_params_state

  def _promote_var(jax_path, var):
    key_tuple = _key_tuple(jax_path)
    if key_tuple not in flat_res or not isinstance(var, nnx.Variable):
      return var
    val = var.get_value()
    if isinstance(val, jax.ShapeDtypeStruct):
      return var.replace(value=jax.ShapeDtypeStruct(val.shape, jnp.float32, sharding=getattr(val, "sharding", None)))
    return var

  return jax.tree_util.tree_map_with_path(
      _promote_var,
      model_params_state,
      is_leaf=lambda x: isinstance(x, nnx.Variable),
  )


def overlay_raw_params_with_residual16(state: Any, raw_model_params: Any) -> bool:
  """Overlays params-only restored weights onto `state`, splitting `(bf16, int16)` for residual16."""
  if not isinstance(state, (nnx.State, dict)) or "model" not in state or "optimizer" not in state:
    return False
  opt = state["optimizer"]
  if not isinstance(opt, (nnx.State, dict)) or "master_residuals" not in opt:
    return False

  target_model = state["model"]
  target_pure = target_model.to_pure_dict() if hasattr(target_model, "to_pure_dict") else target_model
  res_target = opt["master_residuals"]
  res_pure = res_target.to_pure_dict() if hasattr(res_target, "to_pure_dict") else res_target

  target_flat = _flatten_leaves_by_key_tuple(target_pure)
  res_target_flat = _flatten_leaves_by_key_tuple(res_pure)
  if not res_target_flat:
    return False
  raw_flat = _flatten_leaves_by_key_tuple(raw_model_params)

  updated_model_flat: dict[tuple[Any, ...], Any] = {}
  updated_res_flat: dict[tuple[Any, ...], Any] = dict(res_target_flat)
  for k, target_val in target_flat.items():
    if k in raw_flat and not isinstance(raw_flat[k], jax.ShapeDtypeStruct):
      raw_val = raw_flat[k]
      if isinstance(raw_val, nnx.Variable):
        raw_val = raw_val.get_value()
      if k in res_target_flat:
        w_bf16, res_i16 = _split_f32_concrete_for_restore(
            raw_val,
            getattr(target_val, "sharding", None),
            getattr(res_target_flat[k], "sharding", None),
        )
        updated_model_flat[k] = w_bf16
        updated_res_flat[k] = res_i16
      else:
        cast_val = raw_val
        if hasattr(target_val, "dtype") and getattr(raw_val, "dtype", None) != target_val.dtype:
          raw_sharding = getattr(raw_val, "sharding", None)
          if isinstance(raw_sharding, NamedSharding) and raw_sharding.memory_kind == "pinned_host":
            cast_val = _device_put_staged(raw_val, raw_sharding.with_memory_kind("device"))
          cast_val = cast_val.astype(target_val.dtype)
        if hasattr(target_val, "sharding") and target_val.sharding is not None:
          updated_model_flat[k] = _device_put_staged(cast_val, target_val.sharding)
        else:
          updated_model_flat[k] = cast_val
    else:
      updated_model_flat[k] = target_val

  nnx.replace_by_pure_dict(target_model, nnx.traversals.unflatten_mapping(updated_model_flat))
  if isinstance(res_target, nnx.State):
    nnx.replace_by_pure_dict(res_target, nnx.traversals.unflatten_mapping(updated_res_flat))
  elif isinstance(res_target, dict):
    res_target.clear()
    res_target.update(nnx.traversals.unflatten_mapping(updated_res_flat))
  return True


def resplit_model_params_with_residual16(state: Any) -> bool:
  """Splits in-place restored `state["model"]` leaves into `(bf16, int16)` for every key in `master_residuals`.

  Used after model-only in-place restores (such as `lineage_adapter.restore_params`) that overwrite
  `state["model"]` with restored `float32` weights without touching `state["optimizer"]["master_residuals"]`.
  Preserves each model leaf's and each `master_residuals` leaf's existing `sharding` and `memory_kind`
  (including `pinned_host` when `optimizer_memory_host_offload=True`).
  """
  if not isinstance(state, (nnx.State, dict)) or "model" not in state or "optimizer" not in state:
    return False
  opt = state["optimizer"]
  if not isinstance(opt, (nnx.State, dict)) or "master_residuals" not in opt:
    return False

  target_model = state["model"]
  model_pure = target_model.to_pure_dict() if hasattr(target_model, "to_pure_dict") else target_model
  res_target = opt["master_residuals"]
  res_pure = res_target.to_pure_dict() if hasattr(res_target, "to_pure_dict") else res_target

  model_flat = _flatten_leaves_by_key_tuple(model_pure)
  res_flat = _flatten_leaves_by_key_tuple(res_pure)
  if not res_flat:
    return False

  split_model_flat: dict[tuple[Any, ...], Any] = {}
  updated_res_flat: dict[tuple[Any, ...], Any] = dict(res_flat)
  for k, res_leaf in res_flat.items():
    if k not in model_flat:
      continue
    model_val = model_flat[k]
    if isinstance(model_val, nnx.Variable):
      model_val = model_val.get_value()
    if isinstance(model_val, jax.ShapeDtypeStruct):
      continue
    res_val = res_leaf.get_value() if isinstance(res_leaf, nnx.Variable) else res_leaf
    w_bf16, res_i16 = _split_f32_concrete_for_restore(
        model_val,
        getattr(model_val, "sharding", None),
        getattr(res_val, "sharding", None),
    )
    split_model_flat[k] = w_bf16
    updated_res_flat[k] = res_i16

  if not split_model_flat:
    return False

  if isinstance(target_model, nnx.State):
    nnx.replace_by_pure_dict(target_model, nnx.traversals.unflatten_mapping(split_model_flat))
  elif isinstance(target_model, dict):
    merged_model_flat = dict(model_flat)
    merged_model_flat.update(split_model_flat)
    target_model.clear()
    target_model.update(nnx.traversals.unflatten_mapping(merged_model_flat))

  if isinstance(res_target, nnx.State):
    nnx.replace_by_pure_dict(res_target, nnx.traversals.unflatten_mapping(updated_res_flat))
  elif isinstance(res_target, dict):
    res_target.clear()
    res_target.update(nnx.traversals.unflatten_mapping(updated_res_flat))
  return True


# On-disk checkpoint format.
#
# A run saves in the same on-disk layout that Linen used, so old and new checkpoints are
# interchangeable. The NNX state pure dict differs from Linen's in three ways, all
# reshaped below at save time:
#   1. top-level keys: {model, optimizer:{step, opt_state}} -> {params:{params:...}, step, opt_state}
#   2. weights: model/... -> params/params/... (Linen `params` collection)
#   3. opt_state: int-keyed dict (empty entries skipped) -> list with None for EmptyState,
#      and mu/nu wrapped under the `params` collection
# Persistent NNX state Linen's params/opt_state/step layout can't hold -- rngs/dropout,
# batch stats, and any custom non-Param variable -- is split out by Variable type and stored
# under an `nnx_aux` subtree of `items` (see `to_checkpoint_dict`), so it survives a resume
# instead of resetting.

_NNX_RNG_STATE_KEYS = ("rngs", "dropout")


def _cast_step(step, dtype):
  """Casts the step's dtype, handling both concrete arrays and abstract ShapeDtypeStruct.

  NNX stores step as uint32 and Linen as int32; this also runs on the abstract
  (SDS) state when building a restore target, so it can't assume concrete
  values.
  """
  if isinstance(step, jax.ShapeDtypeStruct):
    return jax.ShapeDtypeStruct(step.shape, dtype, sharding=getattr(step, "sharding", None))
  sharding = getattr(step, "sharding", None)
  if isinstance(sharding, NamedSharding) and sharding.memory_kind == "pinned_host":
    out_sharding = sharding if dtype == jnp.uint32 else sharding.with_memory_kind("device")
    return jax.jit(lambda x: jnp.asarray(x, dtype=dtype), in_shardings=sharding, out_shardings=out_sharding)(step)
  return jnp.asarray(step, dtype=dtype)


def _strip_rng_state(tree):
  """Removes the NNX-only 'rngs'/'dropout' subtrees that Linen doesn't carry.

  Subtrees that become empty after stripping are dropped too, so the result has
  no keys Linen wouldn't also have.
  """
  if not isinstance(tree, dict):
    return tree
  out = {}
  for k, v in tree.items():
    if k in _NNX_RNG_STATE_KEYS:
      continue
    stripped = _strip_rng_state(v)
    if isinstance(stripped, dict) and not stripped:
      continue
    out[k] = stripped
  return out


def _wrap_mu_nu_with_params(state):
  """Wraps mu/nu under an inner 'params' key (the Linen collection)."""
  if not isinstance(state, dict):
    return state
  return {k: {"params": v} if k in ("mu", "nu") and isinstance(v, dict) else v for k, v in state.items()}


def _as_chain_index(key):
  """Returns the int index for an int or digit-string key, else None."""
  if isinstance(key, int):
    return key
  if isinstance(key, str) and key.isdigit():
    return int(key)
  return None


def _opt_state_to_linen(opt_state):
  """Reshapes the NNX opt_state to Linen's on-disk layout.

  A chain (optax.adamw) is an int-keyed dict -> list-with-None. An un-chained
  optimizer (adam_pax) is flat (count/mu/nu) and stays a dict.
  """
  if not isinstance(opt_state, dict):
    return opt_state
  indices = [_as_chain_index(k) for k in opt_state.keys()]
  if not indices or any(i is None for i in indices):
    return _wrap_mu_nu_with_params(opt_state)
  chain = [None] * (max(indices) + 1)
  for key, idx in zip(opt_state.keys(), indices):
    chain[idx] = _wrap_mu_nu_with_params(opt_state[key])
  return chain


def to_linen_checkpoint_dict(nnx_pure_dict, *, reconstruct_chunk_bytes: int | None = None):
  """Reshapes a TrainStateNNX pure dict ({model, optimizer}) into the Linen on-disk layout."""
  if not isinstance(nnx_pure_dict, dict):
    return nnx_pure_dict
  result = {}
  optimizer = nnx_pure_dict.get("optimizer")
  flat_residuals = (
      _flatten_leaves_by_key_tuple(optimizer.get("master_residuals", {})) if isinstance(optimizer, dict) else {}
  )
  if "model" in nnx_pure_dict:
    model_params = _strip_rng_state(nnx_pure_dict["model"])
    if flat_residuals:
      model_params = _reconstruct_f32_tree_for_save(
          model_params,
          flat_residuals,
          chunk_bytes=reconstruct_chunk_bytes,
      )
    result["params"] = {"params": model_params}
  if isinstance(optimizer, dict):
    if "step" in optimizer:
      # NNX stores step as uint32; Linen uses int32.
      result["step"] = _cast_step(optimizer["step"], jnp.int32)
    if "opt_state" in optimizer:
      result["opt_state"] = _opt_state_to_linen(optimizer["opt_state"])
  return result


def _strip_mu_nu_params(state):
  """Inverse of `_wrap_mu_nu_with_params`: removes the inner 'params' wrap from mu/nu."""
  if not isinstance(state, dict):
    return state
  return {
      k: (v["params"] if k in ("mu", "nu") and isinstance(v, dict) and "params" in v else v) for k, v in state.items()
  }


def _opt_state_from_linen(opt_state):
  """Inverse of `_opt_state_to_linen`: Linen layout -> NNX opt_state.

  A chain (optax.adamw) is a list -> int-keyed dict. An un-chained state (adam_pax:
  `{count, mu, nu}`) stays flat -- re-wrapping it under a `0` index would not line
  up with the model's opt_state.
  """
  if isinstance(opt_state, list):
    return {i: _strip_mu_nu_params(e) for i, e in enumerate(opt_state) if isinstance(e, dict)}
  if not isinstance(opt_state, dict):
    return opt_state
  return _strip_mu_nu_params(opt_state)


def from_linen_checkpoint_dict(linen_pure_dict):
  """Inverse of `to_linen_checkpoint_dict`: Linen on-disk layout -> NNX layout.

  Doesn't restore NNX-only rngs/dropout (absent from Linen); callers fill those.
  """
  if not isinstance(linen_pure_dict, dict):
    return linen_pure_dict
  result = {}
  params = linen_pure_dict.get("params")
  if isinstance(params, dict) and "params" in params:
    result["model"] = params["params"]
  elif params is not None:
    result["model"] = params
  optimizer = {}
  if "step" in linen_pure_dict:
    # Linen stores step as int32; NNX uses uint32.
    optimizer["step"] = _cast_step(linen_pure_dict["step"], jnp.uint32)
  if "opt_state" in linen_pure_dict:
    optimizer["opt_state"] = _opt_state_from_linen(linen_pure_dict["opt_state"])
  if optimizer:
    result["optimizer"] = optimizer
  return result


def split_for_checkpoint(state: nnx.State):
  """Partitions an nnx.State by its on-disk destination.

  Named by on-disk destination. Returns `(linen_state, aux, ephemeral)`:
    linen_state: trainable weights (nnx.Param) and the optimizer -> Linen params/opt_state/step.
      Its model side is pure nnx.Param, so it maps to the Linen `params` collection cleanly.
    aux:         rngs/dropout, batch stats, and any other persistent model variable (e.g. a
      frozen routing table) -> nnx_aux.
    ephemeral:   caches and intermediates, which are recomputed and so are not checkpointed.

  The type-based catch-all `rest` mixes the optimizer (under "optimizer") with any custom
  persistent variable (under "model"); they route to different places, so they're split apart
  here. Grouping by the ephemeral types rather than whitelisting nnx.Param means a new Variable
  subclass persists (via `aux`) instead of being silently dropped. Save and restore must
  partition identically, so both go through here.
  """
  params, rng_state, batch_stats, caches, intermediates, rest = nnx.split_state(
      state, nnx.Param, nnx.RngState, nnx.BatchStat, nnx.Cache, nnx.Intermediate, ...
  )
  optimizer = nnx.State({"optimizer": rest["optimizer"]}) if "optimizer" in rest else nnx.State({})
  custom = nnx.State({"model": rest["model"]}) if "model" in rest else nnx.State({})
  linen_state = nnx.merge_state(params, optimizer)
  aux = nnx.merge_state(rng_state, batch_stats, custom)
  ephemeral = nnx.merge_state(caches, intermediates)
  return linen_state, aux, ephemeral


def to_checkpoint_dict(state: nnx.State | nnx.Module):
  """Reshapes an nnx.State into the on-disk checkpoint layout.

  Weights (nnx.Param) map to the Linen `params` collection and the optimizer to
  opt_state/step, so NNX and Linen checkpoints stay interchangeable. Everything else that
  must persist -- rngs/dropout, batch stats, and any custom variable -- goes under an `nnx_aux`
  subtree. Works on a concrete state (save) or an abstract state (restore target).
  """
  if isinstance(state, nnx.Module):
    state = nnx.state(state)
  linen_state, aux_state, _ = split_for_checkpoint(state)
  pure = linen_state.to_pure_dict()
  linen_dict = to_linen_checkpoint_dict({"model": pure.get("model", {}), "optimizer": pure.get("optimizer", {})})
  aux = aux_state.to_pure_dict()
  if aux:
    linen_dict["nnx_aux"] = aux
  return linen_dict


def apply_checkpoint_aux(aux_state: nnx.State, nnx_aux: dict[str, Any]) -> None:
  """Fills `aux_state` from a checkpoint's `nnx_aux`, skipping entries it no longer has.

  `split_for_checkpoint` routes every persistent non-Param variable to `nnx_aux`, so a
  checkpoint carries whatever RNG streams, batch stats and custom variables the model held
  when it was written. A model that has since stopped holding one of them -- a module that
  no longer keeps an `nnx.Rngs`, say -- would otherwise fail the resume outright with
  "key in pure_dict not available in state".

  Restore is already tolerant in the other direction: an entry the checkpoint lacks keeps
  its fresh init value. This makes it tolerant both ways. Only `nnx_aux` is filtered;
  weights still go through `replace_by_pure_dict` directly, so a genuinely missing weight
  keeps raising.
  """
  present = {tuple(path) for path, _ in nnx.to_flat_state(aux_state)}
  flat = nnx.traversals.flatten_mapping(nnx_aux)
  kept = {path: value for path, value in flat.items() if path in present}
  dropped = [path for path in flat if path not in present]
  if dropped:
    # Log the paths, not just a count: skipping an RNG stream a module stopped holding
    # is routine, but skipping a batch stat or a custom variable means the model and the
    # checkpoint have genuinely diverged, and that should be visible in the log.
    shown = ", ".join("/".join(str(k) for k in path) for path in sorted(dropped)[:20])
    more = f" (+{len(dropped) - 20} more)" if len(dropped) > 20 else ""
    max_logging.log(f"Skipping {len(dropped)} nnx_aux entries absent from this model: {shown}{more}")
  if kept:
    nnx.replace_by_pure_dict(aux_state, nnx.traversals.unflatten_mapping(kept))
