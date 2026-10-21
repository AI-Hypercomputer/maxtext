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

"""TPU-scale sharded and jitted Zero-Gen runner and Orbax checkpoint restore for Qwen3.8-Flash-Next."""

from collections.abc import Mapping, Sequence
from typing import Any

from etils import epath
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.configs import pyconfig
from maxtext.m3 import models as m3_models
import numpy as np
import orbax.checkpoint


def _load_maxtext_utils() -> Any:
  from maxtext.utils import maxtext_utils  # pylint: disable=g-import-not-at-top,import-outside-toplevel

  return maxtext_utils


def create_device_mesh_for_config(config: Any) -> jax.sharding.Mesh:
  """Creates a jax.sharding.Mesh for the provided configuration."""
  if hasattr(config, "mesh_axes") and hasattr(config, "ici_fsdp_parallelism"):
    return jax.sharding.Mesh(_load_maxtext_utils().create_device_mesh(config), config.mesh_axes)
  return jax.sharding.Mesh(np.asarray(jax.devices()).reshape(-1), ("data",))


def _primary_mesh_axis(mesh: jax.sharding.Mesh) -> tuple[str | None, int]:
  """Determines primary FSDP or data axis name and size from mesh."""
  if "fsdp" in mesh.axis_names and mesh.shape["fsdp"] > 1:
    return "fsdp", int(mesh.shape["fsdp"])
  axes = [(k, int(mesh.shape[k])) for k in mesh.axis_names if mesh.shape[k] > 1]
  return max(axes, key=lambda x: x[1]) if axes else (None, 1)


def compute_param_partition_spec(
    path: tuple[Any, ...],
    shape: tuple[int, ...],
    axis_name: str | None,
    axis_size: int,
) -> jax.sharding.PartitionSpec:
  """Computes 1D parameter PartitionSpec across primary mesh axis."""
  if axis_name is None or axis_size <= 1:
    return jax.sharding.PartitionSpec()
  path_strs = [str(getattr(p, "key", getattr(p, "attr", p))) for p in path]
  start_d = 2 if any(p.endswith("_0_2") for p in path_strs) else (1 if "cycles" in path_strs else 0)
  for d in range(start_d, len(shape)):
    if shape[d] % axis_size == 0:
      return jax.sharding.PartitionSpec(*(axis_name if i == d else None for i in range(len(shape))))
  return jax.sharding.PartitionSpec()


def shard_model_parameters(model: nnx.Module, mesh: jax.sharding.Mesh) -> nnx.Module:
  """Shards model parameters across the provided mesh using 1D PartitionSpec."""
  axis_name, axis_size = _primary_mesh_axis(mesh)
  param_state = nnx.state(model, nnx.Param)
  for path, var in nnx.to_flat_state(param_state):
    pspec = compute_param_partition_spec(path, var.value.shape, axis_name, axis_size)
    var.value = jax.device_put(var.value, jax.sharding.NamedSharding(mesh, pspec))
  nnx.update(model, param_state)
  return model


def _traverse_param_step(curr: Any, step: Any, path: Any) -> Any:
  """Traverses a single checkpoint parameter path step."""
  if isinstance(curr, Mapping):
    for k in (step, str(step), int(step) if str(step).isdigit() else None):
      if k is not None and k in curr:
        return curr[k]
    raise ValueError(f"Parameter {step!r} for {path} not found in checkpoint:" f" {list(curr)[:5]}")
  if isinstance(curr, (list, tuple)):
    try:
      return curr[int(step)]
    except (ValueError, IndexError) as err:
      raise ValueError(f"Invalid index {step!r} for {path}") from err
  raise ValueError(f"Cannot traverse {step!r} into {type(curr)} at {path}")


def restore_model_parameters(
    model: nnx.Module,
    load_parameters_path: str | epath.Path,
    mesh: jax.sharding.Mesh | None = None,
) -> nnx.Module:
  """Restores model parameter state from an Orbax checkpoint."""
  active_mesh = mesh if mesh is not None else getattr(model, "mesh", None)
  axis_name, axis_size = _primary_mesh_axis(active_mesh) if active_mesh is not None else (None, 1)
  checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  path = epath.Path(load_parameters_path)

  has_params_root = False
  try:
    step_meta = checkpointer.metadata(path)
    item_tree = getattr(getattr(step_meta, "item_metadata", None), "tree", None)
    if isinstance(item_tree, Mapping) and "params" in item_tree:
      has_params_root = True
  except Exception:  # pylint: disable=broad-exception-caught
    pass

  param_state = nnx.state(model, nnx.Param)
  restore_args = None
  if active_mesh is not None:
    param_restore_args: dict[str, Any] = {}
    for p, var in nnx.to_flat_state(param_state):
      exp = var.value.shape
      dtype = var.value.dtype
      pspec = compute_param_partition_spec(p, exp, axis_name, axis_size)
      sharding = jax.sharding.NamedSharding(active_mesh, pspec)
      curr = param_restore_args
      for step in p[:-1]:
        curr = curr.setdefault(str(step), {})
      curr[str(p[-1])] = orbax.checkpoint.ArrayRestoreArgs(
          restore_type=jax.Array,
          sharding=sharding,
          global_shape=exp,
          dtype=dtype,
      )
    restore_args = {"params": param_restore_args} if has_params_root else param_restore_args

  if restore_args is not None:
    restored = checkpointer.restore(path, restore_args=restore_args)
  else:
    restored = checkpointer.restore(path)

  if isinstance(restored, Mapping) and "params" in restored:
    restored = restored["params"]

  for path_item, var in nnx.to_flat_state(param_state):
    curr = restored
    for step in path_item:
      curr = _traverse_param_step(curr, step, path_item)

    if (actual := getattr(curr, "shape", None)) != (exp := var.value.shape):
      raise ValueError(f"Checkpoint parameter shape mismatch at {path_item}: {exp} vs" f" {actual}")
    pspec = compute_param_partition_spec(path_item, exp, axis_name, axis_size)
    expected_sharding = jax.sharding.NamedSharding(active_mesh, pspec) if active_mesh is not None else None
    if isinstance(curr, jax.Array) and (
        expected_sharding is None or getattr(curr, "sharding", None) == expected_sharding
    ):
      val = curr
    else:
      val = jnp.asarray(curr, dtype=var.value.dtype)
      if expected_sharding is not None:
        val = jax.device_put(val, expected_sharding)
    var.value = val

  def _init_if_abstract(path_entries, leaf):
    val = leaf.value if hasattr(leaf, "value") else leaf
    if isinstance(val, jax.ShapeDtypeStruct):
      path_tuple = tuple(getattr(k, "key", getattr(k, "attr", k)) for k in path_entries)
      new_val = jnp.zeros(val.shape, dtype=val.dtype)
      if active_mesh is not None:
        pspec = compute_param_partition_spec(path_tuple, val.shape, axis_name, axis_size)
        new_val = jax.device_put(new_val, jax.sharding.NamedSharding(active_mesh, pspec))
      if hasattr(leaf, "value"):
        leaf.value = new_val
        return leaf
      return new_val
    return leaf

  other_state = nnx.state(model).filter(lambda p, var: not isinstance(var, nnx.Param))
  other_state = jax.tree_util.tree_map_with_path(_init_if_abstract, other_state)
  nnx.update(model, param_state, other_state)
  return model


def _make_forward(return_aux: bool, return_mtp: bool) -> Any:
  """Builds a jitted forward function for aux and mtp output flags."""

  @nnx.jit
  def _f(
      m: nnx.Module,
      tok: jax.Array,
      pos: jax.Array | None,
      seg: jax.Array | None,
      nxt: jax.Array | None,
  ) -> Any:
    return m(
        tok,
        decoder_positions=pos,
        decoder_segment_ids=seg,
        return_aux=return_aux,
        return_mtp=return_mtp,
        next_tokens=nxt,
    )

  return _f


_JITTED_FORWARDS = {(aux, mtp): _make_forward(aux, mtp) for aux in (False, True) for mtp in (False, True)}


class ShardedJittedModelRunner:
  """Sharded and jitted model runner wrapping an NNX model inside a device mesh."""

  def __init__(self, model: nnx.Module, mesh: jax.sharding.Mesh):
    self.model = model
    self.mesh = mesh

  def __getattr__(self, name: str) -> Any:
    return getattr(self.model, name)

  def __call__(
      self,
      tokens: jax.Array,
      decoder_positions: jax.Array | None = None,
      decoder_segment_ids: jax.Array | None = None,
      *,
      return_aux: bool = False,
      return_mtp: bool = False,
      next_tokens: jax.Array | None = None,
      **kwargs: Any,
  ) -> Any:
    """Invokes forward pass inside the runner device mesh."""
    del kwargs
    fn = _JITTED_FORWARDS[(bool(return_aux), bool(return_mtp))]
    with jax.set_mesh(self.mesh):
      return fn(
          self.model,
          tokens,
          decoder_positions,
          decoder_segment_ids,
          next_tokens,
      )


def create_sharded_jitted_runner(
    argv: Sequence[str] | None = None,
    *,
    config: Any | None = None,
    mesh: jax.sharding.Mesh | None = None,
    load_parameters_path: str | epath.Path | None = None,
) -> ShardedJittedModelRunner:
  """Builds a mesh, constructs the model, restores weights, returns runner."""
  cfg = config if config is not None else pyconfig.initialize(list(argv or []))
  active_mesh = mesh if mesh is not None else create_device_mesh_for_config(cfg)
  ckpt = load_parameters_path or getattr(cfg, "load_parameters_path", "")
  with jax.set_mesh(active_mesh):
    if ckpt:
      model = nnx.eval_shape(lambda: m3_models.create_model(cfg, mesh=active_mesh, rngs=nnx.Rngs(params=0)))
      restore_model_parameters(model, ckpt, mesh=active_mesh)
    else:
      model = m3_models.create_model(cfg, mesh=active_mesh, rngs=nnx.Rngs(params=0))
      shard_model_parameters(model, active_mesh)
  return ShardedJittedModelRunner(model, active_mesh)
