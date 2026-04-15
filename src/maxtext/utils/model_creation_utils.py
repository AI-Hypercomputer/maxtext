# Copyright 2023–2025 Google LLC
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

# pylint: disable=bare-except, consider-using-generator
""" Utils that are only interesting for creating a model in MaxText. """

import dataclasses
from collections.abc import Sequence
from functools import partial
from typing import overload

from etils import epath
from flax import nnx
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.sharding import AxisType, Mesh
from maxtext.configs import pyconfig
from maxtext.common.common_types import MODEL_MODE_TRAIN, ShardMode
from maxtext.layers import quantizations
from maxtext.models import models
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from orbax import checkpoint as ocp

try:
  from orbax.checkpoint.metadata import ArrayMetadata as _OrbaxArrayMetadata

  def _is_orbax_array_metadata(x):
    return isinstance(x, _OrbaxArrayMetadata)

except ImportError:

  def _is_orbax_array_metadata(x):
    return hasattr(x, "shape") and hasattr(x, "sharding") and hasattr(x, "dtype") and not isinstance(x, jax.Array)


def _expand_checkpoint_to_model_shapes(ckpt_arr, model_arr):
  """Expand ckpt_arr to model_arr's shape and re-shard to model_arr's sharding.

  Used to expand checkpoint KV-head (and similar) arrays that were saved with
  fewer heads than the padded model shape requires (e.g. due to TP/EP padding
  in adapter.py).  Each dimension must divide evenly into the corresponding
  model dimension.

  Uses jnp.repeat so that each original slice is placed adjacent to its copies.
  For GQA with TP, device i needs KV head i//ratio from the original checkpoint,
  so the correct layout is e.g. [h0, h0, h1, h1, h2, h2, h3, h3] rather than
  [h0, h1, h2, h3, h0, h1, h2, h3].
  """
  ckpt_shape = ckpt_arr.shape
  model_shape = model_arr.shape
  if ckpt_shape == model_shape:
    return jax.device_put(ckpt_arr, model_arr.sharding)
  if len(ckpt_shape) != len(model_shape):
    raise ValueError(
        f"Checkpoint and model arrays have different ranks: {ckpt_shape} vs {model_shape}. "
        "If the checkpoint was saved with scan_layers=True (stacked layers), convert it to "
        "unscanned format before loading with vLLM (vllm.yml sets scan_layers=False)."
    )
  result = ckpt_arr
  for axis, (ckpt_dim, model_dim) in enumerate(zip(ckpt_shape, model_shape)):
    if model_dim % ckpt_dim != 0:
      raise ValueError(
          f"Model dimension {model_dim} is not evenly divisible by checkpoint dimension {ckpt_dim}."
          f" Full shapes — checkpoint: {ckpt_shape}, model: {model_shape}"
      )
    if model_dim != ckpt_dim:
      result = jnp.repeat(result, model_dim // ckpt_dim, axis=axis)
  return jax.device_put(result, model_arr.sharding)


def _fix_restore_args_for_shape_mismatch(restore_args, stored_metadata_tree, mesh):
  """Use replicated sharding for arrays whose checkpoint shape differs from the model shape.

  When the model is initialized with padded shapes (e.g. KV heads padded to match
  TP size) but the checkpoint was saved with smaller shapes, Orbax will reject the
  restore because the provided sharding is incompatible with the stored shape.
  For those arrays we switch to a fully-replicated sharding and clear global_shape
  so Orbax loads the array as-written.  _expand_checkpoint_to_model_shapes then
  expands and re-shards the loaded arrays to match the model.

  Uses tree_map_with_path so each ArrayRestoreArgs is looked up by path in the
  metadata dict — avoids ordering/count mismatches from flattening two trees with
  different pytree node types (e.g. nnx.State vs plain dict) independently.
  """
  replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

  def _key_str(key):
    """Extract string name from a JAX path key (DictKey, GetAttrKey, etc.)."""
    if hasattr(key, "key"):
      return str(key.key)
    if hasattr(key, "attr"):
      return str(key.attr)
    return str(key)

  def _lookup_stored_meta(path):
    """Navigate stored_metadata_tree using path keys from the restore_args tree."""
    node = stored_metadata_tree
    for key in path:
      name = _key_str(key)
      if isinstance(node, dict) and name in node:
        node = node[name]
      else:
        return None
    return node

  mismatched_paths = []

  def _fix_one(path, restore_arg):
    if not isinstance(restore_arg, ocp.ArrayRestoreArgs):
      return restore_arg
    stored_meta = _lookup_stored_meta(path)
    if stored_meta is not None and _is_orbax_array_metadata(stored_meta):
      stored_shape = tuple(stored_meta.shape)
      if (
          restore_arg.global_shape is not None
          and restore_arg.global_shape != stored_shape
          and len(stored_shape) == len(restore_arg.global_shape)
      ):
        mismatched_paths.append(
            f"  {'.'.join(_key_str(k) for k in path)}: stored={stored_shape} -> model={restore_arg.global_shape}"
        )
        return dataclasses.replace(
            restore_arg, global_shape=None, shape=None, sharding=replicated, mesh=None, mesh_axes=None
        )
    return restore_arg

  fixed = jax.tree_util.tree_map_with_path(_fix_one, restore_args, is_leaf=lambda x: isinstance(x, ocp.ArrayRestoreArgs))
  if mismatched_paths:
    max_logging.log(
        f"Checkpoint shape mismatches ({len(mismatched_paths)} arrays): loading with replicated "
        "sharding and expanding to model shape after restore.\n" + "\n".join(mismatched_paths)
    )
  return fixed


@overload
def from_config(
    config: pyconfig.HyperParameters,
    devices: Sequence[jax.Device] | None = None,
    mesh: Mesh | None = None,
    *,
    model_mode: str = MODEL_MODE_TRAIN,
) -> nn.Module:
  ...


@overload
def from_config(
    config: pyconfig.HyperParameters,
    devices: Sequence[jax.Device] | None = None,
    mesh: Mesh | None = None,
    *,
    model_mode: str = MODEL_MODE_TRAIN,
    rngs: nnx.Rngs,
) -> models.Transformer:
  ...


def from_config(
    config: pyconfig.HyperParameters,
    devices: Sequence[jax.Device] | None = None,
    mesh: Mesh | None = None,
    *,
    model_mode: str = MODEL_MODE_TRAIN,
    rngs: nnx.Rngs | None = None,
) -> nn.Module | models.Transformer:
  """Load a pretrained MaxText model from checkpoint.

  This function loads a model from a checkpoint.

  Args:
      config: Config object.
      devices: Sequence of devices to use for the model. If None, use all
        available devices.

  Returns:
      Transformer: The loaded model instance (only the model)

  Example:
      model = from_config(config)
  """
  if mesh is None:
    devices_array = maxtext_utils.create_device_mesh(config, devices)

    if config.shard_mode == ShardMode.EXPLICIT:
      axis_types = tuple([AxisType.Explicit] * len(config.mesh_axes))
    else:
      axis_types = tuple([AxisType.Auto] * len(config.mesh_axes))

    mesh = Mesh(devices_array, config.mesh_axes, axis_types=axis_types)

  model = create_model(config, mesh, model_mode=model_mode, rngs=rngs)

  # Return only the model
  return model


def get_transformer_model(config, mesh, quant, model_mode: str = MODEL_MODE_TRAIN, rngs: nnx.Rngs | None = None):
  """Returns the transformer model based on the configuration."""
  if rngs is not None:
    return models.Transformer(config, mesh, quant=quant, rngs=rngs, model_mode=model_mode)
  else:
    return models.transformer_as_linen(config, mesh, quant=quant, model_mode=model_mode)


def create_model(config, mesh, model_mode: str = MODEL_MODE_TRAIN, rngs: nnx.Rngs | None = None):
  """Instantiates and returns the model object, sharded across the mesh."""
  # Model definition
  quant = quantizations.configure_quantization(config)
  model = get_transformer_model(config, mesh, quant, model_mode=model_mode, rngs=rngs)
  model = quantizations.maybe_quantize_model(model, config)
  return model


def create_nnx_abstract_model(config, mesh, model_mode=MODEL_MODE_TRAIN, rng_key=None):
  """Returns (_create_model_partial, abstract_model) for AOT compilation.

  Unlike create_nnx_model, this does not shard parameters or load checkpoints.
  It only builds the abstract shape/dtype structure needed by get_abstract_state
  and optimizer construction (e.g. Muon).

  Args:
    config: the configuration
    mesh: the device mesh
    model_mode: train or inference
    rng_key: optional RNG key

  Returns:
    (_create_model_partial, abstract_model) where _create_model_partial() creates
    a concrete model instance and abstract_model is the eval_shape result.
  """

  def _create_model(rng_key=None):
    if rng_key is None:
      rng_key = jax.random.PRNGKey(config.init_weights_seed)
    rngs = nnx.Rngs(params=rng_key, dropout=1)
    return from_config(config, mesh=mesh, rngs=rngs, model_mode=model_mode)

  _create_model_partial = partial(_create_model, rng_key=rng_key)

  with nn.logical_axis_rules(config.logical_axis_rules):
    abstract_model = nnx.eval_shape(_create_model_partial)

  return _create_model_partial, abstract_model


def create_nnx_model(config, mesh=None, devices=None, model_mode=MODEL_MODE_TRAIN, rng_key=None):
  """Creates a NNX model with sharded parameters, possibly loading from a checkpoint."""

  def _create_model(mesh: Mesh | None = None, model_mode: str = MODEL_MODE_TRAIN, rng_key: jax.Array | None = None):
    if rng_key is None:
      rng_key = jax.random.PRNGKey(config.init_weights_seed)

    if model_mode == MODEL_MODE_TRAIN:
      rngs = nnx.Rngs(params=rng_key, dropout=1)
    else:
      rngs = nnx.Rngs(params=rng_key)  # disable dropout RNG for inference

    return from_config(config, devices, mesh, rngs=rngs, model_mode=model_mode)

  _create_model_partial = partial(_create_model, mesh=mesh, model_mode=model_mode, rng_key=rng_key)

  with nn.logical_axis_rules(config.logical_axis_rules):
    abstract_model = nnx.eval_shape(_create_model_partial)
  graphdef, abstract_state = nnx.split(abstract_model)
  specs = nnx.get_partition_spec(abstract_state)

  if mesh is None:
    mesh = abstract_model.mesh

  # JIT a function that creates the model state with proper sharding from the start.
  # By providing out_shardings, we instruct JAX to produce sharded output directly,
  # avoiding a large intermediate allocation on a single device.
  with nn.logical_axis_rules(config.logical_axis_rules):
    out_shardings = nn.logical_to_mesh_sharding(specs, mesh)

  @partial(jax.jit, out_shardings=out_shardings)
  def create_sharded_state():
    # This will be JIT-compiled. JAX knows the output sharding and can
    # initialize the parameters directly on the target devices in a sharded way.
    model = _create_model_partial()
    return nnx.state(model)

  with mesh:
    # Create the model with sharded parameters.
    with nn.logical_axis_rules(config.logical_axis_rules):
      sharded_state = create_sharded_state()
    model = nnx.merge(graphdef, sharded_state)

    # print weights sharding info under debug sharding mode
    if config.debug_sharding:
      max_utils.print_non_trivial_mesh_axis(model.mesh)
      maxtext_utils.print_shardings_params(
          params=sharded_state,
          params_sharding=out_shardings,
          mesh=model.mesh,
          logical_annotations=specs,
      )

    if config.load_parameters_path:
      try:
        ckptr = ocp.Checkpointer(
            ocp.PyTreeCheckpointHandler(
                restore_concurrent_gb=config.checkpoint_storage_concurrent_gb,
                save_concurrent_gb=config.checkpoint_storage_concurrent_gb,
                use_ocdbt=config.checkpoint_storage_use_ocdbt,
                use_zarr3=config.checkpoint_storage_use_zarr3,
            )
        )

        # This is a memory optimization. We don't want to restore the entire checkpoint - only the params.
        # Rather than passing the entire abstract state, which could unnecessarily restore opt_state and
        # waste memory, we instead restore the params field of the checkpoint (which itself may be a dictionary
        #  containing a key named 'params').

        # Get the structure of checkpoint in `config.load_parameters_path`
        metadata = ckptr.metadata(config.load_parameters_path)

        is_nnx_checkpoint = True
        if (
            "params" in metadata.item_metadata.tree.keys()
            and "params" in metadata.item_metadata.tree.get("params", {}).keys()
        ):
          # structure of linen checkpoint: {'params': {'params': {'decoder': ...}}}
          is_nnx_checkpoint = False
          target_for_restore = jax.tree.map(
              lambda v: v.value,
              sharded_state,
              is_leaf=lambda n: hasattr(n, "value"),
          )

          item_to_restore = {"params": {"params": target_for_restore}}
          base_restore_args = ocp.checkpoint_utils.construct_restore_args(target_for_restore)
          restore_args = {
              "params": {
                  "params": _fix_restore_args_for_shape_mismatch(
                      base_restore_args,
                      metadata.item_metadata.tree["params"]["params"],
                      mesh,
                  )
              }
          }
        else:
          # structure of nnx checkpoint: {'decoder': {'value': ...}}
          target_for_restore = jax.tree.map(
              lambda v: {"value": v.value},
              sharded_state,
              is_leaf=lambda n: isinstance(n, nnx.Variable),
          )
          item_to_restore = target_for_restore
          base_restore_args = ocp.checkpoint_utils.construct_restore_args(target_for_restore)
          restore_args = _fix_restore_args_for_shape_mismatch(
              base_restore_args,
              metadata.item_metadata.tree,
              mesh,
          )

        restored = ckptr.restore(
            epath.Path(config.load_parameters_path),
            item=item_to_restore,
            transforms={},
            restore_args=restore_args,
        )

        if is_nnx_checkpoint:
          checkpoint = jax.tree.map(
              lambda v: v["value"],
              restored,
              is_leaf=lambda x: isinstance(x, dict) and "value" in x and not isinstance(x.get("value"), dict),
          )
        else:
          checkpoint = restored["params"]["params"]

        loaded_count = len(jax.tree_util.tree_leaves(checkpoint))
        expected_count = len(jax.tree_util.tree_leaves(target_for_restore))
        if loaded_count < expected_count:
          raise ValueError(
              f"Checkpoint at '{config.load_parameters_path}' loaded only {loaded_count} of {expected_count} "
              "expected parameter arrays. This usually means a scanned (stacked-layers) checkpoint was provided "
              "where an unscanned checkpoint is required. Please convert the checkpoint to unscanned format first."
          )

        if checkpoint:
          model_arrays = jax.tree.map(
              lambda v: v.value,
              sharded_state,
              is_leaf=lambda n: isinstance(n, nnx.Variable),
          )
          checkpoint = jax.tree.map(_expand_checkpoint_to_model_shapes, checkpoint, model_arrays)
          nnx.update(model, checkpoint)

      except Exception as e:
        raise ValueError(f"Checkpoint loading failed: {e}") from e

    return model, mesh
