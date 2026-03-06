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

from collections.abc import Sequence
from functools import partial
import re
from typing import overload

from etils import epath
from flax import nnx
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.sharding import AxisType, Mesh
from maxtext.configs import pyconfig
from maxtext.common.common_types import DecoderBlockType, MODEL_MODE_TRAIN, ShardMode
from maxtext.layers import quantizations
from maxtext.models import models
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from orbax import checkpoint as ocp


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


def _sort_sublayer_keys(keys):
  """Sort sub-layer key names numerically by trailing digit."""
  return sorted(keys, key=lambda k: int(re.search(r"\d+$", k).group()))


def _get_decoder_layer_groups(config) -> list[tuple[str, int]]:
  """Returns [(layer_name, count), ...] tuples based on decoder block type."""
  if config.decoder_block == DecoderBlockType.DEEPSEEK:
    return [
        ("dense_layers", config.first_num_dense_layers),
        ("moe_layers", config.base_num_decoder_layers - config.first_num_dense_layers),
    ]
  return [("layers", config.base_num_decoder_layers)]


def _is_scanned_checkpoint(metadata_tree, is_nnx_checkpoint, config) -> bool:
  """Detects whether a checkpoint uses scanned (stacked) layer storage."""
  if is_nnx_checkpoint:
    decoder_tree = metadata_tree.get("decoder", {})
  else:
    decoder_tree = metadata_tree.get("params", {}).get("params", {}).get("decoder", {})
  layer_groups = _get_decoder_layer_groups(config)
  return any(name in decoder_tree for name, _ in layer_groups)


def _build_scanned_restore_target(target, layer_groups, scan_axis, is_nnx_format, metadata_decoder_tree):
  """Stacks per-layer restore target entries into a single scanned entry.

  Takes an unscanned target (with `layers_0`, `layers_1`, ... entries) and
  stacks them into a single `layers` entry to match the scanned checkpoint
  structure, so orbax can load the checkpoint into a matching target.

  For ScannableBlock checkpoints, the checkpoint nests sub-layer keys (e.g.,
  `layers_0`, `layers_1`) under the layer group name. This function detects
  that from `metadata_decoder_tree` and groups layers accordingly.
  """
  if is_nnx_format:
    decoder = target["decoder"]
  else:
    decoder = target["params"]["params"]["decoder"]

  for layer_name, num_layers in layer_groups:
    if num_layers == 0:
      continue

    ckpt_subtree = metadata_decoder_tree.get(layer_name, {})
    # A ScannableBlock checkpoint nests sub-layer keys (e.g., layers_0, layer_0)
    # under the layer group name instead of providing a single stacked array.
    sub_layer_names = _sort_sublayer_keys(k for k in ckpt_subtree)

    if sub_layer_names:
      # Nested ScannableBlock: group unscanned layers into per-sub-layer stacks.
      # Mapping: sub_layer j, scan step s → global layer s*K + j
      K = len(sub_layer_names)
      scan_steps = num_layers // K
      sub_layers = {}
      for j, sub_name in enumerate(sub_layer_names):
        per_step = [decoder.pop(f"{layer_name}_{s * K + j}") for s in range(scan_steps)]
        sub_layers[sub_name] = jax.tree.map(lambda *a: jnp.stack(a, axis=scan_axis), *per_step)
      decoder[layer_name] = sub_layers
    else:
      # Flat scan: stack all layers into a single array.
      per_layer = [decoder.pop(f"{layer_name}_{i}") for i in range(num_layers)]
      decoder[layer_name] = jax.tree.map(lambda *a: jnp.stack(a, axis=scan_axis), *per_layer)

  return target


def _unscan_checkpoint_dict(checkpoint, layer_groups, scan_axis):
  """Splits stacked scanned layer arrays into per-layer entries.

  For flat scan, converts `checkpoint["decoder"]["layers"]` (shape [N, ...]) into
  `checkpoint["decoder"]["layers_0"]`, ..., `checkpoint["decoder"]["layers_{N-1}"]`.

  For nested ScannableBlock scan, converts
  `checkpoint["decoder"]["layers"]["layers_j"]` (shape [S, ...]) into
  `checkpoint["decoder"]["layers_{s*K+j}"]` for each scan step s and sub-layer j.
  """
  decoder = checkpoint["decoder"]
  for layer_name, num_layers in layer_groups:
    if layer_name not in decoder:
      continue
    scanned = decoder.pop(layer_name)

    if isinstance(scanned, dict):
      # Nested ScannableBlock: scanned is {sub_name_j: array[scan_steps, ...]}
      sub_layer_names = _sort_sublayer_keys(scanned.keys())
      K = len(sub_layer_names)
      scan_steps = num_layers // K
      for s in range(scan_steps):
        for j, sub_name in enumerate(sub_layer_names):
          decoder[f"{layer_name}_{s * K + j}"] = jax.tree.map(
              lambda x, s=s: jnp.take(x, s, axis=scan_axis), scanned[sub_name]
          )
    else:
      # Flat scan: scanned is array[N, ...]
      for i in range(num_layers):
        decoder[f"{layer_name}_{i}"] = jax.tree.map(
            lambda x, i=i: jnp.take(x, i, axis=scan_axis), scanned
        )


def create_nnx_model(config, mesh=None, devices=None, model_mode=MODEL_MODE_TRAIN, rng_key=None, unscan_checkpoint: bool = False):
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
          restore_args = {"params": {"params": ocp.checkpoint_utils.construct_restore_args(target_for_restore)}}
        else:
          # structure of nnx checkpoint: {'decoder': {'value': ...}}
          target_for_restore = jax.tree.map(
              lambda v: {"value": v.value},
              sharded_state,
              is_leaf=lambda n: isinstance(n, nnx.Variable),
          )
          item_to_restore = target_for_restore
          restore_args = ocp.checkpoint_utils.construct_restore_args(target_for_restore)

        # Extract decoder sub-tree from metadata for ScannableBlock sub-layer detection
        if is_nnx_checkpoint:
          metadata_decoder_tree = metadata.item_metadata.tree.get("decoder", {})
        else:
          metadata_decoder_tree = (
              metadata.item_metadata.tree.get("params", {})
              .get("params", {})
              .get("decoder", {})
          )

        # Detect if checkpoint is scanned but model expects unscanned
        layer_groups = _get_decoder_layer_groups(config)
        checkpoint_is_scanned = (
            unscan_checkpoint
            and not config.scan_layers
            and _is_scanned_checkpoint(metadata.item_metadata.tree, is_nnx_checkpoint, config)
        )

        if checkpoint_is_scanned:
          item_to_restore = _build_scanned_restore_target(
              item_to_restore, layer_groups, config.param_scan_axis, is_nnx_checkpoint, metadata_decoder_tree
          )
          if is_nnx_checkpoint:
            restore_args = ocp.checkpoint_utils.construct_restore_args(item_to_restore)
          else:
            restore_args = {
                "params": {
                    "params": ocp.checkpoint_utils.construct_restore_args(item_to_restore["params"]["params"])
                }
            }

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

        if checkpoint_is_scanned:
          _unscan_checkpoint_dict(checkpoint, layer_groups, config.param_scan_axis)

        if checkpoint:
          nnx.update(model, checkpoint)

      except Exception as e:
        raise ValueError(f"Checkpoint loading failed: {e}") from e

    return model, mesh
