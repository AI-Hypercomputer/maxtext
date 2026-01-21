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
from typing import overload
from functools import partial
from etils import epath

from etils import epath
from flax import nnx
import flax.linen as nn
import jax
from jax.sharding import Mesh
from maxtext.configs import pyconfig
from maxtext.common.common_types import MODEL_MODE_TRAIN, ShardMode
from maxtext.layers import quantizations
from maxtext.models import models
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils, maxtext_utils_nnx, max_utils
from orbax import checkpoint as ocp
from orbax import checkpoint as ocp



@overload
def from_config(
    config: pyconfig.HyperParameters,
    devices: Sequence[jax.Device] | None = None,
    mesh: Mesh | None = None,
    *,
    model_mode: str = MODEL_MODE_TRAIN,
    rngs: None = None,
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
    mesh = maxtext_utils.get_mesh_from_config(config, devices)
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


def create_nnx_model(config, mesh=None, devices=None, model_mode=MODEL_MODE_TRAIN, rng_key=None):
  """Creates a NNX model with sharded parameters, possibly loading from a checkpoint."""
  is_training = model_mode == MODEL_MODE_TRAIN

  def _create_model(mesh: Mesh | None = None, model_mode: str = MODEL_MODE_TRAIN, rng_key: jax.Array | None = None):
    rngs = maxtext_utils_nnx.create_nnx_rngs(config, is_training=is_training, rng_key=rng_key)
    return from_config(config, devices, mesh, rngs=rngs, model_mode=model_mode)

  _create_model_partial = partial(_create_model, mesh=mesh, model_mode=model_mode, rng_key=rng_key)

  with nn.logical_axis_rules(config.logical_axis_rules):
    abstract_model = nnx.eval_shape(_create_model_partial)
  graphdef, abstract_state = nnx.split(abstract_model)
  specs = nnx.get_partition_spec(abstract_state)

  if mesh is None:
    mesh = abstract_model.mesh

  # Note for pure_nnx:
  # Currently, the NNX model returned has a linen decoder wrapped to NNX. So it is not a pure NNX model and
  # we still need to use nn.logical_axis_rules(config.logical_axis_rules) to get the out sharding from the linen
  # LogicallyPartitioned structure.
  # In the future if the pure NNX model is used, with pure NNX's eager sharding, there will be no LogicallyPartitioned
  # structure in the abstract state and we can get the sharded state with the following code:
  #     graphdef, state = nnx.get_abstract_model(_create_model_partial, mesh)
  #     abstract_model = nnx.merge(graphdef, state)
  #     model = maxtext_utils_nnx.create_nnx_sharded_model(abstract_model, _create_model_partial, mesh=mesh)
  #     sharded_state = nnx.state(model)

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

        if checkpoint:
          nnx.update(model, checkpoint)

      except Exception as e:
        raise ValueError(f"Checkpoint loading failed: {e}") from e

    return model, mesh
