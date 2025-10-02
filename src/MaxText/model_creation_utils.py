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
from typing import overload

from flax import nnx
import flax.linen as nn
import jax
from jax.sharding import Mesh
from MaxText import maxtext_utils
from MaxText import pyconfig
from MaxText.layers import quantizations
from MaxText.common_types import MODEL_MODE_TRAIN
from MaxText.layers import models
from orbax import checkpoint as ocp
from functools import partial
from etils import epath


@overload
def from_config(
    config: pyconfig.HyperParameters,
    devices: Sequence[jax.Device] | None = None,
    *,
    model_mode: str = MODEL_MODE_TRAIN,
) -> nn.Module: ...
@overload
def from_config(
    config: pyconfig.HyperParameters,
    devices: Sequence[jax.Device] | None = None,
    *,
    model_mode: str = MODEL_MODE_TRAIN,
    rngs: nnx.Rngs,
) -> models.Transformer: ...
def from_config(
    config: pyconfig.HyperParameters,
    devices: Sequence[jax.Device] | None = None,
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
  devices_array = maxtext_utils.create_device_mesh(config, devices)
  mesh = Mesh(devices_array, config.mesh_axes)
  model = create_model(config, mesh, model_mode=model_mode, rngs=rngs)

  # Return only the model
  return model

def get_transformer_model(config, mesh, quant, model_mode: str = MODEL_MODE_TRAIN, rngs: nnx.Rngs | None = None):
  """Returns the transformer model based on the configuration."""
  if config.model_fsdp_ag_once:
    if rngs is not None:
      raise NotImplementedError
    else:
      return models.ZeroOneTransformer(config, mesh, quant=quant, model_mode=model_mode)
  else:
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


def create_nnx_model(config, devices=None):
  """Creates a NNX model with sharded parameters, possibly loading from a checkpoint."""

  def create_model():
    init_rng = jax.random.PRNGKey(config.init_weights_seed)
    return from_config(config, devices, rngs=nnx.Rngs(params=init_rng, dropout=1))

  abstract_model = nnx.eval_shape(create_model)
  graphdef, abstract_state = nnx.split(abstract_model)
  specs = nnx.get_partition_spec(abstract_state)
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
    model = create_model()
    return nnx.state(model)

  with mesh:
    # Create the model with sharded parameters.
    sharded_state = create_sharded_state()
    model = nnx.merge(graphdef, sharded_state)

    if config.load_parameters_path:
      target_for_restore = jax.tree.map(
          lambda v: v.value,
          sharded_state,
          is_leaf=lambda n: isinstance(n, nnx.Variable),
      )

      try:
        ckptr = ocp.Checkpointer(
            ocp.PyTreeCheckpointHandler(
                restore_concurrent_gb=None,
                save_concurrent_gb=None,
                use_ocdbt=True,
                use_zarr3=True,
            )
        )
        # This is a memory optimization. We don't want to restore the entire checkpoint - only the params.
        # Rather than passing the entire abstract state, which could unnecessarily restore opt_state and
        # waste memory, we instead restore the params field of the checkpoint (which itself may be a dictionary
        #  containing a key named 'params').
        restore_args = ocp.checkpoint_utils.construct_restore_args(target_for_restore)
        restored = ckptr.restore(
            epath.Path(config.load_parameters_path),
            item={"params": {"params": target_for_restore}},
            transforms={},
            restore_args={"params": {"params": restore_args}},
        )
        checkpoint = restored["params"]["params"]

        if checkpoint:
          nnx.update(model, checkpoint)

      except Exception as e:
        raise ValueError(f"Checkpoint loading failed: {e}")

    return model, mesh
