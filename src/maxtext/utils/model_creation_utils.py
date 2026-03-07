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

import os
from collections.abc import Sequence
from functools import partial
from typing import overload

from etils import epath
from flax import nnx
import flax.linen as nn
import jax
from jax.sharding import AxisType, Mesh
from maxtext.configs import pyconfig
from maxtext.common.common_types import MODEL_MODE_TRAIN, ShardMode
from maxtext.layers import quantizations
from maxtext.models import models
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import max_logging
from maxtext.utils.globals import MAXTEXT_PKG_DIR
from maxtext.checkpoint_conversion import to_maxtext
from maxtext.checkpoint_conversion.utils.utils import HF_IDS
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


def from_pretrained(argv: list[str] | None = None, lazy_load_tensors=False, **kwargs):
  """
  Obtain a MaxText model by providing minimal configs
  It at least needs the following argument
    Args:
      argv: List of configs in string format.
      lazy_load_tensors: Whether to use lazy loading of HF tensors.
  """
  if argv is None:
    argv = [""]
  for k, v in kwargs.items():
    argv.append(f"{k}={v}")

  yaml_file = False
  model_name_seen = False
  base_output_directory_seen = False
  load_parameters_path_seen = None
  hf_access_token_seen = False
  tokenizer_path_seen = False

  base_output_directory = os.path.abspath("maxtext_output")
  model_name = None

  for a in argv:
    if "model_name" in a:
      model_name_seen = True
      if "=" in a:
        model_name = a.split("=", 1)[1].strip().strip("\"'")
    if "base_output_directory" in a:
      base_output_directory_seen = True
      if "=" in a:
        base_output_directory = a.split("=", 1)[1].strip().strip("\"'")
    if "load_parameters_path" in a:
      load_parameters_path_seen = True
    if "hf_access_token" in a:
      hf_access_token_seen = True
    if "tokenizer_path" in a:
      tokenizer_path_seen = True
    if ".yml" in a:
      yaml_file = a

  if not yaml_file:
    max_logging.warning(
        "yaml file not provided, using default base.yml. If this is not intended,"
        " then please provide the intended .yml file as an argument. e.g., "
        "src/maxtext/configs/post_train/rl.yml for post-training"
    )
    yaml_file = True
    argv.insert(1, f"{MAXTEXT_PKG_DIR}/configs/base.yml")
  else:
    # verify that the .yml is in index 1 in the list
    if ".yml" not in argv[1]:
      # move yaml_file to index 1 in the list
      argv.insert(1, argv.pop(argv.index(yaml_file)))

  if not model_name_seen:
    raise ValueError("model_name must be provided")
  if not base_output_directory_seen:
    max_logging.warning("base_output_directory is not provided; Using local directory called maxtext_output")
    argv.append(f"base_output_directory={base_output_directory}")

  # take HF_TOKEN from env
  if not hf_access_token_seen:
    hf_access_token = os.environ.get("HF_TOKEN")
    if hf_access_token:
      argv.append(f"hf_access_token={hf_access_token}")

  if not load_parameters_path_seen:
    if not hf_access_token_seen:
      raise ValueError("hf_access_token must be provided when not providing a pre-existing checkpoint")
    max_logging.warning("Checkpoint path is not provided, converting checkpoint to orbax format for MaxText")
    argv.extend(
        [
            "use_multimodal=false",
            "scan_layers=true",
            "skip_jax_distributed_system=True",
        ]
    )

  hf_model_path = None
  revision = None
  simulated_cpu_devices_count = 16

  if not tokenizer_path_seen and model_name:

    model_name_original = model_name.replace("-Instruct", "") if "-Instruct" in model_name else model_name
    tokenizer_path = hf_model_path or HF_IDS.get(model_name_original)
    if tokenizer_path:
      argv.append(f"tokenizer_path={tokenizer_path}")

  prev_xla_flags = os.environ.get("XLA_FLAGS")
  os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={simulated_cpu_devices_count}"

  prev_jax_platforms = jax.config.jax_platforms
  jax.config.update("jax_platforms", "cpu")
  to_maxtext.main(
      argv,
      hf_model_path=hf_model_path,
      revision=revision,
      lazy_load_tensors=lazy_load_tensors,
      simulated_cpu_devices_count=simulated_cpu_devices_count,
  )
  jax.config.update("jax_platforms", prev_jax_platforms)

  if prev_xla_flags is None:
    os.environ.pop("XLA_FLAGS", None)
  else:
    os.environ["XLA_FLAGS"] = prev_xla_flags

    load_parameters_path = os.path.join(base_output_directory, "0", "items")
    argv.append(f"load_parameters_path={load_parameters_path}")

  print(f"Anisha: {argv}")
  config = pyconfig.initialize_pydantic(argv)

  model, mesh = create_nnx_model(config)
  return model, mesh, config


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
