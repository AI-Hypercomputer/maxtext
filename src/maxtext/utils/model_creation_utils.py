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
"""Utils that are only interesting for creating a model in MaxText."""

import dataclasses
import collections
from collections.abc import Sequence
from functools import partial
import os
import subprocess
import sys
from typing import overload
from etils import epath
from flax import nnx
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from maxtext.configs import pyconfig
from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_TRAIN
from maxtext.layers import quantizations
from maxtext.models import models
from maxtext.utils import max_logging
from maxtext.utils import max_utils, maxtext_utils, maxtext_utils_nnx
from maxtext.integration.tunix.tunix_adapter import TunixMaxTextAdapter
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
  rank_mismatched_paths = []
  missing_paths = []  # paths in model that are absent from the checkpoint tree
  found_array_count = [0]

  def _fix_one(path, restore_arg):
    if not isinstance(restore_arg, ocp.ArrayRestoreArgs):
      return restore_arg
    stored_meta = _lookup_stored_meta(path)
    if stored_meta is None:
      missing_paths.append(f"  {'.'.join(_key_str(k) for k in path)}")
      return restore_arg
    if _is_orbax_array_metadata(stored_meta):
      stored_shape = tuple(stored_meta.shape)
      if restore_arg.global_shape is not None and restore_arg.global_shape != stored_shape:
        if len(stored_shape) != len(restore_arg.global_shape):
          rank_mismatched_paths.append(
              f"  {'.'.join(_key_str(k) for k in path)}: "
              f"checkpoint shape {stored_shape} (rank {len(stored_shape)}) "
              f"vs model shape {restore_arg.global_shape} (rank {len(restore_arg.global_shape)})"
          )
        else:
          mismatched_paths.append(
              f"  {'.'.join(_key_str(k) for k in path)}: stored={stored_shape} -> model={restore_arg.global_shape}"
          )
          found_array_count[0] += 1
          return dataclasses.replace(
              restore_arg, global_shape=None, shape=None, sharding=replicated, mesh=None, mesh_axes=None
          )
      else:
        found_array_count[0] += 1
    return restore_arg

  fixed = jax.tree_util.tree_map_with_path(_fix_one, restore_args, is_leaf=lambda x: isinstance(x, ocp.ArrayRestoreArgs))
  if rank_mismatched_paths:
    sample = "\n".join(rank_mismatched_paths[:5])
    more = f"\n  ... and {len(rank_mismatched_paths) - 5} more" if len(rank_mismatched_paths) > 5 else ""
    raise ValueError(
        f"Checkpoint rank mismatches detected ({len(rank_mismatched_paths)} arrays). "
        "This usually means a scanned (scan_layers=True) checkpoint was loaded with "
        "scan_layers=False, or vice versa. Please ensure the checkpoint format matches "
        f"the scan_layers setting.\n{sample}{more}"
    )

  # Detect structural mismatch (e.g. scanned checkpoint loaded into unscanned model).
  # In that case the checkpoint tree has "layers" (all layers stacked) but the model
  # expects "layers_0", "layers_1", etc., so _lookup_stored_meta returns None for every
  # layer parameter and nearly all paths end up in missing_paths.
  total_arrays = found_array_count[0] + len(rank_mismatched_paths) + len(missing_paths)
  if total_arrays > 0 and len(missing_paths) / total_arrays > 0.8:
    sample = "\n".join(missing_paths[:5])
    more = f"\n  ... and {len(missing_paths) - 5} more" if len(missing_paths) > 5 else ""
    raise ValueError(
        f"Checkpoint structure mismatch: {len(missing_paths)} of {total_arrays} model parameter "
        "paths were not found in the checkpoint. "
        "This usually means a scanned (scan_layers=True) checkpoint is being loaded with "
        "scan_layers=False, or vice versa. Please ensure the checkpoint format matches the "
        f"scan_layers setting.\nExample missing paths:\n{sample}{more}"
    )

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


def create_nnx_abstract_model(config, mesh, model_mode=MODEL_MODE_TRAIN, rng_key=None):
  """Returns (_create_model_partial, abstract_model) for AOT compilation.

  This does not shard parameters or load checkpoints. It only builds the
  abstract shape/dtype structure needed by get_abstract_state and optimizer
  construction (e.g. Muon).

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
    rngs = maxtext_utils_nnx.create_nnx_rngs(config, model_mode=model_mode, rng_key=rng_key)
    return from_config(config, mesh=mesh, rngs=rngs, model_mode=model_mode)

  _create_model_partial = partial(_create_model, rng_key=rng_key)

  with nn.logical_axis_rules(config.logical_axis_rules):
    abstract_model = nnx.eval_shape(_create_model_partial)

  return _create_model_partial, abstract_model


def setup_configs_and_devices(argv: list[str] | None = None, kwargs: dict | None = None, **extra_kwargs):
  """Setup device allocation and configs for training and inference.
  This API is particularly useful for Reinforcement Learning where we might split the available
  devices into separate mesh for trainer and sampler
  """
  if argv is None:
    argv = [""]

  combined_kwargs = dict(kwargs) if kwargs else {}
  combined_kwargs.update(extra_kwargs)
  config = pyconfig.initialize_pydantic(argv, **combined_kwargs)
  devices = jax.devices()
  if config.num_trainer_slices == -1 and config.num_samplers_slices == -1:
    max_logging.log("Running on a single slice")
    num_vms = len(devices) // config.chips_per_vm
    trainer_devices = devices
    sampler_devices = devices
    if num_vms >= 2 and config.use_pathways:
      # Multiple hosts with Pathways - potentially split devices for trainer and sampler
      # based on trainer_devices_fraction and sampler_devices_fraction
      max_logging.log(f"{num_vms} VMs detected, allocating trainer and sampler devices, and using Pathways.")
      num_devices = len(devices)
      num_trainer_devices = int(num_devices * config.trainer_devices_fraction)
      num_sampler_devices = int(num_devices * config.sampler_devices_fraction)
      trainer_devices = devices[:num_trainer_devices]
      sampler_devices = devices[num_devices - num_sampler_devices :]
      if config.trainer_devices_fraction != 1.0:
        max_logging.log(f"Using first {len(trainer_devices)} devices as Trainer devices")
      if config.sampler_devices_fraction != 1.0:
        max_logging.log(f"Using last {len(sampler_devices)} devices as Sampler devices")
    trainer_config = config
    sampler_config = config
  elif config.num_trainer_slices > 0 and config.num_samplers_slices > 0:
    max_logging.log("Running with Multislice")
    devices_by_slice = collections.defaultdict(list)
    for d in devices:
      devices_by_slice[d.slice_index].append(d)
    slice_indices = sorted(devices_by_slice.keys())

    if len(slice_indices) < config.num_trainer_slices + config.num_samplers_slices:
      raise ValueError("Not enough slices for trainer and samplers")

    trainer_devices = []
    for i in range(config.num_trainer_slices):
      trainer_devices.extend(devices_by_slice[slice_indices[i]])

    sampler_devices = []
    for i in range(config.num_trainer_slices, config.num_trainer_slices + config.num_samplers_slices):
      sampler_devices.extend(devices_by_slice[slice_indices[i]])

    trainer_devices_per_slice = len(trainer_devices) // config.num_trainer_slices
    trainer_fsdp = trainer_devices_per_slice
    tp = config.ici_tensor_parallelism
    if tp > 1:
      if trainer_devices_per_slice % tp != 0:
        raise ValueError(
            f"trainer_devices_per_slice ({trainer_devices_per_slice}) must be divisible by tensor parallelism ({tp})"
        )
      if config.ici_fsdp_parallelism != -1 and config.ici_fsdp_parallelism * tp != trainer_devices_per_slice:
        raise ValueError(
            f"ici_fsdp_parallelism ({config.ici_fsdp_parallelism}) * ici_tensor_parallelism ({tp}) must equal "
            f"devices_per_slice ({trainer_devices_per_slice})"
        )
      trainer_fsdp = trainer_devices_per_slice // tp

    trainer_kwargs = dict(combined_kwargs)
    trainer_kwargs.update(
        {
            "num_slices": config.num_trainer_slices,
            "ici_fsdp_parallelism": trainer_fsdp,
            "ici_tensor_parallelism": tp,
            "dcn_data_parallelism": config.num_trainer_slices,
        }
    )

    sampler_kwargs = dict(combined_kwargs)
    sampler_kwargs.update(
        {
            "num_slices": config.num_samplers_slices,
            "ici_fsdp_parallelism": len(sampler_devices) // config.num_samplers_slices,
            "ici_tensor_parallelism": -1,
            "dcn_data_parallelism": config.num_samplers_slices,
        }
    )

    trainer_config = pyconfig.initialize_pydantic(argv, **trainer_kwargs)
    sampler_config = pyconfig.initialize_pydantic(argv, **sampler_kwargs)

  else:
    raise ValueError("num_trainer_slices and num_samplers_slices should be both -1 or positive")

  return trainer_config, sampler_config, trainer_devices, sampler_devices


def create_models_and_meshes(trainer_config, sampler_config, trainer_devices, sampler_devices):
  """Create reference and actor models and their respective meshes.
  This API is particularly useful for Reinforcement Learning (RL) where we need 2 models (wrapped in TunixMaxTextAdapter
  so that they are compatible with default Tunix APIs) and meshes for reference, actor and rollout (which can be disjoint
  in case of disaggreggated RL training).
  """
  max_logging.log("Creating reference model and also meshes for reference and rollout")
  reference_model, reference_mesh = from_pretrained(trainer_config, devices=trainer_devices, wrap_with_tunix_adapter=True)
  devices_array = maxtext_utils.create_device_mesh(sampler_config, sampler_devices)
  rollout_mesh = Mesh(devices_array, sampler_config.mesh_axes)

  if trainer_config.load_checkpoint_only_once:
    max_logging.log("Creating policy model by copying reference model instead of restoring from checkpoint again.")
    with reference_mesh:
      actor_base_model = nnx.clone(reference_model.base)
      use_no_op_mappings = "maxtext_config" in trainer_config.vllm_additional_config
      # TunixMaxTextAdapter wraps MaxText models to be compatible with Tunix's default APIs
      # The weight mappings for vllm (which is interfaced to from MaxText via Tunix) are model specific.
      # The mappings are defined inside src/maxtext/integration/tunix/weight_mapping
      actor_model = TunixMaxTextAdapter(base_model=actor_base_model, use_no_op_mappings=use_no_op_mappings)
      actor_model.config = None
    actor_mesh = reference_mesh
  else:
    max_logging.log("Creating policy model with same config as reference model on trainer mesh")
    actor_model, actor_mesh = from_pretrained(trainer_config, devices=trainer_devices, wrap_with_tunix_adapter=True)

  return reference_model, reference_mesh, actor_model, actor_mesh, rollout_mesh


def from_pretrained(
    config, mesh=None, devices=None, model_mode=MODEL_MODE_TRAIN, rng_key=None, wrap_with_tunix_adapter=False
):
  """Creates a NNX model with sharded parameters, possibly loading from a checkpoint."""
  original_mesh = mesh
  if config.convert_checkpoint_if_possible and not config.load_parameters_path:
    if not (epath.Path(config.base_output_directory) / "0" / "items").exists():
      # Try to convert checkpoint on the fly
      if not config.hf_access_token:
        raise ValueError("hf_access_token must be provided when not providing a pre-existing checkpoint")

      # Only process 0 performs the conversion; other processes wait at the barrier below.
      # Otherwise every host would race to download from HF and concurrently write the same
      # GCS checkpoint, wasting work and risking corruption.
      if jax.process_index() == 0:
        max_logging.warning("Checkpoint path is not provided, converting checkpoint to orbax format for MaxText")

        # This is an empirically derived value. This simulated devices is needed such that orbax creates multiple
        # shards of the checkpoint. Without simulating multiple devices, when running on CPU orbax created a single
        # giant checkpoint file, which could lead to OOM on TPU generations with smaller memory.
        simulated_cpu_devices_count = 16

        # Run the conversion in a completely isolated subprocess so its CPU
        # JAX/XLA requirements do not interfere with the parent's Pathways TPU mesh.
        conversion_env = os.environ.copy()
        conversion_env["JAX_PLATFORMS"] = "cpu"
        # conversion_env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={simulated_cpu_devices_count}"

        to_maxtext_cmd = [
            sys.executable,
            "-m",
            "maxtext.checkpoint_conversion.to_maxtext",
        ] + [
            f"model_name={config.model_name}",
            f"base_output_directory={config.base_output_directory}",
            f"scan_layers={config.scan_layers}",
            f"hf_access_token={config.hf_access_token}",
            "use_multimodal=false",
            "skip_jax_distributed_system=True",
            "--lazy_load_tensors=True",
            f"--simulated_cpu_devices_count={simulated_cpu_devices_count}",
        ]

        try:
          subprocess.run(to_maxtext_cmd, env=conversion_env, check=True)
        except subprocess.CalledProcessError as e:
          raise RuntimeError(f"Checkpoint conversion failed with exit code {e.returncode}") from e

      jax.experimental.multihost_utils.sync_global_devices("from_pretrained_convert_checkpoint")
    load_parameters_path = epath.Path(config.base_output_directory) / "0" / "items"
    # Create a copied Pydantic model with the updated values
    pydantic_config = getattr(config, "_pydantic_config", config)
    new_config = pydantic_config.model_copy(
        update={
            "load_parameters_path": load_parameters_path,
        }
    )
    config = pyconfig.HyperParameters(new_config)

  def _create_model(mesh: Mesh | None = None, model_mode: str = MODEL_MODE_TRAIN, rng_key: jax.Array | None = None):
    rngs = maxtext_utils_nnx.create_nnx_rngs(config, model_mode=model_mode, rng_key=rng_key)
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
        if metadata is None or metadata.item_metadata is None:
          max_logging.log(
              f"ERROR: No valid Orbax checkpoint found at '{config.load_parameters_path}'. "
              "Please check your load_parameters_path, the path may be missing, empty, "
              "or point to a parent directory rather than the checkpoint step directory "
          )
          raise ValueError(
              f"No valid Orbax checkpoint found at '{config.load_parameters_path}'. "
              "Please check your load_parameters_path."
          )

        def _adjust_target_for_moe_fusion(target, meta_tree, is_nnx):
          if not hasattr(target, "items") or not hasattr(meta_tree, "items"):
            return target
          new_target = {}
          for k, v in target.items():
            if k == "wi" and "wi" not in meta_tree and "wi_0" in meta_tree and "wi_1" in meta_tree:
              if not is_nnx:
                arr = v
                half_dim = arr.shape[-1] // 2
                new_target["wi_0"] = jax.ShapeDtypeStruct(
                    shape=arr.shape[:-1] + (half_dim,), dtype=arr.dtype, sharding=arr.sharding
                )
                new_target["wi_1"] = jax.ShapeDtypeStruct(
                    shape=arr.shape[:-1] + (half_dim,), dtype=arr.dtype, sharding=arr.sharding
                )
              else:
                arr = v["value"]
                half_dim = arr.shape[-1] // 2
                new_target["wi_0"] = {
                    "value": jax.ShapeDtypeStruct(
                        shape=arr.shape[:-1] + (half_dim,), dtype=arr.dtype, sharding=arr.sharding
                    )
                }
                new_target["wi_1"] = {
                    "value": jax.ShapeDtypeStruct(
                        shape=arr.shape[:-1] + (half_dim,), dtype=arr.dtype, sharding=arr.sharding
                    )
                }
            else:
              new_target[k] = _adjust_target_for_moe_fusion(v, meta_tree.get(k, {}), is_nnx)

          return new_target

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

          target_for_restore = _adjust_target_for_moe_fusion(
              target_for_restore, metadata.item_metadata.tree["params"]["params"], False
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
          # NNX checkpoint: {'decoder': {'value': ...}}, or NNX-RL with extra 'base' nesting.
          # Restore only nnx.Param — RNG variable shapes may differ between checkpoint and model.
          target_for_restore = jax.tree.map(
              lambda v: {"value": v.value},
              sharded_state,
              is_leaf=lambda n: isinstance(n, nnx.Variable),
          )
          has_base_key = "base" in metadata.item_metadata.tree
          meta_tree_for_params = metadata.item_metadata.tree.get("base", metadata.item_metadata.tree)
          target_for_restore = _adjust_target_for_moe_fusion(target_for_restore, meta_tree_for_params, True)
          item_to_restore = {"base": target_for_restore} if has_base_key else target_for_restore
          restore_args = _fix_restore_args_for_shape_mismatch(
              ocp.checkpoint_utils.construct_restore_args(target_for_restore), meta_tree_for_params, mesh
          )
          restore_args = {"base": restore_args} if has_base_key else restore_args

        restored = ckptr.restore(
            epath.Path(config.load_parameters_path),
            item=item_to_restore,
            transforms={},
            restore_args=restore_args,
        )

        if is_nnx_checkpoint:
          restored_root = restored["base"] if has_base_key else restored
          checkpoint = jax.tree.map(
              lambda v: v["value"],
              restored_root,
              is_leaf=lambda x: isinstance(x, dict) and "value" in x and not isinstance(x.get("value"), dict),
          )
        else:
          checkpoint = restored["params"]["params"]

        if checkpoint:
          model_arrays = jax.tree.map(
              lambda v: v.value,
              sharded_state,
              is_leaf=lambda n: isinstance(n, nnx.Variable),
          )

          def to_dict(tree):
            if hasattr(tree, "items"):
              return {k: to_dict(v) for k, v in tree.items()}
            return tree

          model_arrays = to_dict(model_arrays)
          checkpoint = to_dict(checkpoint)

          def _fuse_moe_weights(ckpt_tree, model_arrays_tree):
            if not hasattr(ckpt_tree, "items") or not hasattr(model_arrays_tree, "items"):
              return ckpt_tree
            new_ckpt = {}
            for k, v in ckpt_tree.items():
              if k in ("wi_0", "wi_1") and "wi" in model_arrays_tree:
                continue
              new_ckpt[k] = _fuse_moe_weights(v, model_arrays_tree.get(k, {}))

            if "wi" in model_arrays_tree and "wi_0" in ckpt_tree and "wi_1" in ckpt_tree:
              wi_0 = ckpt_tree["wi_0"]
              wi_1 = ckpt_tree["wi_1"]
              new_ckpt["wi"] = np.concatenate([wi_0, wi_1], axis=-1)

            return new_ckpt

          checkpoint = _fuse_moe_weights(checkpoint, model_arrays)
          # Release the raw restored buffers now that wi_0/wi_1 have been fused (if needed).
          # This prevents the replicated intermediate copies from persisting until function return.
          del restored

          def _filter_to_model_keys(ckpt, model):
            """Recursively keep only keys present in model, dropping checkpoint-only fields (e.g. to_nnx__rngs)."""
            if not hasattr(ckpt, "items") or not hasattr(model, "items"):
              return ckpt
            return {k: _filter_to_model_keys(ckpt[k], model[k]) for k in model if k in ckpt}

          checkpoint = _filter_to_model_keys(checkpoint, model_arrays)
          checkpoint = jax.tree.map(_expand_checkpoint_to_model_shapes, checkpoint, model_arrays)
          nnx.update(model, checkpoint)
        else:
          raise ValueError(
              f"Checkpoint restore from '{config.load_parameters_path}' yielded no parameters. "
              "This usually means the checkpoint format is incompatible with the model configuration "
              "(e.g. a scanned checkpoint loaded with scan_layers=False, or vice versa). "
              "Please ensure the checkpoint format matches the scan_layers setting."
          )

      except Exception as e:
        raise ValueError(f"Checkpoint loading failed: {e}") from e

    if wrap_with_tunix_adapter:
      with mesh:
        use_no_op_mappings = "maxtext_config" in config.vllm_additional_config
        model = TunixMaxTextAdapter(base_model=model, use_no_op_mappings=use_no_op_mappings)
        model.config = None

    if original_mesh:
      return model
    else:
      return model, mesh


def setup_decode_state_from_nnx(model, config, rng, mesh):
  """Setup decode state by loading an NNX or NNX-RL checkpoint into a linen TrainState.

  Calls from_pretrained (which handles NNX and NNX-RL 'base'-nested checkpoints and
  applies mesh sharding internally), then extracts nnx.Param values into a plain dict
  for the linen TrainState. For linen checkpoints, use maxtext_utils.setup_decode_state instead.

  Args:
    model: the flax linen model to initialize
    config: config object
    rng: jax.prng key
    mesh: jax.devices() mesh

  Returns:
    state: linen TrainState with params loaded from the NNX checkpoint
    state_mesh_annotations: the mesh annotations for the state
  """
  init_state_fn = partial(maxtext_utils.init_initial_state, model, None, config, False, rng)
  _, state_mesh_annotations, _ = maxtext_utils.get_abstract_state(config, mesh, init_state_fn, False)

  # Load the NNX model; from_pretrained handles sharding via jax.jit(out_shardings=...).
  nnx_model = from_pretrained(config, mesh=mesh, model_mode=MODEL_MODE_AUTOREGRESSIVE)

  # Extract nnx.Param values, converting the State pytree to a plain nested dict.
  def _state_to_dict(tree):
    if isinstance(tree, nnx.Variable):
      return tree.value
    if hasattr(tree, "items") and not isinstance(tree, jax.Array):
      return {k: _state_to_dict(v) for k, v in tree.items()}
    return tree

  nnx_param_state = nnx.state(nnx_model, nnx.Param)
  raw_params = _state_to_dict(nnx_param_state)
  del nnx_model, nnx_param_state  # free memory

  params = {"params": raw_params}

  state = maxtext_utils.init_decode_state(model.apply, params)
  return state, state_mesh_annotations
