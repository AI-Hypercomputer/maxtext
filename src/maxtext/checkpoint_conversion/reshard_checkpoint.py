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

"""
This script re-shards a MaxText checkpoint on CPU.
- The Orbax checkpoint is loaded into the target sharded layout on a simulated CPU mesh, and then saved to a new
  checkpoint.
- The goal is to pre-shard checkpoints (source) to accelerate loading on TPUs (target) by reducing re-sharding overhead.
  E.g., when target sharding is fsdp=64, checkpoint loading time varies across source sharding (fsdp=16 < ep=16)

How it works:
1. Builds an abstract train-mode NNX model for the target mesh, so the output sharding is what training restores into.
2. Restores each weight in a sharding aligned with its stored chunks, so each chunk is read once, then reshards it in
   memory to the target's shard shapes, one weight at a time and without replicas.
3. Saves in the Linen on-disk layout (`{"params": {"params": ...}, "step": 0}`), readable by NNX and Linen loaders.

Flat MoE (`moe_flat_fsdp_weights=True`): 3D/4D routed-expert weights are converted to the flat layout while loading;
an already flat source is re-sharded as is. Pass the same model, MoE and quantization flags as the training run.
Qwix adds no weights, so the checkpoint stays unquantized.

Key Parameters:
- `--simulated_cpu_devices_count` (defaults to 16): must equal the number of devices in the target mesh.
- `custom_mesh_and_rule`: the target's mesh axes and rules, e.g. `fsdp-as-dp-for-attn`. Use the training run's value.
- `weight_dtype`: the dtype of the abstract model. The saved weights keep the source checkpoint's dtypes (e.g. a
  float16 checkpoint with a float32 router bias), so nothing is rounded; training casts them on restore.
- `load_parameters_path` / `base_output_directory`: input checkpoint and output directory (GCS or local). The output
  checkpoint path will be `<base_output_directory>/0/items`.

Memory Requirements: for X billion parameters stored in 2 bytes each (e.g. bf16/fp16), slightly over 2X GB RAM, plus a
host copy of the largest weight (~435 GB for DeepSeek-V3 routed experts). DeepSeek-V3 671B at fsdp=256 x ep=16 peaks at
~2.1 TB.

Example Command:

# DeepSeek-V3 with MTP, fsdp=256 x ep=16.
python3 -m maxtext.checkpoint_conversion.reshard_checkpoint \\
  model_name=deepseek3-671b mtp_num_layers=1 custom_mesh_and_rule=fsdp-as-dp-for-attn \\
  scan_layers=True weight_dtype=bfloat16 \\
  load_parameters_path=<input_ckpt_path> base_output_directory=<output_ckpt_dir> \\
  checkpoint_storage_concurrent_gb=1024 checkpoint_storage_use_ocdbt=True checkpoint_storage_use_zarr3=True \\
  skip_jax_distributed_system=True ici_fsdp_parallelism=256 ici_expert_parallelism=16 \\
  --simulated_cpu_devices_count=4096
"""


import argparse
import math
import os
import sys
import time
from typing import Sequence
from absl import app

from etils import epath
from flax import nnx
from flax.core.spmd import logical_axis_rules
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import numpy as np
from orbax.checkpoint import v1 as ocp

from maxtext.common import checkpointing
from maxtext.common import train_state_nnx
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.layers import quantizations
from maxtext.utils import max_utils, max_logging
from maxtext.utils import maxtext_utils
from maxtext.utils import maxtext_utils_nnx
from maxtext.utils import model_creation_utils
from maxtext.utils import sharding
from maxtext.checkpoint_conversion.utils.utils import print_peak_memory


def create_abstract_params(config, mesh: Mesh) -> nnx.State:
  """Returns the model's weights as `ShapeDtypeStruct`s with the target `NamedSharding`s.

  The weights are every persistent variable training saves, not only `nnx.Param` (e.g. DeepSeek-V3's routed gate bias
  is a `MoEBiasVar`).

  Builds the train-mode model, the one training restores checkpoints into, without materializing any weights.

  Args:
    config: MaxText config of the target model and mesh.
    mesh: The target device mesh.
  """
  if not (config.quantization and config.use_qwix_quantization):
    _, abstract_model = model_creation_utils.create_nnx_abstract_model(config, mesh=mesh, model_mode=MODEL_MODE_TRAIN)
  else:
    # Qwix quantization intercepts a forward pass when the model is created, which needs the target accelerator's
    # kernels (e.g. tokamax gmm_v2 for moe_flat_fsdp_weights) and so fails on CPU. It adds no weights, so build the
    # model without it; this mirrors create_nnx_abstract_model otherwise.
    def _create_model():
      rngs = maxtext_utils_nnx.create_nnx_rngs(config, model_mode=MODEL_MODE_TRAIN)
      quant = quantizations.configure_quantization(config)
      return model_creation_utils.get_transformer_model(config, mesh, quant, model_mode=MODEL_MODE_TRAIN, rngs=rngs)

    with logical_axis_rules(config.logical_axis_rules):
      graphdef, abstract_state = nnx.split(nnx.eval_shape(_create_model))
      named_shardings = sharding.nnx_construct_named_sharding(abstract_state, mesh)
      abstract_state = jax.tree.map(
          lambda a, s: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=s), abstract_state, named_shardings
      )
    abstract_model = nnx.merge(graphdef, abstract_state)
  # The weights training saves and restores from `load_parameters_path`: every persistent variable, not only
  # nnx.Param (e.g. DeepSeek's routed gate bias is a MoEBiasVar).
  return checkpointing._abstract_params(nnx.state(abstract_model))  # pylint: disable=protected-access


def _stored_weights_metadata(load_parameters_path):
  """Returns the checkpoint's metadata tree of bare weights (below `params/params`), or None if unreadable."""
  path_str = str(load_parameters_path).rstrip("/")
  root = epath.Path(path_str.removesuffix("/items") if path_str != "items" else ".")
  checkpointable_name = "items" if (root / "items").exists() else None
  try:
    stored = ocp.metadata(root, checkpointable_name=checkpointable_name).metadata
  except Exception as e:  # pylint: disable=broad-except
    max_logging.log(f"Could not read checkpoint metadata ({e}); loading straight into the target sharding.")
    return None
  weights = stored.get("params") if isinstance(stored, dict) else None
  if isinstance(weights, dict) and isinstance(weights.get("params"), dict):
    weights = weights["params"]
  return weights if isinstance(weights, dict) else None


def _lookup(tree, path):
  for key in path:
    if not isinstance(tree, dict):
      return None
    tree = tree.get(key, tree.get(str(key)))
  return tree


def chunk_aligned_sharding(
    shape: Sequence[int], chunk_shape: Sequence[int] | None, devices: Sequence[jax.Device]
) -> NamedSharding | None:
  """Returns a sharding whose per-device blocks are whole stored chunks, so each chunk is read exactly once.

  The array is split only along axes where the stored chunks are smaller than the array, on the first
  `prod(splits) <= len(devices)` devices; it is never replicated. E.g. DeepSeek-V3 routed experts stored as
  [256, 58, 7168, 2048] arrays in [1, 58, 7168, 2048] chunks get one expert per device on 256 devices.

  Args:
    shape: The stored array shape.
    chunk_shape: The stored chunk shape, or None if unknown.
    devices: The devices to place the array on.

  Returns:
    The sharding, or None if `chunk_shape` is unknown or does not match `shape`.
  """
  if chunk_shape is None or len(chunk_shape) != len(shape):
    return None
  # Number of chunks along each axis; an axis with a partial last chunk is kept whole on each device.
  grid = [s // c if c and s % c == 0 else 1 for s, c in zip(shape, chunk_shape)]
  splits = [1] * len(shape)
  budget = len(devices)
  for axis in sorted(range(len(shape)), key=lambda a: -grid[a]):
    splits[axis] = max(d for d in range(1, grid[axis] + 1) if grid[axis] % d == 0 and d <= budget)
    budget //= splits[axis]
  split_axes = [a for a in range(len(shape)) if splits[a] > 1]
  if not split_axes:
    return NamedSharding(Mesh(np.asarray(devices[:1]), ("chunks",)), PartitionSpec())
  names = tuple(f"chunks_{a}" for a in split_axes)
  mesh_devices = np.asarray(devices[: math.prod(splits)]).reshape([splits[a] for a in split_axes])
  spec = PartitionSpec(*(f"chunks_{a}" if splits[a] > 1 else None for a in range(len(shape))))
  return NamedSharding(Mesh(mesh_devices, names), spec)


def _reshard_on_host(x: jax.Array, target_sharding: NamedSharding) -> jax.Array:
  """Moves `x` to `target_sharding` through host memory and frees `x`; the shardings may use different devices."""
  if x.sharding == target_sharding:
    return x
  host = np.asarray(x)
  x.delete()
  return jax.device_put(host, target_sharding)


def _to_flat_moe_on_host(x: jax.Array, flat_target: jax.ShapeDtypeStruct) -> jax.Array:
  """Converts a routed-expert weight [E, (L,) rows, cols] to the flat [E * rows, (L,) cols] layout of `flat_target`.

  Each target shard is copied out of a host copy of `x`, with no XLA computation: XLA's CPU client treats simulated
  devices with id >= 2048 as another process and refuses computations on them, so a jit on a 4096-device mesh fails.
  Frees `x`.

  Args:
    x: The weight in the stored 3D/4D layout.
    flat_target: The flat shape, dtype and sharding.
  """
  host = np.asarray(x)
  x.delete()
  num_experts, rows = host.shape[0], host.shape[-2]
  # [E, rows, (L,) cols], a view: flat row e * rows + i is src[e, i].
  src = np.moveaxis(host, -2, 1)

  def _block(index):
    start, stop, _ = index[0].indices(num_experts * rows)
    parts = []
    while start < stop:
      expert, lo = divmod(start, rows)
      hi = min(rows, lo + stop - start)
      parts.append(src[(expert, slice(lo, hi)) + tuple(index[1:])])
      start += hi - lo
    return np.concatenate(parts).astype(flat_target.dtype, copy=False)

  return jax.make_array_from_callback(flat_target.shape, flat_target.sharding, _block)


def without_replicas(target_sharding: NamedSharding) -> NamedSharding:
  """Returns `target_sharding` restricted to one device per distinct shard.

  The shard shapes are unchanged. On CPU every replica is a separate host buffer, e.g. DeepSeek-V3 attention weights
  sharded only on `expert` would be held 256 times on an fsdp=256 x ep=16 mesh. Without replicas Orbax writes each
  shard as one chunk; with them it would split each shard across its replicas (replica-parallel save).

  Args:
    target_sharding: The target sharding.

  Returns:
    The same spec on the target mesh without its replicated (unused, size > 1) axes, or `target_sharding` itself if
    it has no replicas.
  """
  mesh, spec = target_sharding.mesh, target_sharding.spec
  used = {name for entry in spec if entry is not None for name in ((entry,) if isinstance(entry, str) else entry)}
  replicated = [name for name in mesh.axis_names if name not in used and mesh.shape[name] > 1]
  if not replicated:
    return target_sharding
  index = tuple(slice(0, 1) if name in replicated else slice(None) for name in mesh.axis_names)
  return NamedSharding(Mesh(mesh.devices[index], mesh.axis_names), spec)


def load_resharded_params(config, mesh: Mesh) -> nnx.State:
  """Restores the checkpoint at `config.load_parameters_path` into the shard shapes of `mesh`.

  Restoring straight into the target sharding reads every stored chunk once per device that holds a slice of
  it (e.g. 256x for routed experts stored one expert per chunk and split 256-way on fsdp). Instead, each array is
  restored in a sharding aligned with its stored chunks (`chunk_aligned_sharding`), so the checkpoint is read once,
  and is then resharded to the target in memory, one weight at a time, without replicas (`without_replicas`).

  Args:
    config: MaxText config of the target model and mesh.
    mesh: The target device mesh.

  Returns:
    An `nnx.State` of the model's weights, as `jax.Array`s with the target's shard shapes, one device per shard.
  """
  target = create_abstract_params(config, mesh)
  load_kwargs = {
      "use_ocdbt": config.checkpoint_storage_use_ocdbt,
      "use_zarr3": config.checkpoint_storage_use_zarr3,
  }
  stored = _stored_weights_metadata(config.load_parameters_path)
  if stored is None:
    # Without chunk metadata, restore straight into the target: it reads chunks once per device holding a slice of
    # them, materializes replicas and casts to `weight_dtype`.
    return checkpointing.load_params_from_path(
        config.load_parameters_path,
        target,
        config.checkpoint_storage_concurrent_gb,
        convert_flat_moe_weights=bool(config.moe_flat_fsdp_weights),
        **load_kwargs,
    )

  target_weights = nnx.traversals.flatten_mapping(target.to_pure_dict())
  # Flat MoE: {path: (flat target, stored 3D/4D shape)} for routed-expert weights the checkpoint stores 3D/4D.
  flat_moe_plan = {}
  if config.moe_flat_fsdp_weights:
    plan = checkpointing._flat_moe_restore_plan(  # pylint: disable=protected-access
        nnx.traversals.unflatten_mapping(target_weights), stored
    )
    flat_moe_plan = {tuple(path): value for path, value in plan.items()}

  # 1. Restore every array in a chunk-aligned staging sharding; flat MoE weights in the stored 3D/4D shape. Each
  # array keeps its stored dtype (e.g. float16 weights and a float32 router bias): `weight_dtype` only builds the
  # model, and casting here would round the saved checkpoint.
  devices = list(mesh.devices.flat)
  staging, num_aligned = {}, 0
  for path, want in target_weights.items():
    if path in flat_moe_plan:
      want = flat_moe_plan[path][1]
    meta = _lookup(stored, path)
    staging_sharding = None
    if meta is not None and tuple(getattr(meta, "shape", ()) or ()) == tuple(want.shape):
      storage = getattr(meta, "storage_metadata", None)
      staging_sharding = chunk_aligned_sharding(want.shape, getattr(storage, "chunk_shape", None), devices)
    num_aligned += staging_sharding is not None
    dtype = getattr(meta, "dtype", None) or want.dtype
    staging[path] = jax.ShapeDtypeStruct(want.shape, dtype, sharding=staging_sharding or want.sharding)
  max_logging.log(f"Restoring {num_aligned}/{len(staging)} weights in a chunk-aligned staging sharding.")
  staging_state = jax.tree.map(lambda x: x, target)
  nnx.replace_by_pure_dict(staging_state, nnx.traversals.unflatten_mapping(staging))
  start = time.time()
  restored = checkpointing.load_params_from_path(
      config.load_parameters_path,
      staging_state,
      config.checkpoint_storage_concurrent_gb,
      convert_flat_moe_weights=False,
      **load_kwargs,
  )
  max_logging.log(f"Elapse for staged checkpoint restore: {(time.time() - start) / 60:.2f} min")

  # 2. Reshard to the target in memory, one weight at a time, freeing each staging copy.
  start = time.time()
  restored_weights = nnx.traversals.flatten_mapping(restored.to_pure_dict())
  del restored
  resharded = {}
  for path, want in target_weights.items():
    x = restored_weights.pop(path)
    target_sharding = without_replicas(want.sharding)
    if path in flat_moe_plan:
      flat_target = jax.ShapeDtypeStruct(want.shape, x.dtype, sharding=target_sharding)
      resharded[path] = _to_flat_moe_on_host(x, flat_target)
    else:
      resharded[path] = _reshard_on_host(x, target_sharding)
  max_logging.log(f"Elapse for in-memory reshard to the target: {(time.time() - start) / 60:.2f} min")
  params = jax.tree.map(lambda x: x, target)
  nnx.replace_by_pure_dict(params, nnx.traversals.unflatten_mapping(resharded))
  return params


def save_params(config, params: nnx.State, step: int = 0) -> str:
  """Saves `params` to `<config.base_output_directory>/<step>/items` in the Linen on-disk layout.

  Args:
    config: MaxText config with the output directory and storage options.
    params: The weights to save.
    step: The step number of the saved checkpoint.

  Returns:
    The path of the saved checkpoint items.
  """
  checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
      config.base_output_directory,
      enable_checkpointing=True,
      use_async=False,
      save_interval_steps=1,
      use_ocdbt=config.checkpoint_storage_use_ocdbt,
      use_zarr3=config.checkpoint_storage_use_zarr3,
      # Sets the Orbax write timeout to 60 minutes (up from 20) for large models.
      enable_continuous_checkpointing=True,
      checkpoint_storage_concurrent_gb=config.checkpoint_storage_concurrent_gb,
  )
  if checkpoint_manager is None:
    raise RuntimeError("Failed to create Orbax checkpoint manager.")

  # {"params": {"params": <weights>}, "step": step}, the layout training checkpoints use.
  state = train_state_nnx.to_linen_checkpoint_dict({"model": params.to_pure_dict(), "optimizer": {"step": step}})
  if not checkpointing.save_checkpoint(checkpoint_manager, step, state):
    raise RuntimeError(f"Checkpoint for step {step} was not saved; does it already exist in the output directory?")
  # Upon preemption, exit when and only when all ongoing saves are complete.
  checkpointing.wait_until_finished(checkpoint_manager)
  return os.path.join(config.base_output_directory, str(step), "items")


def main(argv: Sequence[str]) -> None:
  config = pyconfig.initialize(argv)
  max_utils.print_system_information()
  max_logging.log(
      f"Reshard checkpoint with weight dtype: {config.weight_dtype},"
      f" moe_flat_fsdp_weights: {config.moe_flat_fsdp_weights}"
  )

  # 1. Set up the target mesh on the simulated CPU devices.
  mesh = Mesh(maxtext_utils.create_device_mesh(config), config.mesh_axes)
  max_logging.log(f"Target mesh: {dict(mesh.shape)}")

  # 2. Load parameters and reshard with the mesh.
  start = time.time()
  params = load_resharded_params(config, mesh)
  max_logging.log(f"Elapse for checkpoint load (with reshard): {(time.time() - start) / 60:.2f} min")

  # 3. Save checkpoint.
  start = time.time()
  save_ckpt_path = save_params(config, params)
  max_logging.log(f"Saved checkpoint: {save_ckpt_path}")
  max_logging.log(f"Elapse for checkpoint save: {(time.time() - start) / 60:.2f} min")
  print_peak_memory()


if __name__ == "__main__":
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # Suppress TensorFlow logging

  # Define local parser
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--simulated_cpu_devices_count",
      type=int,
      required=False,
      default=16,
      help="Number of simulated CPU devices for sharding the checkpoint",
  )

  # Parse known args returns the namespace AND the list of remaining arguments
  local_args, remaining_args = parser.parse_known_args()

  # Reconstruct model_args (script name + the args MaxText needs)
  model_args = [sys.argv[0]] + remaining_args

  # Set JAX environment
  jax.config.update("jax_platforms", "cpu")
  # Simulate CPU devices as virtual mesh
  os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={local_args.simulated_cpu_devices_count}"

  app.run(main, argv=model_args)
