#  Copyright 2023–2026 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

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
from typing import Callable, overload
from functools import partial
import os
import subprocess
import sys
from etils import epath
from flax import nnx
import flax.linen as nn
import jax
import jax.numpy as jnp
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


# Logical axis names whose padding semantics are "replicate the existing values"
# (e.g. KV-head replication for GQA with TP > num_kv_heads).
_VLLM_REPEAT_AXES = frozenset({"kv_heads", ("expert", "model")})

# Logical axis names whose padding semantics are "append zeros at the end"
# (e.g. MoE MLP-dim padding to satisfy the GMM_v2 kernel's per-shard size
# constraint, set in src/maxtext/integration/vllm/maxtext_vllm_adapter/adapter.py).
_VLLM_ZERO_PAD_AXES = frozenset({"mlp_moe", "activation_mlp", ("attn_dp", "model")})


def _normalize_logical_axes(axes):
  """Coerce an axes value (PartitionSpec / tuple / None / leaf marker) to a plain tuple or None.

  ``nnx.get_partition_spec`` returns a tree of :class:`jax.sharding.PartitionSpec`
  leaves whose entries are logical axis names, ``None`` (unsharded), or nested
  tuples (multi-axis sharding).  PartitionSpec is iterable, so ``tuple(spec)``
  yields the per-axis entries directly.
  """
  if axes is None:
    return None
  if isinstance(axes, jax.sharding.PartitionSpec):
    return tuple(axes)
  if isinstance(axes, (tuple, list)):
    return tuple(axes)
  return None


def _zero_pad_axis(arr, axis, extra):
  """Append ``extra`` zeros at the end of ``axis``.

  When ``arr`` is sharded along ``axis`` with a NamedSharding, this pads each
  *local* shard via ``shard_map`` so we never materialize the full replicated
  tensor — critical for large MoE weights where the global pad would re-OOM.
  The resulting layout has zeros interleaved at each shard's tail rather than
  all at the global tail. Both layouts are mathematically equivalent for any
  matmul along the padded axis with a matching pad on the consuming weight,
  which is exactly the MoE wi/wo + GMM_v2 kernel case (the only consumer that
  triggers ``_ZERO_PAD_AXES`` today).

  When ``arr`` is unsharded (or sharded along a different axis), falls back to
  ``jnp.pad`` — same as before this change.
  """
  if extra == 0:
    return arr

  sharding = getattr(arr, "sharding", None)
  pad_width = [(0, 0)] * arr.ndim

  if isinstance(sharding, jax.sharding.NamedSharding):
    spec = sharding.spec
    partition = spec[axis] if axis < len(spec) else None
    shards_along_axis = _partition_size(partition, sharding.mesh)
    if shards_along_axis > 1:
      if extra % shards_along_axis != 0:
        raise ValueError(
            f"Cannot per-shard zero-pad axis {axis}: extra={extra} not divisible by "
            f"shards_along_axis={shards_along_axis} (sharding spec entry {partition!r})."
        )
      per_shard_extra = extra // shards_along_axis
      pad_width[axis] = (0, per_shard_extra)

      def _pad_local(x):
        return jnp.pad(x, pad_width)

      return jax.shard_map(_pad_local, mesh=sharding.mesh, in_specs=spec, out_specs=spec, check_vma=False)(arr)

  pad_width[axis] = (0, extra)
  return jnp.pad(arr, pad_width)


def _align_checkpoint_to_model_shapes(ckpt_arr, model_arr, logical_axes=None):
  """Align ckpt_arr to model_arr's shape and re-shard to model_arr's sharding.

  Per-axis dispatch (driven by ``logical_axes``, the tuple of logical axis names
  for ``model_arr``):

    - axis name in ``_VLLM_REPEAT_AXES`` (e.g. ``"kv_heads"``): replicate the
      checkpoint values along that axis with ``jnp.repeat``. For GQA with TP,
      device ``i`` needs KV head ``i // ratio``, so the correct layout is e.g.
      ``[h0, h0, h1, h1]`` rather than ``[h0, h1, h0, h1]``. Requires the
      model dim to be an integer multiple of the checkpoint dim.

    - axis name in ``_VLLM_ZERO_PAD_AXES`` (e.g. ``"mlp_moe"``,
      ``"activation_mlp"``): append zeros at the end of the axis with
      ``jnp.pad``. This is the right semantics for MoE MLP-dim padding done by
      the vLLM adapter to satisfy the GMM_v2 kernel's size constraint.

    - any other axis name (or when ``logical_axes`` is None): fall back to
      ``jnp.repeat`` if the model dim divides evenly (preserving prior
      behavior), otherwise raise.

  Logical axis names are defined by the rules in ``configs/base.yml``. New
  axes that need a non-default expansion semantics must be registered in
  ``_VLLM_REPEAT_AXES`` or ``_VLLM_ZERO_PAD_AXES`` above.
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
  axes = _normalize_logical_axes(logical_axes)
  if axes is None or len(axes) != len(model_shape):
    axes = (None,) * len(model_shape)

  result = ckpt_arr
  for axis, (ckpt_dim, model_dim, axis_name) in enumerate(zip(ckpt_shape, model_shape, axes)):
    if model_dim == ckpt_dim:
      continue
    if axis_name in _VLLM_ZERO_PAD_AXES:
      if model_dim < ckpt_dim:
        raise ValueError(
            f"axis {axis} (logical={axis_name!r}): model_dim={model_dim} smaller than "
            f"ckpt_dim={ckpt_dim}; shapes ckpt={ckpt_shape} model={model_shape}"
        )
      result = _zero_pad_axis(result, axis, model_dim - ckpt_dim)
    elif axis_name in _VLLM_REPEAT_AXES:
      if model_dim % ckpt_dim != 0:
        raise ValueError(
            f"axis {axis} (logical={axis_name!r}): model_dim={model_dim} not divisible by "
            f"ckpt_dim={ckpt_dim}; shapes ckpt={ckpt_shape} model={model_shape}"
        )
      result = jnp.repeat(result, model_dim // ckpt_dim, axis=axis)
    else:
      if model_dim % ckpt_dim != 0:
        raise ValueError(
            f"Cannot align axis {axis} (logical={axis_name!r}): model_dim={model_dim} not "
            f"divisible by ckpt_dim={ckpt_dim}, and axis is not registered in _VLLM_REPEAT_AXES "
            f"({sorted(str(x) for x in _VLLM_REPEAT_AXES)}) or "
            f"_VLLM_ZERO_PAD_AXES ({sorted(str(x) for x in _VLLM_ZERO_PAD_AXES)}). "
            f"Full shapes ckpt={ckpt_shape} model={model_shape}"
        )
      result = jnp.repeat(result, model_dim // ckpt_dim, axis=axis)
  return jax.device_put(result, model_arr.sharding)


def _fuse_moe_weights(ckpt_tree, model_arrays_tree):
  """Fuse separate wi_0/wi_1 checkpoint entries into a single wi when model uses fused layout.

  This properly interleaves the gate and up projections based on the target tensor
  parallelism (TP) sharding, ensuring that each device receives its respective
  slice of both wi_0 and wi_1. It also applies any necessary MLP-dim padding
  on a per-shard basis to satisfy kernel constraints.
  """

  def _is_fusion_site(node):
    """A ckpt-side dict that holds wi_0/wi_1 leaf siblings — the parent of a fusion."""
    return (
        isinstance(node, dict)
        and "wi_0" in node
        and "wi_1" in node
        and not isinstance(node["wi_0"], dict)
        and not isinstance(node["wi_1"], dict)
    )

  def _key_str(key):
    if hasattr(key, "key"):
      return key.key
    if hasattr(key, "attr"):
      return key.attr
    return key

  def _lookup_model(path):
    node = model_arrays_tree
    for key in path:
      name = _key_str(key)
      if isinstance(node, dict) and name in node:
        node = node[name]
      else:
        return None
    return node

  def _maybe_fuse(path, ckpt_node):
    if not _is_fusion_site(ckpt_node):
      return ckpt_node
    model_node = _lookup_model(path)
    if not isinstance(model_node, dict) or "wi" not in model_node:
      return ckpt_node

    wi_model = model_node["wi"]
    axis = wi_model.ndim - 1

    # Determine the number of shards (TP degree) along the concatenated axis
    n_shards = 1
    sharding = getattr(wi_model, "sharding", None)
    if isinstance(sharding, jax.sharding.NamedSharding):
      spec = sharding.spec
      partition = spec[axis] if axis < len(spec) else None
      n_shards = _partition_size(partition, sharding.mesh)

    # Target size for a single half (wi_0 or wi_1) AFTER padding
    target_half_dim = wi_model.shape[-1] // 2

    # Helper to pad per-shard and reshape for interleaving
    def _pad_and_chunk(arr, target_total_size):
      shape = arr.shape
      current_total_size = shape[-1]

      # Calculate per-shard chunk sizes
      chunk_size = current_total_size // n_shards
      target_chunk_size = target_total_size // n_shards
      pad_amount = target_chunk_size - chunk_size

      # Reshape to expose the per-shard chunk: (..., n_shards, chunk_size)
      arr_reshaped = arr.reshape(*shape[:-1], n_shards, chunk_size)

      # Pad each chunk individually if necessary
      if pad_amount > 0:
        pad_widths = [(0, 0)] * arr_reshaped.ndim
        pad_widths[-1] = (0, pad_amount)
        arr_reshaped = jnp.pad(arr_reshaped, pad_widths)

      return arr_reshaped

    # Apply per-shard padding and chunking
    padded_chunked_wi_0 = _pad_and_chunk(ckpt_node["wi_0"], target_half_dim)
    padded_chunked_wi_1 = _pad_and_chunk(ckpt_node["wi_1"], target_half_dim)

    # Concatenate along the inner chunk dimension to interleave the shards
    # Shape becomes: (..., n_shards, target_chunk_size * 2)
    wi_interleaved = jnp.concatenate([padded_chunked_wi_0, padded_chunked_wi_1], axis=-1)

    # Flatten the n_shards dimension back out to match the final model shape, drop wi_0/wi_1.
    new_node = {k: v for k, v in ckpt_node.items() if k not in ("wi_0", "wi_1")}
    new_node["wi"] = wi_interleaved.reshape(*wi_model.shape)
    return new_node

  return jax.tree_util.tree_map_with_path(_maybe_fuse, ckpt_tree, is_leaf=_is_fusion_site)


def _partition_size(partition, mesh):
  """Total mesh-axis size used to shard a single tensor axis.

  ``partition`` is a single PartitionSpec entry: ``None`` (unsharded), a single
  mesh-axis name (str), or a tuple of mesh-axis names.
  """
  if partition is None:
    return 1
  names = (partition,) if isinstance(partition, str) else tuple(partition)
  size = 1
  for n in names:
    size *= mesh.shape[n]
  return size


def _stored_shape_evenly_shardable(restore_arg, stored_shape):
  """Whether the restore_arg's NamedSharding evenly partitions the stored shape.

  When True, we can load the checkpoint with the model's logical sharding intact
  (each device receives only its local slice), avoiding the multi-GB replicated
  fanout that fully-replicated loading produces for large MoE weights.
  """
  sharding = restore_arg.sharding
  if not isinstance(sharding, jax.sharding.NamedSharding):
    return False
  spec = sharding.spec
  for axis_idx, dim in enumerate(stored_shape):
    partition = spec[axis_idx] if axis_idx < len(spec) else None
    if dim % _partition_size(partition, sharding.mesh) != 0:
      return False
  return True


def _fix_restore_args_for_shape_mismatch(restore_args, stored_metadata_tree, mesh):
  """Adjust restore_args for arrays whose checkpoint shape differs from the model shape.

  When the model is initialized with padded shapes (e.g. KV heads padded to match
  TP size, or MoE MLP dim padded for the GMM_v2 kernel) but the checkpoint was
  saved with smaller shapes, Orbax will reject the restore because the provided
  ``global_shape`` is incompatible with the stored shape.

  Two cases:

  1. **Stored shape divides evenly across the model's NamedSharding** (typical
     for MoE MLP-dim padding: stored 768 / TP=4 = 192). Keep the model's
     sharding and pass the stored shape so Orbax loads each device's local
     slice directly — no replicated fanout.
     ``_align_checkpoint_to_model_shapes`` then pads each local shard.

  2. **Stored shape doesn't divide evenly** (typical for KV-head padding:
     stored 2 KV heads / TP=8 = 0.25). Fall back to fully-replicated loading
     and let alignment expand-then-reshard. This matches the original behavior
     and is required for correctness when sharded loading is impossible.

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

  mismatched_paths_sharded = []
  mismatched_paths_replicated = []
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
        # Check for scanned vs unscanned rank mismatch
        if len(stored_shape) != len(restore_arg.global_shape):
          rank_mismatched_paths.append(
              f"  {'.'.join(_key_str(k) for k in path)}: "
              f"checkpoint shape {stored_shape} (rank {len(stored_shape)}) "
              f"vs model shape {restore_arg.global_shape} (rank {len(restore_arg.global_shape)})"
          )
        else:
          # Handle the shape mismatch logic for padding/sharding
          found_array_count[0] += 1
          path_str = f"  {'.'.join(_key_str(k) for k in path)}: stored={stored_shape} -> model={restore_arg.global_shape}"
          if _stored_shape_evenly_shardable(restore_arg, stored_shape):
            mismatched_paths_sharded.append(path_str)
            return dataclasses.replace(restore_arg, global_shape=stored_shape, shape=stored_shape)

          mismatched_paths_replicated.append(path_str)
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

  if mismatched_paths_sharded:
    max_logging.log(
        f"Checkpoint shape mismatches ({len(mismatched_paths_sharded)} arrays): loading sharded at "
        "stored shape and padding each local shard after restore.\n" + "\n".join(mismatched_paths_sharded)
    )
  if mismatched_paths_replicated:
    max_logging.log(
        f"Checkpoint shape mismatches ({len(mismatched_paths_replicated)} arrays): loading with replicated "
        "sharding (stored shape not evenly partitionable across mesh) and expanding after restore.\n"
        + "\n".join(mismatched_paths_replicated)
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


def get_nnx_create_model_fn(config, mesh=None, devices=None, model_mode=MODEL_MODE_TRAIN, rng_key=None) -> Callable:

  def _create_model():
    rngs = maxtext_utils_nnx.create_nnx_rngs(config, model_mode=model_mode, rng_key=rng_key)
    return from_config(config, devices, mesh, rngs=rngs, model_mode=model_mode)

  return _create_model


def create_nnx_abstract_model(
    config, mesh=None, devices=None, model_mode=MODEL_MODE_TRAIN, rng_key=None
) -> tuple[Callable, nnx.Module]:
  """Creates an abstract NNX model.

  Returns:
    A tuple containing (create_model_fn, abstract_model):
      create_model_fn: A zero-argument callable that produces a new model instance.
      abstract_model: The stateful NNX model instance in an abstract state.
  """

  with nn.logical_axis_rules(config.logical_axis_rules):
    _create_model = get_nnx_create_model_fn(config, mesh, devices, model_mode, rng_key)
    if mesh is None:
      _tmp = nnx.eval_shape(_create_model)
      mesh = _tmp.mesh
    # Use nnx.eval_shape + our scan-axis-aware sharding helper instead of
    # nnx.get_abstract_model, which uses get_var_pspec internally and ignores
    # param_scan_axis / nnx.PARTITION_NAME metadata set by _create_scanned_layers,
    # causing the stacked layers axis to be missing from the PartitionSpec.
    with jax.set_mesh(mesh):
      abs_model = nnx.eval_shape(_create_model)
    graphdef, abs_var_state = nnx.split(abs_model)
    named_sharding_state = maxtext_utils.get_nnx_named_sharding_with_scan_axis(abs_var_state, mesh)
    abstract_state = jax.tree.map(
        lambda a, s: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=s),
        abs_var_state,
        named_sharding_state,
    )
    return _create_model, nnx.merge(graphdef, abstract_state)


def create_nnx_sharded_model_hybrid(config, mesh=None, devices=None, model_mode=MODEL_MODE_TRAIN, rng_key=None):
  """Creates a sharded model for hybrid NNX modules containing Linen sub-modules.

  DEPRECATED: This function is a transitional utility for the Linen-to-NNX
  migration. It should be removed once all model components are ported to
  pure NNX modules.

  This function specifically handles the complexity of "mixed" state initialization,
  where logical sharding annotations must be resolved for both NNX native
  Parameters and legacy Linen variables wrapped via the NNX-Linen bridge.
  It ensures that both systems correctly respect the provided mesh and
  logical axis rules during the abstraction/sharding planning phase.
  """
  _create_model_partial = get_nnx_create_model_fn(config, mesh, devices, model_mode, rng_key)

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
    return model


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
        if config.hf_access_token:
          conversion_env["HF_TOKEN"] = config.hf_access_token

        to_maxtext_cmd = [
            sys.executable,
            "-m",
            "maxtext.checkpoint_conversion.to_maxtext",
        ] + [
            f"model_name={config.model_name}",
            f"base_output_directory={config.base_output_directory}",
            f"scan_layers={config.scan_layers}",
            "use_multimodal=false",
            "skip_jax_distributed_system=True",
            "--lazy_load_tensors=True",
            f"--simulated_cpu_devices_count={simulated_cpu_devices_count}",
            f"checkpoint_storage_use_ocdbt={config.checkpoint_storage_use_ocdbt}",
            f"checkpoint_storage_use_zarr3={config.checkpoint_storage_use_zarr3}",
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

  if config.pure_nnx:
    _create_model, abstract_model = create_nnx_abstract_model(config, mesh, devices, model_mode, rng_key)
    model = maxtext_utils_nnx.create_nnx_sharded_model(abstract_model, _create_model, mesh=mesh)
    # TODO: print debug_sharding info
  else:
    model = create_nnx_sharded_model_hybrid(config, mesh, devices, model_mode, rng_key)

  # Compute logical-axis specs for downstream checkpoint alignment.
  # The model-creation helpers above resolve specs internally for sharding, but
  # the checkpoint-loading branch below needs the logical PartitionSpec tree
  # (axis names like "kv_heads", "mlp_moe") for repeat/zero-pad dispatch in
  # _align_checkpoint_to_model_shapes. nnx.eval_shape is cheap (abstract trace).
  _create_model_for_specs = get_nnx_create_model_fn(config, mesh, devices, model_mode, rng_key)
  with nn.logical_axis_rules(config.logical_axis_rules):
    _abs_model_for_specs = nnx.eval_shape(_create_model_for_specs)
  _, _abs_state_for_specs = nnx.split(_abs_model_for_specs)
  specs = nnx.get_partition_spec(_abs_state_for_specs)

  sharded_state = nnx.state(model)

  if mesh is None:
    mesh = model.mesh

  with mesh:
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
              lambda v: v[...],
              sharded_state,
              is_leaf=lambda n: isinstance(n, nnx.Variable),
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
              lambda v: {"value": v[...]},
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

        # Free memory used by initial sharded_state before restore, to make room for the incoming checkpoint arrays.
        def _free_device_memory(node):
          if isinstance(node, nnx.Variable) and not isinstance(node, nnx.RngState):
            val = node[...]
          else:
            val = node

          if isinstance(val, jax.Array) and not val.is_deleted():
            val.delete()

          return node

        jax.tree_util.tree_map(_free_device_memory, sharded_state, is_leaf=lambda n: isinstance(n, nnx.Variable))

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
              lambda v: v[...],
              sharded_state,
              is_leaf=lambda n: isinstance(n, nnx.Variable),
          )
          # ``specs`` (nnx.get_partition_spec(abstract_state) at the top of from_pretrained)
          # is the source of truth for logical axis names — it's the input to
          # nn.logical_to_mesh_sharding.  Each leaf is a PartitionSpec whose entries are
          # logical axis names (or None / nested tuples).  Reuse it for repeat/zero-pad
          # dispatch in _align_checkpoint_to_model_shapes.
          # nnx.get_partition_spec returns Variables wrapping PartitionSpecs at the leaves;
          # unwrap to raw PartitionSpecs so _normalize_logical_axes can read them.
          logical_axes_tree = jax.tree.map(
              lambda v: v.get_value(),
              specs,
              is_leaf=lambda n: isinstance(n, nnx.Variable),
          )

          def to_dict(tree):
            if hasattr(tree, "items"):
              return {k: to_dict(v) for k, v in tree.items()}
            return tree

          model_arrays = to_dict(model_arrays)
          checkpoint = to_dict(checkpoint)
          logical_axes_tree = to_dict(logical_axes_tree)

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

          def _walk_align(ckpt, model_arr, axes):
            if isinstance(ckpt, dict):
              return {
                  k: _walk_align(
                      v,
                      model_arr[k],
                      axes.get(k) if isinstance(axes, dict) else None,
                  )
                  for k, v in ckpt.items()
              }
            return _align_checkpoint_to_model_shapes(ckpt, model_arr, axes)

          checkpoint = _walk_align(checkpoint, model_arrays, logical_axes_tree)
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
      return tree.get_value()
    if hasattr(tree, "items") and not isinstance(tree, jax.Array):
      return {k: _state_to_dict(v) for k, v in tree.items()}
    return tree

  nnx_param_state = nnx.state(nnx_model, nnx.Param)
  raw_params = _state_to_dict(nnx_param_state)
  del nnx_model, nnx_param_state  # free memory

  params = {"params": raw_params}

  state = maxtext_utils.init_decode_state(model.apply, params)
  return state, state_mesh_annotations
