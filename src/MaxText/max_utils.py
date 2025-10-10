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

""" Common Max Utils needed by multiple modules.
All the functions include MaxText modules, such as Pyconfig, should be moved to MaxText utils file."""

import collections
from collections.abc import Sequence
import functools
from functools import partial
import os
import socket
import subprocess
import time
from typing import Any

from etils import epath
import flax
import jax
from jax.experimental import mesh_utils
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from orbax.checkpoint.experimental.emergency.multi_tier_checkpointing import initialization
import psutil
from tensorboardX import writer

from MaxText import max_logging
from MaxText.common_types import MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_TRAIN

initialize_multi_tier_checkpointing = initialization.initialize_multi_tier_checkpointing
HYBRID_RING_64X4 = "hybrid_ring_64x4"
HYBRID_RING_32X8 = "hybrid_ring_32x8"

# pylint: disable=too-many-positional-arguments


def with_memory_kind(t, memory_kind):
  return jax.tree_util.tree_map(lambda x: x.with_memory_kind(kind=memory_kind), t)


def cast_dtype_from_to(nest, src, dst):
  """All items in nest with dtype src are casted to dtype dst."""
  return jax.tree_util.tree_map(lambda t: t.astype(dst) if t.dtype == src else t, nest)


def find_nans_and_infs(pytree):
  def finder(x):
    return jnp.any(jnp.isinf(x) | jnp.isnan(x))

  bad_pytree = jax.tree_util.tree_map(finder, pytree)
  return jax.tree_util.tree_flatten(bad_pytree)


def l2norm_pytree(x):
  """L2 norm of a pytree of arrays."""
  return jnp.sqrt(jax.tree_util.tree_reduce(lambda x, y: x + jnp.sum(jnp.square(y)), x, initializer=0.0))


def calculate_num_params_from_pytree(params):
  params_sizes = jax.tree_util.tree_map(jax.numpy.size, params)
  total_parameters = jax.tree_util.tree_reduce(lambda x, y: x + y, params_sizes)
  assert total_parameters >= 0
  return total_parameters


def device_space():
  """Version guard for jax.memory.Space.Device."""
  # See b/436565838 for more.
  if jax.__version__ >= "0.7.1":
    return jax.memory.Space.Device  # pytype: disable=module-attr
  else:
    return jax._src.sharding_impls.TransferToMemoryKind("device")  # pylint: disable=protected-access # pytype: disable=module-attr


def calculate_total_params_per_chip(params):
  """Calculate total params per chip."""

  def calculate_leaf_params_per_chip(arr):
    shard = arr.addressable_shards[0]
    return np.prod(shard.data.shape)

  params_sizes_per_chip = jax.tree_util.tree_map(calculate_leaf_params_per_chip, params)
  total_parameters_per_chip = jax.tree_util.tree_reduce(lambda x, y: x + y, params_sizes_per_chip)
  return total_parameters_per_chip


def _bytes_of(x):
  """Return the number of bytes used by a single leaf in a pytree.
  Handles concrete arrays (NumPy/JAX), abstract shapes, scalars, and None.
  Unknown types default to 0.
  """
  # Abstract JAX values: compute bytes from shape × dtype size.
  if isinstance(x, jax.ShapeDtypeStruct):
    # jnp.dtype() normalizes to a consistent dtype object (e.g., handles bfloat16)
    return int(np.prod(x.shape)) * int(jnp.dtype(x.dtype).itemsize)

  # Concrete arrays (NumPy, JAX): rely on their native nbytes property.
  if hasattr(x, "nbytes"):
    return int(x.nbytes)

  # Python scalars (int, float, bool): convert to a NumPy array to measure size.
  if isinstance(x, (int, float, bool)):
    return int(np.array(x).nbytes)

  # None or unsupported leaf types: count as zero bytes.
  if x is not None:
    max_logging.log(f"Unsupported leaf type in calculate_bytes_from_pytree: {type(x)}")

  return 0


def calculate_bytes_from_pytree(params):
  """Return the total memory footprint (in bytes) of all leaves in a pytree.

  Each leaf is measured using `_bytes_of`. Non-array or unsupported types
  contribute 0 unless they are scalars.
  """
  return sum(map(_bytes_of, jax.tree_util.tree_leaves(params)))


def summarize_size_from_pytree(params):
  num_params = calculate_num_params_from_pytree(params)
  num_bytes = calculate_bytes_from_pytree(params)
  return num_params, num_bytes, num_bytes / num_params


def initialize_summary_writer(tensorboard_dir, run_name):
  summary_writer_path = os.path.join(tensorboard_dir, run_name)
  return writer.SummaryWriter(summary_writer_path) if jax.process_index() == 0 else None


def close_summary_writer(summary_writer):
  if jax.process_index() == 0:
    summary_writer.close()


def add_text_to_summary_writer(key, value, summary_writer):
  """Writes given key-value pair to tensorboard as text/summary."""
  if jax.process_index() == 0:
    summary_writer.add_text(key, value)






def get_num_slices(raw_keys):
  """Calculate num_slices based on number of devices."""
  if raw_keys["hardware"] == "cpu":
    max_logging.log(" Setting num_slices=1 for CPU hardware type")
    return 1
  if int(raw_keys["compile_topology_num_slices"]) > 0:
    return raw_keys["compile_topology_num_slices"]
  else:
    devices = jax.devices()
    try:
      return 1 + max(d.slice_index for d in devices)
    except (ValueError, AttributeError):
      return 1




def fill_unspecified_mesh_axes(parallelism_vals, target_product, parallelism_type):
  """Evaluates unspecified DCN/ICI parallelism values"""
  if -1 in parallelism_vals:
    assert (
        parallelism_vals.count(-1) == 1
    ), f"Found unspecified values (-1) for more than one {parallelism_type}\
      parallelism axis. At most one axis can be unspecified."

    determined_val = target_product / np.prod(parallelism_vals) * -1

    assert (
        determined_val >= 1 and determined_val.is_integer
    ), f"Unspecified value unable to be determined with the given\
      {parallelism_type} parallelism values"

    parallelism_vals[parallelism_vals.index(-1)] = int(determined_val)

  target_type = "slices" if parallelism_type == "DCN" else "devices per slice"
  assert np.prod(parallelism_vals) == target_product, (
      f"Number of {target_type} {target_product} does not match"
      f" the product of the {parallelism_type} parallelism {np.prod(parallelism_vals)}"
  )

  return parallelism_vals


def reshape_mesh_to_rings(a, strategy):
  """Reshape device mesh to rings for 64x4 or 32x8 mesh shape"""
  b = []
  if strategy == HYBRID_RING_64X4:
    for i in range(8):
      b.append([])
      for j in range(8):
        a_i = i * 2
        a_j = j * 2
        # forms a ring of size 4
        b[i].append([a[a_i, a_j], a[a_i, a_j + 1], a[a_i + 1, a_j + 1], a[a_i + 1, a_j]])
    b = np.array(b)
    b = np.reshape(b, (64, 4))
  elif strategy == HYBRID_RING_32X8:
    for i in range(8):
      b.append([])
      for j in range(4):
        a_i = i * 2
        a_j = j * 4
        # forms a ring of size 8
        b[i].append(
            [
                a[a_i, a_j],
                a[a_i, a_j + 1],
                a[a_i, a_j + 2],
                a[a_i, a_j + 3],
                a[a_i + 1, a_j + 3],
                a[a_i + 1, a_j + 2],
                a[a_i + 1, a_j + 1],
                a[a_i + 1, a_j],
            ]
        )
    b = np.array(b)
    b = np.reshape(b, (32, 8))
  else:
    raise ValueError(f"The strategy {strategy} to reshape the mesh is not implemented.")
  return b


def create_custom_device_mesh(
    mesh_shape: Sequence[int],
    dcn_mesh_shape: Sequence[int],
    devices: Sequence[Any],
    custom_strategy: str,
    process_is_granule: bool = False,
    should_sort_granules_by_key: bool = True,
) -> np.ndarray:
  """Custom device mesh for 64x4 ici parallelism"""
  assert len(devices) % 256 == 0, f"This custom mesh is not valid for {len(devices)} devices"
  attr = "process_index" if process_is_granule else "slice_index"
  if not hasattr(devices[0], attr):
    raise ValueError(f"Device {devices[0]} does not have attribute {attr}. See" " `process_is_granule` option.")
  granule_dict = collections.defaultdict(list)
  for dev in devices:
    granule_dict[getattr(dev, attr)].append(dev)
  granules = (
      [granule_dict[key] for key in sorted(granule_dict.keys())] if should_sort_granules_by_key else granule_dict.values()
  )
  if np.prod(dcn_mesh_shape) != len(granules):
    raise ValueError(f"Number of slices {len(granules)} must equal the product of " f"dcn_mesh_shape {dcn_mesh_shape}")
  per_granule_meshes = [
      mesh_utils.create_device_mesh(
          [16, 16],
          granule,
          allow_split_physical_axes=False,
      )
      for granule in granules
  ]

  per_granule_meshes = [np.reshape(reshape_mesh_to_rings(x, custom_strategy), mesh_shape) for x in per_granule_meshes]
  # TODO(jekbradbury): handle non-uniform DCN topologies
  granule_mesh = np.arange(len(granules)).reshape(dcn_mesh_shape)
  blocks = np.vectorize(lambda i: per_granule_meshes[i], otypes=[object])(granule_mesh)
  device_mesh = np.block(blocks.tolist())
  return device_mesh


def is_valid_custom_mesh(ici_parallelism, strategy):
  """Checks if the given strategy and ICI parallelism are valid."""
  if not strategy:
    return False

  valid_strategies = {
      HYBRID_RING_64X4: [1, 4, 64],
      HYBRID_RING_32X8: [1, 8, 32],
  }

  if strategy in valid_strategies:
    if sorted(set(ici_parallelism)) == valid_strategies[strategy]:
      return True
    else:
      raise ValueError(f"Invalid custom_mesh:{strategy} chosen for ICI mesh shape {ici_parallelism}")
  else:
    raise ValueError(f"The strategy {strategy} to reshape the mesh is invalid.")


def optimize_mesh_for_tpu_v6e(mesh, devices):
  """Apply transformations to the mesh to optimize for TPU v6e"""
  if devices[0].device_kind != "TPU v6 lite":
    return mesh
  num_devices = len(devices)
  mesh_is_1d_ring = num_devices in mesh.shape
  if not mesh_is_1d_ring:
    return mesh
  # check that the physical topology is 2x4
  device_coords = [d.coords for d in devices]
  coord_size = len(device_coords[0])
  max_coords = tuple(max(dc[i] for dc in device_coords) for i in range(coord_size))
  min_coords = tuple(min(dc[i] for dc in device_coords) for i in range(coord_size))
  dims = tuple(h - l + 1 for (h, l) in zip(max_coords, min_coords))
  if dims != (2, 4, 1):
    return mesh
  axis_idx = mesh.shape.index(num_devices)
  new_mesh = np.moveaxis(mesh, axis_idx, 0)
  new_mesh[4:] = new_mesh[-1:3:-1]
  new_mesh = np.moveaxis(new_mesh, 0, axis_idx)
  max_logging.log("Optimized the mesh for TPU v6e")
  return new_mesh


def unbox_logicallypartioned(boxed_pytree):
  """Unboxes the flax.LogicallyPartitioned pieces

  Args:
    boxed_pytree: a pytree that includes LogicallyPartitioned
      leaves.
  Returns:
    a pytree where all all LogicallyPartitioned leaves have been unboxed.
  """
  return jax.tree_util.tree_map(
      lambda x: x.unbox() if isinstance(x, flax.linen.spmd.LogicallyPartitioned) else x,
      boxed_pytree,
      is_leaf=lambda k: isinstance(k, flax.linen.spmd.LogicallyPartitioned),
  )


# Cross entropy implementation is taken from original T5X codebase:
# https://github.com/google-research/t5x/blob/ace831eea1e2742b4299cd1a9af7e4f302038351/t5x/losses.py#L25-L101
@jax.custom_vjp
def cross_entropy_with_logits(
    logits: jnp.ndarray, targets: jnp.ndarray, z_loss: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Computes cross entropy loss with stable custom gradient.
  Computes a stabilized-gradient version of:
    -jnp.sum(targets * nn.log_softmax(logits), axis=-1)
  If z_loss > 0, then an auxiliary loss equal to z_loss*log(z)^2
  will be added to the cross entropy loss (z = softmax normalization constant).
  The two uses of z_loss are:
  1. To keep the logits from drifting too far from zero, which can cause
     unacceptable roundoff errors in bfloat16.
  2. To encourage the logits to be normalized log-probabilities.
  Args:
    logits: [batch, length, num_classes] float array.
    targets: categorical one-hot targets [batch, length, num_classes] float
      array.
    z_loss: coefficient for auxiliary z-loss loss term.
  Returns:
    tuple with the total loss and the z_loss, both
    float arrays with shape [batch, length].
  """
  logits_sum = jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
  log_softmax = logits - logits_sum
  loss = -jnp.sum(targets * log_softmax, axis=-1)
  # Add auxiliary z-loss term.
  log_z = jnp.squeeze(logits_sum, axis=-1)
  total_z_loss = z_loss * jax.lax.square(log_z)
  loss += total_z_loss
  return loss, total_z_loss


def _cross_entropy_with_logits_fwd(logits: jnp.ndarray, targets: jnp.ndarray, z_loss: float = 0.0) -> tuple[
    tuple[jnp.ndarray, jnp.ndarray],
    tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ],
]:
  """Forward-mode of `cross_entropy_with_logits`."""
  max_logit = logits.max(axis=-1, keepdims=True)
  shifted = logits - max_logit
  exp_shifted = jnp.exp(shifted)
  sum_exp = jnp.sum(exp_shifted, axis=-1, keepdims=True)
  log_softmax = shifted - jnp.log(sum_exp)
  loss = -jnp.sum(targets * log_softmax, axis=-1)
  # Add auxiliary z-loss term.
  log_z = jnp.squeeze(jnp.log(sum_exp) + max_logit, axis=-1)
  total_z_loss = z_loss * jax.lax.square(log_z)
  loss += total_z_loss
  return (loss, total_z_loss), (
      logits,
      targets,
      z_loss,
      exp_shifted,
      sum_exp,  # pytype: disable=bad-return-type  #jax-ndarray
      log_softmax,
      log_z,
  )


def _cross_entropy_with_logits_bwd(
    res: tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ],
    g: tuple[jnp.ndarray, jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Backward-mode of `cross_entropy_with_logits`."""
  g = g[0]  # Ignore z_loss component as that is only used for logging.
  logits, targets, z_loss, exp_shifted, sum_exp, log_softmax, log_z = res
  # z-loss term adds the (2 * z_loss * log_z) factor.
  deriv = jnp.expand_dims(1 + 2 * z_loss * log_z, -1) * exp_shifted / sum_exp - targets
  g_logits = jnp.expand_dims(g, axis=-1) * deriv
  g_targets = -jnp.expand_dims(g, axis=-1) * log_softmax
  return (
      jnp.asarray(g_logits, logits.dtype),
      jnp.asarray(g_targets, targets.dtype),
      jnp.array(0.0),
  )  # sets z-loss coeff gradient to 0


cross_entropy_with_logits.defvjp(_cross_entropy_with_logits_fwd, _cross_entropy_with_logits_bwd)


def print_pytree_shape(print_str, ptree):
  print("\n")
  print(print_str)
  print(jax.tree_util.tree_map(lambda x: x.shape, ptree))


def print_model_vars(print_str, model_vars):
  for k in model_vars:
    print(f"{print_str} key{k}:")
    print(f"\t {model_vars[k]}")


def get_project():
  """Get project"""
  completed_command = subprocess.run(["gcloud", "config", "get", "project"], check=True, capture_output=True)
  project_outputs = completed_command.stdout.decode().strip().split("\n")
  if len(project_outputs) < 1 or project_outputs[-1] == "":
    max_logging.log("You must specify config.vertex_tensorboard_project or set 'gcloud config set project <project>'")
    return None
  return project_outputs[-1]


def delete_pytree(p):
  def delete_leaf(leaf):
    if isinstance(leaf, jax.Array):
      leaf.delete()
    del leaf

  jax.tree_util.tree_map(delete_leaf, p)


def summarize_pytree_data(params, name="Params", raw=False):
  """Generate basic metrics of a given Pytree."""
  num_params, total_param_size, avg_param_size = summarize_size_from_pytree(params)
  if not raw:
    num_params_in_billions = num_params / 1e9
    total_param_size_in_gb = total_param_size / 1e9
    print(
        f"{name} stats: \n"
        f"\tTotal number of params: {num_params_in_billions:.3f} billion \n"
        f"\tTotal memory usage: {total_param_size_in_gb:.3f} GB \n"
        f"\tAvg size: {avg_param_size:.3f} bytes\n"
    )
  else:
    print(
        f"{name} stats: \n"
        f"\tTotal number of params: {num_params:.3f} \n"
        f"\tTotal memory usage: {total_param_size:.3f} bytes \n"
        f"\tAvg size: {avg_param_size:.3f} bytes\n"
    )
  return num_params, total_param_size, avg_param_size


def print_mem_stats(label: str):
  max_logging.log(f"\nMemstats: {label}:")
  try:
    for d in jax.local_devices():
      stats = d.memory_stats()
      used = round(stats["bytes_in_use"] / 2**30, 2)
      limit = round(stats["bytes_limit"] / 2**30, 2)
      max_logging.log(f"\tUsing (GB) {used} / {limit} ({used/limit:%}) on {d}")
  except (RuntimeError, KeyError, TypeError) as ex:
    max_logging.log(f"\tMemstats unavailable, error: {ex}")


def print_cpu_ram_stats(label: str):
  """Print stats of CPU RAM usage/availability."""
  max_logging.log(f"\nRAMstats: {label}:")
  try:
    ram = psutil.virtual_memory()

    total = round(ram.total / 2**30, 2)
    available = round(ram.available / 2**30, 2)
    used = round(ram.used / 2**30, 2)

    max_logging.log(f"\tUsing (GB) {used} / {total} ({used/total:%}) -->  Available:{available}")
  except (RuntimeError, KeyError, TypeError) as ex:
    max_logging.log(f"\tRAM stats unavailable, error: {ex}")


def print_compiled_memory_stats(compiled_stats):
  """Prints a summary of the compiled memory statistics."""
  if compiled_stats is None:
    return

  def bytes_to_gb(num_bytes):
    return num_bytes / (1024**3)

  output_gb = bytes_to_gb(compiled_stats.output_size_in_bytes)
  temp_gb = bytes_to_gb(compiled_stats.temp_size_in_bytes)
  argument_gb = bytes_to_gb(compiled_stats.argument_size_in_bytes)
  alias_gb = bytes_to_gb(compiled_stats.alias_size_in_bytes)
  host_temp_gb = bytes_to_gb(compiled_stats.host_temp_size_in_bytes)
  total_gb = output_gb + temp_gb + argument_gb - alias_gb

  max_logging.log(
      f"Total memory size: {total_gb:.1f} GB, Output size: {output_gb:.1f} GB, Temp size: {temp_gb:.1f} GB, "
      f"Argument size: {argument_gb:.1f} GB, Host temp size: {host_temp_gb:.1f} GB."
  )


def print_system_information():
  """Print system information of the current environment.
  Note that this will initialize the JAX backend."""
  max_logging.log(f"System Information: Jax Version: {jax.__version__}")
  max_logging.log(f"System Information: Jaxlib Version: {jax.lib.__version__}")
  max_logging.log(f"System Information: Jax Backend: {jax.extend.backend.get_backend().platform_version}")


def permute_to_match_maxtext_rope(arr):
  """Permutes the Huggingface Rope to match the MaxText logic."""
  assert arr.shape[-1] % 2 == 0, "The last dimension for rope has to be even."
  evens, odds = np.split(arr, 2, axis=arr.ndim - 1)  # pylint: disable=W0632
  x = np.empty_like(arr)
  x[..., ::2] = evens
  x[..., 1::2] = odds
  return x


def unpermute_from_match_maxtext_rope(arr, model_size):
  """
  Function to get the RoPE values in correct ordering
  """
  if model_size[:8] != "llama3.1":
    return arr
  evens = arr[..., ::2]
  odds = arr[..., 1::2]
  return jax.numpy.concatenate((evens, odds), axis=arr.ndim - 1)


@partial(jax.jit, static_argnames=("cp_size", "seq_dim", "to_contiguous"))
def reorder_sequence(tensor, cp_size: int, seq_dim: int = 1, to_contiguous: bool = False):
  """Reorders the sequence of the tensor. For example, with cp_size=2,
  [0, 1, 2, 3, 4, 5, 6, 7] -> [0, 1, 6, 7, 2, 3, 4, 5]
  and backward
  [0, 1, 6, 7, 2, 3, 4, 5] -> [0, 1, 2, 3, 4, 5, 6, 7]
  """

  if tensor is None:
    return tensor

  seq_len = tensor.shape[seq_dim]
  group_size = seq_len // (2 * cp_size)

  if cp_size % 2 != 0:
    raise ValueError(f"{cp_size=} must be a multiple of 2.")

  # Need to ensure we have 2 pairs to swap for balancing between cp ranks
  if seq_len % (cp_size * 2) != 0:
    raise ValueError(f"{tensor.shape=} is not a multiple of {cp_size*2=}")

  # [B, S, H, D]: [B, 2*cp_size, S/2*cp_size, H, D] -> [B, 2, S/2*cp_size, H, D]
  # [S, B, H, D]: [2*cp_size, S/2*cp_size, B, H, D] -> [2, S/2*cp_size, B, H, D]
  ori_tensor_shape = tensor.shape
  reshaped = tensor.reshape(
      *ori_tensor_shape[:seq_dim],
      2 * cp_size,
      group_size,
      *ori_tensor_shape[seq_dim + 1 :],
  )

  if not to_contiguous:
    # Create first and second halves
    first_half = jnp.arange(cp_size)
    second_half = jnp.arange(2 * cp_size - 1, cp_size - 1, -1)

    # Stack and reshape to interleave
    src_indices = jnp.stack([first_half, second_half], axis=1).reshape(-1)

  else:

    half = cp_size // 2

    # Build the 1st and 2nd groups of contiguous‑pair indices:
    first_pair = [4 * r for r in range(half)]  # [0, 4, 8, …]
    second_pair = [4 * r + 2 for r in range(half)]  # [2, 6, 10, …]
    third_pair = [2 * cp_size - 1 - 4 * r for r in range(half)]  # [2*cp_size-1, 2*cp_size-5, …]
    fourth_pair = [i - 2 for i in third_pair]  # [2*cp_size-3, 2*cp_size-7, …]

    # Concatenate so each rank’s two indices sit next to each other:
    # e.g. [0,2, 4,6, …, (2cp‑1),(2cp‑3), …]
    first_block = first_pair + third_pair
    second_block = second_pair + fourth_pair

    # Stack into shape (2*cp_size//2, 2) → then flatten → length=2*cp_size
    src_indices = jnp.stack([jnp.array(first_block), jnp.array(second_block)], axis=1).reshape(-1)

  # One gather and one reshape
  reordered = jnp.take(reshaped, src_indices, axis=seq_dim)

  # Reshape back to original dimensions
  return reordered.reshape(ori_tensor_shape)


@partial(jax.jit, static_argnums=1)
def reorder_causal_load_balanced(batch, cp_size):
  """Reorders the example batch sequences"""
  return {
      key: reorder_sequence(
          value,  # Pass each key's value inside batch separately
          cp_size=cp_size,
      )
      if key
      in ["inputs", "targets", "inputs_position", "targets_position", "inputs_segmentation", "targets_segmentation"]
      else value
      for key, value in batch.items()
  }


def shard_reorder_causal_load_balanced(batch, cp_size):
  """Shard the output of the reordered sequence."""
  reordered = reorder_causal_load_balanced(batch, cp_size)
  for _, v in batch.items():
    if isinstance(v, jax.Array):
      reordered = jax.lax.with_sharding_constraint(reordered, v.sharding)
      break
  return reordered


def get_reorder_callable(cp_size):
  """Creates a callable that can be used with map() to reorder batches."""
  return functools.partial(shard_reorder_causal_load_balanced, cp_size=cp_size)


@staticmethod
def reorder_mask_load_balancing(tensor, cp_size: int, seq_dim: int):
  """
  Reorders a tensor for load balancing the compute of causal attention.
  This function works on numpy arrays instead of jax.numpy arrays.
  This is needed because we need the mask to be statically computable.
  So, we need to redefine the same logic as reorder_causal_load_balancing.
  We are still doing [0, 1, 2, 3, 4, 5, 6, 7] -> [0, 1, 6, 7, 2, 3, 4, 5]

  Args:
    tensor: The tensor to reorder.
    cp_size: The size of the compute parallelism.
    seq_dim: The dimension of the sequence.
  """

  seq_len = tensor.shape[seq_dim]
  group_size = seq_len // (2 * cp_size)

  if cp_size % 2 != 0:
    raise ValueError(f"{cp_size=} must be a multiple of 2.")

  # Need to ensure we have 2 pairs to swap for balancing between cp ranks
  if seq_len % (cp_size * 2) != 0:
    raise ValueError(f"{tensor.shape=} is not a multiple of {cp_size*2=}")

  # [B, S, H, D]: [B, 2*cp_size, S/2*cp_size, H, D] -> [B, 2, S/2*cp_size, H, D]
  # [S, B, H, D]: [2*cp_size, S/2*cp_size, B, H, D] -> [2, S/2*cp_size, B, H, D]
  ori_tensor_shape = tensor.shape
  reshaped = tensor.reshape(
      *ori_tensor_shape[:seq_dim],
      2 * cp_size,
      group_size,
      *ori_tensor_shape[seq_dim + 1 :],
  )

  # Create first and second halves
  first_half = np.arange(cp_size)
  second_half = np.arange(2 * cp_size - 1, cp_size - 1, -1)

  # Stack and reshape to interleave
  src_indices = np.stack([first_half, second_half], axis=1).reshape(-1)

  # One gather and one reshape
  reordered = np.take(reshaped, src_indices, axis=seq_dim)

  # Reshape back to original dimensions
  return reordered.reshape(ori_tensor_shape)


def parse_custom_args(argv):
  """Load multiple YAML config files from command line arguments."""
  configs = []
  current_argv = []
  python_script = argv[0]
  for arg in argv[1:]:
    if arg.endswith((".yaml", ".yml")):
      if current_argv:
        configs.append(current_argv)
      current_argv = [python_script, arg]
    else:
      current_argv.append(arg)
  if current_argv:
    configs.append(current_argv)
  return configs


def unscan_train_state_params(params, sharding, mesh, scan_axis, layer_groups):
  """
  Unrolls scanned parameter groups into per-layer entries.

  Args:
    train_state: training state with scanned `params`
    mesh: the mesh to use for sharding output
    scan_axis: axis along which scanning was applied (usually 0)
    layer_groups: list of tuples like:
      [("dense_layers", 4), ("moe_layers", 12)]
  """
  decoder = params["params"]["decoder"]
  sharding = sharding["params"]["decoder"]

  for layer_name, num_layers in layer_groups:
    scanned_layers = decoder[layer_name]

    def strip_axis(pspec):
      return jax.sharding.PartitionSpec(*(pspec[:scan_axis] + pspec[scan_axis + 1 :]))

    old_spec = jax.tree_util.tree_map(lambda x: x.spec, sharding[layer_name])
    new_spec = jax.tree_util.tree_map(strip_axis, old_spec)
    new_sharding = jax.tree_util.tree_map(lambda ps: jax.sharding.NamedSharding(mesh, ps), new_spec)

    def slice_layer(arr, i):
      return jax.tree_util.tree_map(lambda x: jnp.take(x, i, axis=scan_axis), arr)

    p_slice_layer = jax.jit(slice_layer, out_shardings=new_sharding)

    for i in range(num_layers):
      per_layer = p_slice_layer(scanned_layers, i)
      decoder[f"{layer_name}_{i}"] = per_layer

    del decoder[layer_name]  # Free memory


def rescan_train_state_params(params, source_shardings, scan_axis, layer_groups):
  """
  Reconstruct scanned layers from per-layer entries using minimal HBM.

  Args:
    train_state: training state with unrolled {layer_name}_{i} entries
    scan_axis: axis to scan over
    layer_groups: list of (layer_name, num_layers)
    mesh: jax.sharding.Mesh for out_shardings
  """
  decoder = params["params"]["decoder"]
  sharding = source_shardings["params"]["decoder"]

  for layer_name, num_layers in layer_groups:

    def stack_layers(*layers):
      return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=scan_axis), *layers)

    # Create a wrapper that allows pjit + donation
    compiled_stack = jax.jit(
        stack_layers,
        out_shardings=sharding[layer_name],
        # donate_argnums=tuple(range(num_layers)),
    )

    # Collect per-layer entries for stacking
    layer_list = [decoder.pop(f"{layer_name}_{i}") for i in range(num_layers)]

    # Stack them with donation
    scanned = compiled_stack(*layer_list)

    # Store result and clear temporary memory
    decoder[layer_name] = scanned


def get_batch_seq_len_for_mode(config, model_mode):
  """
  Resolves the batch size and sequence length based on the model's operational mode.

  Args:
    config: A configuration object with model parameters.
    model_mode: The current operational mode
                (e.g., PREFILL, AUTOREGRESSIVE, TRAIN).

  Returns:
    A tuple of (batch_size, seq_len).
  """
  if model_mode == MODEL_MODE_PREFILL:
    # Prefill mode: Process one full-length prompt.
    batch_size = 1
    seq_len = config.max_prefill_predict_length

  elif model_mode == MODEL_MODE_AUTOREGRESSIVE:
    # Autoregressive/decode mode: Generate one token at a time for a batch.
    batch_size = config.micro_batch_size_to_train_on
    seq_len = 1

  elif model_mode == MODEL_MODE_TRAIN:
    # Training mode: Process a full batch of full-length sequences.
    batch_size = config.micro_batch_size_to_train_on
    seq_len = config.max_target_length

  else:
    # Explicitly handle unknown modes instead of falling back to a default.
    raise ValueError(f"Unknown model_mode: {model_mode}")

  return batch_size, seq_len
