
import os
import sys
from typing import Optional, Tuple

import jax
from jax.sharding import Mesh


def initialize_tensor_parallelism(
    tp_plan: Optional[str], tp_size: Optional[int] = None
) -> Tuple[None, None, Optional[Mesh], Optional[int]]:
  """
    Sets up the device mesh for tensor parallelism.

    This function is called when the model is loaded and the TP plan is set to 'auto'.
    In JAX, this primarily involves creating a jax.sharding.Mesh.

    Args:
        tp_plan: The tensor parallelism plan. If None, the function returns early.
        tp_size: The size of the tensor parallelism. If None, it defaults to the
            total number of available devices.

    Returns:
        A tuple containing:
        - None: Placeholder for tp_device, not applicable in JAX.
        - None: Placeholder for device_map, not applicable in JAX.
        - device_mesh: A JAX Mesh object for tensor parallelism.
        - tp_size: The size of the tensor parallelism.
    """
  if tp_plan is None:
    return None, None, None, None

  # In JAX, distributed initialization is typically handled by the environment.
  # We check if we are in a multi-process environment if tp_size suggests it.
  if jax.process_count() == 1 and (tp_size is not None and tp_size > 1):
    raise OSError(
        "Tensor parallelism is requested with tp_size > 1, but JAX is not in a"
        " multi-process environment. Please ensure jax.distributed.initialize()"
        " is called or the script is run in a distributed setting."
    )

  # Silence output for non-primary ranks to reduce log spam.
  if jax.process_index() > 0:
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")

  # Determine tensor parallelism size. Default to all available devices.
  if tp_size is None:
    tp_size = jax.device_count()

  devices = jax.devices()
  if len(devices) < tp_size:
    raise ValueError(f"Not enough devices for the requested tp_size={tp_size}. " f"Found {len(devices)} devices.")

  # Create a 1D mesh for tensor parallelism.
  # The original PyTorch code creates a mesh of `tp_size`.
  device_mesh = Mesh(devices[:tp_size], axis_names=("tp",))

  # The concepts of `tp_device` and `device_map` from the PyTorch DTensor API
  # do not have direct equivalents in JAX's sharding model. The `Mesh` object
  # serves the purpose of mapping computations to devices. We return None for
  # these to maintain signature compatibility.
  tp_device = None
  device_map = None

  return tp_device, device_map, device_mesh, tp_size

import jax.numpy as jnp
from jax import Array


def repack_weights(
    packed_parameter: Array,
    sharded_dim: int,  # The dimension index in the global tensor that was sharded
    world_size: int,
    num_blocks: int = 2,
) -> Array:
  """Reorders a tensor that was reconstructed from sharded packed weights into its canonical packed format.

  For example, if a weight was packed (e.g., gate_proj and up_proj) and then
  sharded, a gather operation might produce an interleaved layout like
  [G0, U0, G1, U1, ...] along the sharded dimension. This function reorders it
  to [G0, G1, ..., U0, U1, ...]. This is an inverse operation to
  get_packed_weights.

  Args:
    packed_parameter: The tensor reconstructed from sharded weights (e.g., via a
      gather operation).
    sharded_dim: The dimension index in the reconstructed_tensor that was
      originally sharded.
    world_size: The tensor parallel world size.
    num_blocks: The number of projections that were packed together (e.g., 2 for
      gate_up_proj).

  Returns:
    The reordered tensor in canonical packed format.
  """

  if num_blocks != 2:
    raise ValueError(
        "Num blocks different from 2 is not supported yet. This is most likely a"
        " bug in your implementation as we only pack gate and up projections"
        " together."
    )

  actual_sharded_dim = (
      sharded_dim if sharded_dim >= 0 else sharded_dim + packed_parameter.ndim
  )
  total_size_on_sharded_dim = packed_parameter.shape[actual_sharded_dim]
  original_block_size_on_dim = total_size_on_sharded_dim // num_blocks
  shard_chunk_size = original_block_size_on_dim // world_size

  prefix_shape = packed_parameter.shape[:actual_sharded_dim]
  suffix_shape = packed_parameter.shape[actual_sharded_dim + 1 :]

  tensor_view = packed_parameter.reshape(
      *prefix_shape,
      world_size,
      num_blocks,
      shard_chunk_size,
      *suffix_shape,
  )

  # Permute to bring num_blocks first, then world_size, then shard_chunk_size
  # This groups all chunks of G together, then all chunks of U together.
  # Target order of these middle dimensions: (num_blocks, world_size, shard_chunk_size)
  # Current order of view's middle dimensions: (world_size, num_blocks, shard_chunk_size)
  # Absolute indices of the dimensions to be permuted (world_size, num_blocks)
  axis_ws_abs = len(prefix_shape)
  axis_npp_abs = len(prefix_shape) + 1

  permute_order = list(range(tensor_view.ndim))
  (
      permute_order[axis_ws_abs],
      permute_order[axis_npp_abs],
  ) = (
      permute_order[axis_npp_abs],
      permute_order[axis_ws_abs],
  )

  tensor_permuted = jnp.transpose(tensor_view, axes=tuple(permute_order))

  # Reshape back to the original tensor's ndim, with the sharded dimension now
  # correctly ordered as [G_all, U_all]. The final shape should be the same as
  # packed_parameter.
  final_ordered_tensor = tensor_permuted.reshape(packed_parameter.shape)

  return final_ordered_tensor

from __future__ import annotations
import functools
from typing import Any

from flax.experimental import nnx
import jax
import jax.numpy as jnp

from maxtext.utils import logging
# The following imports are assumed to be converted and available in the new codebase.
# from .parallel_layers import ALL_PARALLEL_STYLES
# from .parallel_layers import _get_parameter_tp_plan


# This is a new helper function required to replicate `model.get_submodule` functionality in JAX/NNX.
def _get_submodule(model: nnx.Module, name: str) -> nnx.Module:
  """Traverses the module hierarchy to find a submodule by name."""
  if not name:
    return model
  return functools.reduce(getattr, name.split("."), model)


def shard_and_distribute_param(
    model: nnx.Module,
    param: jax.Array,
    empty_param: jax.Array,
    parameter_name: str,
    param_casting_dtype: jnp.dtype,
    is_contiguous: bool,
    rank: int,
    device_mesh: jax.sharding.Mesh,
    set_param: bool = True,
) -> jax.Array:
  r"""
  This function is called in `from_pretrained` when loading a model's checkpoints.
  It receives the pointer to the parameter (or the parameter itself) and takes care of "sharding".
  All process run this function, so they just load the partition of the tensor that they require.

  Main uses cases:
  - column / rowise parallelism, you just shard all the weights of the layer (weight and bias)
  - packed layers: you slice the weights, then shard like above
  - custom operation:
      - you want to add an all-gather at the end of a local layer.
      - you want to have a layer that is isolated from the rest of the world

  Args:
    model: The model object.
    param: The parameter tensor to be sharded.
    empty_param: A placeholder parameter on the target device with the correct final shape.
    parameter_name: The full name of the parameter (e.g., `layers.0.attention.wq.weight`).
    param_casting_dtype: The target data type for the parameter.
    is_contiguous: A boolean flag (unused in JAX version, kept for API consistency).
    rank: The rank of the current process.
    device_mesh: The device mesh for tensor parallelism.
    set_param: A flag to control if the parameter is set on the submodule.

  Returns:
    The sharded parameter as a JAX array.
  """
  if "." in parameter_name:
    param_name, param_type = parameter_name.rsplit(".", 1)
  else:
    param_name = ""
    param_type = parameter_name

  tp_plan = model._tp_plan or {}
  # In PyTorch, this also updated from a class-level plan. We assume the instance plan is sufficient.
  # tp_plan.update(getattr(type(model), "_tp_plan", None) or {})
  module_to_tp = _get_submodule(model, param_name)
  rank = int(rank)
  current_shard_plan = _get_parameter_tp_plan(parameter_name, tp_plan)

  if jax.process_index() == 0:
    if current_shard_plan is None:
      logging.info(f"Tensor sharding plan for {param_name} not found, using default 'replicate' plan.")
    else:
      logging.info(f"Tensor sharding plan for {param_name}: {current_shard_plan}")

  if current_shard_plan is not None:
    try:
      tp_layer = ALL_PARALLEL_STYLES[current_shard_plan]
      # The `is_contiguous` argument is not relevant in JAX.
      param = tp_layer.partition_tensor(param, empty_param, param_type, param_casting_dtype, rank, device_mesh)
    except NotImplementedError as e:
      print(
          f"Trying to prepare {parameter_name}, but it's not supported. "
          f"Corresponding module: {module_to_tp} Fix its TP plan, "
          f"current layer: {tp_layer} : {e}"
      )
  else:
    param = param.astype(param_casting_dtype)

  # SUPER IMPORTANT we have to use setattr
  # otherwise loading is crazy slow
  if set_param:
    # In NNX, trainable parameters are wrapped in nnx.Param.
    # The `requires_grad` logic is implicit in JAX based on dtype.
    setattr(module_to_tp, param_type, nnx.Param(param))

  return param

import re
from typing import Optional
from absl import logging


def verify_tp_plan(expected_keys: list[str], tp_plan: Optional[dict[str, str]]):
  """Verifies the TP plan, logging warnings for unsharded layers and unused rules.

  Args:
    expected_keys: A list of all parameter names in the model.
    tp_plan: A dictionary mapping parameter name patterns to sharding rules.
  """

  if tp_plan is None:
    return

  generic_keys = {re.sub(r"\d+", "*", key) for key in expected_keys}
  unsharded_layers = set(generic_keys)
  unused_rules = tp_plan.copy()

  for key in generic_keys:
    param_name = key.rsplit(".", 1)[0] if "." in key else key
    generic_param_name = re.sub(r"\d+", "*", param_name)

    if generic_param_name in unused_rules:
      unused_rules.pop(generic_param_name)
      unsharded_layers.discard(key)
    elif "." in generic_param_name and (parent_param_name := generic_param_name.rsplit(".", 1)[0]) in unused_rules:
      unused_rules.pop(parent_param_name)
      unsharded_layers.discard(key)
    else:
      pass  # we couldn't find the rule for this parameter, so it's not sharded

  if unused_rules:
    logging.warning("The following TP rules were not applied on any of the layers: %s", unused_rules)
  if unsharded_layers:
    logging.warning("The following layers were not sharded: %s", ", ".join(unsharded_layers))

# This is a PyTorch-specific global flag for DeepSpeed integration and has no equivalent in JAX/MaxText.

import jax

_jax_distributed_available = jax.process_count() > 1

from typing import Union
import jax

Device = jax.Device


def is_accelerator_device(device: Union[str, int, Device]) -> bool:
  """Check if the device is an accelerator. We need to function, as device_map can be "disk" as well, which is not
    a proper `jax.Device`.
    """
  if isinstance(device, str):
    if device == "disk":
      return False
    device_type = device.split(":")[0]
    # "meta" is a PyTorch-specific device type for tensors without storage.
    # JAX does not have a direct equivalent device type.
    return device_type != "cpu"
  elif isinstance(device, int):
    # Integers are assumed to be accelerator indices.
    return True
  elif isinstance(device, Device):
    return device.platform != "cpu"
  else:
    # The original PyTorch function would attempt `torch.device(device)` which raises a TypeError for unsupported types.
    # We'll do the same for clarity and to flag unexpected inputs.
    raise TypeError(f"Unsupported device type for is_accelerator_device: {type(device)}")

import os
import jax


def is_local_dist_rank_0():
  """Checks if the current process is the local rank 0 in a distributed setup."""
  return (
      jax.distributed.is_initialized()
      and int(os.environ.get("LOCAL_RANK", "-1")) == 0
  )

from contextlib import contextmanager

# This global variable is defined at the module level.
# _is_ds_init_called = False

# Skip recursive calls to deepspeed.zero.Init to avoid pinning errors.
# This issue occurs with ZeRO stage 3 when using NVMe offloading.
# For more details, refer to issue #34429.
@contextmanager
def set_zero3_state():
  """Context manager to avoid recursive calls to deepspeed.zero.Init."""
  global _is_ds_init_called
  _is_ds_init_called = True
  try:
    yield
  finally:
    _is_ds_init_called = False

# Copyright 2024 Google LLC
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

"""
This function is a port of a HuggingFace Transformers utility.
Deepspeed is a PyTorch-specific library, so its features are not applicable in JAX/MaxText.
This function is maintained for API compatibility during the porting process
but will always return False.
"""

def is_deepspeed_zero3_enabled() -> bool:
  """
  Checks if DeepSpeed ZeRO Stage 3 is enabled.

  Since DeepSpeed is a PyTorch-specific library, this feature is not
  applicable in a JAX environment. This function always returns False.
  """
  return False

from __future__ import annotations
from contextlib import contextmanager
from jax.sharding import Mesh


@contextmanager
def init_on_device(mesh: Mesh):
  """A context manager for initializing Flax models on a specific JAX mesh.

  In JAX, device placement for parameters is controlled by sharding annotations
  (e.g., PartitionSpec) and the `jax.sharding.Mesh` context. This function
  provides a semantic equivalent to PyTorch's `init_on_device` by entering
  the provided mesh context. Any `model.init` call within this context will
  create sharded parameters directly on the devices of the mesh according to
  the model's sharding rules.

  The `include_buffers` argument from the original PyTorch function is not
  needed in JAX, as all variable collections (parameters, buffers like
  batch_stats, etc.) are handled by the same sharding mechanism.

  Args:
    mesh: The device mesh to initialize all parameters on.

  Example:
    .. code-block:: python

      from jax.sharding import Mesh, PartitionSpec
      from jax.experimental import mesh_utils
      import flax.linen as nn
      import jax
      import jax.numpy as jnp

      devices = mesh_utils.create_device_mesh((jax.device_count(),))
      mesh = Mesh(devices, axis_names=('data',))

      class SimpleModel(nn.Module):
        @nn.compact
        def __call__(self, x):
          # In a real model, sharding is defined here or via logical rules.
          kernel_init = nn.with_partitioning(
              nn.initializers.lecun_normal(), ('data', 'model'))
          return nn.Dense(features=4, kernel_init=kernel_init)(x)

      with init_on_device(mesh=mesh):
        model = SimpleModel()
        # The `init` call inside the context creates sharded parameters.
        params = model.init(jax.random.PRNGKey(0), jnp.ones((2, 3)))
        # `params` is now a PyTree of sharded JAX arrays.
  """
  with mesh:
    yield

import jax

# Cache this result as it may involve communication or environment variable parsing.
_jax_distributed_available = jax.process_count() > 1

from flax.linen import Module as nn_Module
from jax.sharding import Mesh
from typing import Optional, Dict, Any

# In JAX/Flax, there is no direct equivalent to the PyTorch `ALL_PARALLEL_STYLES`
# and the concept of adding hooks. Parallelism strategies are defined declaratively
# within the modules themselves during initialization.

def add_tensor_parallel_hooks_to_module(
    model: nn_Module,
    module: nn_Module,
    tp_plan: Dict[str, Any],
    layer_name: str,
    current_module_plan: Optional[str],
    device_mesh: Mesh,
    parameter_name: Optional[str] = None,
):
  r"""
  This function is called in `PretrainedModel.post_init()`. It is responsible of adding hooks
  to the modules of the `model`, based on the `PretrainedModel._tp_plan`.

  This is the place where we add the `pre_forward` and `post_forwards` hooks. These are defined
  for each `TensorParallelLayer` as `_prepare_input_fn` and `_prepare_output_fn`.
  """
  if current_module_plan is not None:
    # In JAX/Flax, parallelism is handled declaratively within the module's
    # definition and initialization, not by adding hooks post-creation.
    # The `prepare_module_tp` call from the original PyTorch code, which
    # registers forward hooks, has no direct equivalent. That functionality
    # is already part of the module's `__call__` method in a JAX implementation.
    # The `try...except NotImplementedError` block is omitted as there is no
    # equivalent operation to attempt.

    # Attach the TP plan to the module for introspection.
    module._hf_tp_plan = current_module_plan

    # Note: Dynamically modifying __repr__ on a Flax module is not standard
    # practice but is preserved for structural similarity.
    original_repr_method = module.__repr__
    module.__repr__ = lambda: f"{original_repr_method()}\nTP Plan: {current_module_plan}"

from typing import Any, Dict, Optional
import flax.nnx as nnx
from flax.nnx import graph
from jax.sharding import Mesh

# From generated_code.Qwen3MoeForCausalLM.distributed_utils
from generated_code.Qwen3MoeForCausalLM.distributed_utils import (
    ALL_PARALLEL_STYLES,
    _jax_distributed_available,
    add_tensor_parallel_hooks_to_module,
)
# From generated_code.Qwen3MoeForCausalLM.model_utils
from generated_code.Qwen3MoeForCausalLM.model_utils import _get_parameter_tp_plan


def distribute_model(
    model: nnx.Module,
    distributed_config: Optional[Dict[str, Any]],
    device_mesh: Mesh,
    tp_size: int,
):
  """
  Distributes the model according to the provided configuration and device mesh.

  This function sets up the model for tensor parallelism (TP) or expert
  parallelism (EP) by consolidating the parallelism plan and annotating
  each submodule with its respective plan. In JAX/NNX, unlike PyTorch, this
  process is about configuration and annotation rather than applying functional
  hooks, as parallelism is handled declaratively within the modules.

  Args:
    model: The model to be distributed.
    distributed_config: Configuration for distributed execution.
    device_mesh: The JAX device mesh for sharding.
    tp_size: The tensor parallelism size.
  Returns:
    The distributed model.
  """
  _plan = '_tp_plan'
  tp_plan = (getattr(model, '_tp_plan', None) or {}).copy()
  model._tp_plan = getattr(model.config, 'base_model_tp_plan', {}).copy()
  model._tp_plan.update(tp_plan)
  model._tp_size = tp_size
  model._device_mesh = device_mesh

  enable_expert_parallel = False
  if distributed_config is not None:
    if isinstance(distributed_config, dict):
      enable_expert_parallel = distributed_config.get(
          'enable_expert_parallel', False
      )
    else:
      # Assuming distributed_config is an object with an enable_expert_parallel attribute
      enable_expert_parallel = getattr(
          distributed_config, 'enable_expert_parallel', False
      )

  if enable_expert_parallel:
    _plan = '_ep_plan'
    model._tp_plan = getattr(
        model.config, 'base_model_ep_plan', model._tp_plan
    ).copy()

  # now fetch my childrens (direct submodules)
  for name, module in vars(model).items():
    if isinstance(module, nnx.Module):
      if plan := getattr(module, _plan, getattr(module, 'tp_plan', None)):
        model._tp_plan.update(
            {f'{name}.{k}': v for k, v in plan.copy().items()}
        )
      if hasattr(module, 'config'):
        plan = getattr(module.config, f'base_model{_plan}', {})
        if not plan:
          plan = getattr(module.config, 'base_model_tp_plan', {})
        model._tp_plan.update(
            {f'{name}.{k}': v for k, v in plan.copy().items()}
        )

  if model._tp_plan is not None and _jax_distributed_available:
    for v in model._tp_plan.values():
      if v not in ALL_PARALLEL_STYLES:
        raise ValueError(
            f'Unsupported tensor parallel style {v}. Supported styles are'
            f' {ALL_PARALLEL_STYLES}'
        )

    flat_graph = graph.flatten(model)
    for path, module in flat_graph.items():
      if isinstance(module, nnx.Module):
        name_parts = []
        for p in path:
          if isinstance(p, graph.Attr):
            name_parts.append(p.key)
          elif isinstance(p, graph.GetItem):
            name_parts.append(str(p.key))
        name = '.'.join(name_parts)

        if not getattr(module, '_is_hooked', False):
          plan = _get_parameter_tp_plan(
              parameter_name=name, tp_plan=model._tp_plan, is_weight=False
          )
          add_tensor_parallel_hooks_to_module(
              model=model,
              module=module,
              tp_plan=model._tp_plan,
              layer_name='',
              current_module_plan=plan,
              device_mesh=device_mesh,
          )
        module._is_hooked = True
  return model

from typing import Dict
from jax import Array
from jax.sharding import Mesh, PartitionSpec
from jax.experimental.pjit import with_sharding_constraint

# Re-used from generated_code.Qwen3MoeForCausalLM.model_utils._get_parameter_tp_plan
from .model_utils import _get_parameter_tp_plan


def convert_local_tensor_to_dtensor(
    parameter: Array, parameter_name: str, device_mesh: Mesh, tp_plan: Dict[str, str]
) -> Array:
  """
  Converts a local variant of weights to a sharded JAX array with corresponding placements.
  Shouldn't be done ever except of before saving the model.
  """
  if "." in parameter_name:
    _, param_type = parameter_name.rsplit(".", 1)
  else:
    param_type = parameter_name

  tp_style = _get_parameter_tp_plan(parameter_name, tp_plan)
  if not tp_style:
    return parameter

  if tp_style not in ["local_packed_rowwise", "local_rowwise", "local_colwise"]:
    return parameter

  # Assume the tensor parallelism axis is named 'tp'. This is a common MaxText convention.
  tp_axis_name = "tp"

  placements = None
  # TODO: this logic should be wrapped in a function, this is copied from corresponding tp classes.
  if tp_style == "local_packed_rowwise":
    # Corresponds to Shard(-1)
    spec = [None] * parameter.ndim
    spec[-1] = tp_axis_name
    placements = PartitionSpec(*spec)
  elif tp_style == "local_rowwise":
    if param_type == "bias":
      # Corresponds to Replicate()
      placements = PartitionSpec()
    else:
      # Corresponds to Shard(-1)
      spec = [None] * parameter.ndim
      spec[-1] = tp_axis_name
      placements = PartitionSpec(*spec)
  elif tp_style == "local_colwise":
    if param_type == "bias":
      # Corresponds to Shard(-1)
      spec = [None] * parameter.ndim
      spec[-1] = tp_axis_name
      placements = PartitionSpec(*spec)
    else:
      # Corresponds to Shard(-2)
      spec = [None] * parameter.ndim
      spec[-2] = tp_axis_name
      placements = PartitionSpec(*spec)

  if placements is not None:
    # with_sharding_constraint is the JAX way to express the desired layout
    # for an intermediate tensor within a pjit'd computation.
    # This assumes the function is called within a pjit context.
    return with_sharding_constraint(parameter, placements)
  else:
    return parameter

from typing import Dict

import jax
from jax.experimental.pjit import with_sharding_constraint
from jax.sharding import Mesh, PartitionSpec

# The following import is assumed to exist and correspond to the
# `_get_parameter_tp_plan` function in the original PyTorch file.
# from MaxText.layers.distributed_utils import _get_parameter_tp_plan


def convert_local_tensor_to_dtensor(
    parameter: jax.Array,
    parameter_name: str,
    device_mesh: Mesh,
    tp_plan: Dict[str, str],
) -> jax.Array:
  """Converts a local JAX array to a sharded JAX array with corresponding placements.

  This is the JAX equivalent of converting a torch.Tensor to a DTensor. It
  applies a sharding constraint based on the tensor parallelism plan. This
  function should be called within a pjit context.

  Args:
    parameter: The local JAX array to be sharded.
    parameter_name: The name of the parameter, used to look up the TP plan.
    device_mesh: The JAX device mesh. Unused in JAX implementation as the mesh
      is implicit in the pjit context, but kept for API consistency.
    tp_plan: A dictionary mapping parameter names to TP styles.

  Returns:
    A JAX array, potentially with a sharding constraint applied.
  """
  del device_mesh  # Unused in JAX, mesh is implicit from context
  param_type = (
      parameter_name.rsplit(".", 1)[-1]
      if "." in parameter_name
      else parameter_name
  )
  # _get_parameter_tp_plan is not provided, but is required for this function to work
  tp_style = _get_parameter_tp_plan(parameter_name, tp_plan)

  if not tp_style or tp_style not in [
      "local_packed_rowwise",
      "local_rowwise",
      "local_colwise",
  ]:
    return parameter

  placements = None
  # Assuming the tensor parallel mesh axis is named 'tp'
  if tp_style == "local_packed_rowwise":
    placements = PartitionSpec(*(None for _ in range(parameter.ndim - 1)), "tp")
  elif tp_style == "local_rowwise":
    if param_type == "bias":
      placements = PartitionSpec()  # Replicate
    else:
      placements = PartitionSpec(
          *(None for _ in range(parameter.ndim - 1)), "tp"
      )
  elif tp_style == "local_colwise":
    if param_type == "bias":
      placements = PartitionSpec(
          *(None for _ in range(parameter.ndim - 1)), "tp"
      )
    else:
      # Shard on the second to last dimension
      placements = PartitionSpec(
          *(None for _ in range(parameter.ndim - 2)), "tp", None
      )

  if placements is not None:
    return with_sharding_constraint(parameter, placements)
  else:
    return parameter


def replace_state_dict_local_with_dtensor(
    state_dict: Dict[str, jax.Array],
    tp_plan: Dict[str, str],
    device_mesh: Mesh,
) -> Dict[str, jax.Array]:
  """Replaces local arrays with sharded arrays based on the TP plan.

  This is the JAX equivalent of replacing torch.Tensors with DTensors for
  tensors that were sharded with a `local_*` strategy. It makes determining
  their proper size possible and applies sharding constraints.

  Args:
    state_dict: A dictionary mapping parameter names to JAX arrays.
    tp_plan: A dictionary mapping parameter names to TP styles.
    device_mesh: The JAX device mesh.

  Returns:
    The state_dict with sharding constraints applied to relevant arrays.
  """
  for key, value in state_dict.items():
    # In JAX, there's no DTensor class. We check if the value is a JAX array.
    # The conversion function is idempotent for already-sharded or non-local
    # tensors.
    if isinstance(value, jax.Array):
      state_dict[key] = convert_local_tensor_to_dtensor(
          value, key, device_mesh, tp_plan
      )
  return state_dict

# Used src.MaxText.layers.distributed_utils._jax_distributed_available
from ..distributed_utils import _jax_distributed_available

# In JAX, the equivalent of DTensor (sharded jax.Array) is a core feature.
# Its availability is tied to whether the distributed environment is available,
# unlike in PyTorch where it was introduced in a specific version.
_is_dtensor_available = _jax_distributed_available

import jax

# Reused from generated_code.Qwen3MoeForCausalLM.distributed_utils._jax_distributed_available
_jax_distributed_available = jax.process_count() > 1

# Copyright 2024 The HuggingFace Team. All rights reserved.
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
""" JAX - PyTorch general utilities."""
from __future__ import annotations

from typing import TYPE_CHECKING

import jax


if TYPE_CHECKING:
  from maxtext.common_types import Config


def is_fsdp_managed_module(config: Config) -> bool:
  """
  Checks if the model is configured to use an FSDP-like strategy.

  In JAX/MaxText, FSDP is a parameter sharding strategy controlled by the global
  configuration, rather than a wrapper around a module instance as in PyTorch.
  This function checks for that configuration.

  Args:
    config: The MaxText configuration object.

  Returns:
    A boolean indicating if an FSDP sharding strategy is active in a
    distributed environment.
  """
  if jax.process_count() <= 1:
    return False

  # In MaxText, FSDP is a sharding strategy for parameters defined in the config.
  is_fsdp_strategy = getattr(config, "fsdp_sharding_strategy", "") == "fsdp"

  return is_fsdp_strategy
CCL_IMPORT_ERROR = """
{0} requires a communication library for distributed training that was not found in your environment.
For JAX on NVIDIA GPUs, this is typically NCCL. Please check the JAX installation instructions
for your specific hardware and environment to ensure all necessary libraries are installed.
Please note that you may need to restart your runtime after installation.
"""
def is_ccl_available():
  """
  Checks if the torch ccl library is available.
  oneCCL is a PyTorch-specific library for distributed communication on Intel hardware.
  It is not used in JAX, so this check will always be False.
  """
  return _is_ccl_available
