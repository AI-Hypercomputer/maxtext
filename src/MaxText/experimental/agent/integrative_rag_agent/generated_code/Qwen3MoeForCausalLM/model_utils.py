
# The _CAN_RECORD_REGISTRY is part of a PyTorch-specific mechanism for capturing
# intermediate model outputs by monkey-patching `forward` methods. This pattern
# does not directly translate to JAX's functional paradigm. In JAX/Flax,
# capturing intermediates is typically handled by explicitly returning them from
# the model's `__call__` method or by using `flax.linen.sow`.
_CAN_RECORD_REGISTRY = {}

from collections import defaultdict
from typing import Any, Dict, List

from flax.traverse_util import flatten_dict


def find_tied_parameters(params: Dict[str, Any]) -> List[List[str]]:
  """
  Find the tied parameters in a given model's parameter PyTree.

  In Flax, parameters are tied if they point to the exact same JAX array object in memory. This function finds
  such cases by grouping parameter paths by the ID of their corresponding array.

  Args:
      params (`Dict[str, Any]`): The PyTree of model parameters.

  Returns:
      `List[List[str]]`: A list of lists of parameter names that are all tied together.

  Example:

  
from collections import defaultdict
from typing import Any, List

from flax.traverse_util import flatten_dict


# A PyTree is a nested structure of containers and leaves.
PyTree = Any


def find_tied_parameters(params: PyTree) -> List[List[str]]:
  """
  Finds the tied parameters in a given PyTree of parameters.

  In Flax, parameters are considered tied if they are the exact same JAX array
  object in memory, meaning two different paths in the PyTree point to the same
  underlying array.

  Args:
      params (PyTree): A PyTree of parameters (e.g., from a Flax model's `state.params`).

  Returns:
      list[list[str]]: A list of lists of parameter names being all tied together.

  Example:

  
import re
from typing import Dict, Optional


def _get_parameter_tp_plan(
    parameter_name: str, tp_plan: Dict[str, str], is_weight: bool = True
) -> Optional[str]:
  """Get the TP style for a parameter from the TP plan.

  The TP plan is a dictionary that maps parameter names to TP styles.
  The parameter name can be a generic name with wildcards (e.g. "*.weight") or
  a specific name (e.g. "layer_1.weight").

  The `is_weight` is important because for weights, we want to support
  `.weights` and `.bias` cases seamlessly! but not parrent classes for
  `post_init` calls
  """
  generic_param_name = re.sub(r"\d+", "*", parameter_name)
  if generic_param_name in tp_plan:
    return tp_plan[generic_param_name]
  elif "." in generic_param_name and generic_param_name.rsplit(".", 1)[0] in tp_plan and is_weight:
    return tp_plan[generic_param_name.rsplit(".", 1)[0]]
  return None

from typing import Optional
import flax.linen as nn


class EmbeddingAccessMixin:
  """
    Base utilities to regroup getters and setters for embeddings.
    Introduces the `input_layer_embed` attribute, which indicates
    where the input embeddings come from and where they
    should be set.
    """

  _input_embed_layer = "embed_tokens"  # default layer that holds input embeddings.

  def get_input_embeddings(self) -> nn.Module:
    """
    Returns the model's input embeddings.

        Returns:
            `nn.Module`: A Flax module mapping vocabulary to hidden states.
        """

    # 1) Check if the model has an attribute named 'embed_tokens' (the standard input embedding layer
    #  for most NLP models), and if so, return it.

    name = getattr(self, "_input_embed_layer", "embed_tokens")

    if (default_embedding := getattr(self, name, None)) is not None:
      return default_embedding
    # 2) encoder/decoder and VLMs like `Gemma3nForConditionalGeneration`

    if hasattr(self, "model") and hasattr(self.model, "embed_tokens"):
      return self.model.embed_tokens

    # 3) vanilla decoder‑only architectures
    elif hasattr(self, "embed_tokens"):
      return self.embed_tokens
    else:
      base_model = getattr(self, "base_model_prefix", None)
      if base_model is not None:
        base_model = getattr(self, base_model, None)
        if base_model is not None and base_model is not self:
          return base_model.get_input_embeddings()
      raise NotImplementedError(
          f"`get_input_embeddings` not auto‑handled for {self.__class__.__name__}; "
          "please override in the subclass."
      )

  def set_input_embeddings(self, value: nn.Module):
    """Fallback setter that handles **~70 %** of models in the code‑base.

        Order of attempts:
        1. `self.model.embed_tokens`
        2. `self.embed_tokens`
        3. delegate to the *base model* if one exists
        4. otherwise raise `NotImplementedError` so subclasses still can (and
            should) override for exotic layouts.
        """

    # 1) encoder/decoder and VLMs like `Gemma3nForConditionalGeneration`
    name = getattr(self, "_input_embed_layer", "embed_tokens")
    if hasattr(self, "model") and hasattr(self.model, name):
      setattr(self.model, name, value)
    # 2) as well as vanilla decoder‑only architectures
    elif hasattr(self, name):
      setattr(self, name, value)
    # 3) recurse once into the registered *base* model (e.g. for encoder/decoder)
    elif getattr(self, self.base_model_prefix, self) is not self:
      base_model = getattr(self, self.base_model_prefix, self)
      base_model.set_input_embeddings(value)
    else:
      raise NotImplementedError(
          f"`set_input_embeddings` not auto‑handled for {self.__class__.__name__}; please override in the subclass."
      )

  def get_output_embeddings(self) -> Optional[nn.Module]:
    """Returns the model's output embeddings."""
    if not hasattr(self, "lm_head"):
      return None
    try:
      # Speech / vision backbones raise here, so we return None.
      # Legit use of get_input_embs?
      self.get_input_embeddings()
    except NotImplementedError:
      return None
    return self.lm_head

  def set_output_embeddings(self, new_embeddings: nn.Module):
    """
        Sets the model's output embedding, defaulting to setting new_embeddings to lm_head.
        """
    if hasattr(self, "lm_head"):
      self.lm_head = new_embeddings

from typing import TypeVar

# This is a generic type variable that is used for type hinting in the PreTrainedModel class.
# It's bound to 'PreTrainedModel' to indicate that it can be any subclass of PreTrainedModel.
# The string forward reference is used to handle the case where PreTrainedModel is not yet defined.
SpecificPreTrainedModelType = TypeVar("SpecificPreTrainedModelType", bound="PreTrainedModel")

from typing import Optional


def _add_variant(weights_name: str, variant: Optional[str] = None) -> str:
  """Adds a variant to a weights name, e.g. pytorch_model.bin -> pytorch_model.fp16.bin."""
  if variant is not None:
    path, name = weights_name.rsplit(".", 1)
    weights_name = f"{path}.{variant}.{name}"
  return weights_name

from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import unfreeze
from flax.traverse_util import flatten_dict
from safetensors import safe_open
from safetensors.flax import load_file as safe_load_file


# This is a new helper function required by the converted code.
# It is a simplified JAX equivalent of the PyTorch `load_state_dict`
# function found in the full source file, with support for `map_location="meta"`.
def load_state_dict(
    checkpoint_file: str,
    is_quantized: bool = False,
    map_location: Optional[str] = "cpu",
    weights_only: bool = True,
) -> Dict[str, Any]:
  """
    Reads a `safetensor` checkpoint file. We load the checkpoint on "cpu" by default.
    The `map_location="meta"` option loads only shape and dtype information.
    """
  # Unused arguments to maintain signature consistency with the original PyTorch function.
  del is_quantized, weights_only

  if not checkpoint_file.endswith(".safetensors"):
    raise NotImplementedError("Only .safetensors loading is implemented for this JAX conversion.")

  if map_location == "meta":
    state_dict = {}
    with safe_open(checkpoint_file, framework="flax") as f:
      for k in f.keys():
        tensor_slice = f.get_slice(k)
        try:
          # Safetensors uses dtypes like 'F16', 'BF16', etc. jnp.dtype can parse most of them.
          dtype_str = tensor_slice.get_dtype()
          # Manual mapping for common abbreviations if needed
          dtype_map = {"F16": "float16", "BF16": "bfloat16", "F32": "float32"}
          dtype = jnp.dtype(dtype_map.get(dtype_str, dtype_str))
        except TypeError:
          raise ValueError(f"Cannot map safetensors dtype {tensor_slice.get_dtype()} to JAX dtype.")
        shape = tensor_slice.get_shape()
        state_dict[k] = jax.ShapeDtypeStruct(shape, dtype)
    return state_dict
  else:
    return safe_load_file(checkpoint_file)


def _find_mismatched_keys(
    model_params: Dict[str, Any],
    state_dict: Optional[Dict[str, Any]],
    checkpoint_files: Optional[List[str]],
    ignore_mismatched_sizes: bool,
    keys_to_rename_mapping: Dict[str, str],
    is_quantized: bool,
    weights_only: bool,
) -> Tuple[List[str], List[Tuple[Tuple[int, ...], Tuple[int, ...]]]]:
  """
    Find potential shape mismatch between the different state dicts and the model parameters, but only if `ignore_mismatched_sizes`
    is True. Otherwise, return immediately and any shape mismatch that may exist will be raised later on. This avoids checking
    every parameter in advance, as shape mismatch are extremely rare in practice. If we want to ignore them however, we do
    need to check in advance as we need to know which parameters we need to move back from meta to cpu, and initialize
    correctly. Indeed, as our model initialization takes place at the module level, and not the weight level, in the
    case of a sharded checkpoint we cannot correctly initialize the weights according to `model._init_weights()` if we perform
    this check on each state dict at loading time (after the first loaded checkpoint, there are no way to initialize only the
    mismatched weights if any, without overwriting the previously loaded weights as well because all the module will be
    initialized, not only the weights that are mismatched).
    """

  # An error will be raised later on anyway if there is a mismatch - this avoids running the rest of this function
  # if there are no mismatch (which is almost always the case)
  if not ignore_mismatched_sizes:
    return [], []

  if state_dict is not None:
    checkpoint_files = [""]

  model_state_dict = {".": ".join(map(str, k)), "": v for k, v in flatten_dict(unfreeze(model_params)).items()}

  mismatched_keys = []
  mismatched_shapes = []
  for shard_file in checkpoint_files:
    # If shard_file is "", we use the existing state_dict instead of loading it
    if shard_file != "":
      current_state_dict = load_state_dict(
          shard_file, is_quantized=is_quantized, map_location="meta", weights_only=weights_only
      )
    else:
      current_state_dict = state_dict

    # Fix the key names
    new_state_dict = {
        keys_to_rename_mapping[k]: v for k, v in current_state_dict.items() if k in keys_to_rename_mapping
    }

    for key, tensor_info in new_state_dict.items():
      if key in model_state_dict and tensor_info.shape != model_state_dict[key].shape:
        # tensor_info can be a JAX array or a ShapeDtypeStruct
        ckpt_numel = np.prod(tensor_info.shape)
        model_numel = model_state_dict[key].size

        # This skips size mismatches for 4-bit weights. Two 4-bit values share an 8-bit container, causing size differences.
        # Without matching with module type or parameter type it seems like a practical way to detect valid 4bit weights.
        if not (
            is_quantized
            and len(tensor_info.shape) > 0
            and tensor_info.shape[-1] == 1
            and ckpt_numel * 2 == model_numel
        ):
          mismatched_keys.append(key)
          mismatched_shapes.append((tensor_info.shape, model_state_dict[key].shape))

  return mismatched_keys, mismatched_shapes

import flax.linen as nn
from flax.experimental import nnx
from typing import Any, List

def _get_tied_weight_keys(module: Any, prefix: str = "") -> List[str]:
  """
  Recursively retrieves the keys of tied weights from a Flax module and its submodules.
  This is the JAX/Flax equivalent of the PyTorch utility function.
  """
  tied_weight_keys: List[str] = []

  # Check for tied weights keys on the current module
  if hasattr(module, "_tied_weights_keys") and getattr(module, "_tied_weights_keys") is not None:
    names = [f"{prefix}.{k}" if prefix else k for k in module._tied_weights_keys]
    tied_weight_keys.extend(names)
  if hasattr(module, "_dynamic_tied_weights_keys") and getattr(module, "_dynamic_tied_weights_keys") is not None:
    names = [f"{prefix}.{k}" if prefix else k for k in module._dynamic_tied_weights_keys]
    tied_weight_keys.extend(names)

  # In Flax, submodules are attributes of the parent module. We inspect them
  # to find children, similar to PyTorch's `named_children()`.
  # This handles modules assigned directly to attributes, as well as lists, tuples,
  # and dicts of modules.
  if hasattr(module, "__dict__"):
    for name, attr in vars(module).items():
      if name.startswith("_"):
        continue

      if isinstance(attr, (nn.Module, nnx.Module)):
        local_prefix = f"{prefix}.{name}" if prefix else name
        tied_weight_keys.extend(_get_tied_weight_keys(attr, prefix=local_prefix))
      elif isinstance(attr, (list, tuple)):
        for i, item in enumerate(attr):
          if isinstance(item, (nn.Module, nnx.Module)):
            # Mimic PyTorch ModuleList naming convention (e.g., "layers.0")
            local_prefix = f"{prefix}.{name}.{i}" if prefix else f"{name}.{i}"
            tied_weight_keys.extend(_get_tied_weight_keys(item, prefix=local_prefix))
      elif isinstance(attr, dict):
        for key, item in attr.items():
          if isinstance(item, (nn.Module, nnx.Module)):
            # Mimic PyTorch ModuleDict naming convention (e.g., "layers.attention")
            local_prefix = f"{prefix}.{name}.{key}" if prefix else f"{name}.{key}"
            tied_weight_keys.extend(_get_tied_weight_keys(item, prefix=local_prefix))

  return tied_weight_keys

from flax.linen import Module
from typing import List

# From src.MaxText.layers.modeling_utils._get_tied_weight_keys
def _get_tied_weight_keys(module: Module, prefix: str = "") -> List[str]:
  """Recursively retrieves the keys of tied weights from a Flax module."""
  tied_weight_keys = []
  # Check for explicitly declared tied weight keys on the current module
  if getattr(module, "_tied_weights_keys", None) is not None:
    names = [f"{prefix}.{k}" if prefix else k for k in module._tied_weights_keys]
    tied_weight_keys.extend(names)
  if getattr(module, "_dynamic_tied_weights_keys", None) is not None:
    names = [
        f"{prefix}.{k}" if prefix else k
        for k in module._dynamic_tied_weights_keys
    ]
    tied_weight_keys.extend(names)

  # Recurse into submodules. In Flax, submodules are attributes, which can be
  # nn.Module instances, or lists/dicts of them.
  for name, child in vars(module).items():
    # Skip private attributes and non-module containers to avoid infinite recursion
    # or traversing things like config objects.
    if name.startswith("_"):
      continue

    if isinstance(child, Module):
      local_prefix = f"{prefix}.{name}" if prefix else name
      tied_weight_keys.extend(_get_tied_weight_keys(child, prefix=local_prefix))
    elif isinstance(child, (list, tuple)):
      for i, sub_child in enumerate(child):
        if isinstance(sub_child, Module):
          local_prefix = f"{prefix}.{name}.{i}" if prefix else f"{name}.{i}"
          tied_weight_keys.extend(
              _get_tied_weight_keys(sub_child, prefix=local_prefix)
          )
    elif isinstance(child, dict):
      for key, sub_child in child.items():
        if isinstance(sub_child, Module):
          local_prefix = f"{prefix}.{name}.{key}" if prefix else f"{name}.{key}"
          tied_weight_keys.extend(
              _get_tied_weight_keys(sub_child, prefix=local_prefix)
          )

  return tied_weight_keys

from contextlib import contextmanager

# internal global flag used when loading quantized models
_is_quantized = False


@contextmanager
def set_quantized_state():
  """Context manager to globally set the quantization state."""
  global _is_quantized
  _is_quantized = True
  try:
    yield
  finally:
    _is_quantized = False

import flax.linen as nn


def unwrap_model(model: nn.Module) -> nn.Module:
  """Recursively unwraps a model from potential containers.

  In JAX, there isn't a direct equivalent of PyTorch's distributed wrappers
  like DDP which wrap the model in a `.module` attribute. This function
  provides a basic unwrapping mechanism for compatibility or for custom
  wrappers that might follow a similar pattern.

  Args:
    model (`flax.linen.Module`): The model to unwrap.

  Returns:
    `flax.linen.Module`: The unwrapped model.
  """
  while hasattr(model, "module"):
    model = model.module
  return model

from typing import Any


def is_timm_config_dict(config_dict: dict[str, Any]) -> bool:
  """Checks whether a config dict is a timm config dict."""
  return "pretrained_cfg" in config_dict
from jax import Array
import jax.numpy as jnp


def find_packed_sequence_indices(position_ids: Array) -> Array:
  """Finds the indices of the sequence to which each new query token in the sequence belongs when using packed
    tensor format (i.e. several sequences packed in the same batch dimension).

  Args:
    position_ids (`jax.Array`): A 2D tensor of shape (batch_size,
      query_length) indicating the positions of each token in the sequences.

  Returns:
    A 2D tensor where each similar integer indicates that the tokens belong to
    the same sequence. For example, if we pack 3 sequences of 2, 3 and 1 tokens
    respectively along a single batch dim, this will return [[0, 0, 1, 1, 1, 2]].
  """
  # What separate different sequences is when 2 consecutive positions_ids are
  # separated by more than 1. So taking the diff (by prepending the first
  # value - 1 to keep correct indexing) and applying cumsum to the result gives
  # exactly the sequence indices Note that we assume that a single sequence
  # cannot span several batch dimensions, i.e. 1 single sequence cannot be part
  # of the end of the first batch dim and the start of the 2nd one for example
  first_dummy_value = position_ids[:, :1] - 1  # We just need the diff on this first value to be 1
  position_diff = jnp.diff(position_ids, prepend=first_dummy_value, axis=-1)
  packed_sequence_mask = (position_diff != 1).cumsum(axis=-1)

  # Here it would be nice to return None if we did not detect packed sequence
  # format, i.e. if `packed_sequence_mask[:, -1] == 0` but it causes issues
  # with export
  return packed_sequence_mask
from collections import defaultdict
from typing import List

from flax.traverse_util import flatten_dict
from jax.typing import PyTree


def find_tied_parameters(params: PyTree, **kwargs) -> List[List[str]]:
  """Find the tied parameters in a given PyTree of parameters.

  <Tip warning={true}>

  The signature accepts keyword arguments, but they are for the recursive part of
  this function and you should ignore them.

  </Tip>

  Args:
    params (`PyTree`): The PyTree of parameters to inspect.

  Returns:
    list[list[str]]: A list of lists of parameter names being all tied
    together.

  Example:

  
from typing import Any, Dict, List
from collections import defaultdict

from flax.traverse_util import flatten_dict
from flax.core import unfreeze


def find_tied_parameters(params: Dict[str, Any]) -> List[List[str]]:
  """
  Find the tied parameters in a given model's parameter PyTree.

  Args:
      params (`Dict[str, Any]`): The PyTree of model parameters.

  Returns:
      list[list[str]]: A list of lists of parameter names being all tied together.

  Example:

  
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def init_empty_weights(include_buffers: bool = False) -> Iterator[None]:
  """A context manager for initializing empty models.

  In JAX, creating an "empty" model (with abstract shapes and dtypes instead
  of allocated arrays) is achieved by wrapping the initialization function with
  `jax.eval_shape`. Since this is an explicit functional wrapper rather than a
  global state that can be manipulated by a context manager, this context
  manager is a no-op. The user is expected to use `jax.eval_shape` on their
  model's `.init` call within this context.

  Args:
    include_buffers: This argument is ignored in the JAX implementation as
      `jax.eval_shape` handles all array-like objects.

  Yields:
    None.

  Example:

  
from __future__ import annotations
from typing import Union

import flax.linen as nn
import jax
from jax.tree_util import tree_leaves


def get_parameter_device(parameter: nn.Module) -> jax.Device:
  """Gets the device of the first available parameter or buffer in the module.

  This is a JAX implementation of the PyTorch `get_parameter_device` function.
  It assumes the module has been initialized and has variables.

  Args:
    parameter: The Flax module to inspect.

  Returns:
    The JAX device of the first found parameter or buffer.

  Raises:
    ValueError: If the module has no variables or no array leaves in its
      variables.
  """
  if not hasattr(parameter, "variables") or not parameter.variables:
    # This can happen if the module is not initialized.
    raise ValueError("The module has no variables. It may not be initialized.")

  # jax.tree_util.tree_leaves will get all arrays from all collections
  # (params, batch_stats, etc.), which is equivalent to finding all tensors
  # in the PyTorch module.
  all_arrays = tree_leaves(parameter.variables)

  if not all_arrays:
    # This case is equivalent to the StopIteration in the PyTorch code's
    # primary path and the fallback path finding no tensors.
    raise ValueError("Could not find any parameters or buffers in the module.")

  # Get the first array we find.
  first_array = all_arrays[0]

  # .devices() returns a set of devices. For an unsharded array, it's one.
  # For a sharded array, we return the first one to match the original's
  # behavior of returning a single device.
  return next(iter(first_array.devices()))

import collections
from typing import Dict, List, Set, Tuple

from jax import Array


def _find_identical(
    tensors: List[Set[str]], state_dict: Dict[str, Array]
) -> Tuple[List[Set[str]], List[Set[str]]]:
  """
  Finds identical tensors from a list of sets of tensor names.

  In JAX, we check for object identity using id() instead of memory pointers,
  as JAX arrays are immutable and memory management is abstracted away. Tensors
  are considered identical if they are the same JAX array object.

  Args:
    tensors: A list of sets, where each set contains names of tensors that
      might share memory.
    state_dict: A dictionary mapping tensor names to JAX arrays.

  Returns:
    A tuple containing:
      - A list of sets of tensor names that are shared but not identical.
      - A list of sets of tensor names that are identical (point to the same
        JAX array object).
  """
  shared_tensors = []
  identical = []
  for shared in tensors:
    if len(shared) < 2:
      continue

    # Group tensor names by the object ID of their corresponding JAX array.
    areas = collections.defaultdict(set)
    for name in shared:
      tensor = state_dict[name]
      # The key is the object ID of the JAX array.
      area = id(tensor)
      areas[area].add(name)

    if len(areas) == 1:
      # All tensors in this set are the same object.
      identical.append(shared)
    else:
      # Tensors in this set are not all identical objects.
      shared_tensors.append(shared)
  return shared_tensors, identical

import re
from typing import Any, Dict, List, Optional, Tuple

from flax.core import unfreeze
from flax.traverse_util import flatten_dict
from flax.linen import Module as FlaxModule

# from .quantizers import HfQuantizer # Placeholder for HfQuantizer import
# Reused from generated_code.Qwen3MoeForCausalLM.model_utils.find_tied_parameters
from generated_code.Qwen3MoeForCausalLM.model_utils import find_tied_parameters


def _find_missing_and_unexpected_keys(
    model: FlaxModule,
    original_checkpoint_keys: List[str],
    checkpoint_keys: List[str],
    loading_base_model_from_task_state_dict: bool,
    hf_quantizer: Optional[Any],  # Using Any for HfQuantizer placeholder
    device_map: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    """Find missing keys (keys that are part of the model parameters but were NOT found in the loaded state dict keys) and unexpected keys
    (keys found in the loaded state dict keys, but that are NOT part of the model parameters)
    """
    prefix = model.config.base_model_prefix

    # Compute expected keys, i.e. keys that the FULL model (not model_to_load) expects
    # In Flax, variables are stored in collections. We flatten them to get dot-separated keys.
    variables = unfreeze(model.variables)
    expected_keys = []
    model_buffers_list = []
    for collection, params_in_collection in variables.items():
        flat_params = flatten_dict(params_in_collection, sep=".")
        keys = list(flat_params.keys())
        expected_keys.extend(keys)
        if collection != "params":
            model_buffers_list.extend(keys)
    model_buffers = set(model_buffers_list)

    if hf_quantizer is not None:
        expected_keys = hf_quantizer.update_expected_keys(model, expected_keys, checkpoint_keys)

    # Adjust prefix of the keys to make them match loaded keys before removing them
    missing_keys = sorted(list(set(expected_keys) - set(checkpoint_keys)))
    unexpected_keys_set = set(checkpoint_keys) - set(expected_keys)
    # If a module has the same name under the base and task specific model, we have to re-add it to unexpected keys
    if loading_base_model_from_task_state_dict:
        task_specific_keys = [k for k in original_checkpoint_keys if not k.startswith(f"{prefix}.")]
        unexpected_keys_set.update(task_specific_keys)

    # Remove nonpersistent buffers from unexpected keys: they are not in the expected keys (model state dict), but
    # may be in the loaded keys. Note that removing all buffers does the job, as they were part of the expected keys anyway
    unexpected_keys = sorted(list(unexpected_keys_set - model_buffers))

    # Old checkpoints may have keys for rotary_emb.inv_freq for each layer, however we moved this buffer to the main model
    # (so the buffer name has changed). Remove them in such a case
    has_inv_freq_buffers = any("rotary_emb.inv_freq" in buffer for buffer in model_buffers)
    if has_inv_freq_buffers:
        unexpected_keys = [k for k in unexpected_keys if "rotary_emb.inv_freq" not in k]

    # The JAX find_tied_parameters function expects the 'params' collection PyTree
    tied_params = find_tied_parameters(model.variables["params"])
    for group in tied_params:
        missing_in_group = [k for k in missing_keys if k in group]
        if len(missing_in_group) > 0 and len(missing_in_group) < len(group):
            missing_keys = [k for k in missing_keys if k not in missing_in_group]

    if hf_quantizer is not None:
        missing_keys = hf_quantizer.update_missing_keys(model, missing_keys, prefix)
        unexpected_keys = hf_quantizer.update_unexpected_keys(model, unexpected_keys, prefix)

    # Model-specific exceptions for missing and unexpected keys (e.g. if the modeling change over time, or any other reason...)
    if hasattr(model.config, "_keys_to_ignore_on_load_missing") and model.config._keys_to_ignore_on_load_missing is not None:
        for pattern in model.config._keys_to_ignore_on_load_missing:
            missing_keys = [k for k in missing_keys if re.search(pattern, k) is None]

    if (
        hasattr(model.config, "_keys_to_ignore_on_load_unexpected")
        and model.config._keys_to_ignore_on_load_unexpected is not None
    ):
        for pattern in model.config._keys_to_ignore_on_load_unexpected:
            unexpected_keys = [k for k in unexpected_keys if re.search(pattern, k) is None]

    return missing_keys, unexpected_keys

from typing import Any, Dict

import jax.numpy as jnp


def get_state_dict_dtype(state_dict: Dict[str, Any]) -> jnp.dtype:
  """
  Returns the first found floating dtype in `state_dict` if there is one, otherwise returns the first dtype.
  """
  for t in state_dict.values():
    if jnp.issubdtype(t.dtype, jnp.floating):
      return t.dtype

  # if no floating dtype was found return whatever the first dtype is
  return next(iter(state_dict.values())).dtype

import os
from threading import Thread
from typing import Dict, List, Optional, Union

from absl import logging
# HF-specific utils are assumed to be in a common hub_utils module
# Some of these were found in the JAX_MODULES_DICT, others are assumed to exist
# for a complete implementation.
from huggingface_hub import (
    cached_file,
    hf_hub_download as download_url,
)

# Reused from generated_code.Qwen3MoeForCausalLM.hub_utils.auto_conversion
# Reused from generated_code.Qwen3MoeForCausalLM.hub_utils.get_checkpoint_shard_files
# Assumed to exist in the same hub_utils module for a complete implementation
from generated_code.Qwen3MoeForCausalLM.hub_utils import (
    auto_conversion,
    get_checkpoint_shard_files,
    has_file,
    is_offline_mode,
    is_remote_url,
)
# Reused from generated_code.Qwen3MoeForCausalLM.constants
from generated_code.Qwen3MoeForCausalLM.constants import (
    FLAX_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
)
# Reused from generated_code.Qwen3MoeForCausalLM.model_utils._add_variant
from generated_code.Qwen3MoeForCausalLM.model_utils import _add_variant


def _get_resolved_checkpoint_files(
    pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
    subfolder: str,
    variant: Optional[str],
    gguf_file: Optional[str],
    from_tf: bool,
    from_flax: bool,
    use_safetensors: bool,
    cache_dir: str,
    force_download: bool,
    proxies: Optional[Dict[str, str]],
    local_files_only: bool,
    token: Optional[Union[str, bool]],
    user_agent: Dict,
    revision: str,
    commit_hash: Optional[str],
    is_remote_code: bool,  # Because we can't determine this inside this function, we need it to be passed in
    transformers_explicit_filename: Optional[str] = None,
) -> tuple[Optional[List[str]], Optional[Dict]]:
  """Get all the checkpoint filenames based on `pretrained_model_name_or_path`, and optional metadata if the
    checkpoints are sharded.
    This function will download the data if necessary.
    """
  is_sharded = False
  resolved_archive_file = None
  filename = None

  if pretrained_model_name_or_path is not None and gguf_file is None:
    pretrained_model_name_or_path = str(pretrained_model_name_or_path)
    is_local = os.path.isdir(pretrained_model_name_or_path)
    if is_local:
      if transformers_explicit_filename is not None:
        # If the filename is explicitly defined, load this by default.
        archive_file = os.path.join(
            pretrained_model_name_or_path,
            subfolder,
            transformers_explicit_filename,
        )
        is_sharded = transformers_explicit_filename.endswith(
            ".safetensors.index.json"
        )
      elif from_tf and os.path.isfile(
          os.path.join(
              pretrained_model_name_or_path, subfolder, TF_WEIGHTS_NAME + ".index"
          )
      ):
        # Load from a TF 1.0 checkpoint in priority if from_tf
        archive_file = os.path.join(
            pretrained_model_name_or_path, subfolder, TF_WEIGHTS_NAME + ".index"
        )
      elif from_tf and os.path.isfile(
          os.path.join(pretrained_model_name_or_path, subfolder, TF2_WEIGHTS_NAME)
      ):
        # Load from a TF 2.0 checkpoint in priority if from_tf
        archive_file = os.path.join(
            pretrained_model_name_or_path, subfolder, TF2_WEIGHTS_NAME
        )
      elif from_flax and os.path.isfile(
          os.path.join(pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME)
      ):
        # Load from a Flax checkpoint in priority if from_flax
        archive_file = os.path.join(
            pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME
        )
      elif use_safetensors is not False and os.path.isfile(
          os.path.join(
              pretrained_model_name_or_path,
              subfolder,
              _add_variant(SAFE_WEIGHTS_NAME, variant),
          )
      ):
        # Load from a safetensors checkpoint
        archive_file = os.path.join(
            pretrained_model_name_or_path,
            subfolder,
            _add_variant(SAFE_WEIGHTS_NAME, variant),
        )
      elif use_safetensors is not False and os.path.isfile(
          os.path.join(
              pretrained_model_name_or_path,
              subfolder,
              _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
          )
      ):
        # Load from a sharded safetensors checkpoint
        archive_file = os.path.join(
            pretrained_model_name_or_path,
            subfolder,
            _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
        )
        is_sharded = True
      elif not use_safetensors and os.path.isfile(
          os.path.join(
              pretrained_model_name_or_path,
              subfolder,
              _add_variant(WEIGHTS_NAME, variant),
          )
      ):
        # Load from a PyTorch checkpoint
        archive_file = os.path.join(
            pretrained_model_name_or_path,
            subfolder,
            _add_variant(WEIGHTS_NAME, variant),
        )
      elif not use_safetensors and os.path.isfile(
          os.path.join(
              pretrained_model_name_or_path,
              subfolder,
              _add_variant(WEIGHTS_INDEX_NAME, variant),
          )
      ):
        # Load from a sharded PyTorch checkpoint
        archive_file = os.path.join(
            pretrained_model_name_or_path,
            subfolder,
            _add_variant(WEIGHTS_INDEX_NAME, variant),
        )
        is_sharded = True
      # At this stage we don't have a weight file so we will raise an error.
      elif not use_safetensors and (
          os.path.isfile(
              os.path.join(
                  pretrained_model_name_or_path,
                  subfolder,
                  TF_WEIGHTS_NAME + ".index",
              )
          )
          or os.path.isfile(
              os.path.join(
                  pretrained_model_name_or_path, subfolder, TF2_WEIGHTS_NAME
              )
          )
      ):
        raise OSError(
            f"Error no file named {_add_variant(WEIGHTS_NAME, variant)} found in"
            f" directory {pretrained_model_name_or_path} but there is a file for"
            " TensorFlow weights. Use `from_tf=True` to load this model from"
            " those weights."
        )
      elif not use_safetensors and os.path.isfile(
          os.path.join(pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME)
      ):
        raise OSError(
            f"Error no file named {_add_variant(WEIGHTS_NAME, variant)} found in"
            f" directory {pretrained_model_name_or_path} but there is a file for"
            " Flax weights. Use `from_flax=True` to load this model from those"
            " weights."
        )
      elif use_safetensors:
        raise OSError(
            f"Error no file named {_add_variant(SAFE_WEIGHTS_NAME, variant)} found"
            f" in directory {pretrained_model_name_or_path}."
        )
      else:
        raise OSError(
            f"Error no file named {_add_variant(WEIGHTS_NAME, variant)},"
            f" {_add_variant(SAFE_WEIGHTS_NAME, variant)}, {TF2_WEIGHTS_NAME},"
            f" {TF_WEIGHTS_NAME + '.index'} or {FLAX_WEIGHTS_NAME} found in"
            f" directory {pretrained_model_name_or_path}."
        )
    elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)):
      archive_file = pretrained_model_name_or_path
      is_local = True
    elif os.path.isfile(
        os.path.join(subfolder, pretrained_model_name_or_path + ".index")
    ):
      if not from_tf:
        raise ValueError(
            "We found a TensorFlow checkpoint at"
            f" {pretrained_model_name_or_path + '.index'}, please set from_tf to"
            " True to load from this checkpoint."
        )
      archive_file = os.path.join(
          subfolder, pretrained_model_name_or_path + ".index"
      )
      is_local = True
    elif is_remote_url(pretrained_model_name_or_path):
      filename = pretrained_model_name_or_path
      resolved_archive_file = download_url(pretrained_model_name_or_path)
    else:
      # set correct filename
      if transformers_explicit_filename is not None:
        filename = transformers_explicit_filename
        is_sharded = transformers_explicit_filename.endswith(
            ".safetensors.index.json"
        )
      elif from_tf:
        filename = TF2_WEIGHTS_NAME
      elif from_flax:
        filename = FLAX_WEIGHTS_NAME
      elif use_safetensors is not False:
        filename = _add_variant(SAFE_WEIGHTS_NAME, variant)
      else:
        filename = _add_variant(WEIGHTS_NAME, variant)

      try:
        # Load from URL or cache if already cached
        cached_file_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "local_files_only": local_files_only,
            "token": token,
            "user_agent": user_agent,
            "revision": revision,
            "subfolder": subfolder,
            "_raise_exceptions_for_gated_repo": False,
            "_raise_exceptions_for_missing_entries": False,
            "_commit_hash": commit_hash,
        }
        resolved_archive_file = cached_file(
            pretrained_model_name_or_path, filename, **cached_file_kwargs
        )

        # Since we set _raise_exceptions_for_missing_entries=False, we don't get an exception but a None
        # result when internet is up, the repo and revision exist, but the file does not.
        if (
            resolved_archive_file is None
            and filename == _add_variant(SAFE_WEIGHTS_NAME, variant)
        ):
          # Maybe the checkpoint is sharded, we try to grab the index name in this case.
          resolved_archive_file = cached_file(
              pretrained_model_name_or_path,
              _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
              **cached_file_kwargs,
          )
          if resolved_archive_file is not None:
            is_sharded = True
          elif use_safetensors:
            if revision == "main":
              resolved_archive_file, revision, is_sharded = auto_conversion(
                  pretrained_model_name_or_path, **cached_file_kwargs
              )
            cached_file_kwargs["revision"] = revision
            if resolved_archive_file is None:
              raise OSError(
                  f"{pretrained_model_name_or_path} does not appear to have a"
                  " file named"
                  f" {_add_variant(SAFE_WEIGHTS_NAME, variant)} or"
                  f" {_add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)} and thus"
                  " cannot be loaded with `safetensors`. Please make sure that"
                  " the model has been saved with `safe_serialization=True` or"
                  " do not set `use_safetensors=True`."
              )
          else:
            # This repo has no safetensors file of any kind, we switch to PyTorch.
            filename = _add_variant(WEIGHTS_NAME, variant)
            resolved_archive_file = cached_file(
                pretrained_model_name_or_path, filename, **cached_file_kwargs
            )
        if (
            resolved_archive_file is None
            and filename == _add_variant(WEIGHTS_NAME, variant)
        ):
          # Maybe the checkpoint is sharded, we try to grab the index name in this case.
          resolved_archive_file = cached_file(
              pretrained_model_name_or_path,
              _add_variant(WEIGHTS_INDEX_NAME, variant),
              **cached_file_kwargs,
          )
          if resolved_archive_file is not None:
            is_sharded = True
        if not local_files_only and not is_offline_mode():
          if resolved_archive_file is not None:
            if filename in [WEIGHTS_NAME, WEIGHTS_INDEX_NAME]:
              # If the PyTorch file was found, check if there is a safetensors file on the repository
              # If there is no safetensors file on the repositories, start an auto conversion
              safe_weights_name = (
                  SAFE_WEIGHTS_INDEX_NAME if is_sharded else SAFE_WEIGHTS_NAME
              )
              has_file_kwargs = {
                  "revision": revision,
                  "proxies": proxies,
                  "token": token,
                  "cache_dir": cache_dir,
                  "local_files_only": local_files_only,
              }
              cached_file_kwargs = {
                  "cache_dir": cache_dir,
                  "force_download": force_download,
                  "local_files_only": local_files_only,
                  "user_agent": user_agent,
                  "subfolder": subfolder,
                  "_raise_exceptions_for_gated_repo": False,
                  "_raise_exceptions_for_missing_entries": False,
                  "_commit_hash": commit_hash,
                  **has_file_kwargs,
              }
              if not has_file(
                  pretrained_model_name_or_path,
                  safe_weights_name,
                  **has_file_kwargs,
              ) and not is_remote_code:
                Thread(
                    target=auto_conversion,
                    args=(pretrained_model_name_or_path,),
                    kwargs={
                        "ignore_errors_during_conversion": True,
                        **cached_file_kwargs,
                    },
                    name="Thread-auto_conversion",
                ).start()
          else:
            # Otherwise, no PyTorch file was found, maybe there is a TF or Flax model file.
            # We try those to give a helpful error message.
            has_file_kwargs = {
                "revision": revision,
                "proxies": proxies,
                "token": token,
                "cache_dir": cache_dir,
                "local_files_only": local_files_only,
            }
            if has_file(
                pretrained_model_name_or_path, TF2_WEIGHTS_NAME, **has_file_kwargs
            ):
              raise OSError(
                  f"{pretrained_model_name_or_path} does not appear to have a"
                  " file named"
                  f" {_add_variant(WEIGHTS_NAME, variant)} but there is a file"
                  " for TensorFlow weights. Use `from_tf=True` to load this"
                  " model from those weights."
              )
            elif has_file(
                pretrained_model_name_or_path, FLAX_WEIGHTS_NAME, **has_file_kwargs
            ):
              raise OSError(
                  f"{pretrained_model_name_or_path} does not appear to have a"
                  " file named"
                  f" {_add_variant(WEIGHTS_NAME, variant)} but there is a file"
                  " for Flax weights. Use `from_flax=True` to load this model"
                  " from those weights."
              )
            elif variant is not None and has_file(
                pretrained_model_name_or_path, WEIGHTS_NAME, **has_file_kwargs
            ):
              raise OSError(
                  f"{pretrained_model_name_or_path} does not appear to have a"
                  " file named"
                  f" {_add_variant(WEIGHTS_NAME, variant)} but there is a file"
                  f" without the variant {variant}. Use `variant=None` to load"
                  " this model from those weights."
              )
            else:
              raise OSError(
                  f"{pretrained_model_name_or_path} does not appear to have a"
                  " file named"
                  f" {_add_variant(WEIGHTS_NAME, variant)},"
                  f" {_add_variant(SAFE_WEIGHTS_NAME, variant)},"
                  f" {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or"
                  f" {FLAX_WEIGHTS_NAME}."
              )

      except OSError:
        # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
        # to the original exception.
        raise
      except Exception as e:
        # For any other exception, we throw a generic error.
        raise OSError(
            f"Can't load the model for '{pretrained_model_name_or_path}'. If you"
            " were trying to load it from 'https://huggingface.co/models', make"
            " sure you don't have a local directory with the same name."
            f" Otherwise, make sure '{pretrained_model_name_or_path}' is the"
            " correct path to a directory containing a file named"
            f" {_add_variant(WEIGHTS_NAME, variant)}, {TF2_WEIGHTS_NAME},"
            f" {TF_WEIGHTS_NAME} or {FLAX_WEIGHTS_NAME}."
        ) from e

    if is_local:
      logging.info(f"loading weights file {archive_file}")
      resolved_archive_file = archive_file
    else:
      logging.info(
          f"loading weights file {filename} from cache at {resolved_archive_file}"
      )

  elif gguf_file:
    # Case 1: the GGUF file is present locally
    if os.path.isfile(gguf_file):
      resolved_archive_file = gguf_file
    # Case 2: The GGUF path is a location on the Hub
    # Load from URL or cache if already cached
    else:
      cached_file_kwargs = {
          "cache_dir": cache_dir,
          "force_download": force_download,
          "proxies": proxies,
          "local_files_only": local_files_only,
          "token": token,
          "user_agent": user_agent,
          "revision": revision,
          "subfolder": subfolder,
          "_raise_exceptions_for_gated_repo": False,
          "_raise_exceptions_for_missing_entries": False,
          "_commit_hash": commit_hash,
      }

      resolved_archive_file = cached_file(
          pretrained_model_name_or_path, gguf_file, **cached_file_kwargs
      )

  # We now download and resolve all checkpoint files if the checkpoint is sharded
  sharded_metadata = None
  if is_sharded:
    checkpoint_files, sharded_metadata = get_checkpoint_shard_files(
        pretrained_model_name_or_path,
        resolved_archive_file,
        cache_dir=cache_dir,
        force_download=force_download,
        proxies=proxies,
        local_files_only=local_files_only,
        token=token,
        user_agent=user_agent,
        revision=revision,
        subfolder=subfolder,
        _commit_hash=commit_hash,
    )
  else:
    checkpoint_files = (
        [resolved_archive_file]
        if pretrained_model_name_or_path is not None
        else None
    )

  return checkpoint_files, sharded_metadata

import re
from typing import Optional, Tuple

import jax.numpy as jnp
from flax.linen import Module

from transformers.utils.quantization_config import QuantizationMethod
# Assuming HfQuantizer is defined in a similar way in the JAX ecosystem.
# from .quantizers import HfQuantizer


def _infer_parameter_dtype(
    model: Module,
    param_name: str,
    empty_param: jnp.ndarray,
    keep_in_fp32_regex: Optional[re.Pattern] = None,
    hf_quantizer: Optional["HfQuantizer"] = None,
) -> Tuple[bool, Optional[jnp.dtype]]:
    """Infers the parameter's dtype and if it is contiguous.

    Args:
      model: The model instance.
      param_name: The name of the parameter.
      empty_param: An abstract or concrete array representing the parameter from a checkpoint,
        used to infer shape and dtype.
      keep_in_fp32_regex: A regex pattern to identify parameters that should be kept in float32.
      hf_quantizer: An optional quantizer object that may influence dtype decisions.

    Returns:
      A tuple containing:
        - A boolean indicating if the parameter is contiguous (always True in JAX).
        - The inferred casting dtype for the parameter, or None if no casting is needed.
    """
    try:
        # In JAX, parameters are typically stored in a separate PyTree.
        # We assume the model has a method to retrieve abstract parameter info.
        old_param = model.get_parameter_or_buffer(param_name)
    except Exception as e:
        if hf_quantizer is not None and hf_quantizer.quantization_config.quant_method in {
            QuantizationMethod.HQQ,
            QuantizationMethod.QUARK,
            QuantizationMethod.MXFP4,
        }:
            # JAX arrays are always contiguous.
            return True, None
        else:
            raise e

    is_jnp_e4m3fn_available = hasattr(jnp, "float8_e4m3fn")

    casting_dtype = None
    is_param_float8_e4m3fn = (
        is_jnp_e4m3fn_available and hasattr(jnp, "float8_e4m3fn") and empty_param.dtype == jnp.float8_e4m3fn
    )

    if jnp.issubdtype(empty_param.dtype, jnp.floating) and not is_param_float8_e4m3fn:
        # First fp32 if part of the exception list
        if keep_in_fp32_regex is not None and keep_in_fp32_regex.search(param_name):
            casting_dtype = jnp.float32
        # Then dtype that was instantiated in the meta model -- note that this respects subconfigs dtypes
        elif hf_quantizer is not None:
            casting_dtype = model.config._pre_quantization_dtype
        else:
            casting_dtype = old_param.dtype

    # JAX arrays are always contiguous.
    is_contiguous = old_param is not None
    return is_contiguous, casting_dtype

from flax.core import PyTree
import jax
import jax.numpy as jnp


def get_parameter_dtype(parameters: PyTree) -> jnp.dtype | None:
  """Returns the first found floating dtype in parameters if there is one.

  If no floating dtype is found, returns the last dtype it encountered.

  Args:
    parameters: A PyTree of model parameters or variables.

  Returns:
    The determined jax.numpy.dtype, or None if the PyTree is empty.
  """
  last_dtype = None
  # jax.tree_util.tree_leaves provides a flat list of all arrays in the PyTree.
  for t in jax.tree_util.tree_leaves(parameters):
    # We only care about arrays, not other potential leaves like strings or None.
    if hasattr(t, "dtype"):
      last_dtype = t.dtype
      if jnp.issubdtype(t.dtype, jnp.floating):
        return t.dtype

  # if no floating dtype was found return whatever the last dtype is
  return last_dtype

import re
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze
from flax.linen import Module
from jax.sharding import Mesh
from safetensors import safe_open

from transformers.modeling_flax_utils import FlaxPreTrainedModel
from transformers.utils.quantization_config import QuantizationMethod

# Re-used from src/MaxText/layers/distributed_utils.py
from src.MaxText.layers.distributed_utils import is_local_dist_rank_0
# Re-used from src/MaxText/layers/distributed_utils.py
from src.MaxText/layers/distributed_utils import shard_and_distribute_param as shard_and_distribute_module
# Re-used from src/MaxText/layers/distributed_utils.py
from src.MaxText/layers/distributed_utils import is_deepspeed_zero3_enabled
# Re-used from generated_code/Qwen3MoeForCausalLM/model_utils.py
from generated_code.Qwen3MoeForCausalLM.model_utils import _infer_parameter_dtype
# Re-used from generated_code/Qwen3MoeForCausalLM/dynamic_module_utils.py
from generated_code.Qwen3MoeForCausalLM.dynamic_module_utils import _load_parameter_into_model
# Re-used from generated_code/Qwen3MoeForCausalLM/dynamic_module_utils.py
from generated_code.Qwen3MoeForCausalLM.dynamic_module_utils import get_module_from_name
# This is a new implementation for a function that does not have a direct equivalent in the existing JAX modules.
from transformers.utils import strtobool
import os


# This is a new implementation for a function that does not have a direct equivalent in the existing JAX modules.
def is_fsdp_enabled():
  """Checks if FSDP is enabled by checking environment variables."""
  return (
      jax.distributed.is_initialized()
      and strtobool(os.environ.get("ACCELERATE_USE_FSDP", "False"))
      and strtobool(os.environ.get("FSDP_CPU_RAM_EFFICIENT_LOADING", "False"))
  )


# This is a new implementation for a function that does not have a direct equivalent in the existing JAX modules.
def offload_weight(
    param: jax.Array, weight_name: str, offload_folder: str, offload_index: Dict
) -> Dict:
  """
    Simulates accelerate's offload_weight by saving a JAX array to disk and updating an index.
    Note: This involves side-effects (writing to disk) and is not idiomatic JAX,
    but is included for functional equivalence with the PyTorch source.
    """
  if offload_index is None:
    offload_index = {}
  param_np = np.array(param)
  file_path = os.path.join(offload_folder, f"{weight_name}.dat")
  np.save(file_path, param_np)
  offload_index[weight_name] = {"shape": param.shape, "dtype": str(param.dtype)}
  return offload_index


def _load_state_dict_into_meta_model(
    model: FlaxPreTrainedModel,
    state_dict: Dict[str, Any],
    shard_file: str,
    expected_keys: List[str],
    reverse_renaming_mapping: Dict[str, str],
    device_map: Optional[Dict[str, Any]] = None,
    disk_offload_folder: Optional[str] = None,
    disk_offload_index: Optional[Dict] = None,
    cpu_offload_folder: Optional[str] = None,
    cpu_offload_index: Optional[Dict] = None,
    hf_quantizer: Optional[Any] = None,
    is_safetensors: bool = False,
    keep_in_fp32_regex: Optional[re.Pattern] = None,
    unexpected_keys: Optional[List[str]] = None,
    device_mesh: Optional[Mesh] = None,
) -> Tuple[Optional[Dict], Optional[Dict]]:
  """
    Load parameters from `meta_state_dict` into the model. The parameters of the `meta_state_dict` are on the meta
    device in order to easily infer the shapes and dtypes that they will have. Then proper parameters are then loaded
    from `shard_file`, which is the actual state dict file on disk.
    This function takes care of correctly casting dtypes, devices, and sharding tensors in case of tensor parallelism.
    """
  tensor_device = "cpu"
  if device_map is not None and device_map.get("", None) is not None:
    if device_map[""] not in ("cpu", jax.devices("cpu")[0]):
      tensor_device = device_map[""]

  if device_map is not None:
    device_map_regex = "|".join(
        [re.escape(k) for k in sorted(device_map.keys(), reverse=True)]
    )

  is_quantized = hf_quantizer is not None
  is_hqq_or_bnb = is_quantized and hf_quantizer.quantization_config.quant_method in {
      QuantizationMethod.HQQ,
      QuantizationMethod.BITS_AND_BYTES,
  }
  is_meta_state_dict = shard_file.endswith(".safetensors") and not is_hqq_or_bnb
  file_pointer = None
  if is_meta_state_dict:
    # In JAX, we load to host memory (numpy) first.
    file_pointer = safe_open(shard_file, framework="numpy", device=tensor_device)

  for param_name, empty_param in state_dict.items():
    if param_name not in expected_keys:
      continue

    if is_meta_state_dict:
      serialized_param_name = reverse_renaming_mapping[param_name]
      param = file_pointer.get_slice(serialized_param_name)[:]  # Load as numpy array
    else:
      param = np.array(empty_param)

    to_contiguous, casting_dtype = _infer_parameter_dtype(
        model,
        param_name,
        empty_param,
        keep_in_fp32_regex,
        hf_quantizer,
    )

    if device_mesh is not None:
      if (
          not is_quantized
          or (not hf_quantizer.requires_parameters_quantization)
          or (
              not hf_quantizer.check_quantized_param(
                  model,
                  param,
                  param_name,
                  state_dict,
                  device_map=device_map,
              )
          )
      ):
        shard_and_distribute_module(
            model,
            param,
            empty_param,
            param_name,
            casting_dtype,
            to_contiguous,
            jax.process_index(),
            device_mesh,
        )
      else:
        sharding_kwargs = {
            "empty_param": empty_param,
            "casting_dtype": casting_dtype,
            "to_contiguous": to_contiguous,
            "rank": jax.process_index(),
            "device_mesh": device_mesh,
        }
        hf_quantizer.create_quantized_param(
            model,
            param,
            param_name,
            jax.process_index(),
            state_dict,
            unexpected_keys,
            **sharding_kwargs,
        )
    else:
      param = jnp.array(param)
      if casting_dtype is not None:
        param = param.astype(casting_dtype)

      if device_map is None:
        param_device = "cpu"
      else:
        module_layer = re.search(device_map_regex, param_name)
        if not module_layer:
          raise ValueError(f"{param_name} doesn't have any device set.")
        else:
          param_device = device_map[module_layer.group()]

      if param_device == "disk":
        if not is_safetensors:
          disk_offload_index = offload_weight(
              param, param_name, disk_offload_folder, disk_offload_index
          )
      elif param_device == "cpu" and cpu_offload_index is not None:
        cpu_offload_index = offload_weight(
            param, param_name, cpu_offload_folder, cpu_offload_index
        )
      elif (
          not is_quantized
          or (not hf_quantizer.requires_parameters_quantization)
          or (
              not hf_quantizer.check_quantized_param(
                  model,
                  param,
                  param_name,
                  state_dict,
                  param_device=param_device,
                  device_map=device_map,
              )
          )
      ):
        if is_fsdp_enabled():
          # In JAX, 'meta' device doesn't exist. We keep params on CPU for non-rank 0.
          # The actual placement happens during sharding.
          if not is_local_dist_rank_0():
            param_device = "cpu"

        target_device = (
            jax.devices(param_device)[0]
            if isinstance(param_device, str)
            else param_device
        )
        param = jax.device_put(param, target_device)
        model.params = _load_parameter_into_model(model.params, param_name, param)

      else:
        # This part modifies the model in-place which is not very JAX-like.
        # For functional equivalence, we replicate this behavior.
        model.params = hf_quantizer.create_quantized_param(
            model.params, param, param_name, param_device, state_dict, unexpected_keys
        )

        if is_fsdp_enabled() or is_deepspeed_zero3_enabled():
          param_name = hf_quantizer.update_param_name(param_name)
          # In JAX, we don't modify modules in-place like this.
          # This logic is highly specific to PyTorch's distributed paradigms.
          # We will update the parameter in the model's state PyTree.
          module_path, param_type = get_module_from_name(model, param_name)
          current_params = model.params
          for part in module_path.split("."):
            if part:
              current_params = current_params[part]
          value = current_params[param_type]

          param_to = "cpu"
          if is_fsdp_enabled() and not is_local_dist_rank_0():
            # 'meta' device equivalent in JAX for this purpose is keeping it on CPU
            # until sharded.
            param_to = "cpu"

          target_device = jax.devices(param_to)[0]
          new_value = jax.device_put(value, target_device)

          # Update the model's params PyTree
          keys = module_path.split(".") + [param_type] if module_path else [param_type]
          current_level = model.params
          for i, key in enumerate(keys[:-1]):
            if i == 0 and not key:
              continue
            if not isinstance(current_level, dict) and hasattr(
                current_level, "unfreeze"
            ):
              current_level = current_level.unfreeze()
            current_level = current_level[key]

          if not isinstance(current_level, dict) and hasattr(current_level, "unfreeze"):
            current_level = current_level.unfreeze()
          current_level[keys[-1]] = new_value
          model.params = freeze(model.params)

  if file_pointer is not None:
    file_pointer.close()

  return disk_offload_index, cpu_offload_index

import os
from typing import Dict, Optional, Union

import jax
import jax.numpy as jnp
from safetensors import safe_open

# Reused from generated_code.Qwen3MoeForCausalLM.package_utils.is_safetensors_available
from transformers.utils import is_safetensors_available
# Reused from generated_code.Qwen3MoeForCausalLM.tensor_utils.str_to_jnp_dtype
from transformers.utils.tensor_utils import str_to_jnp_dtype


def load_state_dict(
    checkpoint_file: Union[str, os.PathLike],
    is_quantized: bool = False,
    map_location: Optional[str] = "cpu",
) -> Dict[str, Union[jnp.ndarray, jax.ShapeDtypeStruct]]:
  """
  Reads a `.safetensors` checkpoint file. We can load the checkpoint on "cpu"
  (as numpy arrays) or "meta" (as ShapeDtypeStructs).
  """
  # Use safetensors if possible
  if str(checkpoint_file).endswith(".safetensors") and is_safetensors_available():
    with safe_open(checkpoint_file, framework="numpy") as f:
      metadata = f.metadata()

      if metadata is not None and metadata.get("format") not in ["pt", "tf", "flax", "mlx"]:
        raise OSError(
            f"The safetensors archive passed at {checkpoint_file} does not contain the valid metadata. Make sure "
            "you save your model with the `save_pretrained` method."
        )
      state_dict = {}
      for k in f.keys():
        if map_location == "meta":
          _slice = f.get_slice(k)
          k_dtype_str = _slice.get_dtype()
          if k_dtype_str in str_to_jnp_dtype:
            dtype = str_to_jnp_dtype[k_dtype_str]
          else:
            raise ValueError(f"Cannot load safetensors of unknown dtype {k_dtype_str}")
          state_dict[k] = jax.ShapeDtypeStruct(shape=_slice.get_shape(), dtype=dtype)
        else:
          state_dict[k] = jnp.asarray(f.get_tensor(k))
      return state_dict

  # Fallback for non-safetensors files
  try:
    with open(checkpoint_file, "r", encoding="utf-8") as f:
      if f.read(7) == "version":
        raise OSError(
            "You seem to have cloned a repository without having git-lfs installed. Please install "
            "git-lfs and run `git lfs install` followed by `git lfs pull` in the folder "
            "you cloned."
        )
      else:
        raise OSError(
            f"Unable to load weights from checkpoint file for '{checkpoint_file}'. "
            f"This JAX function only supports loading from '.safetensors' files. "
            "Loading PyTorch '.bin' files is not supported."
        )
  except (UnicodeDecodeError, IsADirectoryError, FileNotFoundError) as e:
    raise OSError(
        f"Unable to load weights from checkpoint file for '{checkpoint_file}' "
        f"at '{checkpoint_file}'. "
        "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True."
    ) from e
  except Exception as e:  # Catch other potential open() errors
    raise OSError(f"Unable to read file '{checkpoint_file}'. It may be corrupted or not a valid checkpoint.") from e

from flax.linen import initializers

JAX_INIT_FUNCTIONS = {
    "uniform_": initializers.uniform,
    "normal_": initializers.normal,
    "trunc_normal_": initializers.truncated_normal,
    "constant_": initializers.constant,
    "xavier_uniform_": initializers.xavier_uniform,
    "xavier_normal_": initializers.xavier_normal,
    "kaiming_uniform_": initializers.kaiming_uniform,
    "kaiming_normal_": initializers.kaiming_normal,
    "uniform": initializers.uniform,
    "normal": initializers.normal,
    "xavier_uniform": initializers.xavier_uniform,
    "xavier_normal": initializers.xavier_normal,
    "kaiming_uniform": initializers.kaiming_uniform,
    "kaiming_normal": initializers.kaiming_normal,
}

import contextlib
from typing import Any, Callable, Generator

import jax.numpy as jnp

# From generated_code.Qwen3MoeForCausalLM.constants._init_weights
# From generated_code.Qwen3MoeForCausalLM.model_utils.JAX_INIT_FUNCTIONS
from .. import constants, model_utils


@contextlib.contextmanager
def no_init_weights() -> Generator[None, None, None]:
  """Context manager to globally disable weight initialization to speed up loading large models.

  This context manager has two effects:
  1. It sets a global flag (`_init_weights`) to `False`, which is checked by the
     `init_weights` method of the base model class to skip custom initialization.
  2. It temporarily replaces the standard Flax initializers in the project's
     `JAX_INIT_FUNCTIONS` mapping with a function that returns zeros. This prevents
     costly random initialization in layers that might be initialized directly.
  """
  old_init_weights = constants._init_weights
  constants._init_weights = False

  def _skip_init_factory(*args: Any, **kwargs: Any) -> Callable[..., jnp.ndarray]:
    """A factory that returns a function that initializes with zeros."""

    def _skip_init(key, shape, dtype=jnp.float32):
      return jnp.zeros(shape, dtype=dtype)

    return _skip_init

  # Save and replace the original initialization functions
  original_inits = model_utils.JAX_INIT_FUNCTIONS.copy()
  for name in original_inits:
    model_utils.JAX_INIT_FUNCTIONS[name] = _skip_init_factory

  try:
    yield
  finally:
    # Restore the original state
    constants._init_weights = old_init_weights
    model_utils.JAX_INIT_FUNCTIONS.clear()
    model_utils.JAX_INIT_FUNCTIONS.update(original_inits)

from collections import defaultdict
from typing import Any, List, Mapping

from flax.core import unfreeze
from flax.traverse_util import flatten_dict


def find_tied_parameters(params: Mapping[str, Any], **kwargs) -> List[List[str]]:
  """
  Find the tied parameters in a given model's parameter PyTree.

  <Tip warning={true}>

  The signature accepts keyword arguments, but they are for the recursive part of this function and you should ignore
  them.

  </Tip>

  Args:
      params (`Mapping[str, Any]`): The PyTree of model parameters to inspect.

  Returns:
      list[list[str]]: A list of lists of parameter names being all tied together.

  Example:

  
from collections import defaultdict
from typing import Any, List

from flax.core import unfreeze
from flax.traverse_util import flatten_dict
import jax.numpy as jnp


# PyTree definition for type hinting
PyTree = Any


def find_tied_parameters(params: PyTree, **kwargs) -> List[List[str]]:
  """
  Finds the tied parameters in a given Flax parameter PyTree.

  <Tip warning={true}>

  The signature accepts keyword arguments, but they are for the recursive part of this function and you should ignore
  them.

  </Tip>

  Args:
      params (`PyTree`): The parameter PyTree to inspect.

  Returns:
      list[list[str]]: A list of lists of parameter names being all tied together.

  Example:

  
from collections import defaultdict
from typing import Any, List

import flax.traverse_util
from flax.core import freeze
import jax.numpy as jnp

# Define PyTree for typing
PyTree = Any


def find_tied_parameters(params: PyTree) -> List[List[str]]:
  """
  Finds the tied parameters in a given JAX/Flax parameter PyTree.

  Tied parameters are parameters that share the same underlying array object.

  Args:
    params (`PyTree`): The parameter PyTree to inspect.

  Returns:
    `list[list[str]]`: A list of lists of parameter names being all tied together.

  Example:

  
# Copyright 2025 The HuggingFace Team. All rights reserved.
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
The `find_tied_parameters` was copied from `accelerate.utils.modeling.py`
"""
from collections import defaultdict
from typing import Any, List

from flax.traverse_util import flatten_dict


def find_tied_parameters(model: Any, **kwargs) -> List[List[str]]:
  """
  Find the tied parameters in a given model's parameter PyTree.

  <Tip warning={true}>

  The signature accepts keyword arguments, but they are for the recursive part of this function and you should ignore
  them.

  </Tip>

  Args:
      model (`Any`): The model's parameter PyTree to inspect.

  Returns:
      list[list[str]]: A list of lists of parameter names being all tied together.

  Example:

  
import os
import warnings
from typing import Any, Optional, Tuple, Union

import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
from flax.core import unfreeze
from jax import Array

from transformers.utils import logging


logger = logging.get_logger(__name__)


class ModuleUtilsMixin:
  """A few utilities for Flax Modules, to be used as a mixin.

  This is a JAX/Flax conversion of the PyTorch ModuleUtilsMixin.
  Memory-hooking methods are not supported in JAX's functional paradigm and have been removed.
  The `device` and `dtype` properties have been omitted; JAX handles device placement automatically,
  and the module is expected to have a `dtype` attribute.
  Methods requiring access to parameters have been updated to accept a `params`
  PyTree as an argument.
  """

  def invert_attention_mask(self, encoder_attention_mask: Array) -> Array:
    """Inverts an attention mask (e.g., switches 0. and 1.).

    Args:
      encoder_attention_mask (`jax.Array`): An attention mask.

    Returns:
      `jax.Array`: The inverted attention mask.
    """
    if encoder_attention_mask.ndim == 3:
      encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
    elif encoder_attention_mask.ndim == 2:
      encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
    else:
      raise ValueError(
          "Wrong shape for encoder_attention_mask (shape"
          f" {encoder_attention_mask.shape})"
      )

    encoder_extended_attention_mask = encoder_extended_attention_mask.astype(
        self.dtype
    )  # fp16 compatibility
    encoder_extended_attention_mask = (
        1.0 - encoder_extended_attention_mask
    ) * jnp.finfo(self.dtype).min

    return encoder_extended_attention_mask

  @staticmethod
  def create_extended_attention_mask_for_decoder(
      input_shape: Tuple[int, int], attention_mask: Array
  ) -> Array:
    batch_size, seq_length = input_shape
    seq_ids = jnp.arange(seq_length)
    causal_mask = seq_ids[None, None, :] <= seq_ids[None, :, None]
    # broadcast to create a (1, seq_length, seq_length) mask
    causal_mask = jnp.broadcast_to(
        causal_mask, (batch_size, seq_length, seq_length)
    )
    causal_mask = causal_mask.astype(attention_mask.dtype)

    if causal_mask.shape[1] < attention_mask.shape[1]:
      prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
      causal_mask = jnp.concatenate(
          [
              jnp.ones(
                  (batch_size, seq_length, prefix_seq_len),
                  dtype=causal_mask.dtype,
              ),
              causal_mask,
          ],
          axis=-1,
      )

    extended_attention_mask = (
        causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
    )
    return extended_attention_mask

  def get_extended_attention_mask(
      self,
      attention_mask: Array,
      input_shape: Tuple[int, ...],
      dtype: jnp.dtype = None,
  ) -> Array:
    """Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
      attention_mask (`jax.Array`):
        Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
      input_shape (`Tuple[int]`):
        The shape of the input to the model.

    Returns:
      `jax.Array` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
    """
    if dtype is None:
      dtype = self.dtype

    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.ndim == 3:
      extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.ndim == 2:
      # Provided a padding mask of dimensions [batch_size, seq_length]
      # - if the model is a decoder, apply a causal mask in addition to the padding mask
      # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
      if self.config.is_decoder:
        extended_attention_mask = (
            ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
                input_shape, attention_mask
            )
        )
      else:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
      raise ValueError(
          "Wrong shape for input_ids (shape"
          f" {input_shape}) or attention_mask (shape {attention_mask.shape})"
      )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and the dtype's smallest value for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.astype(
        dtype
    )  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * jnp.finfo(
        dtype
    ).min
    return extended_attention_mask

  def get_head_mask(
      self,
      head_mask: Optional[Array],
      num_hidden_layers: int,
      is_attention_chunked: bool = False,
  ) -> Array:
    """Prepares the head mask if needed.

    Args:
      head_mask (`jax.Array` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
        The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
      num_hidden_layers (`int`):
        The number of hidden layers in the model.
      is_attention_chunked (`bool`, *optional*, defaults to `False`):
        Whether or not the attentions scores are computed by chunks or not.

    Returns:
      `jax.Array` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
      `[None]` for each layer.
    """
    if head_mask is not None:
      head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
      if is_attention_chunked is True:
        head_mask = jnp.expand_dims(head_mask, -1)
    else:
      head_mask = [None] * num_hidden_layers

    return head_mask

  def _convert_head_mask_to_5d(
      self, head_mask: Array, num_hidden_layers: int
  ) -> Array:
    """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
    if head_mask.ndim == 1:
      head_mask = jnp.expand_dims(head_mask, (0, 1, 3, 4))
      head_mask = jnp.broadcast_to(
          head_mask, (num_hidden_layers, -1, -1, -1, -1)
      )
    elif head_mask.ndim == 2:
      head_mask = jnp.expand_dims(
          head_mask, (1, 3, 4)
      )  # We can specify head_mask for each layer
    assert (
        head_mask.ndim == 5
    ), f"head_mask.ndim != 5, instead {head_mask.ndim}"
    head_mask = head_mask.astype(
        self.dtype
    )  # switch to float if need + fp16 compatibility
    return head_mask

  def num_parameters(
      self,
      params: Any,
      only_trainable: bool = False,
      exclude_embeddings: bool = False,
  ) -> int:
    """Get number of (optionally, trainable or non-embeddings) parameters in the module.

    Args:
      params (`Any`):
        A PyTree of model parameters.
      only_trainable (`bool`, *optional*, defaults to `False`):
        Whether or not to return only the number of trainable parameters. In JAX, this is typically
        handled by parameter partitioning and is not straightforward to determine from parameters alone.
        This argument is currently ignored.
      exclude_embeddings (`bool`, *optional*, defaults to `False`):
        Whether or not to return only the number of non-embeddings parameters.

    Returns:
      `int`: The number of parameters.
    """
    if only_trainable:
      warnings.warn(
          "The `only_trainable` flag is not supported in this JAX implementation and will be ignored."
      )

    flat_params = flatten_dict(unfreeze(params))
    total_numel = 0
    for path, param in flat_params.items():
      if exclude_embeddings:
        path_str = "/".join(path)
        # Heuristic to identify embedding layers by name
        if "embedding" in path_str.lower():
          continue
      total_numel += param.size
    return total_numel

  def estimate_tokens(self, input_dict: dict[str, Union[Array, Any]]) -> int:
    """Helper function to estimate the total number of tokens from the model inputs.

    Args:
      inputs (`dict`): The model inputs.

    Returns:
      `int`: The total number of tokens.
    """
    if not hasattr(self, "warnings_issued"):
      self.warnings_issued = {}
    if self.main_input_name in input_dict:
      return input_dict[self.main_input_name].size
    elif "estimate_tokens" not in self.warnings_issued:
      logger.warning(
          "Could not estimate the number of tokens of the input, floating-point operations will not be computed"
      )
      self.warnings_issued["estimate_tokens"] = True
    return 0

  def floating_point_ops(
      self,
      params: Any,
      input_dict: dict[str, Union[Array, Any]],
      exclude_embeddings: bool = True,
  ) -> int:
    """Get number of (optionally, non-embeddings) floating-point operations for the forward and backward passes of a
    batch with this transformer model. Default approximation neglects the quadratic dependency on the number of
    tokens (valid if `12 * d_model << sequence_length`) as laid out in [this
    paper](https://huggingface.co/papers/2001.08361) section 2.1. Should be overridden for transformers with parameter
    re-use e.g. Albert or Universal Transformers, or if doing long-range modeling with very high sequence lengths.

    Args:
      params (`Any`):
        A PyTree of model parameters.
      input_dict (`dict`):
        The model inputs.
      exclude_embeddings (`bool`, *optional*, defaults to `True`):
        Whether or not to count embedding and softmax operations.

    Returns:
      `int`: The number of floating-point operations.
    """
    return 6 * self.estimate_tokens(input_dict) * self.num_parameters(
        params, exclude_embeddings=exclude_embeddings
    )

from typing import Any, Dict, List, Optional, Tuple
import re

from flax.linen import Module as FlaxModule
import jax
from jax.sharding import Mesh

# From generated_code.Qwen3MoeForCausalLM.model_utils.load_state_dict
from .model_utils import load_state_dict
# From generated_code.Qwen3MoeForCausalLM.model_utils.Syntax error in code: invalid syntax (<unknown>, line 19)
from .model_utils import _load_state_dict_into_meta_model
# From generated_code.Qwen3MoeForCausalLM.distributed_utils.is_deepspeed_zero3_enabled
from .distributed_utils import is_deepspeed_zero3_enabled
# From generated_code.Qwen3MoeForCausalLM.distributed_utils.is_fsdp_managed_module
from .distributed_utils import is_fsdp_managed_module
# From generated_code.Qwen3MoeForCausalLM.distributed_utils.is_local_dist_rank_0
from .distributed_utils import is_local_dist_rank_0
# From generated_code.Qwen3MoeForCausalLM.quantizers.quantization_config import QuantizationMethod
from .quantizers.quantization_config import QuantizationMethod
# From generated_code.Qwen3MoeForCausalLM.quantizers.hf_quantizer import HfQuantizer
from .quantizers.hf_quantizer import HfQuantizer


def load_shard_file(args: Tuple) -> Tuple[List[str], Optional[Dict], Optional[Dict]]:
  """Loads a single shard file into the model."""
  (
      shard_file,
      state_dict,
      disk_only_shard_files,
      is_hqq_or_bnb,
      is_quantized,
      device_map,
      hf_quantizer,
      key_renaming_mapping,
      weights_only,
      model_to_load,
      expected_keys,
      reverse_key_renaming_mapping,
      disk_offload_folder,
      disk_offload_index,
      cpu_offload_folder,
      cpu_offload_index,
      is_offloaded_safetensors,
      keep_in_fp32_regex,
      unexpected_keys,
      device_mesh,
  ) = args

  # Skip the load for shards that only contain disk-offloaded weights
  if shard_file in disk_only_shard_files:
    return [], disk_offload_index, cpu_offload_index

  map_location = "cpu"
  # From generated_code.Qwen3MoeForCausalLM.distributed_utils.is_deepspeed_zero3_enabled
  if shard_file.endswith(".safetensors") and not is_hqq_or_bnb and not (is_deepspeed_zero3_enabled() and not is_quantized):
    map_location = "meta"

  # If shard_file is "", we use the existing state_dict instead of loading it
  if shard_file != "":
    # From generated_code.Qwen3MoeForCausalLM.model_utils.load_state_dict
    state_dict = load_state_dict(
        shard_file, is_quantized=is_quantized, map_location=map_location, weights_only=weights_only
    )

  # Fix the key names
  state_dict = {key_renaming_mapping[k]: v for k, v in state_dict.items() if k in key_renaming_mapping}

  error_msgs = []

  # DeepSpeed is a PyTorch-specific library, so the is_deepspeed_zero3_enabled branch is removed.
  # The main logic is handled by _load_state_dict_into_meta_model.

  # Skip it with fsdp on ranks other than 0
  # From generated_code.Qwen3MoeForCausalLM.distributed_utils.is_fsdp_managed_module
  # From generated_code.Qwen3MoeForCausalLM.distributed_utils.is_local_dist_rank_0
  if not (is_fsdp_managed_module(model_to_load.config) and not is_local_dist_rank_0() and not is_quantized):
    # From generated_code.Qwen3MoeForCausalLM.model_utils.Syntax error in code: invalid syntax (<unknown>, line 19)
    disk_offload_index, cpu_offload_index = _load_state_dict_into_meta_model(
        model_to_load,
        state_dict,
        shard_file,
        expected_keys,
        reverse_key_renaming_mapping,
        device_map=device_map,
        disk_offload_folder=disk_offload_folder,
        disk_offload_index=disk_offload_index,
        cpu_offload_folder=cpu_offload_folder,
        cpu_offload_index=cpu_offload_index,
        hf_quantizer=hf_quantizer,
        is_safetensors=is_offloaded_safetensors,
        keep_in_fp32_regex=keep_in_fp32_regex,
        unexpected_keys=unexpected_keys,
        device_mesh=device_mesh,
    )

  return error_msgs, disk_offload_index, cpu_offload_index

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

from absl import logging
from tqdm.auto import tqdm

# The JAX equivalent of `load_shard_file` will be in `model_utils`
from . import model_utils


def load_shard_files_with_threadpool(
    args_list: List[Tuple[Any, ...]]
) -> Tuple[List[str], Dict[str, Any], Dict[str, Any]]:
  """Loads model weights in parallel using a thread pool.

  Args:
    args_list: A list of argument tuples, where each tuple contains the
      arguments for a single call to `load_shard_file`.

  Returns:
    A tuple containing:
      - A list of error messages encountered during loading.
      - The disk offload index from the last completed shard.
      - The CPU offload index from the last completed shard.
  """
  num_workers = int(os.environ.get("HF_PARALLEL_LOADING_WORKERS", "8"))

  # Do not spawn more workers than you need
  num_workers = min(len(args_list), num_workers)

  logging.info(f"Loading model weights in parallel with {num_workers} workers...")

  error_msgs = []
  disk_offload_index = None
  cpu_offload_index = None

  with ThreadPoolExecutor(max_workers=num_workers) as executor:
    with tqdm(
        total=len(args_list), desc="Loading checkpoint shards"
    ) as pbar:
      # The JAX equivalent of `load_shard_file` is assumed to be in `model_utils`
      futures = [
          executor.submit(model_utils.load_shard_file, arg)
          for arg in args_list
      ]
      for future in as_completed(futures):
        result = future.result()
        (
            _error_msgs,
            disk_offload_index,
            cpu_offload_index,
        ) = result

        error_msgs.extend(_error_msgs)

        pbar.update(1)

  return error_msgs, disk_offload_index, cpu_offload_index

import re
from typing import Optional


def fuzzy_match_size(config_name: str) -> Optional[str]:
  """
  Extract the size digit from strings like "4weight", "8weight".
  Returns the digit as an integer if found, otherwise None.
  """
  config_name = config_name.lower()

  str_match = re.search(r"(\d)weight", config_name)

  if str_match:
    return str_match.group(1)

  return None

from typing import Any

# Re-used from generated_code.Qwen3MoeForCausalLM.quantization._quantization_type
from generated_code.Qwen3MoeForCausalLM.quantization import _quantization_type


def _linear_extra_repr(self: Any) -> str:
  """Generates a string representation for a linear layer with quantization info."""
  weight = _quantization_type(self.weight)
  if weight is None:
    return f"in_features={self.weight.shape[1]}, out_features={self.weight.shape[0]}, weight=None"
  else:
    return f"in_features={self.weight.shape[1]}, out_features={self.weight.shape[0]}, weight={weight}"
from dataclasses import dataclass
from typing import Optional, Type

from flax.linen import Module as nn_Module


@dataclass
class OutputRecorder:
  """Configuration for recording outputs from a model via hooks.

  Attributes:
    target_class: The class (e.g., flax.linen.Module) to which the hook will be
      attached.
    index: If the output is a tuple/list, optionally record only at a specific
      index.
    layer_name: Name of the submodule to target (if needed), e.g.,
      "transformer.layer.3.attn".
    class_name: Name of the class to which the hook will be attached. Could be
      the suffix of class name in some cases.
  """

  target_class: Type[nn_Module]
  index: Optional[int] = 0
  layer_name: Optional[str] = None
  class_name: Optional[str] = None
from collections import defaultdict
from functools import wraps
from typing import Any, Callable

import flax.linen as nn
from flax.linen import module as flax_module

# Assuming _CAN_RECORD_REGISTRY is defined globally as in the original file.
# This registry would map model class names to dictionaries that configure
# which intermediates can be captured.
_CAN_RECORD_REGISTRY = {}

# Assuming a logger is available, similar to the original implementation.
# from ..utils import logging
# logger = logging.get_logger(__name__)
# As a placeholder:
import logging

logger = logging.getLogger(__name__)


def check_model_inputs(func: Callable) -> Callable:
  """
  Decorator to intercept specific layer outputs using Flax's sow/reap mechanism.

  This is a JAX-native re-implementation of the intent behind the original
  PyTorch decorator. It replaces the monkey-patching mechanism with the idiomatic
  `capture_intermediates` context, which relies on submodules using `self.sow()`
  to record their outputs.
  """

  @wraps(func)
  def wrapper(self: nn.Module, *args: Any, **kwargs: Any) -> Any:
    use_cache = kwargs.get("use_cache")
    if use_cache is None:
      use_cache = getattr(self.config, "use_cache", False)

    return_dict = kwargs.pop("return_dict", None)
    if return_dict is None:
      return_dict = getattr(self.config, "return_dict", True)

    # In JAX/Flax, training mode is controlled by the `deterministic` flag.
    is_training = not kwargs.get("deterministic", True)
    if getattr(self, "gradient_checkpointing", False) and is_training and use_cache:
      # Use a simple print for the warning if a logger is not configured.
      print("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
      use_cache = False

    kwargs["use_cache"] = use_cache

    all_args = kwargs.copy()
    if "kwargs" in all_args:
      for k, v in all_args["kwargs"].items():
        all_args[k] = v

    capture_flags = _CAN_RECORD_REGISTRY.get(str(self.__class__), {})
    recordable_keys = {
        f"output_{k}": all_args.get(
            f"output_{k}",
            getattr(
                self.config,
                f"output_{k}",
                all_args.get("output_attentions", getattr(self.config, "output_attentions", False)),
            ),
        )
        for k in capture_flags
    }

    collections_to_capture = {k.replace("output_", "") for k, v in recordable_keys.items() if v}

    collected_outputs = defaultdict(tuple)

    if collections_to_capture:
      # Use the internal context manager to capture intermediates sown by submodules.
      # This assumes that the underlying layers are instrumented with `self.sow()` calls.
      with flax_module.capture_intermediates(
          lambda collection, key: collection in collections_to_capture
      ) as captured_intermediates:
        outputs = func(self, *args, **kwargs)

      collected_vars = captured_intermediates.get()

      # Process collected_vars into the expected format (a tuple of tensors per key).
      for collection_name, keys_and_values in collected_vars.items():
        # Sort by key to maintain layer order, assuming keys are simple numeric strings like ('0',), ('1',).
        try:
          # Attempt to sort numerically if keys are string representations of integers
          sorted_items = sorted(keys_and_values.items(), key=lambda item: int(item[0][0]))
        except (ValueError, IndexError):
          # Fallback to lexicographical sort if keys are not simple integers
          sorted_items = sorted(keys_and_values.items())

        for _, value_tuple in sorted_items:
          collected_outputs[collection_name] += value_tuple
    else:
      outputs = func(self, *args, **kwargs)

    # Inject collected outputs into model output
    # This assumes `outputs` is a mutable, dict-like object (e.g., ModelOutput).
    for key in collected_outputs:
      if key == "hidden_states":
        final_hidden_states = collected_outputs[key]
        if len(final_hidden_states) > 0:
          final_hidden_states = final_hidden_states[:-1]

        if hasattr(outputs, "vision_hidden_states"):
          final_hidden_states += (outputs.vision_hidden_states,)
        elif hasattr(outputs, "last_hidden_state"):
          final_hidden_states += (outputs.last_hidden_state,)

        outputs[key] = final_hidden_states
      elif key == "attentions":
        if isinstance(capture_flags.get(key), list) and len(capture_flags[key]) == 2:
          outputs[key] = collected_outputs[key][0::2]
          outputs["cross_" + key] = collected_outputs[key][1::2]
        else:
          outputs[key] = collected_outputs[key]
      else:
        outputs[key] = collected_outputs[key]

    if return_dict is False and hasattr(outputs, "to_tuple"):
      outputs = outputs.to_tuple()

    return outputs

  return wrapper
