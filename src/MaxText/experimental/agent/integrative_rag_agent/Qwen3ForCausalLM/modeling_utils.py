
from packaging import version
import jax

# This flag corresponds to a PyTorch version check (>=2.6) that enables more modern features.
# In JAX, these features (e.g., advanced vmap) are generally mature. We keep the flag for
# structural consistency during translation, but the specific version checked against is
# chosen as a baseline modern JAX version and may not gate the same features.
_is_jax_greater_or_equal_than_0_4_20 = version.parse(jax.__version__) >= version.parse("0.4.20")
def causal_mask_function(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
  """This creates a basic lower-diagonal causal mask."""
  return kv_idx <= q_idx
from functools import wraps
from typing import Callable

import jax.numpy as jnp
from flax.linen import Module


def dynamic_rope_update(rope_forward: Callable) -> Callable:
  """Decorator function to update the RoPE parameters in the forward pass.

  This is for models using a dynamic RoPE (i.e. a RoPE implementation that may
  recompute its frequencies in the forward pass).

  Args:
    rope_forward: The forward pass of the RoPE implementation.

  Returns:
    The decorated forward pass.
  """

  def longrope_frequency_update(self: Module, position_ids: jnp.ndarray):
    """Longrope uses long factor if sequence is larger than original pretraining length, short otherwise."""
    seq_len = jnp.max(position_ids) + 1
    if hasattr(self.config, "original_max_position_embeddings"):
      original_max_position_embeddings = self.config.original_max_position_embeddings
    else:
      original_max_position_embeddings = self.config.max_position_embeddings

    # In JAX, lazy initialization like in the original PyTorch code is not idiomatic.
    # We assume `long_inv_freq` is pre-computed and available as a variable in the module's setup.
    # The logic simplifies to selecting between the long and original frequencies.
    use_long_freq = seq_len > original_max_position_embeddings

    # This assumes `inv_freq`, `long_inv_freq`, and `original_inv_freq` are mutable variables
    # in a collection like 'buffers'. The update happens via `.value`.
    self.inv_freq.value = jnp.where(
        use_long_freq,
        self.long_inv_freq.value,
        self.original_inv_freq.value,
    )

  def dynamic_frequency_update(self: Module, position_ids: jnp.ndarray):
    """dynamic RoPE layers should recompute `inv_freq` in the following situations:

    1 - growing beyond the cached sequence length (allow scaling)
    2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
    """
    seq_len = jnp.max(position_ids) + 1

    # Growth phase
    needs_growth = seq_len > self.max_seq_len_cached.value
    # NOTE: rope_init_fn must be a pure function for this to be JIT-compatible.
    grown_inv_freq, grown_attention_scaling = self.rope_init_fn(self.config, seq_len=seq_len)

    inv_freq_after_growth = jnp.where(needs_growth, grown_inv_freq, self.inv_freq.value)
    attn_scale_after_growth = jnp.where(needs_growth, grown_attention_scaling, self.attention_scaling.value)
    max_len_after_growth = jnp.where(needs_growth, seq_len, self.max_seq_len_cached.value)

    # Reset phase
    needs_reset = (seq_len < self.original_max_seq_len) & (max_len_after_growth > self.original_max_seq_len)

    final_inv_freq = jnp.where(needs_reset, self.original_inv_freq.value, inv_freq_after_growth)
    final_max_len = jnp.where(needs_reset, self.original_max_seq_len, max_len_after_growth)

    # Apply the final state updates. This requires the decorated method to be called
    # in a mutable context (e.g., mutable=['buffers']).
    self.inv_freq.value = final_inv_freq
    self.attention_scaling.value = attn_scale_after_growth
    self.max_seq_len_cached.value = final_max_len

  @wraps(rope_forward)
  def wrapper(self: Module, x: jnp.ndarray, position_ids: jnp.ndarray):
    # The state update functions must be called within a mutable context.
    # In Flax, attempting to mutate state outside a mutable context will raise an error.
    # The caller of the decorated function is responsible for enabling mutability.
    if "dynamic" in self.rope_type:
      dynamic_frequency_update(self, position_ids)
    elif self.rope_type == "longrope":
      longrope_frequency_update(self, position_ids)
    return rope_forward(self, x, position_ids)

  return wrapper

import flax.linen as nn
from typing import Optional


class EmbeddingAccessMixin:
  """Base utilities to regroup getters and setters for embeddings.

  Introduces the `input_layer_embed` attribute, which indicates
  where the input embeddings come from and where they
  should be set.
  """

  _input_embed_layer = "embed_tokens"  # default layer that holds input embeddings.

  def get_input_embeddings(self) -> nn.Module:
    """Returns the model's input embeddings.

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
    """Sets the model's output embedding, defaulting to setting new_embeddings to lm_head."""
    if getattr(self, "lm_head"):
      self.lm_head = new_embeddings

from flax.linen import Module
from typing import List


def _get_tied_weight_keys(module: Module, prefix: str = "") -> List[str]:
  """Recursively retrieves the keys of tied weights from a Flax module and its submodules."""
  tied_weight_keys = []
  if getattr(module, "_tied_weights_keys", None) is not None:
    names = [f"{prefix}.{k}" if prefix else k for k in module._tied_weights_keys]
    tied_weight_keys.extend(names)
  if getattr(module, "_dynamic_tied_weights_keys", None) is not None:
    names = [
        f"{prefix}.{k}" if prefix else k
        for k in module._dynamic_tied_weights_keys
    ]
    tied_weight_keys.extend(names)

  # In Flax, submodules are attributes of the parent module.
  # We iterate through the attributes and check if they are instances of `Module`.
  for name, submodule in vars(module).items():
    if isinstance(submodule, Module):
      local_prefix = f"{prefix}.{name}" if prefix else name
      tied_weight_keys.extend(
          _get_tied_weight_keys(submodule, prefix=local_prefix)
      )
  return tied_weight_keys

from contextlib import contextmanager

# This global variable is used to control whether weights are initialized.
# In JAX/Flax, initialization is more explicit, but we maintain this flag
# for compatibility with the HF Transformers loading logic, where `init_weights`
# is called and can be skipped based on this flag.
_init_weights = True


@contextmanager
def no_init_weights():
  """
  Context manager to globally disable weight initialization.

  This is used to speed up loading large models when the weights will be
  immediately overwritten by a checkpoint. In the JAX/Flax paradigm, this
  is controlled by skipping the `init_weights` call based on the `_init_weights`
  global flag, rather than monkey-patching individual initializer functions
  as is done in PyTorch.
  """
  global _init_weights
  old_init_weights = _init_weights
  _init_weights = False
  try:
    yield
  finally:
    _init_weights = old_init_weights

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

"""A few utilities for `flax.linen.Modules`, to be used as a mixin."""

import warnings
from typing import Any, Optional, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import Array

from MaxText.common_types import DType, PyTree


class ModuleUtilsMixin:
  """A few utilities for `flax.linen.Modules`, to be used as a mixin."""

  # `add_memory_hooks` and `reset_memory_hooks_state` are skipped as they are for debugging.
  # `device` and `dtype` properties are skipped as they are handled differently in JAX.

  def invert_attention_mask(self, encoder_attention_mask: Array) -> Array:
    """Inverts an attention mask (e.g., switches 0. and 1.).

    Args:
      encoder_attention_mask (`Array`): An attention mask.

    Returns:
      `Array`: The inverted attention mask.
    """
    if encoder_attention_mask.ndim == 3:
      encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
    elif encoder_attention_mask.ndim == 2:
      encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
    else:
      raise ValueError(f"Wrong shape for encoder_attention_mask (shape {encoder_attention_mask.shape})")

    if not hasattr(self, "dtype"):
      raise ValueError("Module must have a `dtype` attribute to use `invert_attention_mask`.")

    encoder_extended_attention_mask = encoder_extended_attention_mask.astype(self.dtype)  # fp16 compatibility
    encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * jnp.finfo(self.dtype).min

    return encoder_extended_attention_mask

  @staticmethod
  def create_extended_attention_mask_for_decoder(input_shape: tuple[int, int], attention_mask: Array) -> Array:
    """Creates an extended attention mask for a decoder."""
    batch_size, seq_length = input_shape
    seq_ids = jnp.arange(seq_length)
    causal_mask = seq_ids[None, None, :] <= seq_ids[None, :, None]
    # in case past_key_values are used we need to add a prefix ones mask to the causal mask
    causal_mask = causal_mask.astype(attention_mask.dtype)

    if causal_mask.shape[1] < attention_mask.shape[1]:
      prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
      causal_mask = jnp.concatenate(
          [
              jnp.ones((batch_size, seq_length, prefix_seq_len), dtype=causal_mask.dtype),
              causal_mask,
          ],
          axis=-1,
      )

    extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
    return extended_attention_mask

  def get_extended_attention_mask(
      self, attention_mask: Array, input_shape: tuple[int], dtype: DType | None = None
  ) -> Array:
    """Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Args:
      attention_mask (`Array`): Mask with ones indicating tokens to attend to,
        zeros for tokens to ignore.
      input_shape (`tuple[int]`): The shape of the input to the model.
      dtype (`DType`, *optional*): The dtype of the resulting mask.

    Returns:
      `Array` The extended attention mask, with a the same dtype as
      `attention_mask.dtype`.
    """
    if dtype is None:
      if not hasattr(self, "dtype"):
        raise ValueError("Module must have a `dtype` attribute or `dtype` must be provided.")
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
        extended_attention_mask = ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
            input_shape, attention_mask
        )
      else:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
      raise ValueError(f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})")

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and the dtype's smallest value for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.astype(dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * jnp.finfo(dtype).min
    return extended_attention_mask

  def get_head_mask(
      self, head_mask: Optional[Array], num_hidden_layers: int, is_attention_chunked: bool = False
  ) -> Union[Array, list[None]]:
    """Prepares the head mask if needed.

    Args:
      head_mask (`Array` with shape `[num_heads]` or `[num_hidden_layers x
        num_heads]`, *optional*): The mask indicating if we should keep the
        heads or not (1.0 for keep, 0.0 for discard).
      num_hidden_layers (`int`): The number of hidden layers in the model.
      is_attention_chunked (`bool`, *optional*, defaults to `False`): Whether or
        not the attentions scores are computed by chunks or not.

    Returns:
      `Array` with shape `[num_hidden_layers x batch x num_heads x seq_length x
      seq_length]` or list with `[None]` for each layer.
    """
    if head_mask is not None:
      head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
      if is_attention_chunked:
        head_mask = jnp.expand_dims(head_mask, axis=-1)
    else:
      head_mask = [None] * num_hidden_layers

    return head_mask

  def _convert_head_mask_to_5d(self, head_mask: Array, num_hidden_layers: int) -> Array:
    """Converts a head mask to 5D.

    Resulting shape: `[num_hidden_layers x batch x num_heads x seq_length x
    seq_length]`
    """
    if head_mask.ndim == 1:
      head_mask = jnp.expand_dims(head_mask, axis=(0, 1, 3, 4))
      target_shape = (num_hidden_layers, 1, head_mask.shape[2], 1, 1)
      head_mask = jnp.broadcast_to(head_mask, target_shape)
    elif head_mask.ndim == 2:
      head_mask = jnp.expand_dims(head_mask, axis=(1, 3, 4))  # We can specify head_mask for each layer

    if head_mask.ndim != 5:
      raise AssertionError(f"head_mask.ndim != 5, instead {head_mask.ndim}")

    if not hasattr(self, "dtype"):
      raise ValueError("Module must have a `dtype` attribute to use `_convert_head_mask_to_5d`.")
    head_mask = head_mask.astype(self.dtype)  # switch to float if need + fp16 compatibility
    return head_mask

  def num_parameters(self, params: PyTree, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
    """Gets number of (optionally, trainable or non-embeddings) parameters in the module.

    Args:
      params (`PyTree`): The PyTree of model parameters.
      only_trainable (`bool`, *optional*, defaults to `False`): Whether or not
        to return only the number of trainable parameters.
      exclude_embeddings (`bool`, *optional*, defaults to `False`): Whether or
        not to return only the number of non-embeddings parameters.

    Returns:
      `int`: The number of parameters.
    """
    # This is a simplified JAX version. The original PyTorch version has complex logic
    # for bitsandbytes quantization, which is not directly translatable.
    # `only_trainable` and `exclude_embeddings` would require traversing the PyTree
    # with path and filtering based on names or other metadata (e.g., from flax.core.param_dict).
    if only_trainable:
      # In Flax, trainability is often managed by freezing parts of the params PyTree
      # or using optimizers that only update a subset of params. This function
      # doesn't have that context.
      warnings.warn("`only_trainable` is not implemented.")

    leaves_with_paths = jax.tree_util.tree_leaves_with_path(params)

    if exclude_embeddings:
      # This requires a convention for naming embedding layers.
      # A common convention is to have 'embedding' in the parameter name.
      warnings.warn("`exclude_embeddings` is partially implemented using a name-based heuristic.")
      return sum(p.size for path, p in leaves_with_paths if "embedding" not in ".".join(k.key for k in path))

    return sum(p.size for p in jax.tree_util.tree_leaves(params))

  def estimate_tokens(self, input_dict: dict[str, Union[Array, Any]]) -> int:
    """Helper function to estimate the total number of tokens from the model inputs.

    Args:
      input_dict (`dict`): The model inputs.

    Returns:
      `int`: The total number of tokens.
    """
    if not hasattr(self, "warnings_issued"):
      self.warnings_issued = {}
    if self.main_input_name in input_dict:
      return input_dict[self.main_input_name].size
    elif "estimate_tokens" not in self.warnings_issued:
      warnings.warn("Could not estimate the number of tokens of the input, floating-point operations will not be computed")
      self.warnings_issued["estimate_tokens"] = True
    return 0

  def floating_point_ops(
      self, params: PyTree, input_dict: dict[str, Union[Array, Any]], exclude_embeddings: bool = True
  ) -> int:
    """Gets number of (optionally, non-embeddings) floating-point operations for the forward and backward passes.

    Args:
      params (`PyTree`): The PyTree of model parameters.
      input_dict (`dict`): The model inputs.
      exclude_embeddings (`bool`, *optional*, defaults to `True`): Whether or not
        to count embedding and softmax operations.

    Returns:
      `int`: The number of floating-point operations.
    """
    return 6 * self.estimate_tokens(input_dict) * self.num_parameters(params, exclude_embeddings=exclude_embeddings)

from typing import Any


def get_module_from_name(module: Any, tensor_name: str) -> tuple[Any, str]:
  """Traverses a module hierarchy to find a submodule from a dot-separated name.

  This is a utility function to navigate nested modules, similar in purpose to
  PyTorch's `get_submodule`. It splits a tensor name like
  "layers_0.attention.kernel" into a module path "layers_0.attention" and a
  final name "kernel", then traverses the module attributes to find the
  target submodule.

  Args:
    module: The top-level module to start traversal from.
    tensor_name: The dot-separated name of the tensor or submodule path.

  Returns:
    A tuple containing the target submodule and the remaining part of the name.
  """
  if "." in tensor_name:
    module_name, tensor_name = tensor_name.rsplit(".", 1)
    for part in module_name.split("."):
      module = getattr(module, part)
  return module, tensor_name

from functools import lru_cache
import importlib.util


@lru_cache
def is_flash_attn_3_available():
  """Checks if flash-attention 3 is available."""
  if importlib.util.find_spec("flash_attn_3") is None:
    return False

  try:
    import jax

    if not jax.devices("gpu"):
      return False
  except ImportError:
    return False

  # TODO: Check for a minimum version when FA3 is stable
  # return version.parse(importlib.metadata.version("flash_attn_3")) >= version.parse("3.0.0")

  return True

# The variable `_flax_available` is assumed to be defined at the module level,
# similar to the original PyTorch file.

def is_flax_available() -> bool:
  """Returns True if Flax is available."""
  return _flax_available

# This function is a JAX-based replacement for a similar utility in PyTorch.
# In a JAX environment, PyTorch is not the primary backend, so this function
# consistently returns False. This approach is taken for compatibility with
# codebases that might still contain checks for PyTorch availability.

def is_torch_available() -> bool:
  """
  Returns `False` as PyTorch is not used in this JAX-based environment.

  This function is provided for compatibility with original transformers utilities.
  """
  return False

import jax.numpy as jnp
from jax import Array


def find_packed_sequence_indices(position_ids: Array) -> Array:
  """Finds the indices of the sequence to which each new query token in the sequence belongs when using packed
    tensor format (i.e. several sequences packed in the same batch dimension).

  Args:
    position_ids (`Array`): A 2D tensor of shape (batch_size, query_length) indicating the positions of each token in
      the sequences.

  Returns:
    A 2D tensor where each similar integer indicates that the tokens belong to the same sequence. For example, if we
    pack 3 sequences of 2, 3 and 1 tokens respectively along a single batch dim, this will return
    [[0, 0, 1, 1, 1, 2]].
  """
  # What separate different sequences is when 2 consecutive positions_ids are separated by more than 1. So
  # taking the diff (by prepending the first value - 1 to keep correct indexing) and applying cumsum to the result
  # gives exactly the sequence indices
  # Note that we assume that a single sequence cannot span several batch dimensions, i.e. 1 single sequence
  # cannot be part of the end of the first batch dim and the start of the 2nd one for example
  first_dummy_value = position_ids[:, :1] - 1  # We just need the diff on this first value to be 1
  position_diff = jnp.diff(position_ids, prepend=first_dummy_value, axis=-1)
  packed_sequence_mask = (position_diff != 1).cumsum(axis=-1)

  # Here it would be nice to return None if we did not detect packed sequence format, i.e. if
  # `packed_sequence_mask[:, -1] == 0` but it causes issues with export
  return packed_sequence_mask

from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp

# All references to `torch.Tensor` are replaced by `jax.Array`.
# `BlockMask` is a PyTorch-specific type for flex_attention, which is not present in JAX.
# We will treat it as a standard `jax.Array`.
# Assumed imports for PretrainedConfig and Cache, mirroring the original structure.
from ..transformers.configuration_utils import PretrainedConfig
from .cache_utils import Cache

# Reused from Qwen3ForCausalLM.modeling_utils.find_packed_sequence_indices
from ..qwen3.modeling_utils import find_packed_sequence_indices


# This is a set of attention implementations that require mask creation.
# Custom backends that handle masking internally are not included.
# This mirrors the logic of `ALL_MASK_ATTENTION_FUNCTIONS` in the original PyTorch code.
ALL_MASK_ATTENTION_IMPLEMENTATIONS = {"sdpa", "eager", "flash_attention_2", "flex_attention"}


def _preprocess_mask_arguments(
    config: PretrainedConfig,
    input_embeds: jax.Array,
    attention_mask: Optional[jax.Array],
    cache_position: jax.Array,
    past_key_values: Optional[Cache],
    position_ids: Optional[jax.Array],
    layer_idx: Optional[int],
) -> Tuple[bool, Optional[jax.Array], Optional[jax.Array], Optional[int], Optional[int]]:
    """
    Perform some common pre-processing of the mask arguments we get from the modeling code. Mostly determine the
    key-value length and offsets, and if we should early exit or not.

    Args:
        config (`PretrainedConfig`):
            The model config.
        input_embeds (`jax.Array`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is used only to infer the
            batch size, query length and dtype.
        attention_mask (`jax.Array`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length).
            It can also be an already prepared 4D mask, in which case it is returned as-is.
        cache_position (`jax.Array`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        past_key_values (`Cache`, optional):
            The past key values, if we use a cache.
        position_ids (`jax.Array`, optional)
            A 2D tensor of shape (batch_size, query_length) indicating the positions of each token in the sequences.
        layer_idx (`int`, optional):
            If `past_key_values` is not None, this is the layer index of the cache from which to get the key-value
            length and offset. Indeed, for hybrid caches, different layers may return different lengths.

    Returns:
        early_exit (`bool`):
            Whether we should early exit mask creation, and return the mask as-is.
        attention_mask (`jax.Array` or `None`):
            The attention mask to either return immediately, or to use in downstream mask creation.
        packed_sequence_mask (`jax.Array`, optional):
            In case we detected packed sequence format, this is a tensor where each similar integer indicates that
            the tokens belong to the same sequence.
        kv_length (`int`):
            The size that the key and value states will have during the attention computation.
        kv_offset (`int`):
            An offset to indicate at which first position the key and values states will refer to.
    """
    # If the mask is already 4D, simply return as-is (it was already prepared, or it is custom)
    if attention_mask is not None and attention_mask.ndim == 4:
        return True, attention_mask, None, None, None

    # For TGI/vLLM backends, or other custom attention without equivalent mask creation: we don't need a mask!
    if config._attn_implementation not in ALL_MASK_ATTENTION_IMPLEMENTATIONS:
        return True, None, None, None, None

    # Potentially switch dtype for efficiency
    if attention_mask is not None and attention_mask.ndim == 2:
        attention_mask = attention_mask.astype(jnp.bool_)

    # If using a cache, it can give all information about mask sizes based on seen tokens
    if past_key_values is not None:
        kv_length, kv_offset = past_key_values.get_mask_sizes(cache_position, layer_idx)
    # Otherwise, the sizes are simply the input sizes
    else:
        kv_length, kv_offset = input_embeds.shape[1], 0

    # We check the position_ids for potential packed sequence format (only if the 2D attention mask is explicitly None,
    # and we don't have past_key_values, i.e. generally a training setup)
    packed_sequence_mask = None
    if position_ids is not None and attention_mask is None and past_key_values is None:
        batch_size = input_embeds.shape[0]
        # The position ids are sometimes just unsqueezed, without being expanded
        if batch_size != position_ids.shape[0]:
            position_ids = jnp.broadcast_to(position_ids, (batch_size, position_ids.shape[-1]))
        # Reused from Qwen3ForCausalLM.modeling_utils.find_packed_sequence_indices
        packed_sequence_mask = find_packed_sequence_indices(position_ids)

    return False, attention_mask, packed_sequence_mask, kv_length, kv_offset

from typing import Optional, Tuple

import jax.numpy as jnp
from jax import Array

from MaxText.common_types import Config


def _compute_default_rope_parameters(
    config: Optional[Config] = None,
    seq_len: Optional[int] = None,
) -> Tuple[Array, float]:
  """Computes the inverse frequencies according to the original RoPE implementation.

  Args:
    config: The model configuration.
    seq_len: The current sequence length. Unused for this type of RoPE.

  Returns:
    A tuple containing the inverse frequencies for the RoPE embeddings and the
    post-processing scaling factor applied to the computed cos/sin (unused in
    this type of RoPE).
  """
  base = config.rope_theta
  partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
  head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
  dim = int(head_dim * partial_rotary_factor)

  attention_factor = 1.0  # Unused in this type of RoPE

  # Compute the inverse frequencies
  inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
  return inv_freq, attention_factor

from typing import Optional, Tuple, Union

import jax.numpy as jnp
from jax import Array

from .configuration_utils import PretrainedConfig


def _compute_dynamic_ntk_parameters(
    config: Optional[PretrainedConfig] = None,
    seq_len: Optional[Union[int, Array]] = None,
) -> Tuple[Array, float]:
    """
    Computes the inverse frequencies with NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla
    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        seq_len (`int` or `Array`, *optional*):
            The current sequence length, used to update the dynamic RoPE at inference time.
    Returns:
        Tuple of (`Array`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    # TODO (joao): use the new `original_max_position_embeddings` from rope_scaling
    base = config.rope_theta
    partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)
    max_position_embeddings = config.max_position_embeddings
    factor = config.rope_scaling["factor"]

    attention_factor = 1.0  # Unused in this type of RoPE

    # seq_len: default to max_position_embeddings, e.g. at init time
    if seq_len is None:
        seq_len = max_position_embeddings
    elif isinstance(seq_len, jnp.ndarray):
        seq_len = jnp.maximum(
            seq_len,
            jnp.array(max_position_embeddings, dtype=seq_len.dtype),
        )
    else:
        seq_len = max(seq_len, max_position_embeddings)

    # Compute the inverse frequencies
    base = base * ((factor * seq_len / max_position_embeddings) - (factor - 1)) ** (dim / (dim - 2))
    inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    return inv_freq, attention_factor

import math
from typing import Optional, Tuple

import jax.numpy as jnp
from jax import Array

from maxtext.common_types import Config
# The following import is assumed to exist in the target codebase,
# as it is a dependency of the function to be converted but not provided.
# It is the JAX equivalent of the original PyTorch `_compute_default_rope_parameters`.
from maxtext.layers.rope import _compute_default_rope_parameters


def _compute_llama3_parameters(
    config: Config, seq_len: Optional[int] = None
) -> Tuple[Array, float]:
  """Computes the inverse frequencies for llama 3.1.

  Args:
    config: The model configuration.
    seq_len: The current sequence length. Unused for this type of RoPE.

  Returns:
    Tuple of (Array, float), containing the inverse frequencies for the RoPE
    embeddings and the post-processing scaling factor applied to the computed
    cos/sin.
  """
  # Gets the default RoPE parameters
  # Assumed to be implemented in: maxtext.layers.rope._compute_default_rope_parameters
  inv_freq, attention_factor = _compute_default_rope_parameters(
      config, seq_len=seq_len
  )

  factor = config.rope_scaling["factor"]  # `8` in the original implementation
  low_freq_factor = config.rope_scaling[
      "low_freq_factor"
  ]  # `1` in the original implementation
  high_freq_factor = config.rope_scaling[
      "high_freq_factor"
  ]  # `4` in the original implementation
  old_context_len = config.rope_scaling[
      "original_max_position_embeddings"
  ]  # `8192` in the original implementation

  low_freq_wavelen = old_context_len / low_freq_factor
  high_freq_wavelen = old_context_len / high_freq_factor

  wavelen = 2 * math.pi / inv_freq
  # wavelen < high_freq_wavelen: do nothing
  # wavelen > low_freq_wavelen: divide by factor
  inv_freq_llama = jnp.where(
      wavelen > low_freq_wavelen, inv_freq / factor, inv_freq
  )
  # otherwise: interpolate between the two, using a smooth factor
  smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
      high_freq_factor - low_freq_factor
  )
  smoothed_inv_freq = (
      1 - smooth_factor
  ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
  is_medium_freq = jnp.logical_and(
      wavelen <= low_freq_wavelen, wavelen >= high_freq_wavelen
  )
  inv_freq_llama = jnp.where(
      is_medium_freq, smoothed_inv_freq, inv_freq_llama
  )

  return inv_freq_llama, attention_factor

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
"""
Functions to compute RoPE parameters, adapted from HuggingFace transformers.
"""

import math
from typing import Optional, Tuple

import jax.numpy as jnp
from jax import Array

from MaxText.common_types import Config


def _compute_yarn_parameters(config: Config, seq_len: Optional[int] = None) -> Tuple[Array, float]:
  """Computes the inverse frequencies with NTK scaling.

  Please refer to the [original paper](https://huggingface.co/papers/2309.00071).

  Args:
    config: The model configuration.
    seq_len: The current sequence length. Unused for this type of RoPE.

  Returns:
    A tuple of (Array, float), containing the inverse frequencies for the RoPE
    embeddings and the post-processing scaling factor applied to the computed
    cos/sin.
  """

  base = config.rope_theta
  partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
  head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
  dim = int(head_dim * partial_rotary_factor)
  factor = config.rope_scaling["factor"]
  attention_factor = config.rope_scaling.get("attention_factor")
  mscale = config.rope_scaling.get("mscale")
  mscale_all_dim = config.rope_scaling.get("mscale_all_dim")

  # NOTE: DeekSeek-V3 (and potentially other models) modify `max_position_embeddings` and have a
  # `original_max_position_embeddings` field containing the pretrained value. They use the ratio between these two
  # values to compute the default attention scaling factor, instead of using `factor`.
  if "original_max_position_embeddings" in config.rope_scaling:
    original_max_position_embeddings = config.rope_scaling["original_max_position_embeddings"]
    factor = config.max_position_embeddings / original_max_position_embeddings
  else:
    original_max_position_embeddings = config.max_position_embeddings

  def get_mscale(scale, mscale=1):
    if scale <= 1:
      return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0

  # Sets the attention factor as suggested in the paper
  if attention_factor is None:
    if mscale and mscale_all_dim:
      attention_factor = float(get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim))
    else:
      attention_factor = get_mscale(factor)

  # Optional config options
  # beta_fast/beta_slow: as suggested in the paper, default to 32/1 (correspondingly)
  beta_fast = config.rope_scaling.get("beta_fast") or 32
  beta_slow = config.rope_scaling.get("beta_slow") or 1

  # Compute the inverse frequencies
  def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
    """Inverse dimension formula to find the dimension based on the number of rotations"""
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

  def find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings, truncate):
    """Find dimension range bounds based on rotations"""
    low = find_correction_dim(low_rot, dim, base, max_position_embeddings)
    high = find_correction_dim(high_rot, dim, base, max_position_embeddings)
    if truncate:
      low = math.floor(low)
      high = math.ceil(high)
    return max(low, 0), min(high, dim - 1)

  def linear_ramp_factor(min_val, max_val, dim):
    if min_val == max_val:
      max_val += 0.001  # Prevent singularity

    linear_func = (jnp.arange(dim, dtype=jnp.float32) - min_val) / (max_val - min_val)
    ramp_func = jnp.clip(linear_func, 0, 1)
    return ramp_func

  # Note on variable naming: "interpolation" comes from the original technique, where we interpolate the position IDs
  # to expand the possible context length. In other words, interpolation = apply scaling factor.
  pos_freqs = base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
  inv_freq_extrapolation = 1.0 / pos_freqs
  inv_freq_interpolation = 1.0 / (factor * pos_freqs)

  truncate = config.rope_scaling.get("truncate", True)
  low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_max_position_embeddings, truncate)

  # Get n-dimensional rotational scaling corrected for extrapolation
  inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, dim // 2)
  inv_freq = (
      inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
      + inv_freq_extrapolation * inv_freq_extrapolation_factor
  )
  return inv_freq, attention_factor

from typing import Callable, Optional

_flash_varlen_fn: Optional[Callable] = None

from typing import Optional, Tuple

import jax.numpy as jnp
from jax import Array


# `globals()` is not compatible with JIT, hence we have do define them in global scope ourselves
_flash_fn = None
_flash_varlen_fn = None
_pad_fn = None
_unpad_fn = None


def _index_first_axis(tensor: Array, indices: Array) -> Array:
  """A local implementation of the PyTorch indexing operation `tensor[indices]` on the first axis,
    after flattening the first two dimensions of the tensor. This is functionally equivalent to
    FA2's `index_first_axis` and replaces the need to import it.
    """
  # The input tensor is expected to be of shape (batch, seq_len, ...). We flatten the first
  # two dimensions to get (total_tokens, ...) before indexing.
  reshaped_tensor = tensor.reshape(-1, *tensor.shape[2:])
  return reshaped_tensor[indices]


def _unpad_input(
    hidden_states: Array, attention_mask: Array, unused_mask: Optional[Array] = None
) -> Tuple[Array, Array, Array, Array, Array]:
  """unpad_input function for flash attention variants that do not have them within their pkg themselves, e.g. fa3.

    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
        unused_mask: (batch, seqlen), bool / int, 1 means the element is allocated but unused.

    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens selected in attention_mask + unused_mask.
        indices: (total_nnz), the indices of masked tokens from the flattened input sequence.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
        seqused: (batch), returns the number of tokens selected in attention_mask + unused_mask.
    """
  if unused_mask is not None:
    all_masks = attention_mask + unused_mask
  else:
    all_masks = attention_mask

  seqlens_in_batch = all_masks.sum(axis=-1, dtype=jnp.int32)
  used_seqlens_in_batch = attention_mask.sum(axis=-1, dtype=jnp.int32)
  indices = jnp.where(all_masks.flatten())[0]
  max_seqlen_in_batch = seqlens_in_batch.max()
  cu_seqlens = jnp.pad(jnp.cumsum(seqlens_in_batch, axis=0, dtype=jnp.int32), ((1, 0),))

  return (
      _index_first_axis(hidden_states, indices),
      indices,
      cu_seqlens,
      max_seqlen_in_batch,
      used_seqlens_in_batch,
  )


def _pad_input(hidden_states: Array, indices: Array, batch: int, seqlen: int) -> Array:
  """pad_input function for flash attention variants that do not have them within their pkg themselves, e.g. fa3.

    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices that represent the non-masked tokens of the original padded input sequence.
        batch: int, batch size for the padded sequence.
        seqlen: int, maximum sequence length for the padded sequence.

    Return:
        hidden_states: (batch, seqlen, ...)
    """
  dim = hidden_states.shape[1:]
  output = jnp.zeros((batch * seqlen, *dim), dtype=hidden_states.dtype)
  output = output.at[indices].set(hidden_states)
  return output.reshape(batch, seqlen, *dim)

from typing import Callable, Optional

_unpad_fn: Optional[Callable] = None

from typing import Any
import jax
from flax.core import FrozenDict

PyTree = Any


def get_parameter_device(parameters: PyTree) -> jax.Device:
  """Gets the device of the first parameter in a PyTree.

  This is a JAX equivalent of a common PyTorch pattern. It assumes all
  parameters are on the same device or sharded across a set of devices, and
  returns the first device from the first parameter's device set.

  Args:
    parameters: A PyTree of JAX arrays (e.g., model parameters or state).

  Returns:
    The jax.Device of the first parameter found.

  Raises:
    StopIteration: If the PyTree contains no arrays (leaves).
  """
  try:
    # jax.tree_util.tree_leaves returns a flat list of all arrays in the PyTree.
    first_param = jax.tree_util.tree_leaves(parameters)[0]
    # A JAX array can be on multiple devices if sharded. We return the first one
    # to be analogous to the PyTorch function's behavior.
    return next(iter(first_param.devices()))
  except (StopIteration, IndexError) as e:
    # This block is entered if tree_leaves returns an empty list.
    raise StopIteration("Could not find any arrays in the PyTree.") from e

from typing import Optional

import jax
import jax.numpy as jnp

from MaxText.common_types import PyTree


def get_parameter_dtype(variables: PyTree) -> Optional[jnp.dtype]:
  """
  Returns the first found floating dtype in parameters if there is one,
  otherwise returns the last dtype it found.

  Args:
    variables: A PyTree of model variables, typically a dict with 'params'
      and 'buffers'.

  Returns:
    The determined jax.numpy.dtype or None if no parameters/buffers are found.
  """
  last_dtype = None

  # 1. Check 'params' collection first, similar to PyTorch's model.parameters()
  if "params" in variables:
    for param in jax.tree_util.tree_leaves(variables["params"]):
      if hasattr(param, "dtype"):
        last_dtype = param.dtype
        if jnp.issubdtype(param.dtype, jnp.floating):
          return param.dtype

  # 2. Fallback to 'buffers' collection, similar to PyTorch's model.buffers()
  if "buffers" in variables:
    for buffer in jax.tree_util.tree_leaves(variables["buffers"]):
      if hasattr(buffer, "dtype"):
        last_dtype = buffer.dtype
        if jnp.issubdtype(buffer.dtype, jnp.floating):
          return buffer.dtype

  # 3. If no floating point type was found in either, return the last dtype seen.
  return last_dtype

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may
# obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A few utilities for `flax.linen.Modules`, to be used as a mixin."""

import warnings
from typing import Any, Dict, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp

# Reused from src.MaxText.common_types
from maxtext.common_types import Array, DType, PyTree


# Reused from src.MaxText.layers.models.Qwen3ForCausalLM.modeling_utils.ModuleUtilsMixin
class ModuleUtilsMixin:
  """A few utilities for `flax.linen.Modules`, to be used as a mixin."""

  def invert_attention_mask(self, encoder_attention_mask: Array) -> Array:
    """
    Invert an attention mask (e.g., switches 0. and 1.).

    Args:
        encoder_attention_mask (`Array`): An attention mask.

    Returns:
        `Array`: The inverted attention mask.
    """
    if encoder_attention_mask.ndim == 3:
      encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
    elif encoder_attention_mask.ndim == 2:
      encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
    else:
      encoder_extended_attention_mask = encoder_attention_mask

    encoder_extended_attention_mask = encoder_extended_attention_mask.astype(self.dtype)
    encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * jnp.finfo(self.dtype).min

    return encoder_extended_attention_mask

  @staticmethod
  def create_extended_attention_mask_for_decoder(input_shape: Tuple[int, ...], attention_mask: Array):
    batch_size, seq_length = input_shape
    seq_ids = jnp.arange(seq_length)
    causal_mask = seq_ids[None, None, :].repeat(batch_size, axis=0) <= seq_ids[None, :, None]
    # in case past_key_values are used we need to add a prefix ones mask to the causal mask
    causal_mask = causal_mask.astype(attention_mask.dtype)

    if causal_mask.shape[1] < attention_mask.shape[1]:
      prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
      causal_mask = jnp.concatenate(
          [
              jnp.ones((batch_size, seq_length, prefix_seq_len), dtype=causal_mask.dtype),
              causal_mask,
          ],
          axis=-1,
      )

    extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
    return extended_attention_mask

  def get_extended_attention_mask(
      self, attention_mask: Array, input_shape: Tuple[int, ...], dtype: DType = None
  ) -> Array:
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (`Array`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (`Tuple[int]`):
            The shape of the input to the model.

    Returns:
        `Array` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
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
        extended_attention_mask = ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
            input_shape, attention_mask
        )
      else:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
      raise ValueError(f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})")

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and the dtype's smallest value for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.astype(dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * jnp.finfo(dtype).min
    return extended_attention_mask

  def get_head_mask(
      self, head_mask: Optional[Array], num_hidden_layers: int, is_attention_chunked: bool = False
  ) -> Union[Array, list]:
    """
    Prepare the head mask if needed.

    Args:
        head_mask (`Array` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
            The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
        num_hidden_layers (`int`):
            The number of hidden layers in the model.
        is_attention_chunked (`bool`, *optional*, defaults to `False`):
            Whether or not the attentions scores are computed by chunks or not.

    Returns:
        `Array` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
        `[None]` for each layer.
    """
    if head_mask is not None:
      head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
      if is_attention_chunked is True:
        head_mask = jnp.expand_dims(head_mask, -1)
    else:
      head_mask = [None] * num_hidden_layers

    return head_mask

  def _convert_head_mask_to_5d(self, head_mask: Array, num_hidden_layers: int) -> Array:
    """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
    if head_mask.ndim == 1:
      head_mask = jnp.expand_dims(head_mask, (0, 1, 3, 4))
      head_mask = jnp.broadcast_to(head_mask, (num_hidden_layers, -1, -1, -1, -1))
    elif head_mask.ndim == 2:
      head_mask = jnp.expand_dims(head_mask, (1, 3, 4))  # We can specify head_mask for each layer
    assert head_mask.ndim == 5, f"head_mask.ndim != 5, instead {head_mask.ndim}"
    head_mask = head_mask.astype(self.dtype)
    return head_mask

  def num_parameters(self, params: PyTree, exclude_embeddings: bool = False) -> int:
    """
    Get number of (optionally, trainable or non-embeddings) parameters in the module.

    Args:
        params (`PyTree`):
            The PyTree of model parameters.
        exclude_embeddings (`bool`, *optional*, defaults to `False`):
            Whether or not to return only the number of non-embeddings parameters

    Returns:
        `int`: The number of parameters.
    """
    if "params_axes" in params:
      params = params["params"]

    all_parameters = jax.tree_util.tree_leaves_with_path(params)

    if exclude_embeddings:
      embedding_param_names = [
          f"{name}.weight" for name, module_type in self.named_modules() if isinstance(module_type, nn.Embed)
      ]
      total_parameters = [
          parameter for path, parameter in all_parameters if ".".join(path) not in embedding_param_names
      ]
    else:
      total_parameters = [parameter for _, parameter in all_parameters]

    # The following logic for 4-bit models is not applicable in JAX/Flax in the same way.
    # MaxText uses AQT for quantization, which has a different parameter structure.
    # We will just count the number of elements.
    total_numel = [param.size for param in total_parameters]

    return sum(total_numel)

  def estimate_tokens(self, input_dict: Dict[str, Union[Array, Any]]) -> int:
    """
    Helper function to estimate the total number of tokens from the model inputs.

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
      warnings.warn("Could not estimate the number of tokens of the input, floating-point operations will not be computed")
      self.warnings_issued["estimate_tokens"] = True
    return 0

  def floating_point_ops(
      self, params: PyTree, input_dict: Dict[str, Union[Array, Any]], exclude_embeddings: bool = True
  ) -> int:
    """
    Get number of (optionally, non-embeddings) floating-point operations for the forward and backward passes of a
    batch with this transformer model. Default approximation neglects the quadratic dependency on the number of
    tokens (valid if `12 * d_model << sequence_length`) as laid out in [this
    paper](https://huggingface.co/papers/2001.08361) section 2.1. Should be overridden for transformers with parameter
    re-use e.g. Albert or Universal Transformers, or if doing long-range modeling with very high sequence lengths.

    Args:
        params (`PyTree`):
            The PyTree of model parameters.
        input_dict (`Dict[str, Union[Array, Any]]`):
            The model inputs.
        exclude_embeddings (`bool`, *optional*, defaults to `True`):
            Whether or not to count embedding and softmax operations.

    Returns:
        `int`: The number of floating-point operations.
    """
    return 6 * self.estimate_tokens(input_dict) * self.num_parameters(params, exclude_embeddings=exclude_embeddings)

import importlib.metadata
import importlib.util
from packaging import version

def is_flash_attn_2_available():
    """Checks if flash-attention 2 is available and compatible with the JAX backend."""
    if importlib.util.find_spec("flash_attn") is None:
        return False

    # Check for a compatible JAX backend (GPU)
    try:
        import jax
        import jax.lib

        if not any(d.platform == "gpu" for d in jax.devices()):
            return False
    except (ImportError, RuntimeError):
        # JAX not installed or no backend found
        return False

    flash_attn_version = version.parse(importlib.metadata.version("flash_attn"))
    backend = jax.lib.xla_bridge.get_backend()

    if backend.platform == "cuda":
        return flash_attn_version >= version.parse("2.1.0")
    elif backend.platform == "rocm":
        # TODO: Bump the requirement to 2.1.0 once released in https://github.com/ROCmSoftwarePlatform/flash-attention
        return flash_attn_version >= version.parse("2.0.4")
    else:
        # The original PyTorch code also checked for MLU, which is not applicable to JAX.
        # Other platforms like TPU are not covered by this specific check.
        return False

from packaging import version

# from transformers.utils.import_utils import is_torch_available, _torch_version
# In a JAX environment, is_torch_available() would return False and _torch_version would be "N/A".
# Therefore, this function will always return False. For simplicity, we can directly return False.

def is_torch_flex_attn_available() -> bool:
  """
  Checks if PyTorch's flex attention is available.

  This is a PyTorch-specific feature. In a JAX environment, this will always
  return False.
  """
  return False

import importlib.util
from functools import lru_cache


@lru_cache
def is_torch_npu_available(check_device=False):
    "Checks if a JAX NPU backend is available and potentially if an NPU is in the environment"
    # In a JAX context, we check for Flax/JAX availability and a hypothetical JAX NPU plugin.
    # This assumes a hypothetical jax_npu_plugin, as there's no standard JAX NPU backend.
    if not is_flax_available() or importlib.util.find_spec("jax_npu_plugin") is None:
        return False

    import jax
    import jax_npu_plugin  # noqa: F401

    if check_device:
        try:
            # jax.devices() will raise a RuntimeError if the backend is not found or fails to initialize.
            return len(jax.devices("npu")) > 0
        except RuntimeError:
            return False

    # Check if the 'npu' backend is registered with JAX and can be initialized.
    try:
        _ = jax.lib.xla_bridge.get_backend("npu")
        return True
    except RuntimeError:
        return False

import jax.numpy as jnp
from MaxText.common_types import Array


def rotate_half(x: Array) -> Array:
  """Rotates half the hidden dims of the input."""
  x1 = x[..., : x.shape[-1] // 2]
  x2 = x[..., x.shape[-1] // 2 :]
  return jnp.concatenate((-x2, x1), axis=-1)

from typing import Optional, Tuple

from jax import Array
import jax.numpy as jnp

# Re-used from Qwen3ForCausalLM.modeling_utils.rotate_half
from ..modeling_utils import rotate_half


def apply_rotary_pos_emb(
    q: Array,
    k: Array,
    cos: Array,
    sin: Array,
    position_ids: Optional[Array] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[Array, Array]:
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`Array`): The query tensor.
        k (`Array`): The key tensor.
        cos (`Array`): The cosine part of the rotary embedding.
        sin (`Array`): The sine part of the rotary embedding.
        position_ids (`Array`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(Array)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = jnp.expand_dims(cos, axis=unsqueeze_dim)
    sin = jnp.expand_dims(sin, axis=unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

from jax import numpy as jnp

from maxtext.common_types import Array


def repeat_kv(hidden_states: Array, n_rep: int) -> Array:
  """
  This is the equivalent of jnp.repeat(x, repeats=n_rep, axis=1). The hidden states go from (batch,
  num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
  """
  if n_rep == 1:
    return hidden_states
  return jnp.repeat(hidden_states, repeats=n_rep, axis=1)

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax.linen import nn

# From src.MaxText.common_types
from typing import Any, Callable, Iterable, Literal, Sequence, Union
Array = jax.Array
DType = jax.typing.DTypeLike
PyTree = Any
# Re-used from Qwen3ForCausalLM.modeling_utils.repeat_kv
from Qwen3ForCausalLM.modeling_utils import repeat_kv


def eager_attention_forward(
    query: Array,
    key: Array,
    value: Array,
    attention_mask: Optional[Array],
    dropout_layer: nn.Module,
    num_key_value_groups: int,
    scaling: float,
    deterministic: bool,
    **kwargs,
) -> Tuple[Array, Array]:
  """
  JAX implementation of eager attention forward pass.

  Args:
    query: The query tensor.
    key: The key tensor.
    value: The value tensor.
    attention_mask: The attention mask.
    dropout_layer: The dropout layer to apply.
    num_key_value_groups: The number of key-value groups.
    scaling: The scaling factor for attention scores.
    deterministic: Whether to run in deterministic mode (no dropout).
    **kwargs: Additional arguments.

  Returns:
    A tuple containing the attention output and attention weights.
  """
  # Re-used from Qwen3ForCausalLM.modeling_utils.repeat_kv
  key_states = repeat_kv(key, num_key_value_groups)
  value_states = repeat_kv(value, num_key_value_groups)

  attn_weights = jnp.matmul(query, jnp.swapaxes(key_states, -2, -1)) * scaling

  if attention_mask is not None:
    causal_mask = attention_mask[..., : key_states.shape[-2]]
    attn_weights = attn_weights + causal_mask

  attn_weights = jax.nn.softmax(attn_weights, axis=-1, dtype=jnp.float32).astype(query.dtype)
  attn_weights = dropout_layer(attn_weights, deterministic=deterministic)
  attn_output = jnp.matmul(attn_weights, value_states)
  attn_output = jnp.swapaxes(attn_output, 1, 2)

  return attn_output, attn_weights
from typing import Any, Optional


def infer_framework_from_repr(x: Any) -> Optional[str]:
  """Tries to guess the framework of an object `x` from its repr.

  This is brittle but will help in `is_tensor` to try the frameworks in a
  smart order, without the need to import the frameworks.

  Args:
    x: The object to inspect.

  Returns:
    A string representing the framework ('pt', 'tf', 'jax', 'np', 'mlx') or None.
  """
  representation = str(type(x))
  if representation.startswith("<class 'torch."):
    return "pt"
  elif representation.startswith("<class 'tensorflow."):
    return "tf"
  elif representation.startswith("<class 'jax"):
    return "jax"
  elif representation.startswith("<class 'numpy."):
    return "np"
  elif representation.startswith("<class 'mlx."):
    return "mlx"
  return None
from typing import Any
import numpy as np


def is_numpy_array(x: Any) -> bool:
  """Tests if `x` is a numpy array or not."""
  return isinstance(x, np.ndarray)

from . import is_flax_available


def is_flax_fx_available():
  return is_flax_available()

from typing import Any
import jax

# is_flax_available is defined in the same file, so no import is needed.


def is_torch_fx_proxy(x: Any) -> bool:
  """
  Checks if x is a JAX tracer. This is the JAX equivalent of checking for a
  torch.fx.Proxy object.
  """
  if is_flax_available():
    return isinstance(x, jax.core.Tracer)
  return False

from typing import Optional

from MaxText import max_logging as logging
from MaxText.common_types import Config
# from .rope_utils import _check_received_keys is an implicit dependency
# This function is assumed to be available in the same module.


def _validate_dynamic_scaling_rope_parameters(config: Config, ignore_keys: Optional[set] = None):
  """Validate dynamic scaling RoPE parameters."""
  rope_scaling = config.rope_scaling
  rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
  required_keys = {"rope_type", "factor"}
  # TODO (joao): update logic for the inclusion of `original_max_position_embeddings`
  optional_keys = {"original_max_position_embeddings"}
  received_keys = set(rope_scaling.keys())
  _check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)

  factor = rope_scaling["factor"]
  if factor is None or not isinstance(factor, float) or factor < 1.0:
    logging.warning(f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")

from collections import OrderedDict
from typing import Any, Dict, Tuple, Type


class ClassInstantier(OrderedDict):
  """A dictionary-like class that instantiates a class when an item is accessed."""

  def __getitem__(self, key: str) -> Any:
    """Retrieves the content for a key and returns an instantiated class.

    The values in the dictionary can be either a class type or a tuple of
    (class_type, kwargs_dict). When this method is called, it retrieves the
    value, unpacks the class and its keyword arguments, and returns a new
    instance of that class.

    Args:
      key: The key to look up in the dictionary.

    Returns:
      An instance of the class associated with the key.
    """
    content = super().__getitem__(key)
    cls: Type
    kwargs: Dict[str, Any]
    cls, kwargs = content if isinstance(content, tuple) else (content, {})
    return cls(**kwargs)

from typing import Sequence
import jax.numpy as jnp

Array = jnp.ndarray


def _set_aux_loss(outputs_class: Sequence[Array], outputs_coord: Sequence[Array]) -> list[dict[str, Array]]:
  """
  This is a helper function to structure auxiliary loss outputs.

  This is a workaround to make torchscript happy, as torchscript
  doesn't support dictionary with non-homogeneous values, such
  as a dict having both a Tensor and a list.
  """
  # this is a workaround to make torchscript happy, as torchscript
  # doesn't support dictionary with non-homogeneous values, such
  # as a dict having both a Tensor and a list.
  return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class, outputs_coord)]

import typing as tp

from maxtext.common_types import Array


def _set_aux_loss(
    outputs_class: tp.Sequence[Array], outputs_coord: tp.Sequence[Array]
) -> list[dict[str, Array]]:
  """A helper function to structure auxiliary loss outputs.

  This is a workaround to make torchscript happy, as torchscript
  doesn't support dictionary with non-homogeneous values, such
  as a dict having both a Tensor and a list.
  """
  return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

import os
from typing import Dict, Optional

from jax import Array


def _process_flash_attention_kwargs(
    query_length: int,
    key_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: Optional[bool] = None,
    s_aux: Optional[Array] = None,
    supports_mapping: Optional[Dict[str, bool]] = None,
    **kwargs,
) -> Dict:
  """Processes kwargs for the flash attention function.

  Returns a set of kwargs that are passed down to the according flash attention
  function based on requested features and whether it is supported. This depends
  on the version and kernel implementation which is dynamically configured at
  `lazy_import_flash_attention`. The (un)supported features can be inspected in
  `supports_mapping`, see `_lazy_define_process_function` for more details.

  Args:
    query_length: Length of the query states.
    key_length: Length of the key states.
    is_causal: Whether we perform causal (decoder) attention or full attention.
    dropout: Attention dropout.
    softmax_scale: The scaling of QK^T before applying softmax. Default to `1 /
      sqrt(head_dim)`.
    sliding_window: The size of the sliding window, i.e. we look at a max of
      `sliding_window` tokens back.
    use_top_left_mask: Deprecated behavior of older versions of flash attention
      requiring different masking.
    softcap: Softcap for the attention logits, used e.g. in gemma2.
    deterministic: Determines if the deterministic option introduced in
      flash_attn>=2.4.1 is enabled.
    s_aux: Attention sink auxiliary that adds a `bias` to the attention
      calculation via an additional head.
    supports_mapping: A dictionary indicating which optional kwargs are
      supported.
    **kwargs: Additional keyword arguments.

  Returns:
    A dict of kwargs that are requested and supported.
  """
  flash_kwargs = {
      "causal": is_causal and not (use_top_left_mask and query_length == 1),
      "softmax_scale": softmax_scale,
  }

  if supports_mapping["dropout_p"]:
    flash_kwargs["dropout_p"] = dropout

  if supports_mapping["window_size"] and sliding_window is not None and key_length > sliding_window:
    flash_kwargs["window_size"] = (sliding_window, sliding_window)

  if supports_mapping["deterministic"]:
    flash_kwargs["deterministic"] = (
        deterministic if deterministic is not None else os.getenv("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"
    )

  if supports_mapping["softcap"] and softcap is not None:
    flash_kwargs["softcap"] = softcap

  # Only within kernel implementation atm
  if supports_mapping["s_aux"] and s_aux is not None:
    flash_kwargs["s_aux"] = s_aux

  return flash_kwargs

import inspect
from functools import partial
from typing import Any, Callable

# In the JAX/MaxText version of this file, `_process_flash_attention_kwargs` and
# `_hf_api_to_flash_mapping` are assumed to be defined at the module level.

def _lazy_define_process_function(flash_function: Callable[..., Any]) -> partial:
  """Defines a function that processes flash attention kwargs based on support.

  Depending on the version and kernel, some features are not supported. Due to
  limitations in JAX's JIT compilation, we opt to statically determine which
  (optional) kwarg parameters are supported within `_process_flash_attention_kwargs`.

  NOTE: While all supported kwargs are marked as `True`, everything else is
        marked as `False`. This might be confusing for kwargs that we use in
        any case, e.g. `is_causal`.

  Args:
    flash_function: The flash attention function to inspect for supported
      parameters.

  Returns:
    A partial function of `_process_flash_attention_kwargs` with the
    `supports_mapping` argument pre-filled.
  """
  flash_parameters = inspect.signature(flash_function).parameters
  process_parameters = inspect.signature(_process_flash_attention_kwargs).parameters

  supports_mapping = {}
  for param in process_parameters:
    fa_param = _hf_api_to_flash_mapping.get(param, param)
    supports_mapping[fa_param] = fa_param in flash_parameters

  return partial(_process_flash_attention_kwargs, supports_mapping=supports_mapping)

import jax.numpy as jnp
from typing import Tuple

from MaxText.common_types import Array


def _pad_input(hidden_states: Array, indices: Array, batch: int, seqlen: int) -> Array:
  """pad_input function for flash attention variants that do not have them within their pkg themselves, e.g. fa3.

  Arguments:
      hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
      indices: (total_nnz), the indices that represent the non-masked tokens of the original padded input sequence.
      batch: int, batch size for the padded sequence.
      seqlen: int, maximum sequence length for the padded sequence.

  Return:
      hidden_states: (batch, seqlen, ...)
  """
  dim = hidden_states.shape[1:]
  output = jnp.zeros((batch * seqlen, *dim), dtype=hidden_states.dtype)
  output = output.at[indices].set(hidden_states)
  return output.reshape(batch, seqlen, *dim)
from typing import Any
import jax.numpy as jnp


def _is_jax(x: Any) -> bool:
  """Tests if x is a JAX ndarray."""
  return isinstance(x, jnp.ndarray)
from typing import Any

import jax.numpy as jnp


def is_jax_tensor(x: Any) -> bool:
  """Tests if `x` is a Jax tensor or not."""
  return isinstance(x, jnp.ndarray)

from typing import Any


def _is_mlx(x: Any) -> bool:
  """Checks if an object is an MLX array."""
  import mlx.core as mx

  return isinstance(x, mx.array)

from typing import Any

def _is_jax(x: Any) -> bool:
  """Helper function to check if a variable is a JAX tensor."""
  import jax.numpy as jnp

  return isinstance(x, jnp.ndarray)
from typing import Any
import jax.numpy as jnp


# Reused from Qwen3ForCausalLM.modeling_utils._is_jax
def _is_jax(x: Any) -> bool:
  """A helper function to check if a given variable is a JAX tensor."""
  return isinstance(x, jnp.ndarray)
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
""" RoPE utilities"""
from typing import Optional, Set

from MaxText import max_logging


def _check_received_keys(
    rope_type: str,
    received_keys: Set[str],
    required_keys: Set[str],
    optional_keys: Optional[Set[str]] = None,
    ignore_keys: Optional[Set[str]] = None,
):
  """Compare the received keys in `config.rope_scaling` against expected keys.

  Args:
    rope_type: The type of RoPE scaling being used.
    received_keys: The set of keys provided in the configuration.
    required_keys: The set of keys that must be provided.
    optional_keys: An optional set of keys that are allowed but not required.
    ignore_keys: An optional set of keys to ignore during validation.
  """
  # BC: "rope_type" was originally "type" -- let's check for "rope_type" when
  # "type" is present
  if "type" in received_keys:
    received_keys -= {"type"}
    required_keys.add("rope_type")

  # Some models need to store model-specific keys, and we don't want to throw
  # warning at them
  if ignore_keys is not None:
    received_keys -= ignore_keys

  missing_keys = required_keys - received_keys
  if missing_keys:
    raise KeyError(
        "Missing required keys in `rope_scaling` for 'rope_type'="
        f"'{rope_type}': {missing_keys}"
    )

  if optional_keys is not None:
    unused_keys = received_keys - required_keys - optional_keys
  else:
    unused_keys = received_keys - required_keys
  if unused_keys:
    max_logging.warning(
        "Unrecognized keys in `rope_scaling` for 'rope_type'="
        f"'{rope_type}': {unused_keys}"
    )

from typing import Optional, Set

from maxtext.common_types import Config


# Reused from Qwen3ForCausalLM.modeling_utils._check_received_keys
def _validate_default_rope_parameters(config: Config, ignore_keys: Optional[Set[str]] = None):
  """Validates the default RoPE parameters."""
  rope_scaling = config.rope_scaling
  rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
  required_keys = {"rope_type"}
  received_keys = set(rope_scaling.keys())
  _check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)

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
"""
This file is a 1-to-1 copy of the validation functions from transformers.
The only changes are:
- `PretrainedConfig` is replaced by `Config`
- `logging` is replaced by `max_logging`
"""

from typing import Optional, Set

from MaxText import max_logging
from MaxText.common_types import Config
# The following import assumes that `_check_received_keys` is defined in the same module.
from .rope_utils import _check_received_keys


def _validate_llama3_parameters(config: Config, ignore_keys: Optional[Set] = None):
  """Validates the Llama3 parameters."""
  rope_scaling = config.rope_scaling
  rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
  required_keys = {"rope_type", "factor", "original_max_position_embeddings", "low_freq_factor", "high_freq_factor"}
  received_keys = set(rope_scaling.keys())
  _check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)

  factor = rope_scaling["factor"]
  if factor is None or not isinstance(factor, float) or factor < 1.0:
    max_logging.warning(f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")

  low_freq_factor = rope_scaling["low_freq_factor"]
  high_freq_factor = rope_scaling["high_freq_factor"]
  if low_freq_factor is None or not isinstance(low_freq_factor, float):
    max_logging.warning(f"`rope_scaling`'s low_freq_factor field must be a float, got {low_freq_factor}")
  if high_freq_factor is None or not isinstance(high_freq_factor, float):
    max_logging.warning(f"`rope_scaling`'s high_freq_factor field must be a float, got {high_freq_factor}")
  if high_freq_factor <= low_freq_factor:
    max_logging.warning(
        "`rope_scaling`'s high_freq_factor field must be greater than low_freq_factor, got high_freq_factor="
        f"{high_freq_factor} and low_freq_factor={low_freq_factor}"
    )

  original_max_position_embeddings = rope_scaling["original_max_position_embeddings"]
  if original_max_position_embeddings is None or not isinstance(original_max_position_embeddings, int):
    max_logging.warning(
        "`rope_scaling`'s original_max_position_embeddings field must be an integer, got "
        f"{original_max_position_embeddings}"
    )
  if original_max_position_embeddings >= config.max_position_embeddings:
    max_logging.warning(
        "`rope_scaling`'s original_max_position_embeddings field must be less than max_position_embeddings, got "
        f"{original_max_position_embeddings} and max_position_embeddings={config.max_position_embeddings}"
    )

from typing import Optional, Set

from MaxText.common_types import Config
from MaxText import max_logging
from MaxText.layers.rope import _check_received_keys


def _validate_longrope_parameters(config: Config, ignore_keys: Optional[Set[str]] = None):
  """Validates the longrope parameters in the config."""
  rope_scaling = config.rope_scaling
  rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
  required_keys = {"rope_type", "short_factor", "long_factor"}
  # TODO(joao): update logic for the inclusion of `original_max_position_embeddings`
  optional_keys = {"attention_factor", "factor", "original_max_position_embeddings"}
  received_keys = set(rope_scaling.keys())
  _check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)

  partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
  head_dim = getattr(config, "head_dim", config.emb_dim // config.num_query_heads)
  dim = int(head_dim * partial_rotary_factor)

  short_factor = rope_scaling.get("short_factor")
  if short_factor is not None:
    # The original logic is `if not isinstance(short_factor, list) and all(...)`.
    # This triggers for non-list iterables like tuples. Keeping it for equivalence.
    if not isinstance(short_factor, list) and all(isinstance(x, (int, float)) for x in short_factor):
      max_logging.warning(f"`rope_scaling`'s short_factor field must be a list of numbers, got {short_factor}")
    if len(short_factor) != dim // 2:
      max_logging.warning(f"`rope_scaling`'s short_factor field must have length {dim // 2}, got {len(short_factor)}")

  long_factor = rope_scaling.get("long_factor")
  if long_factor is not None:
    if not isinstance(long_factor, list) and all(isinstance(x, (int, float)) for x in long_factor):
      max_logging.warning(f"`rope_scaling`'s long_factor field must be a list of numbers, got {long_factor}")
    if len(long_factor) != dim // 2:
      max_logging.warning(f"`rope_scaling`'s long_factor field must have length {dim // 2}, got {len(long_factor)}")

  # Handle Phi3 divergence: prefer the use of `attention_factor` and/or `factor` over
  # `original_max_position_embeddings` to compute internal variables. The latter lives outside `rope_scaling` and is
  # unique to longrope (= undesirable)
  if hasattr(config, "original_max_position_embeddings"):
    # MaxText logging doesn't have warning_once, using regular warning.
    max_logging.warning(
        "This model has set a `original_max_position_embeddings` field, to be used together with "
        "`max_position_embeddings` to determine a scaling factor. Please set the `factor` field of `rope_scaling`"
        "with this ratio instead -- we recommend the use of this field over `original_max_position_embeddings`, "
        "as it is compatible with most model architectures."
    )
  else:
    factor = rope_scaling.get("factor")
    if factor is None:
      max_logging.warning("Missing required keys in `rope_scaling`: 'factor'")
    elif not isinstance(factor, float) or factor < 1.0:
      max_logging.warning(f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")

    attention_factor = rope_scaling.get("attention_factor")
    if attention_factor is not None:
      if not isinstance(attention_factor, float) or attention_factor < 0.0:
        max_logging.warning(
            f"`rope_scaling`'s attention_factor field must be a float greater than 0, got {attention_factor}"
        )

from packaging import version

# It is assumed that ACCELERATE_MIN_VERSION, _accelerate_available, and
# _accelerate_version are defined in the same file, as in the source.


def is_accelerate_available(min_version: str = ACCELERATE_MIN_VERSION) -> bool:
  """Checks if accelerate is available and its version is >= min_version."""
  return _accelerate_available and version.parse(_accelerate_version) >= version.parse(min_version)

# Copyright 2025 The Fairseq Authors and the HuggingFace Inc. team. All rights reserved.
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
This file is a 1-to-1 copy of the PyTorch original file at
https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_flash_attention_utils.py,
with the following changes:
- Replaced torch with jax.numpy
- Removed all torch.compile specific logic
- Removed all lazy loading logic
- Removed all FA3 specific logic
- Removed all NPU specific logic
- Removed all PEFT specific logic
"""
from maxtext.common_types import Array


def _index_first_axis(tensor: Array, indices: Array) -> Array:
  """A local implementation of the PyTorch indexing operation `tensor[indices]` on the first axis.

  This is done after flattening the first two dimensions of the tensor. This is functionally equivalent to
  FA2's `index_first_axis` and replaces the need to import it.

  Args:
    tensor: The tensor to be indexed.
    indices: The indices to use for indexing.

  Returns:
    The indexed tensor.
  """
  # The input tensor is expected to be of shape (batch, seq_len, ...). We flatten the first
  # two dimensions to get (total_tokens, ...) before indexing.
  reshaped_tensor = tensor.reshape(-1, *tensor.shape[2:])
  return reshaped_tensor[indices]

# Copyright 2025 The Fairseq Authors and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may not use this file except in compliance with the License.
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
Flash Attention utilities for JAX.
"""

from typing import Optional, Tuple
import jax.numpy as jnp
from maxtext.common_types import Array


# Ported from transformers.models.llama.modeling_llama._index_first_axis
def _index_first_axis(tensor: Array, indices: Array) -> Array:
  """
  A local implementation of the PyTorch indexing operation `tensor[indices]` on the first axis,
  after flattening the first two dimensions of the tensor. This is functionally equivalent to
  FA2's `index_first_axis` and replaces the need to import it.
  """
  # The input tensor is expected to be of shape (batch, seq_len, ...). We flatten the first
  # two dimensions to get (total_tokens, ...) before indexing.
  reshaped_tensor = tensor.reshape(-1, *tensor.shape[2:])
  return reshaped_tensor[indices]


def _unpad_input(
    hidden_states: Array,
    attention_mask: Array,
    unused_mask: Optional[Array] = None,
) -> Tuple[Array, Array, Array, int, Array]:
  """
  unpad_input function for flash attention variants that do not have them within their pkg themselves, e.g. fa3.

  Arguments:
      hidden_states: (batch, seqlen, ...)
      attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
      unused_mask: (batch, seqlen), bool / int, 1 means the element is allocated but unused.

  Return:
      hidden_states: (total_nnz, ...), where total_nnz = number of tokens selected in attention_mask + unused_mask.
      indices: (total_nnz), the indices of masked tokens from the flattened input sequence.
      cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
      max_seqlen_in_batch: int
      seqused: (batch), returns the number of tokens selected in attention_mask + unused_mask.
  """
  all_masks = (
      attention_mask + unused_mask if unused_mask is not None else attention_mask
  )
  seqlens_in_batch = jnp.sum(all_masks, axis=-1, dtype=jnp.int32)
  used_seqlens_in_batch = jnp.sum(attention_mask, axis=-1, dtype=jnp.int32)

  # In JAX, jnp.nonzero requires a size for JIT compilation.
  # We can calculate the exact size by summing the mask.
  num_valid_tokens = jnp.sum(all_masks).astype(jnp.int32)
  indices = jnp.nonzero(all_masks.flatten(), size=num_valid_tokens)[0]

  max_seqlen_in_batch = int(jnp.max(seqlens_in_batch))
  cu_seqlens = jnp.pad(
      jnp.cumsum(seqlens_in_batch, axis=0, dtype=jnp.int32), ((1, 0),)
  )

  return (
      _index_first_axis(hidden_states, indices),
      indices,
      cu_seqlens,
      max_seqlen_in_batch,
      used_seqlens_in_batch,
  )

from typing import Optional, Callable, Tuple

# Reused from Qwen3ForCausalLM.layers._lazy_imports
from Qwen3ForCausalLM.layers import _lazy_imports
# Reused from Qwen3ForCausalLM.modeling_utils._lazy_define_process_function
from Qwen3ForCausalLM.modeling_utils import _lazy_define_process_function


# Global variables for caching, mirroring the PyTorch implementation's architecture.
_flash_fn: Optional[Callable] = None
_flash_varlen_fn: Optional[Callable] = None
_pad_fn: Optional[Callable] = None
_unpad_fn: Optional[Callable] = None
_process_flash_kwargs_fn: Optional[Callable] = None


def lazy_import_flash_attention(
    implementation: Optional[str],
) -> Tuple[Tuple[Callable, Callable, Callable, Callable], Callable]:
  """
  Lazy loading flash attention and returning the respective functions + flags back.

  NOTE: For fullgraph, this needs to be called before compile while no fullgraph can
        can work without preloading. See `_check_and_adjust_attn_implementation` in `modeling_utils`.
  """
  global _flash_fn, _flash_varlen_fn, _pad_fn, _unpad_fn
  if any(k is None for k in [_flash_fn, _flash_varlen_fn, _pad_fn, _unpad_fn]):
    _flash_fn, _flash_varlen_fn, _pad_fn, _unpad_fn = _lazy_imports(
        implementation
    )

  global _process_flash_kwargs_fn
  if _process_flash_kwargs_fn is None:
    _process_flash_kwargs_fn = _lazy_define_process_function(_flash_varlen_fn)

  return (_flash_fn, _flash_varlen_fn, _pad_fn, _unpad_fn), _process_flash_kwargs_fn
BS4_IMPORT_ERROR = """
{0} requires the Beautiful Soup library but it was not found in your environment. You can install it with pip:
`pip install beautifulsoup4`. Please note that you may need to restart your runtime after installation.
"""CYTHON_IMPORT_ERROR = """
{0} requires the Cython library but it was not found in your environment. You can install it with pip: `pip install
Cython`. Please note that you may need to restart your runtime after installation.
"""FLAX_IMPORT_ERROR = """
{0} requires the FLAX library but it was not found in your environment. Check out the instructions on the
installation page: https://github.com/google/flax and follow the ones that match your environment.
Please note that you may need to restart your runtime after installation.
"""FTFY_IMPORT_ERROR = """
{0} requires the ftfy library but it was not found in your environment. Check out the instructions on the
installation section: https://github.com/rspeer/python-ftfy/tree/master#installing and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
"""KERAS_NLP_IMPORT_ERROR = """
{0} requires the keras_nlp library but it was not found in your environment. You can install it with pip.
Please note that you may need to restart your runtime after installation.
"""# docstyle-ignore
NATTEN_IMPORT_ERROR = """
{0} requires the natten library but it was not found in your environment. You can install it by referring to:
shi-labs.com/natten . You can also install it with pip (may take longer to build):
`pip install natten`. Please note that you may need to restart your runtime after installation.
"""RICH_IMPORT_ERROR = """
{0} requires the rich library but it was not found in your environment. You can install it with pip: `pip install
rich`. Please note that you may need to restart your runtime after installation.
"""SCIPY_IMPORT_ERROR = """
{0} requires the scipy library but it was not found in your environment. You can install it with pip:
`pip install scipy`. Please note that you may need to restart your runtime after installation.
"""TORCHAUDIO_IMPORT_ERROR = """
{0} requires the torchaudio library but it was not found in your environment. Please install it and restart your
runtime.
"""
# docstyle-ignore
TORCHVISION_IMPORT_ERROR = """
{0} requires the Torchvision library but it was not found in your environment. Check out the instructions on the
installation page: https://pytorch.org/get-started/locally/ and follow the ones that match your environment.
Please note that you may need to restart your runtime after installation.
"""

# Copyright 2024 The MaxText Authors.
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

# docstyle-ignore
VISION_IMPORT_ERROR = """
{0} requires the PIL library but it was not found in your environment. You can install it with pip:
`pip install pillow`. Please note that you may need to restart your runtime after installation.
"""

def is_bs4_available() -> bool:
  """Check if Beautiful Soup 4 is available."""
  return _bs4_available
import importlib.util


def is_cython_available() -> bool:
  """Returns True if Cython is available."""
  return importlib.util.find_spec("pyximport") is not None
import importlib.util

_decord_available = importlib.util.find_spec("decord") is not None


def is_decord_available() -> bool:
  """Returns True if decord is available."""
  return _decord_available
def is_detectron2_available() -> bool:
  """Returns True if detectron2 is available."""
  return _detectron2_available
# The global variable `_g2p_en_available` is assumed to be defined in the same module.

def is_g2p_en_available() -> bool:
  """Checks if g2p_en is available."""
  return _g2p_en_available

def is_librosa_available() -> bool:
  """Checks if librosa is available."""
  return _librosa_available

def is_natten_available() -> bool:
  """Check if natten is available."""
  return _natten_available

def is_pydantic_available() -> bool:
  """Checks if pydantic is available."""
  return _pydantic_available

def is_pytesseract_available() -> bool:
  """Check if pytesseract is available."""
  return _pytesseract_available

# No imports needed for this function in a JAX environment.

def is_pytorch_quantization_available():
  """
  Checks for the availability of the `pytorch_quantization` library.

  This function is included for compatibility with the original PyTorch repository
  but will always return False in a JAX environment, as `pytorch_quantization`
  is a PyTorch-specific library.
  """
  return False

import importlib.metadata

try:
  importlib.metadata.version("tensorflow_probability")
  _tensorflow_probability_available = True
except importlib.metadata.PackageNotFoundError:
  _tensorflow_probability_available = False


def is_tensorflow_probability_available() -> bool:
  """Checks if TensorFlow Probability is available."""
  return _tensorflow_probability_available

def is_timm_available():
  """Checks if the timm library is available."""
  return _timm_available

def is_tokenizers_available() -> bool:
  """Returns True if the tokenizers library is available."""
  return _tokenizers_available
# The variable `_torchaudio_available` is assumed to be defined at the module level.
# This is typically done by checking for package availability using `importlib`.

def is_torchaudio_available() -> bool:
  """Checks if torchaudio is available in the environment."""
  return _torchaudio_available
import importlib.util
import importlib.metadata
from functools import lru_cache

from MaxText import max_logging as logging


logger = logging.get_logger(__name__)


@lru_cache
def is_vision_available():
  """Checks if the PIL library is available."""
  _pil_available = importlib.util.find_spec("PIL") is not None
  if _pil_available:
    try:
      package_version = importlib.metadata.version("Pillow")
    except importlib.metadata.PackageNotFoundError:
      try:
        package_version = importlib.metadata.version("Pillow-SIMD")
      except importlib.metadata.PackageNotFoundError:
        return False
    logger.debug(f"Detected PIL version {package_version}")
  return _pil_available

import importlib.util
import importlib.metadata
from typing import Union

# The following helper function and global variable are adapted from the full source file
# to provide the necessary context for the `is_mlx_available` function.

def _is_package_available(pkg_name: str) -> bool:
  """
  Check if a package is available.

  This is a simplified version of the original helper function, tailored for this specific use case.
  """
  package_exists = importlib.util.find_spec(pkg_name) is not None
  if package_exists:
    try:
      # Check for version to ensure it's a real package, not a local directory.
      importlib.metadata.version(pkg_name)
    except importlib.metadata.PackageNotFoundError:
      package_exists = False
  return package_exists


_mlx_available = _is_package_available("mlx")


def is_mlx_available() -> bool:
  """Checks if the 'mlx' library is installed and available."""
  return _mlx_available
# docstyle-ignore
TF_IMPORT_ERROR_WITH_FLAX = """
{0} requires the TensorFlow library but it was not found in your environment.
However, we were able to find a JAX/Flax installation. JAX/Flax classes do not begin
with "TF", but are otherwise identically named to our TF classes.
If you want to use JAX/Flax, please use those classes instead!

If you really do want to use TensorFlow, please follow the instructions on the
installation page https://www.tensorflow.org/install that match your environment.
"""
import importlib.util
import importlib.metadata

ccl_version = "N/A"
# In JAX, the equivalent for Intel's oneCCL is the jax-oneccl-plugin.
_is_ccl_available = importlib.util.find_spec("jax_oneccl_plugin") is not None
try:
    ccl_version = importlib.metadata.version("jax-oneccl-plugin")
except importlib.metadata.PackageNotFoundError:
    _is_ccl_available = False

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
"""
Import utilities: Utilities related to imports and our lazy inits.
"""
import importlib.util
from functools import lru_cache


# Logic reused from Qwen3ForCausalLM.modeling_utils._is_ccl_available
@lru_cache
def is_ccl_available():
  """Checks for the availability of the `jax-oneccl-plugin` library."""
  return importlib.util.find_spec("jax_oneccl_plugin") is not None

# The function `_is_package_available` is assumed to be defined in the same file.
_jinja_available = _is_package_available("jinja2")

import importlib.metadata

try:
  importlib.metadata.version("mistral_common")
  _mistral_common_available = True
except importlib.metadata.PackageNotFoundError:
  _mistral_common_available = False

# The _mistral_common_available variable is defined at the module level.
# from .. import _mistral_common_available

def is_mistral_common_available() -> bool:
  """Check if mistral_common is available."""
  return _mistral_common_available

def is_pretty_midi_available():
  return _pretty_midi_available

import importlib.machinery
import importlib.metadata
import importlib.util
from typing import Union

# The following imports are for the `_is_package_available` utility function.
# In a real MaxText environment, logging would likely be `from MaxText import max_logging as logging`.
import logging


logger = logging.getLogger(__name__)


def _is_package_available(pkg_name: str, return_version: bool = False) -> Union[tuple[bool, str], bool]:
  """Check if a package is available and optionally return its version."""
  # Check if the package spec exists and grab its version to avoid importing a local directory
  package_exists = importlib.util.find_spec(pkg_name) is not None
  package_version = "N/A"
  if package_exists:
    try:
      # TODO: Once python 3.9 support is dropped, `importlib.metadata.packages_distributions()`
      # should be used here to map from package name to distribution names
      # e.g. PIL -> Pillow, Pillow-SIMD; quark -> amd-quark; onnxruntime -> onnxruntime-gpu.
      # `importlib.metadata.packages_distributions()` is not available in Python 3.9.

      # Primary method to get the package version
      package_version = importlib.metadata.version(pkg_name)
    except importlib.metadata.PackageNotFoundError:
      if pkg_name == "quark":
        # TODO: remove once `importlib.metadata.packages_distributions()` is supported.
        try:
          package_version = importlib.metadata.version("amd-quark")
        except Exception:
          package_exists = False
      elif pkg_name == "triton":
        try:
          # import triton works for both linux and windows
          package = importlib.import_module(pkg_name)
          package_version = getattr(package, "__version__", "N/A")
        except Exception:
          try:
            package_version = importlib.metadata.version("pytorch-triton")  # pytorch-triton
          except Exception:
            package_exists = False
      else:
        # For other packages, don't attempt the fallback and set as not available
        package_exists = False
    logger.debug(f"Detected {pkg_name} version: {package_version}")
  if return_version:
    return package_exists, package_version
  else:
    return package_exists


_rjieba_available = _is_package_available("rjieba")

def is_speech_available() -> bool:
  """Checks if a speech processing library is available."""
  # For now this depends on torchaudio but the exact dependency might evolve in the future.
  return _torchaudio_available

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
_tensorflow_text_available = _is_package_available("tensorflow_text")

# In a JAX environment, TensorFlow is not considered available.
_tf_available: bool = False


def is_tf_available() -> bool:
  """Returns whether TensorFlow is available."""
  return _tf_available

import importlib.util

# Global variable to cache the availability of torchcodec
_torchcodec_available = importlib.util.find_spec("torchcodec") is not None


def is_torchcodec_available() -> bool:
  """Checks if torchcodec is available."""
  return _torchcodec_available

from typing import Any, Callable, Dict

# The following dependencies are assumed to be defined elsewhere in the JAX project,
# analogous to the original full_file_code:
# - infer_framework_from_repr
# - is_torch_tensor
# - is_tf_tensor
# - is_jax_tensor
# - is_numpy_array
# - is_mlx_array


def _get_frameworks_and_test_func(x: Any) -> Dict[str, Callable]:
  """
  Returns an (ordered since we are in Python 3.7+) dictionary framework to test function, which places the framework
  we can guess from the repr first, then Numpy, then the others.
  """
  framework_to_test = {
      "pt": is_torch_tensor,
      "tf": is_tf_tensor,
      "jax": is_jax_tensor,
      "np": is_numpy_array,
      "mlx": is_mlx_array,
  }
  preferred_framework = infer_framework_from_repr(x)
  # We will test this one first, then numpy, then the others.
  frameworks = [] if preferred_framework is None else [preferred_framework]
  if preferred_framework != "np":
    frameworks.append("np")
  frameworks.extend(
      [f for f in framework_to_test if f not in [preferred_framework, "np"]]
  )
  return {f: framework_to_test[f] for f in frameworks}

from __future__ import annotations
import jax.numpy as jnp
from jaxtyping import Array


def weighting_function(max_num_bins: int, up: Array, reg_scale: int) -> Array:
  """
    Generates the non-uniform Weighting Function W(n) for bounding box regression.

    Args:
        max_num_bins (int): Max number of the discrete bins.
        up (Array): Controls upper bounds of the sequence,
                     where maximum offset is ±up * H / W.
        reg_scale (float): Controls the curvature of the Weighting Function.
                           Larger values result in flatter weights near the central axis W(max_num_bins/2)=0
                           and steeper weights at both ends.
    Returns:
        Array: Sequence of Weighting Function.
    """
  upper_bound1 = jnp.abs(up[0]) * jnp.abs(reg_scale)
  upper_bound2 = jnp.abs(up[0]) * jnp.abs(reg_scale) * 2
  step = (upper_bound1 + 1) ** (2 / (max_num_bins - 2))

  # Vectorize the list comprehensions from the original PyTorch code
  left_indices = jnp.arange(max_num_bins // 2 - 1, 0, -1)
  left_values = -((step) ** left_indices) + 1

  right_indices = jnp.arange(1, max_num_bins // 2)
  right_values = (step) ** right_indices - 1

  # Prepare all parts as JAX arrays for concatenation
  dtype = up.dtype
  zero_val = jnp.zeros((1,), dtype=dtype)
  lower_bound_val = jnp.array([-upper_bound2], dtype=dtype)
  upper_bound_val = jnp.array([upper_bound2], dtype=dtype)

  # Concatenate all parts to form the final sequence
  values = jnp.concatenate(
      [
          lower_bound_val,
          left_values.astype(dtype),
          zero_val,
          right_values.astype(dtype),
          upper_bound_val,
      ],
      axis=0,
  )
  return values

from typing import Tuple
import jax.numpy as jnp

from maxtext.common_types import Array
# from Qwen3ForCausalLM.modeling_utils.weighting_function is used for weighting_function
from Qwen3ForCausalLM.modeling_utils import weighting_function


def translate_gt(gt: Array, max_num_bins: int, reg_scale: int, up: Array) -> Tuple[Array, Array, Array]:
    """
    Decodes bounding box ground truth (GT) values into distribution-based GT representations.

    This function maps continuous GT values into discrete distribution bins, which can be used
    for regression tasks in object detection models. It calculates the indices of the closest
    bins to each GT value and assigns interpolation weights to these bins based on their proximity
    to the GT value.

    Args:
        gt (Array): Ground truth bounding box values, shape (N, ).
        max_num_bins (int): Maximum number of discrete bins for the distribution.
        reg_scale (float): Controls the curvature of the Weighting Function.
        up (Array): Controls the upper bounds of the Weighting Function.

    Returns:
        tuple[Array, Array, Array]:
            - indices (Array): Index of the left bin closest to each GT value, shape (N, ).
            - weight_right (Array): Weight assigned to the right bin, shape (N, ).
            - weight_left (Array): Weight assigned to the left bin, shape (N, ).
    """
    gt = gt.reshape(-1)
    function_values = weighting_function(max_num_bins, up, reg_scale)

    # Find the closest left-side indices for each value
    diffs = function_values[None, :] - gt[:, None]
    mask = diffs <= 0
    closest_left_indices = jnp.sum(mask, axis=1) - 1

    # Calculate the weights for the interpolation
    indices = closest_left_indices.astype(jnp.float32)

    weight_right = jnp.zeros_like(indices)
    weight_left = jnp.zeros_like(indices)

    valid_idx_mask = (indices >= 0) & (indices < max_num_bins)
    valid_indices = indices[valid_idx_mask].astype(jnp.int32)

    # Obtain distances
    left_values = function_values[valid_indices]
    right_values = function_values[valid_indices + 1]

    left_diffs = jnp.abs(gt[valid_idx_mask] - left_values)
    right_diffs = jnp.abs(right_values - gt[valid_idx_mask])

    # Valid weights
    valid_weight_right = left_diffs / (left_diffs + right_diffs)
    weight_right = weight_right.at[valid_idx_mask].set(valid_weight_right)
    weight_left = weight_left.at[valid_idx_mask].set(1.0 - valid_weight_right)

    # Invalid weights (out of range)
    invalid_idx_mask_neg = indices < 0
    weight_right = jnp.where(invalid_idx_mask_neg, 0.0, weight_right)
    weight_left = jnp.where(invalid_idx_mask_neg, 1.0, weight_left)
    indices = jnp.where(invalid_idx_mask_neg, 0.0, indices)

    invalid_idx_mask_pos = indices >= max_num_bins
    weight_right = jnp.where(invalid_idx_mask_pos, 1.0, weight_right)
    weight_left = jnp.where(invalid_idx_mask_pos, 0.0, weight_left)
    indices = jnp.where(invalid_idx_mask_pos, max_num_bins - 0.1, indices)

    return indices, weight_right, weight_left

import jax
import jax.numpy as jnp


def _upcast(t: jax.Array) -> jax.Array:
  """Protects from numerical overflows in multiplications by upcasting to the equivalent higher type."""
  if jnp.issubdtype(t.dtype, jnp.floating):
    return t if t.dtype in (jnp.float32, jnp.float64) else t.astype(jnp.float32)
  else:
    return t if t.dtype in (jnp.int32, jnp.int64) else t.astype(jnp.int32)

import jax.numpy as jnp
from jax import Array


def _upcast(t: Array) -> Array:
  """
  Protects from numerical overflows in multiplications by upcasting to the equivalent higher type.
  """
  if jnp.issubdtype(t.dtype, jnp.floating):
    return t if t.dtype in (jnp.float32, jnp.float64) else t.astype(jnp.float32)
  else:
    return t if t.dtype in (jnp.int32, jnp.int64) else t.astype(jnp.int32)


def box_area(boxes: Array) -> Array:
  """
    Computes the area of a set of bounding boxes, which are specified by its (x1, y1, x2, y2) coordinates.

    Args:
        boxes (`jax.Array` of shape `(number_of_boxes, 4)`):
            Boxes for which the area will be computed. They are expected to be in (x1, y1, x2, y2) format with `0 <= x1
            < x2` and `0 <= y1 < y2`.

    Returns:
        `jax.Array`: a tensor containing the area for each box.
    """
  boxes = _upcast(boxes)
  return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

from typing import List
import jax.numpy as jnp
from jax import Array

# Re-used from Qwen3ForCausalLM.structures.NestedTensor
from Qwen3ForCausalLM.structures import NestedTensor
# Re-used from Qwen3ForCausalLM.utils._max_by_axis
from Qwen3ForCausalLM.utils import _max_by_axis


def nested_tensor_from_tensor_list(tensor_list: List[Array]) -> NestedTensor:
  """
    Pads a list of tensors to the same shape and combines them into a single tensor.
    Also returns a boolean mask indicating the padded areas.
    """
  if tensor_list[0].ndim == 3:
    # Re-used from Qwen3ForCausalLM.utils._max_by_axis
    max_size = _max_by_axis([list(img.shape) for img in tensor_list])
    batch_shape = [len(tensor_list)] + max_size
    batch_size, _, height, width = batch_shape
    dtype = tensor_list[0].dtype
    tensor = jnp.zeros(batch_shape, dtype=dtype)
    mask = jnp.ones((batch_size, height, width), dtype=jnp.bool_)

    for i, img in enumerate(tensor_list):
      tensor = tensor.at[i, :img.shape[0], :img.shape[1], :
                         img.shape[2]].set(img)
      mask = mask.at[i, :img.shape[1], :img.shape[2]].set(False)
  else:
    raise ValueError("Only 3-dimensional tensors are supported")
  # Re-used from Qwen3ForCausalLM.structures.NestedTensor
  return NestedTensor(tensor, mask)

from . import _keras_nlp_available, is_tensorflow_text_available


def is_keras_nlp_available() -> bool:
  """Checks if KerasNLP and its dependencies are available."""
  return is_tensorflow_text_available() and _keras_nlp_available


from typing import Any, Callable, List, Union

# Note: The `BACKENDS_MAPPING` dictionary and `Backend` class are defined
# in the same file and are assumed to be available in the execution context.


def requires(*, backends: tuple = ()) -> Callable:
  """A decorator to specify backend requirements for a function or class.

  This decorator enables two things:
  - Attaching a `__backends` tuple to an object to see what are the necessary
    backends for it to execute correctly without instantiating it.
  - The '@requires' string is used to dynamically import objects.

  Args:
    backends: A tuple of strings specifying the required backends. Each string
      can be a simple backend name (e.g., 'torch') or a versioned requirement
      (e.g., 'torch>=2.0').

  Returns:
    A decorator that attaches the backend requirements to the decorated object.
  """

  if not isinstance(backends, tuple):
    raise TypeError("Backends should be a tuple.")

  applied_backends: List[Union[str, "Backend"]] = []
  for backend in backends:
    if backend in BACKENDS_MAPPING:
      applied_backends.append(backend)
    else:
      if any(key in backend for key in ["=", "<", ">"]):
        applied_backends.append(Backend(backend))
      else:
        raise ValueError(
            "Backend should be defined in the BACKENDS_MAPPING. "
            f"Offending backend: {backend}"
        )

  def inner_fn(fun: Callable) -> Callable:
    fun.__backends = applied_backends
    return fun

  return inner_fn

from typing import Optional, Type
from flax.struct import dataclass
import flax.linen as nn


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

  target_class: Type[nn.Module]
  index: Optional[int] = 0
  layer_name: Optional[str] = None
  class_name: Optional[str] = None

from typing import Any, Dict, List, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment

# Reused from Qwen3ForCausalLM.bbox_utils.center_to_corners_format
from ..image_transforms import center_to_corners_format
# Reused from Qwen3ForCausalLM.box_utils.generalized_box_iou
from ..utils import generalized_box_iou


class DeformableDetrHungarianMatcher(nn.Module):
  """
  Computes an assignment between targets and predictions of the Deformable DETR model.

  The assignment is done using the Hungarian algorithm.
  """

  class_cost: float
  bbox_cost: float
  giou_cost: float

  def __call__(
      self, outputs: Dict[str, jnp.ndarray], targets: List[Dict[str, jnp.ndarray]]
  ) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Differences:
    - out_prob = outputs["logits"].flatten(0, 1).sigmoid() instead of softmax
    - class_cost uses alpha and gamma
    """
    batch_size, num_queries = outputs["logits"].shape[:2]

    # We flatten to compute the cost matrices in a batch
    out_prob = jax.nn.sigmoid(
        outputs["logits"].reshape(-1, outputs["logits"].shape[-1])
    )  # [batch_size * num_queries, num_classes]
    out_bbox = outputs["pred_boxes"].reshape(-1, 4)  # [batch_size * num_queries, 4]

    # Also concat the target labels and boxes
    target_ids = jnp.concatenate([v["class_labels"] for v in targets])
    target_bbox = jnp.concatenate([v["boxes"] for v in targets])

    # Compute the classification cost.
    alpha = 0.25
    gamma = 2.0
    neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(jnp.log(1 - out_prob + 1e-8)))
    pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(jnp.log(out_prob + 1e-8)))
    class_cost = pos_cost_class[:, target_ids] - neg_cost_class[:, target_ids]

    # Compute the L1 cost between boxes
    bbox_cost = jnp.sum(jnp.abs(out_bbox[:, None, :] - target_bbox[None, :, :]), axis=-1)

    # Compute the giou cost between boxes
    giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))

    # Final cost matrix
    cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
    cost_matrix = cost_matrix.reshape(batch_size, num_queries, -1)

    # Move to CPU for scipy-based linear sum assignment
    cost_matrix_cpu = np.array(cost_matrix)

    sizes = [len(v["boxes"]) for v in targets]

    # The following part is not JIT-compatible, which is expected for Hungarian matching.
    # It splits the cost matrix into per-image chunks and applies the assignment algorithm.
    split_indices = np.cumsum(sizes)[:-1]
    cost_matrix_chunks = np.split(cost_matrix_cpu, split_indices, axis=-1)

    indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix_chunks)]

    return [(jnp.asarray(i, dtype=jnp.int64), jnp.asarray(j, dtype=jnp.int64)) for i, j in indices]
