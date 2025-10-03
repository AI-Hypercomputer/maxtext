
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
A JAX implementation of the ModelOutput class from Hugging Face transformers.
"""

from collections import OrderedDict
from dataclasses import fields, is_dataclass
from typing import Any, Tuple

import jax
import jax.numpy as jnp
from jax.core import Tracer


# This is a simplified version of the is_tensor utility for a JAX-only context.
def is_tensor(x: Any) -> bool:
  """
  Tests if `x` is a `jax.numpy.ndarray` or a `jax.core.Tracer`.
  """
  return isinstance(x, (jnp.ndarray, Tracer))


# The PyTorch-specific `__init_subclass__` is replaced by the JAX pytree registration decorator
# and the implementation of `tree_flatten` and `tree_unflatten` methods.
@jax.tree_util.register_pytree_node_class
class ModelOutput(OrderedDict):
  """
  Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer or slice (like a
  tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular
  python dictionary.

  <Tip warning={true}>

  You can't unpack a `ModelOutput` directly. Use the [`~utils.ModelOutput.to_tuple`] method to convert it to a tuple
  before.

  </Tip>
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    # Subclasses of ModelOutput must use the @dataclass decorator
    # This check is done in __init__ because the @dataclass decorator operates after __init_subclass__
    # issubclass() would return True for issubclass(ModelOutput, ModelOutput) when False is needed
    # Just need to check that the current class is not ModelOutput
    is_modeloutput_subclass = self.__class__ != ModelOutput

    if is_modeloutput_subclass and not is_dataclass(self):
      raise TypeError(
          f"{self.__module__}.{self.__class__.__name__} is not a dataclass."
          " This is a subclass of ModelOutput and so must use the @dataclass decorator."
      )

  def __post_init__(self):
    """Check the ModelOutput dataclass.

    Only occurs if @dataclass decorator has been used.
    """
    class_fields = fields(self)

    # Safety and consistency checks
    if not len(class_fields):
      raise ValueError(f"{self.__class__.__name__} has no fields.")
    if not all(field.default is None for field in class_fields[1:]):
      raise ValueError(f"{self.__class__.__name__} should not have more than one required field.")

    first_field = getattr(self, class_fields[0].name)
    other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

    if other_fields_are_none and not is_tensor(first_field):
      if isinstance(first_field, dict):
        iterator = first_field.items()
        first_field_iterator = True
      else:
        try:
          iterator = iter(first_field)
          first_field_iterator = True
        except TypeError:
          first_field_iterator = False

      # if we provided an iterator as first field and the iterator is a (key, value) iterator
      # set the associated fields
      if first_field_iterator:
        for idx, element in enumerate(iterator):
          if not isinstance(element, (list, tuple)) or len(element) != 2 or not isinstance(element[0], str):
            if idx == 0:
              # If we do not have an iterator of key/values, set it as attribute
              self[class_fields[0].name] = first_field
            else:
              # If we have a mixed iterator, raise an error
              raise ValueError(f"Cannot set key/value for {element}. It needs to be a tuple (key, value).")
            break
          setattr(self, element[0], element[1])
          if element[1] is not None:
            self[element[0]] = element[1]
      elif first_field is not None:
        self[class_fields[0].name] = first_field
    else:
      for field in class_fields:
        v = getattr(self, field.name)
        if v is not None:
          self[field.name] = v

  def __delitem__(self, *args, **kwargs):
    raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

  def setdefault(self, *args, **kwargs):
    raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

  def pop(self, *args, **kwargs):
    raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

  def update(self, *args, **kwargs):
    raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

  def __getitem__(self, k):
    if isinstance(k, str):
      inner_dict = dict(self.items())
      return inner_dict[k]
    else:
      return self.to_tuple()[k]

  def __setattr__(self, name, value):
    if name in self.keys() and value is not None:
      # Don't call self.__setitem__ to avoid recursion errors
      super().__setitem__(name, value)
    super().__setattr__(name, value)

  def __setitem__(self, key, value):
    # Will raise a KeyException if needed
    super().__setitem__(key, value)
    # Don't call self.__setattr__ to avoid recursion errors
    super().__setattr__(key, value)

  def __reduce__(self):
    if not is_dataclass(self):
      return super().__reduce__()
    callable_func, _args, *remaining = super().__reduce__()
    args = tuple(getattr(self, field.name) for field in fields(self))
    return callable_func, args, *remaining

  def to_tuple(self) -> Tuple[Any, ...]:
    """
    Convert self to a tuple containing all the attributes/keys that are not `None`.
    """
    return tuple(self[k] for k in self.keys())

  def tree_flatten(self):
    """Flattens the ModelOutput into a JAX pytree."""
    children = tuple(self.values())
    aux_data = tuple(self.keys())
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    """Unflattens the JAX pytree into a ModelOutput."""
    return cls(zip(aux_data, children))

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
The file is a JAX version of the huggingface `MoeCausalLMOutputWithPast`
"""

from typing import Optional, Tuple, Any
from flax.struct import dataclass
from jax import Array


@dataclass
class MoeCausalLMOutputWithPast:
  """
    Base class for causal language model (or autoregressive) with mixture of experts outputs.

    Args:
        loss (`jax.Array` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).

        logits (`jax.Array` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

        aux_loss (`jax.Array`, *optional*, returned when `labels` is provided):
            aux_loss for the sparse modules.

        router_logits (`tuple(jax.Array)`, *optional*, returned when `output_router_probs=True` and `config.add_router_probs=True` is passed or when `config.output_router_probs=True`):
            Tuple of `jax.Array` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

            Raw router logtis (post-softmax) that are computed by MoE routers, these terms are used to compute the auxiliary
            loss for Mixture of Experts models.

        past_key_values (`Any`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            It is a `~cache_utils.Cache` instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(jax.Array)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jax.Array` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(jax.Array)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jax.Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

  loss: Optional[Array] = None
  aux_loss: Optional[Array] = None
  logits: Optional[Array] = None
  past_key_values: Optional[Any] = None
  hidden_states: Optional[Tuple[Array, ...]] = None
  attentions: Optional[Tuple[Array, ...]] = None
  router_logits: Optional[Tuple[Array, ...]] = None

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
 The file is a JAX/Flax port of the HuggingFace file:
 https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py
"""

from typing import Optional, Tuple, Any
from flax.struct import dataclass
import jax

# Reused from Qwen3ForCausalLM.model_outputs.ModelOutput
from Qwen3ForCausalLM.model_outputs import ModelOutput


@dataclass
class MoeModelOutputWithPast(ModelOutput):
  """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`jax.Array` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(jax.Array)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jax.Array` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(jax.Array)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jax.Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        router_logits (`tuple(jax.Array)`, *optional*, returned when `output_router_probs=True` and `config.add_router_probs=True` is passed or when `config.output_router_probs=True`):
            Tuple of `jax.Array` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

            Raw router logtis (post-softmax) that are computed by MoE routers, these terms are used to compute the auxiliary
            loss for Mixture of Experts models.
    """

  last_hidden_state: Optional[jax.Array] = None
  past_key_values: Optional[Any] = None
  hidden_states: Optional[Tuple[jax.Array, ...]] = None
  attentions: Optional[Tuple[jax.Array, ...]] = None
  router_logits: Optional[Tuple[jax.Array, ...]] = None
