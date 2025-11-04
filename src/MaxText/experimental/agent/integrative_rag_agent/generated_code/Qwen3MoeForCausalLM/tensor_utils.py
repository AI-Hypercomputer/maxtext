
import jax.numpy as jnp
from jax import Array

def _end_ptr(tensor: Array) -> int:
  """Extracts the end of the pointer if the tensor is a slice of a bigger tensor.

  NOTE: This function is fundamentally incompatible with JAX's memory model.
  JAX does not expose memory pointers like PyTorch's `data_ptr()`. This function
  is part of a mechanism to detect shared memory storage in PyTorch, which is
  handled differently in JAX/Flax (e.g., by sharing module instances or
  parameter objects directly). This implementation raises a NotImplementedError
  to indicate that this low-level approach should not be used.
  """
  raise NotImplementedError(
      "JAX does not expose memory pointers. Detecting shared storage via"
      " pointers is not a JAX-native pattern."
  )

from typing import Dict, List, Set, Tuple

import jax.numpy as jnp


# The JAX module `generated_code.Qwen3MoeForCausalLM.tensor_utils._end_ptr` was not used as it raises a NotImplementedError,
# confirming that pointer-based logic is not supported in JAX.
def _find_disjoint(
    tensors: List[Set[str]], state_dict: Dict[str, jnp.ndarray]
) -> Tuple[List[Set[str]], List[str]]:
  """Finds disjoint and shared tensors from a list of sets of tensor names."""
  # In JAX, we cannot inspect memory pointers to determine if tensors share
  # overlapping storage as done in the original PyTorch version. This function
  # is simplified to categorize pre-grouped tensors based on group size,
  # as we cannot break down groups further without memory layout information.
  del state_dict  # Unused in JAX implementation.

  filtered_tensors = tensors

  disjoint_tensors = []
  shared_tensors = []
  for tensors_group in filtered_tensors:
    if len(tensors_group) == 1:
      disjoint_tensors.append(tensors_group.pop())
    else:
      shared_tensors.append(tensors_group)
  return shared_tensors, disjoint_tensors

import jax.numpy as jnp
import ml_dtypes

str_to_jnp_dtype = {
    "BOOL": jnp.bool_,
    "U8": jnp.uint8,
    "I8": jnp.int8,
    "I16": jnp.int16,
    "F16": jnp.float16,
    "BF16": jnp.bfloat16,
    "I32": jnp.int32,
    "F32": jnp.float32,
    "F64": jnp.float64,
    "I64": jnp.int64,
    "F8_E4M3": ml_dtypes.float8_e4m3fn,
    "F8_E5M2": ml_dtypes.float8_e5m2,
    "U16": jnp.uint16,
    "U32": jnp.uint32,
    "U64": jnp.uint64,
}

from typing import Tuple
import jax
from jax import Array


def id_tensor_storage(tensor: Array) -> Tuple[jax.Device, int, int]:
  """Unique identifier for a tensor's underlying buffer.

  Multiple JAX arrays can share the same buffer. This identifier is guaranteed
  to be unique and constant for this tensor's buffer during its lifetime. Two
  tensor buffers with non-overlapping lifetimes may have the same id.

  Args:
    tensor: The JAX array to identify.

  Returns:
    A tuple containing the tensor's device, a unique integer identifier for its
    buffer, and the size of the buffer in bytes.
  """
  # In JAX, all arrays are conceptually similar, removing the need for special
  # handling of XLA or distributed tensors as in the original PyTorch code.
  # The `unsafe_buffer_pointer()` provides a unique integer identifier for the
  # underlying device buffer, analogous to `data_ptr()`.
  unique_id = tensor.unsafe_buffer_pointer()
  device = tensor.device()
  size_in_bytes = tensor.nbytes

  return device, unique_id, size_in_bytes

import jax
import jax.numpy as jnp


def _index_first_axis(tensor: jax.Array, indices: jax.Array) -> jax.Array:
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
Flax DETR model.
"""

from typing import Optional

from flax.struct import dataclass
from jax import Array


@dataclass
class NestedTensor:
  """
  A dataclass to hold a tensor and an optional mask. This is the JAX equivalent of the PyTorch NestedTensor class.
  """

  tensors: Array
  mask: Optional[Array]

  def decompose(self):
    """
    Decomposes the nested tensor into its constituent tensor and mask.

    Returns:
        A tuple containing the tensor and the mask.
    """
    return self.tensors, self.mask

  def __repr__(self):
    """
    Returns the string representation of the tensor.
    """
    return str(self.tensors)

import jax.numpy as jnp
from typing import List, Sequence


def _max_by_axis(the_list: Sequence[Sequence[int]]) -> List[int]:
  """Computes the element-wise maximum of a list of lists.

  Args:
    the_list: A list of lists of integers.

  Returns:
    A list of integers containing the element-wise maximums.
  """
  if not the_list:
    return []
  return jnp.max(jnp.array(the_list), axis=0).tolist()

from typing import List, Optional

import jax.numpy as jnp
from flax.struct import dataclass
from jax import Array


# Reused from generated_code.Qwen3MoeForCausalLM.tensor_utils.NestedTensor
@dataclass
class NestedTensor:
    """
    A dataclass to hold a tensor and an optional mask.
    This is the JAX/Flax equivalent of the PyTorch NestedTensor class.
    """

    tensors: Array
    mask: Optional[Array]

    def decompose(self):
        """Decomposes the nested tensor into its constituent tensor and mask."""
        return self.tensors, self.mask

    def __repr__(self):
        """Returns the string representation of the contained tensor."""
        return str(self.tensors)


def _max_by_axis(the_list: List[List[int]]) -> List[int]:
    """
    Computes the element-wise maximum of a list of lists of integers.
    """
    if not the_list:
        return []
    maxes = jnp.max(jnp.array(the_list), axis=0)
    return maxes.tolist()


def nested_tensor_from_tensor_list(tensor_list: List[Array]) -> NestedTensor:
    """
    Pads a list of tensors to the same size and combines them into a single tensor.
    Also creates a boolean mask indicating the valid (non-padded) areas.
    """
    if tensor_list[0].ndim == 3:
        # Get the maximum size of each dimension across all tensors
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        batch_size, _, height, width = batch_shape
        dtype = tensor_list[0].dtype

        # Create empty tensor and mask
        tensor = jnp.zeros(batch_shape, dtype=dtype)
        mask = jnp.ones((batch_size, height, width), dtype=jnp.bool_)

        # Pad each image and update the mask
        for i, img in enumerate(tensor_list):
            c, h, w = img.shape
            tensor = tensor.at[i, :c, :h, :w].set(img)
            mask = mask.at[i, :h, :w].set(False)
    else:
        raise ValueError("Only 3-dimensional tensors are supported")
    return NestedTensor(tensor, mask)

from jax import Array
import jax.numpy as jnp


def _upcast(t: Array) -> Array:
  """Protects from numerical overflows in multiplications by upcasting to the equivalent higher type."""
  if jnp.issubdtype(t.dtype, jnp.floating):
    return t if t.dtype in (jnp.float32, jnp.float64) else t.astype(jnp.float32)
  else:
    return t if t.dtype in (jnp.int32, jnp.int64) else t.astype(jnp.int32)
