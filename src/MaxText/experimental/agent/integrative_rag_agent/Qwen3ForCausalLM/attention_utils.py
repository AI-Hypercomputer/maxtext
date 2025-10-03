
from typing import Callable
import jax.numpy as jnp


def or_masks(*mask_functions: list[Callable]) -> Callable:
  """Returns a mask function that is the union of provided mask functions"""
  if not all(callable(arg) for arg in mask_functions):
    raise RuntimeError(f"All inputs should be callable mask_functions: {mask_functions}")

  def or_mask(batch_idx: int, head_idx: int, q_idx: jnp.ndarray, kv_idx: jnp.ndarray) -> jnp.ndarray:
    result = jnp.array(False, dtype=bool)
    for mask in mask_functions:
      result = result | mask(batch_idx, head_idx, q_idx, kv_idx)
    return result

  return or_mask

from typing import Callable
import jax
from jax import Array


def packed_sequence_mask_function(packed_sequence_mask: Array) -> Callable:
  """
  This return the mask_function function corresponding to a 2D packed sequence mask.
  """

  def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    return packed_sequence_mask[batch_idx, q_idx] == packed_sequence_mask[batch_idx, kv_idx]

  return inner_mask
