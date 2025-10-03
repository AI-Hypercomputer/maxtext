
from typing import Callable
import jax.numpy as jnp


def and_masks(*mask_functions: Callable) -> Callable:
  """Returns a mask function that is the intersection of provided mask functions."""
  if not all(callable(arg) for arg in mask_functions):
    raise RuntimeError(f"All inputs should be callable mask_functions: {mask_functions}")

  def and_mask(batch_idx, head_idx, q_idx, kv_idx):
    result = jnp.ones((), dtype=jnp.bool_)
    for mask in mask_functions:
      result = result & mask(batch_idx, head_idx, q_idx, kv_idx)
    return result

  return and_mask
