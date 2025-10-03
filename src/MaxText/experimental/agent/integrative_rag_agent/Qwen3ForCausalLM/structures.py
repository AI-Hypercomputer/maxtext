
from typing import Optional, Tuple
import jax.numpy as jnp


class NestedTensor:
  """Data structure for batched tensors and their associated masks."""

  def __init__(self, tensors: jnp.ndarray, mask: Optional[jnp.ndarray]):
    self.tensors = tensors
    self.mask = mask

  def decompose(self) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """Decomposes the nested tensor into tensors and mask."""
    return self.tensors, self.mask

  def __repr__(self) -> str:
    """String representation of the nested tensor."""
    return str(self.tensors)
