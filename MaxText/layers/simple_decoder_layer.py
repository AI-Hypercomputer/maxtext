from jax import numpy as jnp
from flax import linen as nn
from jax.sharding import Mesh
from typing import Optional
from layers import quantizations
import common_types


class SimpleDecoderLayer(nn.Module):
  config: common_types.Config
  mesh: Mesh
  quant: Optional[quantizations.AqtQuantization] = None

  def setup(self):
    self.weight_mat = self.param('weights', nn.initializers.ones, (self.config.emb_dim, self.config.emb_dim))

  def __call__(self, inputs: jnp.ndarray, positions, segmentation, deterministic, model_mode) -> jnp.ndarray:
    return inputs @ self.weight_mat.astype(inputs.dtype)