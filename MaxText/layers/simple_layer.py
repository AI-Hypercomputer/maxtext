"""
Copyright 2024 Google LLC
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
     https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

""" Simple decoder layers for testing and debugging purposes."""

from jax import numpy as jnp
from jax.sharding import Mesh

from flax import nnx
from MaxText.common_types import Config
from MaxText.layers import quantizations,nnx_wrappers
from MaxText.layers.initializers import variable_to_logically_partitioned
from MaxText.layers.quantizations import AqtQuantization as Quant
from MaxText import max_logging

from typing import Any, Optional
# pytype: disable=attribute-error


def set_attrs_from_kwargs(obj, kwargs, *, skip_if_exists: bool = True, warn_on_skip: bool = False):
  for key, value in kwargs.items():
    if skip_if_exists and hasattr(obj, key):
        if warn_on_skip:
          max_logging.log(
              f"Skip overriding existing attribute {key} with value {value} in {obj.__class__.__name__}"
          )
        continue
    setattr(obj, key, value)


class SimpleDecoderLayer(nnx.Module):
  """Decoder layer consisting of a single [embed, embed] weight matrix."""

  def __init__(self, config: Config, mesh: Mesh,* , quant: Optional[quantizations.AqtQuantization] = None,
               rngs: Optional[nnx.Rngs] = None,
               weight_dtype: Any = jnp.float32,
               **kwargs:Any
               )->None:

    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.weight_dtype = weight_dtype
    self.rngs = rngs if rngs is not None else kwargs.get("rngs", nnx.Rngs(0))
    self.weight_mat = nnx.Param(
      nnx.initializers.lecun_normal()(self.rngs.params(),(self.config.emb_dim, self.config.emb_dim,),self.weight_dtype),
      sharding=("embed", "mlp",),
    )

    set_attrs_from_kwargs(self, kwargs, skip_if_exists=True, warn_on_skip=True)

  def __call__(
      self, inputs: jnp.ndarray, positions, segmentation, deterministic, model_mode, previous_chunk=None, page_state=None
  ):
    if self.config.scan_layers:
      return inputs @ self.weight_mat.astype(inputs.dtype), None
    return inputs @ self.weight_mat.astype(inputs.dtype)


def simple_decoder_layer_class(
):
  """Creates a SimpleDecoderLayer class."""
  return nnx_wrappers.to_linen_class(
      SimpleMlpDecoderLayer,
      metadata_fn=variable_to_logically_partitioned,
  )

def simple_decoder_layer(
   *,
   config: Config, 
   mesh: Mesh,
   name: Optional[str] = None,
   quant: Optional[Quant] = None,
   **kwargs:Any,
):
  """Creates a SimpleDecoderLayer object."""
  return nnx_wrappers.to_linen(
      SimpleMlpDecoderLayer,
      config=config,
      mesh=mesh,
      name=name,
      quant=quant,
      **kwargs,
      metadata_fn=variable_to_logically_partitioned,
  )

class SimpleMlpDecoderLayer(nnx.Module):
  """Decoder layer consisting of [embed,mlp] followed by an [mlp,embed] matmul."""

  def __init__(self, config: Config, mesh: Mesh, 
               *,
               quant: Optional[quantizations.AqtQuantization] = None,
               rngs: Optional[nnx.Rngs] = None,
               weight_dtype: Any = jnp.float32,
               **kwargs: Any,
               )->None:

    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.weight_dtype = weight_dtype
    self.rngs = rngs if rngs is not None else kwargs.get("rngs", nnx.Rngs(0))
    self.ff_1 = nnx.Param(
      nnx.initializers.lecun_normal()(self.rngs.params(),(self.config.emb_dim, self.config.emb_dim,),self.weight_dtype),
      sharding=("embed", "mlp",),
    )
    self.ff_2 = nnx.Param(
      nnx.initializers.lecun_normal()(self.rngs.params(),(self.config.emb_dim, self.config.emb_dim,),self.weight_dtype),
      sharding=("embed", "mlp",),
    )

    set_attrs_from_kwargs(self, kwargs, skip_if_exists=True, warn_on_skip=True)

  def __call__(
      self,
      inputs: jnp.ndarray,
      positions,
      segmentation,
      deterministic,
      model_mode,
      previous_chunk=None,
      page_state=None,
      slot=0,
  ):
    intermediate = inputs @ self.ff_1.astype(inputs.dtype)
    output = intermediate @ self.ff_2.astype(inputs.dtype)
    if self.config.scan_layers:
      return output, None
    else:
      return output

def simple_mlp_decoder_layer_class(
):
  """Creates a SimpleMlpDecoderLayer class."""
  return nnx_wrappers.to_linen_class(
      SimpleMlpDecoderLayer,
      metadata_fn=variable_to_logically_partitioned,
  )

def simple_mlp_decoder_layer(
   *,
   config: Config, 
   mesh: Mesh,
   name: Optional[str] = None,
   quant: Optional[Quant] = None,
   **kwargs:Any,
):
  """Creates a SimpleMlpDecoderLayer object."""
  return nnx_wrappers.to_linen(
      SimpleMlpDecoderLayer,
      config=config,
      mesh=mesh,
      name=name,
      quant=quant,
      **kwargs,
      metadata_fn=variable_to_logically_partitioned,
  )