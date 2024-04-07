import jax
from jax import numpy as jnp
from jax import tree_map
from flax import linen as nn
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.experimental import mesh_utils
import os
import argparse

import common_types

def stack_pytrees(*pytrees):
  """Stacks pytrees with identical structure along a new leading dimension."""
  def stacking_fn(*leaves):
    return jnp.stack(leaves)
  return tree_map(stacking_fn, *pytrees)

def create_mesh(n_stages, tp_axis, dp_axis):
  devices = mesh_utils.create_device_mesh((n_stages, tp_axis, dp_axis))
  mesh = Mesh(devices, axis_names=('stage', 'tensor', 'data'))
  return mesh

class SimpleDecoderLayer(nn.Module):
  embed_size: int

  def setup(self):
    self.weight_mat = self.param('weights', nn.initializers.ones, (self.embed_size, self.embed_size))

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    return x @ self.weight_mat

# Pipeline is made up of several SimpleDecoderLayers 
class Pipeline(nn.Module):
  
  embed_size: int
  n_layers: int
  decoder_layer_class: nn.Module
  mesh: common_types.Mesh #jax.sharding.Mesh

  def setup(self):
    decoder_layers = [self.decoder_layer_class(self.embed_size) for _ in range(self.n_layers)]
    self.decoder_layers = decoder_layers

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    # We want to access the variables of the decoder_layer, the below loop fills in the variables dictionary (previously empty dict)
    for decoder in self.decoder_layers:
      #print(dir(decoder))
      _ = decoder(x)

    # decoder.variables is an empty dictionary until the loop above is executed
    decoder_params = [decoder.variables for decoder in self.decoder_layers] 
    stacked_params = stack_pytrees(*decoder_params)
    stacked_inputs = stack_pytrees(*([x] * self.n_layers))
    pipeline_output = jax.vmap(self.decoder_layers[0].apply)(stacked_params, stacked_inputs)
    return pipeline_output

def main() -> None:
  print("hello")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  parser = argparse.ArgumentParser(description='Pipeline Parallelism Options')
  parser.add_argument('--batch_size', type=int, default=24)
  parser.add_argument('--n_stages', type=int, default=4)
  parser.add_argument('--n_microbatches', type=int, default=8)
  parser.add_argument('--dp_axis', type=int, default=1)
  parser.add_argument('--tp_axis', type=int, default=1)
  parser.add_argument('--features', type=int, default=16)
  parser.add_argument('--sequence', type=int, default=16)
  parser.add_argument('--n_repeat', type=int, default=2)

  args = parser.parse_args()
  args.microbatch_size = args.batch_size // args.n_microbatches
  args.layers = args.n_stages * args.n_repeat
  use_circ_storage = args.n_repeat > 1 and args.n_microbatches > args.n_stages

  mesh = create_mesh(args.n_stages, args.tp_axis, args.dp_axis)
  my_pipeline = Pipeline(embed_size=5,n_layers=2, decoder_layer_class=SimpleDecoderLayer, mesh=mesh)
  example_input = jnp.ones([6,5])
  init_pipeline_params = my_pipeline.init(jax.random.PRNGKey(0), example_input)
  my_pipeline.apply(init_pipeline_params, example_input)

if __name__ == "__main__":
  main()