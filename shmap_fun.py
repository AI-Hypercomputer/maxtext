from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map

embed_dim = 16
non_padded_dimension = 8
layers = 2

NUM_DEVICES=4

control = jnp.arange(NUM_DEVICES)
all_weights = [2*jax.numpy.eye(non_padded_dimension) for _ in range(layers)]
input_image = jnp.arange( NUM_DEVICES*embed_dim*embed_dim).reshape(NUM_DEVICES, embed_dim, embed_dim)
input_image *= 0
input_image = input_image.at[0,0,0].set(3)
input_image = input_image.at[1,13,0].set(5)
input_image = input_image.at[2,0,13].set(7)
input_image = input_image.at[3,13,13].set(11)
### input image sets one cell in each of the 4 "real" subimages


devices = mesh_utils.create_device_mesh((4,))
mesh = Mesh(devices, axis_names=('data'))


@partial(shard_map, mesh=mesh, in_specs=(P('data'),P('data'),P('data')),
         out_specs=P('data'), check_rep=False)
def example_application(all_weights, input_image, control):
  functions = [] # we create 4 clips, one per "real" subimage
  functions.append(lambda x : x[0, 0:8, 0:8] )
  functions.append(lambda x : x[0, 8:16, 0:8] )
  functions.append(lambda x : x[0, 0:8, 8:16] )
  functions.append(lambda x : x[0, 8:16, 8:16] )


  clipped_image = jax.lax.switch(control[0], functions, input_image)
  
  running = clipped_image
  for weight in all_weights:
    weight_gathered = jax.lax.all_gather(weight, axis_name='data', axis=0, tiled=True)
    running = weight_gathered @ running
  #z_block = jax.numpy.sum(z_partialsum)
  return jax.numpy.expand_dims(running,0)



output_image = example_application(all_weights, input_image, control)

print(f"{all_weights[0].shape=} + {input_image.shape=} -> {output_image.shape=}")
print(f"the output should be the multiplication of allweights repeatededly with the subimage of interest\n{output_image}")