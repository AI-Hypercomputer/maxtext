import jax
from jax.sharding import Mesh
import numpy as np
from jax.sharding import PartitionSpec as P
from flax import linen as nn

devices = jax.devices()

# mesh  = Mesh(np.array(jax.devices).reshape(4, 2, 8), ('x', 'y', 'z'))
mesh  = Mesh(np.array(devices).reshape(2, 1, 2), ('x', 'y', 'z'))
sharding = jax.sharding.NamedSharding(mesh, P('x'))
print('mesh', mesh)

print(mesh.devices, mesh.axis_names)
# ar_key_axis_order: AxisIdxes = (1, 2, 0, )
axis_order = (1, 0, 2)
ar_shape = (2, 8)
array = np.arange(np.prod(ar_shape)).reshape(ar_shape)
array = jax.device_put(array, sharding)
jax.debug.visualize_array_sharding(array)
