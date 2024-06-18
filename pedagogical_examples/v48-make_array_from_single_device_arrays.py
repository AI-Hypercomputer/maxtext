import jax
import math
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
import numpy as np


mesh_rows = 2
mesh_cols =  jax.device_count() // 2
print('jax.device_count', jax.device_count())
print('mesh columns', mesh_cols)

global_shape = (4, 8)
mesh = Mesh(
  np.array(jax.devices()).reshape(mesh_rows, mesh_cols),
  ('x', 'y'))

print('mesh.shape', mesh.shape)
print('mesh', mesh)

sharding = jax.sharding.NamedSharding(mesh, P('x', 'y'))
inp_data = np.arange(math.prod(global_shape)).reshape(global_shape)
print('inp_data', inp_data.shape, inp_data)
import pdb; pdb.set_trace()
arrays = [
    jax.device_put(inp_data[index], d)
    for d, index in sharding.addressable_devices_indices_map(global_shape).items()]

arr = jax.make_array_from_single_device_arrays(global_shape, sharding, arrays)
assert arr.shape == (4,8) # arr.shape is (8,8) regardless of jax.device_count()



# sharding = jax.sharding.NamedSharding(mesh, P(('x', 'y'),))
# rows_per_device = 2
# feature_length = 32
# per_device_shape = (rows_per_device, feature_length)
# per_host_shape = (rows_per_device * len(mesh.local_devices), feature_length)
# per_host_generator = lambda : np.arange(np.prod(per_host_shape)).reshape(per_host_shape)
# per_host_data = per_host_generator() # replace with your own per-host data pipeline that outputs numpy arrays
