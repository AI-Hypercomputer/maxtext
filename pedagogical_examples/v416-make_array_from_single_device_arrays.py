import jax
import math
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
import numpy as np

print('initializing....')
jax.distributed.initialize()
print('finished initialize')
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


sharding = jax.sharding.NamedSharding(mesh, P(('x', 'y'),))
rows_per_device = 2
feature_length = 8
per_device_shape = (rows_per_device, feature_length)
per_host_shape = (rows_per_device * len(mesh.local_devices),
                  feature_length)
per_host_generator = lambda : np.arange(np.prod(per_host_shape)).reshape(per_host_shape)

per_host_data = per_host_generator() # replace with your own per-host data pipeline that outputs numpy arrays


global_shape = (rows_per_device * len(sharding.device_set), ) + per_device_shape[1:]
print('sharding device_set', sharding.device_set)
print('global shape', rows_per_device * len(sharding.device_set), per_device_shape[1:], global_shape)
print('mesh.devices', mesh.devices)
print('mesh.local_devices', mesh.local_devices)
per_device_data = np.split(per_host_data, len(mesh.local_devices), axis = 0) # per device data, but on host
print('per device data, but on host', per_device_data)
per_device_data_on_device = jax.device_put(per_device_data, mesh.local_devices) # per device data, now on device
print('per device data, now on device', per_device_data_on_device)
output_global_array = jax.make_array_from_single_device_arrays(global_shape, sharding, per_device_data_on_device)
jax.debug.visualize_array_sharding(output_global_array)
print(output_global_array.addressable_data(0).shape == per_device_shape)
print(output_global_array.shape == global_shape)
