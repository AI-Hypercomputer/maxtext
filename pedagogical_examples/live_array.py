import numpy as np
import jax
a = jax.numpy.ones((4, 8))

#jax.live_arrays()
# print(jax.live_arrays())
# print([arr.unsafe_buffer_pointer() for arr in jax.live_arrays()])
# print(sum([np.prod(x.addressable_shards[0].data.shape) for x in jax.live_arrays()]))
#jax.live_arrays()
# print(jax.live_arrays())
# print([arr.unsafe_buffer_pointer() for arr in jax.live_arrays()])
# print(sum([np.prod(x.addressable_shards[0].data.shape) for x in jax.live_arrays()]))

print(len(jax.live_arrays()))
mesh = jax.sharding.Mesh(np.reshape(jax.devices(), (2,2)), ('axis1', 'axis2'))
sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None,('axis1', 'axis2')))
A = jax.device_put(a, sharding)
print(len(jax.live_arrays()))

print({
  arr.addressable_shards[0].data.unsafe_buffer_pointer(): arr.addressable_shards[0].data
  for arr in jax.live_arrays()})

print({
  arr.addressable_shards[0].data.unsafe_buffer_pointer(): arr.addressable_shards[0].data
  for arr in jax.live_arrays()})

print({
  arr.addressable_shards[0].data.unsafe_buffer_pointer(): arr.addressable_shards[0].data
  for arr in jax.live_arrays()})

