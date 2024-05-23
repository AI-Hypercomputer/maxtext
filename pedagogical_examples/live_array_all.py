from collections import defaultdict
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
# print('a shards', a.addressable_shards)

# print('A shard',  A.addressable_shards)

print(len(jax.live_arrays()))

def count_allocated_bytes_per_device():
  totals = defaultdict(dict)
  for arr in jax.live_arrays():
    print('====================')
    for shard in arr.addressable_shards:
      # print(arr.size, arr.itemsize)
      totals[shard.device][shard.data.unsafe_buffer_pointer()] = arr.size * arr.itemsize
  return {key: sum(val.values()) for key, val in totals.items()}
print(len(jax.live_arrays()))

# import pdb; pdb.set_trace()
print(count_allocated_bytes_per_device())
# print({
#   arr.addressable_shards[0].data.unsafe_buffer_pointer(): arr.addressable_shards[0].data
#   for arr in jax.live_arrays()})

# print({
#   arr.addressable_shards[0].data.unsafe_buffer_pointer(): arr.addressable_shards[0].data
#   for arr in jax.live_arrays()})

# print({
#   arr.addressable_shards[0].data.unsafe_buffer_pointer(): arr.addressable_shards[0].data
#   for arr in jax.live_arrays()})

