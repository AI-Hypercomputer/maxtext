from collections import defaultdict
import numpy as np
import jax
a = jax.numpy.ones((4, 8))

print(len(jax.live_arrays()))
mesh = jax.sharding.Mesh(np.reshape(jax.devices(), (2,2)), ('axis1', 'axis2'))
sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None,('axis1', 'axis2')))
A = jax.device_put(a, sharding)

print(len(jax.live_arrays()))

def count_allocated_bytes_per_device():
  totals = defaultdict(dict)
  for arr in jax.live_arrays():
    for shard in arr.addressable_shards:
      totals[shard.device][shard.data.unsafe_buffer_pointer()] = arr.size * arr.itemsize
  return {key: sum(val.values()) for key, val in totals.items()}
print(len(jax.live_arrays()))


print(count_allocated_bytes_per_device())
