import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu import megablox as mblx

tile_size = (512, 1024, 1024)
m = 512
k = 1024
n = 1024
num_groups = 2

lhs = jax.random.uniform(jax.random.PRNGKey(0), (m,k))
rhs = jax.random.uniform(jax.random.PRNGKey(1), (num_groups, k, n))
group_sizes = jnp.array([0,1])


group_offset = jnp.array([0], dtype=jnp.int32)


# print(f'from matmul: group_metadata: {group_metadata}, num_active_tiles = {num_active_tiles}')
out = mblx.gmm(lhs=lhs, 
               rhs=rhs, 
               group_sizes=group_sizes,
              #  group_metadata=group_metadata, 
              #  num_total_groups=group_sizes.shape[0], 
              #  num_active_tiles=num_active_tiles,
               preferred_element_type=jnp.bfloat16, 
               tiling=tile_size,
)
print(out.shape)
print(out)
