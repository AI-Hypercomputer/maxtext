import numpy as np
import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh

GLOBAL_BATCH_SIZE=128
SEQUENCE_LEN = 1024
VOCAB_SIZE=256
EMBED_SIZE=512




mesh_axes = ["data", "fsdp", "tensor"]
dcn_parallelism = [args.dcn_data_parallelism, args.dcn_fsdp_parallelism, args.dcn_tensor_parallelism]
ici_parallelism = [args.ici_data_parallelism, args.ici_fsdp_parallelism, args.ici_tensor_parallelism]

multislice_env = args.dcn_data_parallelism > 1 or args.dcn_fsdp_parallelism > 1 or args.dcn_tensor_parallelism > 1

if multislice_env:
    device_mesh = mesh_utils.create_hybrid_device_mesh(ici_parallelism, dcn_parallelism)
else:
    device_mesh = mesh_utils.create_device_mesh(ici_parallelism)
mesh = Mesh(device_mesh, mesh_axes)



data_shape = (GLOBAL_BATCH_SIZE, SEQUENCE_LEN)
data = jax.numpy.ones(data_shape)
data_partition_spec = jax.sharding.PartitionSpec(("data", "fsdp"), None)
data_sharding = jax.sharding.NamedSharding(mesh, data_partition_spec)

data = jax.lax.with_sharding_constraint(data, data_sharding)

embedding_shape = (VOCAB_SIZE, EMBED_SIZE)
data = jax.random.normal(data_shape)
data_partition_spec = jax.sharding.PartitionSpec(("data", "fsdp"), None)
data_sharding = jax.sharding.NamedSharding(mesh, data_partition_spec)




mesh = jax.sharding.Mesh(np.reshape(  jax.devices(), (2,2)), ["myaxis1", "myaxis2"])
p1 = jax.sharding.PartitionSpec( "myaxis2", "myaxis1")

d2 = mesh.devices[1]
m2 = jax.sharding.Mesh(np.reshape( d2, (1,2)), ["myaxis1", "myaxis2"])
#sharding1 = jax.sharding.NamedSharding(mesh, p1)
sharding1 = jax.sharding.NamedSharding(m2,p1)
#sharding1 = jax.sharding.SingleDeviceSharding(jax.devices()[2])

sharded_A1 = jax.device_put(A, sharding1)
sss = sharded_A1


print(f"{sharded_A1.shape=} {sharded_A1.addressable_shards[0].data.shape=}")

jax.debug.visualize_array_sharding(sharded_A1)

A3 = jax.device_put(A, jax.sharding.NamedSharding(mesh, p1))

def my_add(a,b):
    return a + b

jit_add = jax.jit(my_add)
jit_add(sss, A3)

breakpoint()