import numpy as np
import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax import numpy as jnp


def create_inputs_and_embed():
    mesh_axes = ["data", "fsdp", "tensor"]
    dcn_parallelism = [args.dcn_data_parallelism, args.dcn_fsdp_parallelism, args.dcn_tensor_parallelism]
    ici_parallelism = [args.ici_data_parallelism, args.ici_fsdp_parallelism, args.ici_tensor_parallelism]

    multislice_env = args.dcn_data_parallelism > 1 or args.dcn_fsdp_parallelism > 1 or args.dcn_tensor_parallelism > 1

    if multislice_env:
        device_mesh = mesh_utils.create_hybrid_device_mesh(ici_parallelism, dcn_parallelism)
    else:
        device_mesh = mesh_utils.create_device_mesh(ici_parallelism)
    mesh = Mesh(device_mesh, mesh_axes)



    data_shape = (args.global_batch_size, args.sequence_len)
    data_partition_spec = jax.sharding.PartitionSpec(("data", "fsdp","tensor"),)
    data_sharding = jax.sharding.NamedSharding(mesh, data_partition_spec)

    # Problem: This will first create the data on one device, the shard it. How can we start with the data sharded?
    data = jax.numpy.ones(data_shape, dtype=jnp.int32)
    data = jax.lax.with_sharding_constraint(data, data_sharding)

    # Alternative:
    # Create the sharded array
    # data = global_device_array.GlobalDeviceArray.from_callback(
    #     global_shape=data_shape,
    #     mesh=mesh,
    #     distribution_specs=data_partition_spec,
    #     callback=create_shard



    embed_shape = (args.vocab_size, args.embed_size)
    embed_partition_spec = jax.sharding.PartitionSpec("fsdp", "tensor")
    embed_sharding = jax.sharding.NamedSharding(mesh, embed_partition_spec)

    embed = jax.random.normal(jax.random.PRNGKey(0), embed_shape)
    embed = jax.lax.with_sharding_constraint(embed, embed_sharding)

    inputs = embed[data]
    print(f"Print statement to force inputs to be execute {inputs[0][0][0]}", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Sharding and size settings')
    parser.add_argument('--dcn_data_parallelism', type=int, default=1)
    parser.add_argument('--dcn_fsdp_parallelism', type=int, default=1)
    parser.add_argument('--dcn_tensor_parallelism', type=int, default=1)
    parser.add_argument('--ici_data_parallelism', type=int, default=1)
    parser.add_argument('--ici_fsdp_parallelism', type=int, default=1)
    parser.add_argument('--ici_tensor_parallelism', type=int, default=1)
    parser.add_argument('--global_batch_size', type=int, default=128)
    parser.add_argument('--sequence_len', type=int, default=1024)
    parser.add_argument('--vocab_size', type=int, default=256)
    parser.add_argument('--embed_size', type=int, default=512)
    global args
    args = parser.parse_args()
    
    create_inputs_and_embed()


main()