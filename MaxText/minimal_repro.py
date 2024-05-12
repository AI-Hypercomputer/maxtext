import numpy as np
import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P


def simple_repro():
    mesh_axes = ["axis_1", "axis_2"]
    
    dcn_parallelism = [args.dcn_axis_1_parallelism, args.dcn_axis_2_parallelism]
    ici_parallelism = [args.ici_axis_1_parallelism, args.ici_axis_2_parallelism]

    multislice_env = args.dcn_axis_1_parallelism > 1  or args.dcn_axis_2_parallelism > 1
    if multislice_env:
        device_mesh = mesh_utils.create_hybrid_device_mesh(ici_parallelism, dcn_parallelism)
    else:
        device_mesh = mesh_utils.create_device_mesh(ici_parallelism)
    mesh = Mesh(device_mesh, mesh_axes)
    
    # Shard over both axes
    data_sharding = (("axis_1", "axis_2"),)
    data_pspec = P(*data_sharding)
    data_pspec_shardings = jax.sharding.NamedSharding(mesh, data_pspec)

    def generate_fully_sharded_data():
        def generate_data_base():
            return jax.numpy.zeros((args.global_batch_size), dtype=jax.numpy.int32) # also fails with bf16
        jit_generate_data = jax.jit(generate_data_base, out_shardings=data_pspec_shardings)
        return jit_generate_data()

    def dummy_op_on_data(data):
        return jnp.linalg.norm(data)
    
    print("Creating data...", flush=True)
    data = generate_fully_sharded_data()
    data.block_until_ready()
    print("Data created!", flush=True)
    jit_data_op = jax.jit(dummy_op_on_data, in_shardings=data_pspec_shardings) 
    norm = jit_data_op(data) 
    print(f"Data norm of {norm}", flush=True) # Runtime crash, XlaRuntimeError: UNKNOWN
    # with verbose libtpu logging platforms/asic_sw/driver/2a886c8/common/internal/host_queue.cc:323 is flagged

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Sharding and size settings')
    parser.add_argument('--ici_axis_1_parallelism', type=int, default=1)
    parser.add_argument('--ici_axis_2_parallelism', type=int, default=1)
    parser.add_argument('--dcn_axis_1_parallelism', type=int, default=1)
    parser.add_argument('--dcn_axis_2_parallelism', type=int, default=1)
    parser.add_argument('--global_batch_size', type=int, default=128)
    global args
    args = parser.parse_args()
    
    simple_repro()

main()