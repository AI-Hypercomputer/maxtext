import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P, NamedSharding
import numpy as np
import os
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Initialize JAX distributed system")
parser.add_argument("--num_nodes", type=int, required=True, help="Number of nodes")
parser.add_argument("--type", type=str, required=False, help="Benchmark type")
parser.add_argument("--xprof", type=bool, required=False, help="Turn on xprof or not")
parser.add_argument("--iter", type=int, required=False, help="Turn on xprof or not")

args = parser.parse_args()

number_of_nodes = args.num_nodes
print(f"Running on ${number_of_nodes} nodes", flush=True)
print("Attempting to initialize the jax distributed system for GPU backend...", flush=True)
coordinator_ip = None
if os.environ.get("JAX_COORDINATOR_IP") is not None:
    coordinator_ip = str(os.getenv("JAX_COORDINATOR_IP"))
coordinator_port = str(os.getenv("JAX_COORDINATOR_PORT"))
jax.distributed.initialize(
    coordinator_address=f"{coordinator_ip}:{coordinator_port}",
    num_processes=int(os.getenv("NNODES")),
    process_id=int(os.getenv("NODE_RANK")),
)
print(f"JAX global devices: {jax.devices()}", flush=True)
print("Jax distributed system initialized on GPU!", flush=True)

devices = jax.devices()

if args.type == '1d':
    mesh_shape = (number_of_nodes * 8)
    devices_grid = np.array(devices).reshape(mesh_shape)
    mesh = Mesh(devices_grid, ('fsdp'))

    in_partition_spec = (P('fsdp'))
    out_partition_spec = (P(None))

    data_in = jnp.zeros((2 * 128 * number_of_nodes * 65536), dtype=jnp.bfloat16)
elif args.type == '2d-tp':
    mesh_shape = (number_of_nodes, 8)
    devices_grid = np.array(devices).reshape(mesh_shape)
    mesh = Mesh(devices_grid, ('fsdp', 'tp'))

    in_partition_spec = (P('fsdp', None))
    out_partition_spec = (P(None, None))

    data_in = jnp.zeros((2 * 128 * number_of_nodes, 65536), dtype=jnp.bfloat16)
elif args.type == '2d-dp':
    mesh_shape = (8, number_of_nodes)
    devices_grid = np.array(devices).reshape(mesh_shape)
    mesh = Mesh(devices_grid, ('dp', 'fsdp'))

    in_partition_spec = (P(None, 'fsdp'))
    out_partition_spec = (P(None, None))

    data_in = jnp.zeros((65536, 2 * 128 * number_of_nodes), dtype=jnp.bfloat16)
else:
    # DP, FSDP, TP
    mesh_shape = (1, number_of_nodes, 8)
    devices_grid = np.array(devices).reshape(mesh_shape)
    mesh = Mesh(devices_grid, ('dp', 'fsdp', 'tp'))

    in_partition_spec = (P(None, 'fsdp', None))
    out_partition_spec = (P(None, None, None))

    # data_in = jnp.zeros((2, number_of_nodes * 128, 65536), dtype=jnp.bfloat16)
    data_in = jnp.zeros((2, 128 * number_of_nodes, 65536), dtype=jnp.bfloat16)

def no_op(x):
  return x

in_shardings = jax.tree_util.tree_map(
    lambda ps: NamedSharding(mesh, ps), in_partition_spec
)
out_shardings = jax.tree_util.tree_map(
    lambda ps: NamedSharding(mesh, ps), out_partition_spec
)

no_op = jax.jit(fun=no_op,
    in_shardings=in_shardings,
    out_shardings=out_shardings,
)

result = no_op(data_in)
jax.block_until_ready(result)

if args.xprof and jax.process_index() == 0:
    jax.profiler.start_trace("gs://lancewang-dev-supercomputer-testing/maxtext_gpu/collective_benchmarking")

for i in range(1000):
    result = no_op(data_in)
    if i%100==0:
        print(f"Running {i}'s loop")

jax.block_until_ready(result)

if args.xprof and jax.process_index() == 0:
    jax.profiler.stop_trace()
