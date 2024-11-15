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


# Define the mesh with 1024 devices
devices = jax.devices()
# assert len(devices) == number_of_nodes*8, f"This example requires {devices} devices"
# DP, FSDP, TP
mesh_shape = (1, number_of_nodes, 8)
devices_grid = np.array(devices).reshape(mesh_shape)
mesh = Mesh(devices_grid, ('dp', 'fsdp', 'tp'))


in_partition_spec = (P(None, 'fsdp', None,)) # This is what we want, replica_groups=[4,2]<=[2,2,2] http://xprof/?session_id=lancewang-9867403027673891038
# in_partition_spec = (P('dp', None, 'tp',)) # replica_groups=[2,4]<=[2,2,2] http://xprof/?session_id=lancewang-9867403027673893787
# in_partition_spec = (P('dp', 'fsdp', 'tp',)) #replica_groups=[1,8]<=[8] http://xprof/?session_id=lancewang-9867403027673892440
# out_partition_spec = (P('dp', 'fsdp', 'tp'))
out_partition_spec = (P(None, None, None))

data_in = jnp.zeros((2, number_of_nodes * 128, 65536), dtype=jnp.bfloat16)

# data_in = jnp.zeros((2, 1024, 16384), dtype=jnp.bfloat16)

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

if jax.process_index() == 0:
    jax.profiler.start_trace("gs://lancewang-dev-supercomputer-testing/maxtext_gpu/collective_benchmarking")

for i in range(10):
    result = no_op(data_in)
jax.block_until_ready(result)

if jax.process_index() == 0:
    jax.profiler.stop_trace()
