import argparse
import jax
from jax._src.mesh import Mesh
# from jax.experimental import multihost_utils
from jax._src.partition_spec import PartitionSpec
from jax.experimental.pjit import pjit
import jax.numpy as jnp
import numpy as np
from datetime import datetime
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

parser = argparse.ArgumentParser(description='TPU configuration options')
parser.add_argument('--BATCH_SIZE', type=int, default=2048,
                    help='Global batch size')
parser.add_argument('--EMBED_SIZE', type=int, default=2560,
                    help='The type of the TPU')
parser.add_argument('--TRACE_DIR', type=str, default=f"gs://mattdavidow-maxtext-br/{datetime.now().date()}")
args = parser.parse_args()

print(f"Global Batch Size set to {args.BATCH_SIZE}", flush=True)
print(f"Global Embed Size set to {args.EMBED_SIZE}", flush=True)
print(f"Trace dir of {args.TRACE_DIR}", flush=True)
def gen_data():
  return (3.0 + jax.process_index()) * jax.numpy.ones(
      (args.BATCH_SIZE, args.EMBED_SIZE), dtype=jax.numpy.bfloat16)

def gen_weights(step):
    # key = jax.random.PRNGKey(step)

    # return jnp.arrange
    # return (step + jax.process_index()) * jax.random.uniform(
    #     key,
    #     shape=(args.EMBED_SIZE,args.EMBED_SIZE),
    #     dtype=jax.numpy.bfloat16)

    def create_capped_array(m, n):
        """Creates a JAX array of shape (m, n) with elements 1, 2, ..., m*n, capped at 100."""
        array = step * jnp.arange(1, m * n + 1).reshape(m, n)  # Create array with desired sequence
        array = jnp.minimum(array, 100)  # Cap values at 100
        return array
    return create_capped_array(args.EMBED_SIZE, args.EMBED_SIZE)

# Define array lengths based on device count
num_devices = len(jax.devices())
mesh_shape = [len(jax.devices())]
devices = np.asarray(jax.devices()).reshape(*mesh_shape)
mesh = Mesh(devices, ("x"))
gen_data_sharded = pjit(
    gen_data, in_shardings=None, out_shardings=PartitionSpec("x")
)
gen_weights_sharded = pjit(
    gen_weights, in_shardings=None, out_shardings=PartitionSpec("x")
)

with Mesh(mesh.devices, mesh.axis_names):
  print(f"Starting to create data of global shape [{args.BATCH_SIZE},{args.EMBED_SIZE}]...", flush=True)
  presharded_X = jax.block_until_ready(gen_data_sharded())
  
  print("Global matrix created!!!", flush=True)
  print(f"Creating weights of global shape [{args.EMBED_SIZE}, {args.EMBED_SIZE}]", flush=True)
  weights = jax.block_until_ready(gen_weights_sharded(0))
  print("Weights created", flush=True)


def f(x, step=0):
    weights = jax.block_until_ready(gen_weights_sharded(step))
    return x @ weights

with Mesh(mesh.devices, mesh.axis_names):
    jit_f = jax.jit(f)

    first_f = jit_f(presharded_X)
    jax.profiler.start_trace(args.TRACE_DIR)
    for i in range(3):
        first_f = jit_f(first_f, i)
    jax.profiler.stop_trace()