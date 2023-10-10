from jax.experimental.topologies import get_topology_desc
import jax
from jax.experimental import pjit
from jax.sharding import PartitionSpec as P
import numpy as np
from jax.experimental.serialize_executable import serialize, deserialize_and_load
import pickle
import jax.numpy as jnp
from flax.serialization import to_bytes
import msgpack
from jax import tree_util
import argparse


parser = argparse.ArgumentParser(description='Xaot example options')
parser.add_argument('--save', type=bool, default=True, action=argparse.BooleanOptionalAction)
parser.add_argument('--load', type=bool, default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--run_f', type=bool, default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--version', type=str, default='v4-8')
args = parser.parse_args()


def fun(x):
    return x * x

if args.save:
    ## Use "fake" topology devices to compile f
    if args.version=='v4-8':
        fake_devices = get_topology_desc(
            platform='tpu',
            topology_name=f'v4:2x2x1',
            chip_config_name='megacore',
            chips_per_host_bounds=(2, 2, 1),
            num_slices=1,
        ).devices
    elif args.version=='v4-16':
        fake_devices = get_topology_desc(
        platform='tpu',
        topology_name=f'v4:2x2x2',
        chip_config_name='megacore',
        chips_per_host_bounds=(2, 2, 2),
        num_slices=1,
    ).devices

    # jit, lower, and compile f using fake devices
    with jax.sharding.Mesh(np.array(fake_devices), ('data',)):
        jitted = pjit.pjit(
            fun, in_shardings=P('data'), out_shardings=P(None, 'data')
        )
        lowered = jitted.lower(
            jax.core.ShapedArray(shape=(128, 128), dtype=np.float32)
        )
    orig_compiled = lowered.compile()

    # Serialize and save the compiled object
    serialized, in_tree, out_tree = serialize(orig_compiled)
    with open(f"x_aot_{args.version}.pickle", "wb") as f:
        pickle.dump(serialized, f)

if args.run_f:
    # This is not a sharded array. However we tell pjit above that the input is sharded along the "data" axis.
    # Why does this work?
    ex_input = 2.0 * jnp.ones((128, 128), dtype=jnp.float32)
    if args.load:
        with open(f"x_aot_{args.version}.pickle", "rb") as f:
            serialized_compiled = pickle.load(f)

        # Input shape
        # ex_input_for_shape = jax.core.ShapedArray(shape=(128, 128), dtype=np.float32)
        flat_in_shaped, in_tree_recreated = tree_util.tree_flatten(((ex_input,),{}))

        # Output shape
        out_shaped = jax.eval_shape(fun, ex_input)
        flat_out_shaped, out_tree_recreated = jax.tree_util.tree_flatten(out_shaped)

        compiled = deserialize_and_load(serialized_compiled, in_tree_recreated, out_tree_recreated)

        with jax.sharding.Mesh(np.array(jax.devices()), ('data',)):
            print("Running f after loading...",flush=True)
            cost = compiled.cost_analysis()[0]['flops']
            print(f"{cost=}")
            out = compiled(ex_input)
            print("computed out")
            print(f"{out=}")
            out_gathered = jax.experimental.multihost_utils.process_allgather(out)
            print(f"{out_gathered=}")
            print("Successfully Ran f!",flush=True)
    else:
        with jax.sharding.Mesh(np.array(jax.devices()), ('data',)):
            print("Running the native f",flush=True)
            print(f"{jitted(ex_input)}")
            print("Successfully Ran f!",flush=True)