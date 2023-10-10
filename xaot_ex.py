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

def make_fake_devices():
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
    return fake_devices

def jit_and_compile(mesh, fun, in_shardings, out_shardings):
    # jit, lower, and compile f using fake devices
    with mesh:
        jitted = pjit.pjit(
            fun, in_shardings=P('data'), out_shardings=P(None, 'data')
        )
        lowered = jitted.lower(
            jax.core.ShapedArray(shape=(128, 128), dtype=np.float32)
        )
    compiled = lowered.compile()
    return jitted, lowered, compiled

def save_compiled(compiled, save_name):
    # Serialize and save the compiled object
    serialized, in_tree, out_tree = serialize(compiled)
    with open(save_name, "wb") as f:
        pickle.dump(serialized, f)

def load_compiled(save_name):
    with open(save_name, "rb") as f:
        serialized_compiled = pickle.load(f)
    return serialized_compiled

def get_io_trees(input_args, input_kwargs):
    _, in_tree_recreated = tree_util.tree_flatten((input_args, input_kwargs))
    out_shaped = jax.eval_shape(fun, *input_args, **input_kwargs)
    _, out_tree_recreated = jax.tree_util.tree_flatten(out_shaped)
    return in_tree_recreated, out_tree_recreated

def run_f(f, input_args, input_kwargs, mesh, print_cost=False):
    with mesh:
        if print_cost:
            cost = f.cost_analysis()[0]['flops']
            print(f"{cost=}")

        out = f(*input_args, **input_kwargs)
        print("computed out")
        try:
            print(f"{out=}")
        except:
            out_gathered = jax.experimental.multihost_utils.process_allgather(out)
            print(f"{out_gathered=}")

# def save_compiled_full(f, compiled_name, in_shardings, out_shardings, mesh_axis_names)

# def load_compiled_full(f, compiled_name, input_args, input_kwargs)

# def run_full(f, input_args, input_kwarts, mesh_axis_names)


#### Start code ####

# Shared between save,  load, and run
compiled_name = f"x_aot_{args.version}.pickle"
mesh_axis_names = ('data',)

if args.save:
    print("Saving the compiled function...", flush=True)
    in_shardings=P('data')
    out_shardings=P(None, 'data')
    fake_devices = make_fake_devices()
    fake_device_mesh = jax.sharding.Mesh(np.array(fake_devices), mesh_axis_names)
    jitted, lowered, compiled = jit_and_compile(fake_device_mesh, fun, in_shardings, out_shardings)
    save_compiled(compiled, compiled_name) # Serialize and save the compiled object
    print("Saved compiled function!", flush=True)

if args.load:
    print("Loading the compiled function...", flush=True)
    serialized_compiled = load_compiled(compiled_name)
    ex_input = 2.0 * jnp.ones((128, 128), dtype=jnp.float32)
    input_args = (ex_input,)
    input_kwargs = {}
    in_tree_recreated, out_tree_recreated = get_io_trees(input_args, input_kwargs)
    compiled = deserialize_and_load(serialized_compiled, in_tree_recreated, out_tree_recreated)
    print("Loaded compiled function!", flush=True)

if args.run_f:
    # This will run with the loaded version of f if args.load is true, else runs with currently jitted version/

    # This is not a sharded array. However we tell pjit above that the input is sharded along the "data" axis.
    # Why does this work?
    ex_input = 2.0 * jnp.ones((128, 128), dtype=jnp.float32)

    mesh = jax.sharding.Mesh(np.array(jax.devices()), mesh_axis_names)
    if args.load:
        f = compiled
    elif args.save:
        f = jitted
    else:
        raise Exception("You must run f either by loading a compiled version or jitting it")
    print("Running f...",flush=True)
    run_f(f, input_args, input_kwargs, mesh, print_cost=args.load)
    print("Successfully Ran f!",flush=True)