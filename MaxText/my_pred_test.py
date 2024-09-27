import jax
from jax import numpy as jnp
from functools import partial
from jax import lax

from jax.sharding import NamedSharding, Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map

import numpy as np

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

global mesh
mesh = Mesh(jax.devices(), ('x',))


a_arr = jax.device_put(
    jnp.arange(4 * 4).reshape((4, 4)),
    jax.sharding.NamedSharding(mesh, P('x', None)))

@jax.jit
@partial(
    shard_map, mesh=mesh, in_specs=(P('x', None),), out_specs=P('x', None)
)
def fwd(a_arr):
    axis_size = lax.psum(1, 'x')
    perm = [(j, (j + 1) % axis_size) for j in range(axis_size)]
    return lax.ppermute(a_arr, 'x', perm=perm)
    #return lax.ppermute(a_arr, 'x', perm=perm, pred_mask=[True,True,True,False])

c_arr = fwd(a_arr)

# assert two jax arrays are close
are_close = jnp.isclose(c_arr[1, :], a_arr[0, :], atol=1e-05, rtol=1e-05)
all_close = jnp.all(are_close)

print(f"{a_arr=}")
print(f"{c_arr=}")

def run_permute_scannable(state, xs):
    new_arr = fwd(state['arr'])
    new_iter = state['iter'] + 1
    return {'arr':new_arr,'iter':new_iter}, None

final_arr, _  = jax.lax.scan(run_permute_scannable, {'arr':a_arr,'iter':0},length=2)
print(f"{final_arr=}")


def fwd_dynamic_loop(a_arr):
    for loop_iter in range(2):
        a_arr = fwd_dynamic_perm(a_arr, loop_iter)
    return a_arr

@jax.jit
@partial(
    shard_map, mesh=mesh, in_specs=(P('x', None),P()), out_specs=P('x', None)
)
def fwd_dynamic_perm(a_arr, index):
    axis_size = lax.psum(1, 'x')
    perm = [(j, (j + 1) % axis_size) for j in range(index + 1)]
    #perm = [(j, (j + 1) % axis_size) for j in range(axis_size)]
    return lax.ppermute(a_arr, 'x', perm=perm)

def run_dynamic_perm_scannable(state, xs):
    new_arr = fwd_dynamic_perm(state['arr'], state['iter'])
    new_iter = state['iter'] + 1
    return {'arr':new_arr,'iter':new_iter}, None

# jit_dynamic_perm = jax.jit(fwd_dynamic_loop)
# rawr_output = jit_dynamic_perm(a_arr)
# print(f"{rawr_output=}")

# jit_dynamic_perm = jax.jit(run_dynamic_perm_scannable)
# rawr_output = jit_dynamic_perm({'arr':a_arr,'iter':0}, None)
# print(f"{rawr_output=}")

# final_arr_dp, _  = jax.lax.scan(run_dynamic_perm_scannable, {'arr':a_arr,'iter':0},length=2)
# print(f"{final_arr_dp=}")

global my_axis_size
my_axis_size = 4


def first_n_true(n, k):
  return jnp.arange(k) < n

@jax.jit
@partial(
    shard_map, mesh=mesh, in_specs=(P('x', None),P()), out_specs=P()
)
def test_first_n(a_arr, index):
    axis_size = lax.psum(1, 'x')
    pred_mask = first_n_true(2, axis_size)
    q = jnp.sum(pred_mask)
    return q

out_test_first_n = test_first_n(a_arr, 0)
print(f"{out_test_first_n=}")

@jax.jit
@partial(
    shard_map, mesh=mesh, in_specs=(P('x', None),P()), out_specs=P('x', None)
)
def fwd_dynamic_mask(a_arr, index):
    axis_size = lax.psum(1, 'x')
    perm = [(j, (j + 1) % axis_size) for j in range(axis_size)]
    #pred_mask = [True for _ in range(axis_size)]
    pred_mask = first_n_true(axis_size, axis_size)
    # pred_mask = first_n_true(my_axis_size, my_axis_size)
    #pred_mask = [True, True, False, False]
    #print(f"{pred_mask=}")
    #return lax.ppermute(a_arr, 'x', perm=perm)
    return lax.ppermute(a_arr, 'x', perm=perm, pred_mask=pred_mask)

def run_dynamic_mask_scannable(state, xs):
    new_arr = fwd_dynamic_mask(state['arr'], state['iter'])
    new_iter = state['iter'] + 1
    return {'arr':new_arr,'iter':new_iter}, None

meow = fwd_dynamic_mask(a_arr, 0)
print(f"{meow=}")

# jit_dynamic_mask = jax.jit(run_dynamic_mask_scannable)
# rawr_output = jit_dynamic_mask({'arr':a_arr,'iter':0}, None)
# print(f"{rawr_output=}")

# final_arr_dp, _  = jax.lax.scan(run_dynamic_mask_scannable, {'arr':a_arr,'iter':0},length=2)
# print(f"{final_arr_dp=}")









