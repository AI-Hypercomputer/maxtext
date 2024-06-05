import jax
from jax import numpy as jnp
import timing_util
import os
from jax.ad_checkpoint import checkpoint_name

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"



def my_inner_func(y, x):
    a0 = x
    for _ in range(NUM_INNER_LOOPS):
        x = checkpoint_name(x @ y['wo'], name='wo')
        x = checkpoint_name(x @ y['wi'], name='wi')
    return x
    
def my_outer_func(y, x):
    for _ in range(NUM_OUTER_LOOPS):
        x = my_inner_func(y, x)
    return x

def my_outer_vmap_func(y, x):
    big_x = jax.lax.broadcast(x, [VMAP_DIM])
    def my_broadcast(leaf):
        return jax.lax.broadcast(leaf, [VMAP_DIM])
        
    big_y = jax.tree.map(my_broadcast, y)
    breakpoint()
    vmap_inner = jax.vmap(my_inner_func)
    for _ in range(NUM_OUTER_LOOPS):
        big_x = vmap_inner(big_y, big_x)
    return jnp.sum(big_x, axis=0)
    

def my_loss_func(y,x):
    #x = my_outer_func(y, x)
    x = my_inner_func(y, x)
    #x = my_outer_vmap_func(y, x)
    return jnp.linalg.norm(x - x_targets)

NUM_INNER_LOOPS = 3
NUM_OUTER_LOOPS = 2
VMAP_DIM = 4


BATCH = 2**10
EMBED = 2048
MLP = 4096

key = jax.random.key(0)
key_targets = jax.random.key(0)
x = jax.random.uniform(key, (BATCH, EMBED), dtype=jnp.bfloat16)
x_targets = jax.random.uniform(key_targets, (BATCH, EMBED), dtype=jnp.bfloat16)
wo= jax.random.uniform(key, (EMBED, MLP), dtype=jnp.bfloat16)
wi= jax.random.uniform(key, (MLP, EMBED), dtype=jnp.bfloat16)
y= {'wo': wo, 'wi':wi}



value_func = jax.value_and_grad(my_loss_func)

#policy = jax.checkpoint_policies.save_only_these_names('wi')
policy = jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
my_checkpoint_func = jax.checkpoint(value_func, policy=policy)

jit_func = jax.jit(my_checkpoint_func)

jax.ad_checkpoint.print_saved_residuals(my_checkpoint_func, y, x)
print("\n Now for the jit \n")
jax.ad_checkpoint.print_saved_residuals(jit_func, y, x)

# output = jit_func(y,x)
# print("rawr")
# timing_util.simple_timeit(jit_func, y, x, task="remat_test")
